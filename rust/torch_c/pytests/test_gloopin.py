"""The gloo oracle must measure the collective, not this machine's mDNS.

`test_all_five_reduce_ops_agree_with_upstream_and_the_bitwise_three_refuse`
failed in the gate with::

    collect2/gloo32-3: rank 0 exited 1 at world 3
    RuntimeError: [enforce fail at .../gloo/transport/uv/device.cc:154]
    rp != nullptr.  Unable to find address for: irackui-Macmini.local

Nothing in that test is about the network. Every rank is a subprocess on this
one machine, rendezvousing through `MASTER_ADDR=127.0.0.1`. But `MASTER_ADDR`
only addresses the **store**; gloo builds its *device* separately, and with no
`GLOO_SOCKET_IFNAME` set it does so from `gethostname()` -- so the run went out
to mDNS for this host's own `.local` name. Measured right after the failure,
that name is ambiguous here: `ping` answers 10.9.8.33 while `getaddrinfo`
answers 10.9.8.15. A gate that fails when the machine's own multicast name
drifts across interfaces is reporting on the host, not on the code -- the same
class of defect as the gate losing a suite log, in a different instrument.

So the gloo children pin the device to the loopback interface
(`gloo_loopback.child_pin_source()`), and this file holds that down with
measurements rather than with the claim:

* `test_the_interface_pin_is_what_this_gloo_build_obeys` -- the pin is
  *verified to take effect*, not assumed: a deliberately bogus interface name
  must make device creation fail naming that interface. This gloo is
  `transport/uv`, and the two failures come from different lines of
  `device.cc` (165 for an interface, 154 for a hostname), which is the
  evidence that the interface path is the one being taken.
* `test_a_pinned_child_binds_only_the_loopback_address` -- what the pinned
  child actually binds, read out of `lsof`, must be `127.0.0.1`.
* `test_removing_the_pin_restores_the_dependency_on_host_name_resolution` --
  the nullification. The identical child with the pin removed binds
  `10.9.8.x`, an address that only exists because the host name was resolved.
  If this ever stops showing a difference, the pin above is proving nothing
  and the first two tests are worthless.

Not a skip and not a retry. A hostname that cannot be resolved still fails
loudly; what changed is that a rendezvous between processes on one machine no
longer asks for one.

What this file does NOT claim: that the pin explains the 600-second collective
hang seen earlier in the week. That hang predates the per-rank reporting that
produced the one-line cause above, so there is no recorded evidence tying it to
name resolution, and none of the measurements here reproduce a hang. It stays
open.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import gloo_loopback

HERE = Path(__file__).resolve().parent

_CHILD = r'''
import json, os, subprocess, sys
PIN
import torch
import torch.distributed as dist
port = sys.argv[1]
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = port
dist.init_process_group(backend="gloo", rank=0, world_size=1)
lsof = subprocess.run(
    ["/usr/sbin/lsof", "-nP", "-a", "-p", str(os.getpid()),
     "-iTCP", "-sTCP:LISTEN"],
    capture_output=True, text=True)
print("LISTENERS>>>" + json.dumps(lsof.stdout))
dist.destroy_process_group()
'''


def _free_port():
    import socket
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _run_child(pin_src):
    """Init a one-rank gloo group in a child and report what it listens on."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)          # the oracle, never the shim
    env.pop("GLOO_SOCKET_IFNAME", None)  # the pin must come from the source
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD.replace("PIN", pin_src), str(_free_port())],
        capture_output=True, text=True, env=env, timeout=600)
    out = proc.stdout + proc.stderr
    listeners = []
    for line in out.splitlines():
        if line.startswith("LISTENERS>>>"):
            listeners = json._default_decoder.decode(
                line[len("LISTENERS>>>"):]).splitlines()
    return proc.returncode, out, listeners


import json  # noqa: E402  (used by _run_child above; kept next to its use)

_ADDR = re.compile(r"TCP (\S+):(\d+) \(LISTEN\)")


def _bound_addresses(listeners):
    return [m.group(1) for m in map(_ADDR.search, listeners) if m]


def _routable(addresses):
    """Addresses that are neither loopback nor the wildcard.

    The wildcard (`*`) is the rendezvous TCPStore, which listens on every
    interface by construction in both arms of this comparison and so cannot
    distinguish them. A concrete non-loopback address, by contrast, can only
    have come from resolving this host's name.
    """
    return [a for a in addresses
            if a not in ("127.0.0.1", "::1", "*", "localhost")]


def test_the_interface_pin_is_what_this_gloo_build_obeys():
    """Verify the lever, rather than assuming the documented one is wired.

    A bogus interface name must be *refused*. If it were ignored, gloo would
    fall back to the hostname, the child would succeed, and every other test
    here would be asserting something about a variable nothing reads.
    """
    rc, out, _ = _run_child(
        'os.environ["GLOO_SOCKET_IFNAME"] = "tn_no_such_if0"')
    assert rc != 0, (
        "gloo ignored GLOO_SOCKET_IFNAME: a nonexistent interface was "
        f"accepted, so the pin is not the lever for this build:\n{out}")
    assert "tn_no_such_if0" in out, (
        "the failure does not name the interface that was asked for, so it "
        f"is not the interface path that refused:\n{out}")
    assert "device.cc" in out, out


def test_a_pinned_child_binds_only_the_loopback_address():
    rc, out, listeners = _run_child(gloo_loopback.child_pin_source())
    assert rc == 0, f"the pinned child failed to rendezvous:\n{out}"
    addresses = _bound_addresses(listeners)
    assert addresses, f"no listening socket was observed at all:\n{out}"
    assert not _routable(addresses), (
        "a pinned gloo device bound a routable address "
        f"{_routable(addresses)} -- it went through host name resolution "
        f"after all:\n{out}")
    assert "127.0.0.1" in addresses, (
        f"the pinned device did not bind loopback: {addresses}\n{out}")


def test_removing_the_pin_restores_the_dependency_on_host_name_resolution():
    """The nullification: without the pin, the host's name is back in the loop.

    Either the child fails the way the gate did (`Unable to find address for:
    <host>.local`), or it binds the address that name resolved to. Both are
    the dependency; neither is possible with the pin in place.
    """
    rc, out, listeners = _run_child("")
    if rc != 0:
        assert "Unable to find address for" in out, (
            "the unpinned child failed for some other reason, so this case "
            f"is not measuring the pin:\n{out}")
        return
    routable = _routable(_bound_addresses(listeners))
    assert routable, (
        "removing the pin changed nothing observable -- the unpinned child "
        "bound only loopback too, so the pinned run above proves nothing "
        f"and these tests are worthless as written:\n{out}")


def test_both_gloo_suites_pin_their_children():
    """The pin has to be in the children that actually run in the gate.

    `test_collect2.py` and `test_asyncwork.py` are the two files that spawn
    upstream gloo ranks; the source they hand to `python -c` is where the pin
    has to appear, because those children get no PYTHONPATH and cannot import
    the helper.
    """
    for name in ("test_collect2.py", "test_asyncwork.py"):
        text = (HERE / name).read_text()
        assert "gloo_loopback" in text, (
            f"{name} spawns upstream gloo ranks without pinning the device "
            "to loopback -- one mDNS hiccup fails it for a reason that has "
            "nothing to do with the collective")


def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
