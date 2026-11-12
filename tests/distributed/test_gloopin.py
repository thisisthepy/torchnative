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
  child actually binds, read out of `lsof` on macOS and out of the kernel's
  own socket tables (`/proc/self/net/tcp*`) on Linux, must be `127.0.0.1`.
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

import _skip
import gloo_loopback

HERE = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file()) / "tests"

_CHILD = r'''
import json, os, subprocess, sys
PIN
import torch
import torch.distributed as dist
port = sys.argv[1]
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = port
dist.init_process_group(backend="gloo", rank=0, world_size=1)
LISTENERS_SOURCE
print("LISTENERS>>>" + json.dumps(listeners()))
dist.destroy_process_group()
'''

# Where a child reads its own listening sockets from, per platform.
#
# macOS: `lsof`, at the path the base system installs it. Linux: the kernel's
# own socket tables. The first Linux gate (issue #40) died here on
# `FileNotFoundError: '/usr/sbin/lsof'` -- `ubuntu-24.04` has no lsof at that
# path, so the child never reached the question, and both tests reported a
# rendezvous failure that was really a missing binary. `/proc/self/net/tcp{,6}`
# lists every TCP socket in the namespace with its state and inode, and
# `/proc/self/fd` names the inodes this process owns, so the intersection is
# exactly what `lsof -p <pid> -iTCP -sTCP:LISTEN` reports -- printed in lsof's
# `TCP <addr>:<port> (LISTEN)` shape so the parser below is the same on both.
_LISTENERS_SOURCE = r'''
def _linux_listeners():
    import ipaddress
    owned = set()
    for fd in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink("/proc/self/fd/" + fd)
        except OSError:
            continue
        if target.startswith("socket:["):
            owned.add(target[len("socket:["):-1])
    lines = []
    for table in ("/proc/self/net/tcp", "/proc/self/net/tcp6"):
        try:
            rows = open(table).read().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            cols = row.split()
            if cols[3] != "0A" or cols[9] not in owned:   # 0A is TCP_LISTEN
                continue
            hexaddr, hexport = cols[1].split(":")
            raw = b"".join(bytes.fromhex(hexaddr[i:i + 8])[::-1]
                           for i in range(0, len(hexaddr), 8))
            ip = ipaddress.ip_address(raw)
            if ip.version == 6 and ip.ipv4_mapped is not None:
                ip = ip.ipv4_mapped
            if ip.is_unspecified:
                addr = "*"
            elif ip.version == 6:
                addr = "[%s]" % ip
            else:
                addr = str(ip)
            lines.append("python %d TCP %s:%d (LISTEN)" % (os.getpid(), addr, int(hexport, 16)))
    return "\n".join(lines) + "\n"


def listeners():
    if sys.platform.startswith("linux"):
        return _linux_listeners()
    return subprocess.run(
        ["/usr/sbin/lsof", "-nP", "-a", "-p", str(os.getpid()),
         "-iTCP", "-sTCP:LISTEN"],
        capture_output=True, text=True).stdout
'''
_CHILD = _CHILD.replace("LISTENERS_SOURCE", _LISTENERS_SOURCE)


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
            if a not in ("127.0.0.1", "::1", "[::1]", "*", "localhost")]


_INET = re.compile(r"inet (?:addr:)?(\d+\.\d+\.\d+\.\d+)")


def _read_interface_text():
    """This machine's interface configuration, as text.

    Separated from the parsing so the parsing can be tested with a reader that
    returns nothing -- see
    `test_the_interface_detector_fails_rather_than_reporting_no_addresses`.
    """
    for argv in (["/sbin/ifconfig", "-a"], ["ifconfig", "-a"], ["ip", "-o", "addr"]):
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout
    return ""


def _interface_addresses(reader=_read_interface_text):
    """Every IPv4 address actually configured on one of this host's interfaces.

    **A broken detector must FAIL, not answer "cannot run".** That direction
    matters more than it looks: this function gates a nullification, and a
    detector that quietly returned an empty set would make the gate *always*
    take the cannot-run branch -- an assertion silently switched off forever,
    which is the "verification that cannot fail" shape AGENTS.md §17.5 records
    and which this repository has been bitten by before.

    So the check is a fact about every machine rather than about this one:
    loopback is always configured. If `127.0.0.1` is not among what was
    parsed, the read or the parse did not work, and that is an error and not
    an answer.
    """
    found = set(_INET.findall(reader()))
    if "127.0.0.1" not in found:
        raise AssertionError(
            "could not read this machine's interface addresses: the parse "
            f"produced {sorted(found)}, which does not include 127.0.0.1. "
            "Every host has loopback, so this is a broken detector rather "
            "than a host without local addresses -- failing instead of "
            "reporting 'the nullification cannot run', which would switch "
            "the assertion below off silently and permanently")
    return found


def _hostname_resolution():
    """`(hostname, [(address, family, sockaddr)] in resolver order, error)`.

    **Resolver order, both families, not a sorted IPv4 set** (#132). gloo's
    `lookupAddrForHostname` (`transport/uv/device.cc`) asks `getaddrinfo` with
    `AF_UNSPEC` / `SOCK_STREAM` and binds the *first* result that binds. On
    this Mac (2026-11-13) the name resolves to `10.9.8.15` and `10.9.8.33`,
    both on its interfaces, but the resolver lists `::1` first, so the unpinned
    child binds loopback. The earlier sorted IPv4 set threw that order away and
    said the comparison could run, and the test went red on the host instead
    of on the pin.

    An error is NOT the cannot-run condition: a name that does not resolve is
    exactly the dependency this file is about, and the unpinned child fails
    loudly on it (`Unable to find address for:`), which the first arm of the
    nullification already asserts.
    """
    import socket
    host = socket.gethostname()
    try:
        infos = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except OSError as exc:
        return host, [], exc
    ordered, seen = [], set()
    for family, _type, _proto, _name, sockaddr in infos:
        if sockaddr[0] in seen:
            continue
        seen.add(sockaddr[0])
        ordered.append((sockaddr[0], family, sockaddr))
    return host, ordered, None


def _binds(family, sockaddr):
    """Does a TCP socket bind here? The same test gloo applies to each result."""
    import socket
    try:
        sock = socket.socket(family, socket.SOCK_STREAM)
    except OSError:
        return False
    try:
        sock.bind((sockaddr[0], 0) + tuple(sockaddr[2:]))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _is_loopback(address):
    import ipaddress
    try:
        return ipaddress.ip_address(address.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _nullification_can_run(ordered, bindable):
    """The precondition, as a pure function of the resolver's answer.

    `ordered` is the resolver's addresses in its own order; `bindable(address)`
    says whether one binds on this machine. Returns `[address]`, the address
    gloo would bind without the pin, when that is a non-loopback address, and
    `[]` when the comparison cannot distinguish anything.

    Pure, and separate from the test, for one reason: on a host where the
    resolver answers a non-local address only ONE of its branches ever runs,
    so the other would be unexercised code deciding whether an assertion
    happens. `test_the_precondition_decides_both_ways_on_synthetic_inputs`
    drives both here.

    Loopback does not count, and neither does an address after it. If the
    first address that binds is loopback, the unpinned child binds loopback for
    that reason rather than because of the pin, and the comparison
    distinguishes nothing just as surely as when the answer is off-machine.
    """
    for address in ordered:
        if bindable(address):
            return [] if _is_loopback(address) else [address]
    return []


def test_the_precondition_decides_both_ways_on_synthetic_inputs():
    """Both branches of the gate, on a host that only ever takes one.

    The cannot-run branch is the one that switches an assertion off, so it is
    the one that has to be shown to be narrow: an address on this machine
    must make the nullification run.
    """
    machine = {"127.0.0.1", "::1", "10.9.8.15", "10.9.8.33"}
    cases = [
        (["218.38.137.27"], [], "an ISP wildcard answer -- not on this machine"),
        ([], [], "no resolution at all"),
        (["127.0.0.1"], [], "loopback only -- indistinguishable from the pin"),
        (["10.9.8.15"], ["10.9.8.15"], "a real local address -- the comparison works"),
        (["218.38.137.27", "10.9.8.33"], ["10.9.8.33"],
         "one of each: the first local answer is the one gloo binds"),
        (["::1", "fe80::1", "127.0.0.1", "10.9.8.33", "10.9.8.15"], [],
         "this Mac, 2026-11-13 (#132): ::1 binds first, so gloo binds loopback"),
        (["10.9.8.33", "127.0.0.1"], ["10.9.8.33"],
         "loopback after a local address does not matter: the first one binds"),
    ]
    for resolved, expected, why in cases:
        got = _nullification_can_run(resolved, machine.__contains__)
        assert got == expected, f"{why}: {resolved} -> {got}, expected {expected}"
    print(f"   precondition decides {len(cases)} synthetic cases, both ways")


def test_the_interface_detector_fails_rather_than_reporting_no_addresses():
    """The detector that gates the nullification is itself gated.

    Two halves, and the second is the one that matters:

    * on this machine it reads real addresses, loopback among them;
    * a reader that returns nothing -- a missing `ifconfig`, a changed output
      format, a permissions failure -- makes it **raise**, not return an empty
      set. If it returned one, every future run would take the cannot-run
      branch and the nullification would be off with nothing saying so.
    """
    real = _interface_addresses()
    assert "127.0.0.1" in real, real
    for broken, what in ((lambda: "", "no output at all"),
                         (lambda: "flags=8863<UP,BROADCAST>\n", "no inet lines"),
                         (lambda: "inet 10.9.8.15 netmask 0xffffff00\n",
                          "addresses but no loopback")):
        try:
            _interface_addresses(broken)
        except AssertionError:
            continue
        raise AssertionError(
            f"the detector answered instead of failing when the reader gave {what} "
            "-- a silent cannot-run branch")
    print(f"   detector reads {len(real)} address(es) and refuses three broken readers")


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

    **The comparison has a precondition, and it is measured rather than
    assumed.** The unpinned child can only bind a routable address if this
    host's own name resolves to one that is *actually on one of its
    interfaces*. That was true when this file was written -- the docstring at
    the top records `10.9.8.x` -- and it is not always true: an ISP wildcard
    answering the `.local` lookup returns an address that is not on this
    machine, gloo refuses to treat it as local, and falls back to loopback on
    its own, with

        [W ProcessGroupGloo.cpp:555] Unable to resolve hostname to a (local)
        address. Using the loopback address as fallback.

    Both arms then bind loopback and the comparison cannot distinguish them.

    The same happens when the name resolves to an on-machine address but the
    resolver lists a loopback address first (#132: `::1` first on this Mac),
    because gloo binds the first result that binds.

    **This is a conditioned assertion; the cannot-run branch is a named SKIP**
    (since #132; before that it printed and counted as ok), and:

    * the condition is *measured* -- `getaddrinfo` in resolver order, each
      result bound the way gloo binds it -- and *reported*, naming the resolved
      addresses and the first that binds, so a reader of the gate output can
      see exactly why the comparison did not run;
    * the detector that decides it **fails rather than answering** when it
      cannot read the interfaces at all
      (`test_the_interface_detector_fails_rather_than_reporting_no_addresses`),
      so this cannot become a permanently-off assertion;
    * on a host where the comparison *is* possible, the assertion is
      unchanged and still fails if the pin stops being observable. Nothing
      was widened: "removing the pin changed nothing observable" remains a
      failure wherever it is a statement about the pin rather than about the
      resolver.
    """
    rc, out, listeners = _run_child("")
    if rc != 0:
        assert "Unable to find address for" in out, (
            "the unpinned child failed for some other reason, so this case "
            f"is not measuring the pin:\n{out}")
        print("   nullification RAN: the unpinned child failed on name "
              "resolution, which is the dependency")
        return

    interfaces = _interface_addresses()      # raises if the detector is broken
    host, ordered, err = _hostname_resolution()
    bindable = {a: _binds(fam, sa) for a, fam, sa in ordered}
    resolved = [a for a, _fam, _sa in ordered]
    local = _nullification_can_run(resolved, bindable.__getitem__)
    if not local:
        first = next((a for a in resolved if bindable[a]), None)
        _skip.skip(
            f"nullification cannot run on this host: {host!r} resolves, in "
            f"resolver order, to "
            f"{resolved or ('<' + type(err).__name__ + '>' if err else '<nothing>')}; "
            f"the first that binds is {first or '<none>'}, which is loopback "
            f"or absent (IPv4 interfaces: {sorted(interfaces)}). gloo binds "
            f"that same address without the pin, so both arms bind loopback "
            f"and the comparison distinguishes nothing (#132). The pin is not "
            f"disproved; it is unobservable here.")
        return

    routable = _routable(_bound_addresses(listeners))
    assert routable, (
        "removing the pin changed nothing observable -- the unpinned child "
        "bound only loopback too, so the pinned run above proves nothing "
        f"and these tests are worthless as written:\n{out}")
    print(f"   nullification RAN: {host!r} resolves locally to {local}, and the "
          f"unpinned child bound {routable}")


def test_both_gloo_suites_pin_their_children():
    """The pin has to be in the children that actually run in the gate.

    `test_collect2.py` and `test_asyncwork.py` are the two files that spawn
    upstream gloo ranks; the source they hand to `python -c` is where the pin
    has to appear, because those children get no PYTHONPATH and cannot import
    the helper.
    """
    for name in ("test_collect2.py", "test_asyncwork.py"):
        text = (HERE / "distributed" / name).read_text()
        assert "gloo_loopback" in text, (
            f"{name} spawns upstream gloo ranks without pinning the device "
            "to loopback -- one mDNS hiccup fails it for a reason that has "
            "nothing to do with the collective")


def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        _skip._state["current"] = name
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            if name in _skip._state["skipped"]:
                print(f"SKIP test_gloopin: {name} -- {_skip._state['skipped'][name]}")
            else:
                print(f"ok   {name}")
        _skip._state["current"] = None
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
