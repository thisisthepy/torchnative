"""Pin gloo's device to loopback, so a local collective stays local.

`MASTER_ADDR=127.0.0.1` is not enough, and the difference is where a gate run
went wrong on 2026-09-17::

    collect2/gloo32-3: rank 0 exited 1 at world 3
    RuntimeError: [enforce fail at .../gloo/transport/uv/device.cc:154]
    rp != nullptr.  Unable to find address for: irackui-Macmini.local

`MASTER_ADDR` addresses the **rendezvous store**. The gloo *device* -- the
thing the ranks actually talk over -- is built separately, and with no
`GLOO_SOCKET_IFNAME` in the environment it is built from `gethostname()`. So
several ranks of the same machine, rendezvousing over loopback, went out to
mDNS for this host's own `.local` name to find each other. Measured minutes
after the failure, that name is ambiguous on this host: `ping` answers
10.9.8.33 while `getaddrinfo` answers 10.9.8.15 -- the name drifts across
interfaces, and when it lands on one gloo cannot reach, a test about
`all_reduce` fails with a network error.

Pinning the interface removes host name resolution from the path entirely.
It is not a skip and not a retry: a hostname that genuinely has to be resolved
still fails loudly, and every collective still runs and still has to agree
with upstream. What changes is that ranks on one machine stop asking the
network who this machine is.

The pin is delivered as **source text**, not as an environment entry, because
the ranks that need it are `python -c` children spawned with `PYTHONPATH`
deliberately removed (they must load upstream torch, never the vendored shim),
so they cannot import this module. Setting `os.environ[...]` inside the child
before `import torch.distributed` reaches gloo's `getenv` at device-creation
time just as an inherited variable would; `test_gloopin.py` measures that it
does, rather than assuming it.

`test_gloopin.py` also holds down the other half -- that `GLOO_SOCKET_IFNAME`
is the lever this particular gloo build obeys (a bogus interface name must be
refused, at `device.cc:165`, which is a different line from the hostname
failure at 154), and that removing the pin puts the host's name back in the
loop. Without that nullification the pin would be a comment.
"""

import sys

#: The loopback interface's name, which is not the same everywhere: macOS
#: calls it `lo0` and Linux calls it `lo`. Getting this wrong does not fail
#: quietly -- gloo refuses an unknown interface by name -- but there is no
#: reason to make anyone find that out.
IFNAME = "lo0" if sys.platform == "darwin" else "lo"


def child_pin_source(ifname=IFNAME):
    """Python source that pins gloo to loopback, for a `python -c` child.

    Must be placed **before** the child imports `torch.distributed` and
    initialises its process group: gloo reads the variable when it creates
    the device.
    """
    return (
        'import os as _os\n'
        '# Several ranks of one machine: gloo must not resolve this host\'s\n'
        '# name to find them. MASTER_ADDR only addresses the store.\n'
        '_os.environ["GLOO_SOCKET_IFNAME"] = "%s"\n' % ifname
    )
