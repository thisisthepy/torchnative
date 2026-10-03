"""Vulkan coverage accounting -- so a skip can never be read as a pass.

The shape of the defect this exists for (docs/devices/VULKAN5.md §1): a gate
ran on a machine with no Vulkan loader, every Vulkan test printed a
parenthesised "(skipped ...)" and then returned normally, and the suite runner
printed `ok   test_...` for each of them. The gate came out `0 FAIL` and the
four kernels it was meant to check had never been executed. The skip lines were
in the log; nothing counted them.

So a Vulkan skip here is a **third outcome**, not a quiet `ok`:

  * `vulkan_skip()` records why the current test could not run;
  * `run_tests()` prints `SKIP test_x: <reason>` for it instead of `ok`, and
    ends with one machine-readable line --
    `VULKAN: ran=R ok=K failed=F skipped=S device=...`;
  * `summarize()` (run by `run.sh` over every suite's log) adds those lines up
    and prints `VULKAN COVERAGE: R ran / S skipped`, with an **UNVERIFIED**
    banner when nothing ran. `TORCHNATIVE_REQUIRE_VULKAN=1` turns any skip, or
    zero tests run, into a non-zero exit -- the only way a run can be offered as
    evidence that the Vulkan kernels were measured.

What this cannot see: a Vulkan test that never calls `vulkan_skip()` or
`vulkan_used()` is not counted at all, in either column. `test_vulkancov.py`
holds that down for the two suites that have Vulkan tests by checking that
every test reading `_vulkan_probe()` through their skip helper is tallied.
"""

import os
import re
import sys

_state = {"current": None, "used": set(), "skipped": {}, "device": None}


def vulkan_used(device=None):
    """The current test reached a live Vulkan device."""
    if _state["current"] is not None:
        _state["used"].add(_state["current"])
    if device:
        _state["device"] = device


def vulkan_skip(reason):
    """The current test could not run because there is no Vulkan here."""
    name = _state["current"]
    if name is not None:
        _state["used"].add(name)
        _state["skipped"].setdefault(name, reason)


def run_tests(items):
    """Run `(name, fn)` pairs; print ok / SKIP / FAIL; return the failure count.

    The `VULKAN:` line is printed even when no test touched Vulkan, so its
    absence from a log means "this suite does not use the runner", never
    "nothing was skipped".
    """
    failures = 0
    ran = ok = failed = 0
    for name, fn in items:
        _state["current"] = name
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            outcome = "fail"
        else:
            if name in _state["skipped"]:
                print(f"SKIP {name}: {_state['skipped'][name]}")
                outcome = "skip"
            else:
                print(f"ok   {name}")
                outcome = "ok"
        if name in _state["used"] and outcome != "skip":
            ran += 1
            ok += outcome == "ok"
            failed += outcome == "fail"
        _state["current"] = None
    skipped = len(_state["skipped"])
    print(f"VULKAN: ran={ran} ok={ok} failed={failed} skipped={skipped} "
          f"device={_state['device'] or '-'}")
    return failures


_LINE = re.compile(
    r"^VULKAN: ran=(\d+) ok=(\d+) failed=(\d+) skipped=(\d+) device=(.*)$")
_SKIP = re.compile(r"^SKIP (test_\w+): (.*)$")


def parse(text):
    """Every `VULKAN:` line and every `SKIP` line in one suite's output."""
    lines = [tuple(int(g) for g in m.groups()[:4]) + (m.group(5),)
             for m in map(_LINE.match, text.splitlines()) if m]
    skips = [m.groups() for m in map(_SKIP.match, text.splitlines()) if m]
    return lines, skips


def summarize(logs, require=False):
    """`logs` maps a suite name to its captured output. Returns (text, exit)."""
    ran = ok = failed = skipped = 0
    devices, reasons = set(), []
    for suite, text in sorted(logs.items()):
        lines, skips = parse(text)
        for r, k, f, s, dev in lines:
            ran, ok, failed, skipped = ran + r, ok + k, failed + f, skipped + s
            if dev != "-":
                devices.add(dev)
        reasons += [f"{suite}::{name}: {why}" for name, why in skips]
    out = [f"VULKAN COVERAGE: {ran} ran ({ok} ok, {failed} failed) / "
           f"{skipped} skipped -- device: {', '.join(sorted(devices)) or 'none'}"]
    verified = ran > 0 and skipped == 0 and failed == 0
    if ran == 0:
        out.append(
            "VULKAN COVERAGE: UNVERIFIED -- no Vulkan test executed in this run. "
            "A green gate here says nothing about any Vulkan kernel.")
    elif skipped:
        out.append(
            f"VULKAN COVERAGE: PARTIAL -- {skipped} Vulkan test(s) skipped; "
            "those are unverified by this run.")
    for line in reasons[:5]:
        out.append(f"   skipped {line}")
    if len(reasons) > 5:
        out.append(f"   ... and {len(reasons) - 5} more")
    code = 0
    if require and not verified:
        out.append(
            "VULKAN COVERAGE: FAIL -- TORCHNATIVE_REQUIRE_VULKAN=1 was set and "
            "this run did not execute every Vulkan test successfully.")
        code = 1
    return "\n".join(out), code


def main(argv):
    logs = {}
    for path in argv:
        with open(path, encoding="utf-8", errors="replace") as fh:
            logs[os.path.basename(path)] = fh.read()
    text, code = summarize(logs, os.environ.get("TORCHNATIVE_REQUIRE_VULKAN") == "1")
    print(text)
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
