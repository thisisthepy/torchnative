#!/usr/bin/env python3
"""Re-derive the (dtype x device) matrix in `docs/devices/matrix.md`.

Not a test -- `run.sh` runs `test_*.py` and this is deliberately not one. It is
a sweep, in the shape of `agree_sweep.py` and `decomp_sweep.py`: it produces the
numbers a document quotes, so that the document's cells can be re-measured
instead of re-asserted.

    PYTHONPATH=<stage>:<pytests> python3 dtype_device_matrix.py --drive \\
        --json /tmp/matrix.json --markdown /tmp/matrix.md

**Why this exists rather than the harness that produced the first table.** The
round that built the original matrix captured each op's operands through a
proxy and then placed the *tensor* arguments on the target device. An op with
no tensor arguments -- every factory: `ones`, `full`, `eye`, `arange`,
`linspace`, `scalar_tensor`, `empty`, `hann_window`, `kaiser_window` -- has
nothing to place, so it ran **on the CPU in all sixteen columns** and its CPU
result was recorded under `mps`. That is why those rows read `AGREES` even at
`float64_mps`, a cell measured here to raise `unsupported const-set f64`. A
correct answer computed somewhere other than where the label says is the exact
failure this namespace exists to prevent (CLAUDE.md §4), so this harness reads
each op's **schema** and injects `device=` and `dtype=` whenever the op accepts
them, and records `n/a` -- never a verdict -- when it cannot place the cell on
the device at all.

**The four verdicts, defined so that none of them can absorb another.**

    AGREES    both sides computed and every element matches within
              tools/golden/dtypes.py's tolerance for that dtype. The only
              verdict that is a claim about numbers.
    REACHES   the shim computed and the claim stops there -- either upstream
              refused (so there is no oracle) or both computed and the values
              DISAGREE. The second kind is counted separately and listed by
              name in the document; it is never reported as AGREES.
    REFUSES   the shim raised a refusal. Split in the counts into those that
              name the dtype/device/reason and those that hand back a candle
              symbol, because CLAUDE.md §6 makes only the first kind acceptable.
    BREAKS    anything else: a panic, a hard crash, or a harness that could not
              build the cell. A BREAKS is a statement about this harness as
              much as about the shim, and is never read as a refusal.

**What this harness structurally cannot see**, stated because a matrix invites
the reader to believe it is exhaustive:

  * one shape per op -- the first case `tools/golden/cases.py` builds. A bug
    that needs broadcasting, a non-contiguous input or an empty tensor is
    invisible here.
  * the oracle is asked on the **cpu** even for `mps` cells, following
    `test_dtypedev.py`: the question is whether the shim's number is upstream's
    number, not whether upstream would produce it on that device.
  * `cuda`, `vulkan` and `npu` are not columns. This machine has none of them,
    and a column of "untested" would read as a column of results.
  * casting every tensor operand to the cell's dtype casts *index* operands
    too, so rows like `index_select` and `gather` report on an input upstream
    would reject. Those are marked BREAKS and are a limit of the sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "tools", "golden"))

DTYPES = ("float32", "float16", "bfloat16", "float64",
          "int64", "int32", "int8", "bool")
DEVICES = ("cpu", "mps")

# Message fragments that mean the shim refused in its own vocabulary rather
# than handing back an internal symbol. Kept as data so the document can quote
# the split and a reader can check it.
_NAMED = ("not implemented in torch._C shim", "not implemented for the",
          "MPS framework", "doesn't support", "not implemented for '",
          "shim has no", "refuses")
_SYMBOLS = ("const-set", "contiguous to_dtype", "Error while loading function",
            "mlx matmul", "candle:")
_REFUSALish = ("not implemented", "unsupported", "not supported",
               "cannot be converted", "no kernel", "doesn't support")


def _cell_worker(ops, out_path):
    """Measure every (dtype, device) cell for `ops`, in this process."""
    os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")
    import _C
    import torch
    import cases
    import dtypes as dt_utils
    from loader import resolve_torch_overload
    # The comparison is `tools/golden/compare.py`'s, not a second
    # implementation of it. The first version here was hand-rolled and got
    # infinities wrong -- `inf - inf` is NaN, so `abs(a - b) <= atol` is false
    # for two values that are both `inf`, and `erfinv` on a bool input was
    # reported as disagreeing with upstream while printing the identical list
    # on both sides. A harness that invents disagreements is worse than no
    # harness, and the repository already owns one correct answer to
    # "are these two results the same".
    from compare import _values_close

    def flatten(x, mod):
        """Values as a flat list of floats, read on the host."""
        if hasattr(x, "tolist") and hasattr(x, "device"):
            h = x.cpu() if str(x.device) != "cpu" else x
            h = h.to(mod.float64)
            out = h.flatten().tolist()
            return [float(v) for v in (out if isinstance(out, list) else [out])]
        if isinstance(x, (list, tuple)):
            acc = []
            for v in x:
                acc.extend(flatten(v, mod))
            return acc
        if isinstance(x, bool):
            return [float(x)]
        if isinstance(x, (int, float)):
            return [float(x)]
        return []

    def result_dtype_name(res, fallback):
        """The dtype of the RESULT, which is what the tolerance belongs to.

        `cos` on an int64 input returns float32 on both sides. Looking the
        tolerance up by the *cell's* dtype asks for int64's -- exact, atol and
        rtol both zero -- and that reported thirty last-ulp float32
        differences (torch=1.0 vs shim=0.9999999403953552, |diff|=5.9e-08) as
        divergence from upstream. compare.py reads the result's dtype, and so
        does this.
        """
        d = getattr(res, "dtype", None)
        if d is None and isinstance(res, (list, tuple)) and res:
            d = getattr(res[0], "dtype", None)
        return dt_utils.dtype_name(d) if d is not None else fallback

    class Proxy:
        def __init__(self):
            self.args = self.kwargs = None

        def __call__(self, *args, **kwargs):
            self.args, self.kwargs = args, kwargs
            return None

    def place_torch(x, name, device):
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dt_utils.torch_dtype(torch, name), device=device)
        if isinstance(x, torch.dtype):
            return dt_utils.torch_dtype(torch, name)
        if isinstance(x, (list, tuple)):
            return type(x)(place_torch(v, name, device) for v in x)
        return x

    def place_c(x, name, device):
        if isinstance(x, torch.Tensor):
            h = x.to(dtype=torch.float64).flatten().tolist()
            return _C._tensor_from_flat(
                [float(v) for v in h], list(x.shape),
                dt_utils.c_dtype(_C, name), _C.device(device))
        if isinstance(x, torch.dtype):
            return dt_utils.c_dtype(_C, name)
        if isinstance(x, (list, tuple)):
            return type(x)(place_c(v, name, device) for v in x)
        return x

    def classify_refusal(msg):
        low = msg.lower()
        if not any(f in low for f in _REFUSALish):
            return None
        return "symbol" if any(s in msg for s in _SYMBOLS) else "named"

    results = {}
    for op in ops:
        results[op] = {}
        builder = cases.CASE_BUILDERS.get(op)
        try:
            real_call = resolve_torch_overload(torch, op)
            schema_args = {a.name for a in real_call._schema.arguments}
        except Exception:
            builder = None
            schema_args = set()
        captured = None
        if builder is not None:
            proxy = Proxy()
            try:
                built = builder(torch, _C, proxy)
                if built:
                    built[0].run_torch()
                    captured = (proxy.args or (), dict(proxy.kwargs or {}))
            except Exception:
                captured = None
        for device in DEVICES:
            for name in DTYPES:
                key = "%s_%s" % (name, device)
                if captured is None:
                    results[op][key] = {"verdict": "BREAKS",
                                        "why": "harness could not build the cell"}
                    continue
                t_args, t_kwargs = captured
                has_tensor = any(
                    isinstance(a, torch.Tensor) for a in list(t_args) + list(t_kwargs.values()))
                placed = "device" in schema_args or has_tensor
                if not placed and device != "cpu":
                    # Never a verdict: this op takes neither a tensor nor a
                    # device, so there is no such thing as its mps cell. The
                    # original table recorded the cpu answer here.
                    results[op][key] = {"verdict": "n/a",
                                        "why": "op takes no tensor and no device"}
                    continue
                try:
                    ta = place_torch(list(t_args), name, "cpu")
                    tk = {k: place_torch(v, name, "cpu") for k, v in t_kwargs.items()}
                    ca = place_c(list(t_args), name, device)
                    ck = {k: place_c(v, name, device) for k, v in t_kwargs.items()}
                    if "device" in schema_args:
                        tk["device"] = "cpu"
                        ck["device"] = _C.device(device)
                    # The dtype column has to be expressed somehow, and which
                    # way round depends on the op. A tensor op expresses it
                    # through the dtype of its operands, and adding an
                    # output-dtype kwarg on top would ask a different question.
                    # A factory has no operands, so the kwarg is the only way
                    # to ask at all -- and omitting it is why the original
                    # table's factory rows are identical across all eight dtype
                    # columns, exactly as omitting `device=` made them
                    # identical across both device halves.
                    if "dtype" in schema_args and (not has_tensor or "dtype" in t_kwargs):
                        tk["dtype"] = dt_utils.torch_dtype(torch, name)
                        ck["dtype"] = dt_utils.c_dtype(_C, name)
                except BaseException as e:  # noqa: BLE001
                    msg = str(e).splitlines()[0] if str(e) else type(e).__name__
                    kind = classify_refusal(msg)
                    # Recorded as having happened while *placing the operands*,
                    # not while running the op. Both are real facts about the
                    # cell -- a dtype that cannot exist on a device is a
                    # refusal of that cell -- but they are different facts, and
                    # a harness that merged them would report "this operator
                    # refuses on mps" for an operator it never called.
                    results[op][key] = (
                        {"verdict": "REFUSES", "kind": kind, "stage": "operands",
                         "why": msg}
                        if kind else
                        {"verdict": "BREAKS", "stage": "operands",
                         "why": "operands: " + msg})
                    continue
                t_res = t_exc = c_res = c_exc = None
                try:
                    t_res = real_call(*ta, **tk)
                except BaseException as e:  # noqa: BLE001
                    t_exc = e
                try:
                    c_res = _C._aten_dispatch(op, *ca, **ck)
                except BaseException as e:  # noqa: BLE001
                    c_exc = e
                if c_exc is not None:
                    msg = str(c_exc).splitlines()[0] if str(c_exc) else type(c_exc).__name__
                    kind = classify_refusal(msg)
                    results[op][key] = (
                        {"verdict": "REFUSES", "kind": kind, "stage": "op",
                         "why": msg}
                        if kind else
                        {"verdict": "BREAKS", "stage": "op", "why": msg})
                    continue
                # The shim computed. Did it compute on the device it was asked
                # for? A cpu answer under an mps label is the failure that
                # matters most here, so it is checked rather than assumed.
                where = str(getattr(c_res, "device", device))
                if device != "cpu" and hasattr(c_res, "device") and not where.startswith(device):
                    results[op][key] = {"verdict": "BREAKS",
                                        "why": "answered on %s under a %s label" % (where, device)}
                    continue
                if t_exc is not None:
                    results[op][key] = {"verdict": "REACHES",
                                        "why": "upstream refused: %s"
                                               % str(t_exc).splitlines()[0][:120]}
                    continue
                try:
                    tv, cv = flatten(t_res, torch), flatten(c_res, _C)
                    tol = dt_utils.tolerance_for(result_dtype_name(t_res, name))
                    ok, detail = _values_close(tv, cv, tol.atol, tol.rtol)
                    results[op][key] = (
                        {"verdict": "AGREES"} if ok else
                        {"verdict": "REACHES", "disagrees": True,
                         "why": "values differ (%s): shim=%s upstream=%s"
                                % (detail, cv[:4], tv[:4])})
                except BaseException as e:  # noqa: BLE001
                    results[op][key] = {"verdict": "BREAKS",
                                        "why": "comparison: %s" % e}
    with open(out_path, "w") as fh:
        json.dump(results, fh)


def _drive(args):
    """Run the sweep in chunks, each in its own process.

    Metal can take the whole interpreter down with a driver-level assertion, so
    a chunk that dies is retried one op at a time and only the op that actually
    killed it is recorded as a crash. Results accumulate in the json file, so
    an interrupted sweep resumes.
    """
    os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")
    import _C
    ops = sorted(_C._aten_implemented())
    done = {}
    if os.path.exists(args.json):
        with open(args.json) as fh:
            done = json.load(fh)
    todo = [o for o in ops if o not in done]
    print("%d ops, %d already measured, %d to go" % (len(ops), len(done), len(todo)))
    chunk = args.chunk
    i = 0
    while i < len(todo):
        batch = todo[i:i + chunk]
        i += chunk
        ok = _run_batch(batch, done, args)
        if not ok and len(batch) > 1:
            for one in batch:
                _run_batch([one], done, args, crash_ok=True)
        with open(args.json, "w") as fh:
            json.dump(done, fh, indent=1)
        print("  %d/%d" % (len(done), len(ops)))
    with open(args.json, "w") as fh:
        json.dump(done, fh, indent=1)
    if args.markdown:
        _write_markdown(done, args.markdown)
    return 0


def _run_batch(batch, done, args, crash_ok=False):
    tmp = args.json + ".part"
    try:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker",
             "--ops", ",".join(batch), "--json", tmp],
            capture_output=True, text=True, timeout=args.timeout)
        failed = proc.returncode != 0 or not os.path.exists(tmp)
        detail = (proc.stderr.strip().splitlines()[-1][:120]
                  if proc.stderr.strip() else "exit %d" % proc.returncode)
    except subprocess.TimeoutExpired:
        # A cell that hangs is a fact about that cell, not a reason to lose the
        # whole sweep. It is retried alone and then recorded as BREAKS with the
        # timeout named -- dropping it silently would leave a hole in the table
        # that reads exactly like a cell nobody thought to measure.
        failed, detail = True, "did not finish within %ds" % args.timeout
    if failed:
        if crash_ok:
            why = "crashed or hung the interpreter (%s)" % detail
            for op in batch:
                done[op] = {"%s_%s" % (d, dev): {"verdict": "BREAKS", "why": why}
                            for dev in DEVICES for d in DTYPES}
        return False
    with open(tmp) as fh:
        done.update(json.load(fh))
    os.remove(tmp)
    return True


def _write_markdown(results, path):
    cols = ["%s_%s" % (d, dev) for dev in DEVICES for d in DTYPES]
    lines = ["| Operation | " + " | ".join(cols) + " |",
             "|---|" + "---|" * len(cols)]
    for op in sorted(results):
        row = [results[op].get(c, {}).get("verdict", "BREAKS") for c in cols]
        lines.append("| %s | %s |" % (op, " | ".join(row)))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def tally(results):
    """Counts the document quotes, derived from the json rather than typed."""
    out = {"AGREES": 0, "REACHES": 0, "REFUSES": 0, "BREAKS": 0, "n/a": 0,
           "refuses_named": 0, "refuses_symbol": 0, "refuses_unclassified": 0,
           "disagrees": 0, "cells": 0, "ops": len(results)}
    for op in results:
        for key, cell in results[op].items():
            out["cells"] += 1
            out[cell["verdict"]] = out.get(cell["verdict"], 0) + 1
            if cell["verdict"] == "REFUSES":
                kind = cell.get("kind")
                out["refuses_" + (kind if kind else "unclassified")] += 1
            if cell.get("disagrees"):
                out["disagrees"] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--drive", action="store_true")
    ap.add_argument("--ops", default="")
    ap.add_argument("--json", default="/tmp/dtype_device_matrix.json")
    ap.add_argument("--markdown", default="")
    ap.add_argument("--chunk", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--tally", action="store_true")
    args = ap.parse_args()
    if args.worker:
        _cell_worker([o for o in args.ops.split(",") if o], args.json)
        return 0
    if args.tally:
        with open(args.json) as fh:
            print(json.dumps(tally(json.load(fh)), indent=2))
        return 0
    return _drive(args)


if __name__ == "__main__":
    raise SystemExit(main())
