"""The cells of the sort/argsort/topk tie-order oracle (issue #33).

Shared by `gen_sortties_oracle.py` (which runs UPSTREAM torch and records
the indices) and `tests/ops/test_sortties.py` (which replays the same cells
on the shim). A cell is a plain dict; its `id` is the oracle's key.
"""

import hashlib
import random

NAN = float("nan")
DTYPES = ["float32", "float16", "bfloat16", "float64", "int32", "int64", "uint8", "bool"]
FLOATS = {"float32", "float16", "bfloat16", "float64"}
SIZES = [1, 2, 15, 16, 17, 64, 1000, 5000]
PATTERNS = ["allsame", "twoval", "ties4", "signedzero", "nan", "distinct"]
# (rank layout name, shape builder, dim)
LAYOUTS = {
    "1d": (lambda n: (n,), -1),
    "2d_last": (lambda n: (3, n), -1),
    "2d_dim0": (lambda n: (n, 3), 0),
    "3d_last": (lambda n: (2, 3, n), -1),
    "3d_mid": (lambda n: (2, n, 3), 1),
    "3d_dim0": (lambda n: (n, 2, 3), 0),
}


def pattern_values(pattern, dtype, count, seed):
    """`count` python numbers. None when the pattern does not exist for the dtype."""
    rng = random.Random(seed)
    isf = dtype in FLOATS
    if pattern == "allsame":
        return [1] * count
    if pattern == "twoval":
        return [rng.choice([0, 1]) for _ in range(count)]
    if pattern == "ties4":
        v = [rng.choice([0, 1, 2, 3]) for _ in range(count)]
        return [x % 2 for x in v] if dtype == "bool" else v
    if pattern == "signedzero":
        if not isf:
            return None
        return [rng.choice([0.0, -0.0, 0.0, -0.0, 1.0]) for _ in range(count)]
    if pattern == "nan":
        if not isf:
            return None
        return [rng.choice([NAN, 1.0, 2.0, 1.0, NAN]) for _ in range(count)]
    if pattern == "distinct":
        v = list(range(count))
        rng.shuffle(v)
        if dtype == "uint8":
            v = [x % 256 for x in v]
        if dtype == "bool":
            v = [x % 2 for x in v]
        return v
    raise ValueError(pattern)


def _layouts_for(n, dtype):
    out = ["1d"]
    if n <= 64 or n == 1000:
        out += ["2d_last", "2d_dim0"]
    if n <= 64 and dtype in ("float32", "bfloat16", "int64"):
        out += ["3d_last", "3d_mid", "3d_dim0"]
    return out


def cells():
    """Every cell, in a stable order. Each: id, op, dtype, pattern, n, layout, shape, dim, args."""
    out = []
    for dtype in DTYPES:
        for n in SIZES:
            for layout in _layouts_for(n, dtype):
                shape = LAYOUTS[layout][0](n)
                dim = LAYOUTS[layout][1]
                for pattern in PATTERNS:
                    if pattern_values(pattern, dtype, 1, 0) is None:
                        continue
                    # the wide sweep only where it can show something new
                    wide = layout == "1d" or dtype in ("float32", "int64")
                    ops = []
                    for stable in (None, False, True):
                        for desc in (False, True):
                            ops.append(("sort", dict(stable=stable, descending=desc)))
                    for st in (None, False, True):
                        for desc in (False, True):
                            ops.append(("argsort", dict(stable=st, descending=desc)))
                    ks = sorted({1, max(1, n // 2), n}) if wide else [n]
                    for k in ks:
                        for largest in (True, False):
                            for srt in (True, False):
                                ops.append(("topk", dict(k=k, largest=largest, sorted=srt)))
                    for op, a in ops:
                        tag = ",".join(f"{kk}={vv}" for kk, vv in a.items())
                        cid = f"{op}|{dtype}|{pattern}|{layout}|n{n}|{tag}"
                        out.append(dict(id=cid, op=op, dtype=dtype, pattern=pattern,
                                        n=n, layout=layout, shape=list(shape),
                                        dim=dim, args=a))
    return out


def make_input(torch, c):
    numel = 1
    for s in c["shape"]:
        numel *= s
    seed = hash_seed(c["dtype"], c["pattern"], c["layout"], c["n"])
    vals = pattern_values(c["pattern"], c["dtype"], numel, seed)
    dt = getattr(torch, c["dtype"])
    if c["dtype"] == "bool":
        vals = [bool(v) for v in vals]
    return torch.tensor(vals, dtype=dt).reshape(c["shape"])


def hash_seed(*parts):
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def run_cell(torch, c):
    """Indices (flat python list) the given torch answers for the cell."""
    x = make_input(torch, c)
    a = c["args"]
    dim = c["dim"]
    if c["op"] == "sort":
        if a["stable"] is None:
            idx = torch.sort(x, dim=dim, descending=a["descending"])[1]
        else:
            idx = torch.sort(x, dim=dim, descending=a["descending"], stable=a["stable"])[1]
    elif c["op"] == "argsort":
        if a["stable"] is None:
            idx = torch.argsort(x, dim=dim, descending=a["descending"])
        else:
            idx = torch.argsort(x, dim=dim, descending=a["descending"], stable=a["stable"])
    else:
        idx = torch.topk(x, a["k"], dim=dim, largest=a["largest"], sorted=a["sorted"])[1]
    return [int(v) for v in idx.reshape(-1).tolist()]


def encode(indices):
    """Short lists verbatim; long ones as their length and sha1 (every index still compared)."""
    if len(indices) <= 200:
        return indices
    h = hashlib.sha1(",".join(map(str, indices)).encode()).hexdigest()
    return {"len": len(indices), "sha1": h}


def matches(recorded, indices):
    return encode(indices) == recorded
