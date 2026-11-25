"""Lower through the vendored exir or through executorch 1.4.1, as JSON on stdout (issue #423, step R5-V).

Run as a script by `tests/export/test_et_vendored.py`, twice, in the interpreter
named by `TORCHNATIVE_EXIR_PYTHON` with the vendored torch tree on `PYTHONPATH`
(so `torch` is torchnative's shim, as in `et_serialize_probe.py`):

    et_vendored_probe.py upstream   lowers with the installed executorch
    et_vendored_probe.py vendored   lowers with `torchnative._vendor.executorch`,
                                    with `import executorch` and the third-party
                                    packages the vendored tree must not need
                                    refused, and flatc unreachable

A helper, not a suite: it reports and the suite compares. Each lowering reports
the program's bytes (base64) or the error; the vendored run also reports every
refused import attempt, every `executorch*` module that got into
`sys.modules`, and flatc start attempts. `TMPDIR` is set by the caller to a
directory inside the repository.
"""

import base64
import json
import os
import subprocess
import sys
import traceback

MARK = "ET_VENDORED_JSON "

#: Top-level packages the vendored tree must lower without.
REFUSED = ("executorch", "flatbuffers", "ruamel", "tabulate", "torchao")

VENDORED = "torchnative._vendor.executorch"


def _err(exc):
    tail = traceback.extract_tb(exc.__traceback__)[-4:]
    return {"error": f"{type(exc).__name__}: {exc}"[:600],
            "where": [f"{f.filename.split('site-packages/')[-1]}:{f.lineno} {f.name}" for f in tail]}


def _refuse():
    """Make `import X` fail for every X in REFUSED, the way an absent package
    fails, and record each attempt.

    `sys.modules[X] = None` is what an absent package looks like to both
    `import X` (ModuleNotFoundError) and `importlib.util.find_spec(X)` (None);
    a meta-path finder that raised would break availability probes instead.
    Attempts are recorded by wrapping `__import__`."""
    import builtins

    attempts = []
    for top in REFUSED:
        for name in [m for m in sys.modules if m == top or m.startswith(top + ".")]:
            del sys.modules[name]
        sys.modules[top] = None
    real = builtins.__import__

    def recording(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name.split(".")[0] in REFUSED:
            attempts.append(name)
        return real(name, globals, locals, fromlist, level)

    builtins.__import__ = recording
    return attempts


def _guard_flatc():
    attempts = []
    real = subprocess.Popen

    class _NoFlatc(real):
        def __init__(self, args, *a, **k):
            argv = [args] if isinstance(args, (str, bytes, os.PathLike)) else list(args)
            program = os.path.basename(os.fsdecode(argv[0])) if argv else ""
            if program == "flatc" or program.startswith("flatc."):
                attempts.append([os.fsdecode(x) for x in argv])
                raise FileNotFoundError(2, "flatc is unreachable in this probe", os.fsdecode(argv[0]))
            super().__init__(args, *a, **k)

    subprocess.Popen = _NoFlatc
    return attempts


def _linear():
    import torch

    lin = torch.nn.Linear(8, 4)
    with torch.no_grad():
        lin.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(4, 8) / 32 - 0.5)
        lin.bias.copy_(torch.arange(4, dtype=torch.float32) / 4)
    return torch.nn.Sequential(lin, torch.nn.ReLU()).eval(), (torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16 - 0.5,)


def _scalars():
    import torch

    class Scalars(torch.nn.Module):
        def forward(self, x):
            y = torch.clamp(x, min=-0.25, max=float("inf"))
            z = y.sum(dim=[1], keepdim=True)
            return torch.cat([y * 1.5, z.expand(-1, 8) - 2], dim=0)

    return Scalars().eval(), (torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16 - 0.5,)


def _buffered():
    import torch

    class Counter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("acc", torch.arange(8, dtype=torch.float32))

        def forward(self, x):
            self.acc.add_(x.sum(dim=0))
            return x * self.acc

    return Counter().eval(), (torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16,)


def _xnn_mix():
    import torch

    class Mix(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(8, 8)
            self.b = torch.nn.Linear(8, 3)
            with torch.no_grad():
                self.a.weight.copy_(torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64 - 0.5)
                self.a.bias.zero_()
                self.b.weight.copy_(torch.arange(24, dtype=torch.float32).reshape(3, 8) / 24 - 0.5)
                self.b.bias.copy_(torch.arange(3, dtype=torch.float32))

        def forward(self, x):
            h = torch.nn.functional.hardtanh(self.a(x), -0.5, 0.5)
            return self.b(h * x + x).sigmoid()

    return Mix().eval(), (torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16 - 0.5,)


#: The lowerings of et_serialize_probe.py, so both suites speak of the same programs.
LOWERINGS = [
    ("portable_linear", "portable", _linear, None),
    ("portable_linear_stacktrace", "portable", _linear, {"emit_stacktrace": True}),
    ("portable_scalars", "portable", _scalars, None),
    ("portable_buffer", "portable", _buffered, None),
    ("xnnpack_linear", "xnnpack", _linear, None),
    ("xnnpack_linear_inline", "xnnpack", _linear, {"extract_delegate_segments": False}),
    ("xnnpack_mix", "xnnpack", _xnn_mix, None),
    ("xnnpack_mix_aligned", "xnnpack", _xnn_mix, {"segment_alignment": 4096, "constant_tensor_alignment": 64,
                                                   "delegate_alignment": 32}),
]


def _lower(pkg, kind, factory, config_kw):
    import importlib

    import torch

    exir = importlib.import_module(pkg + ".exir")
    model, args = factory()
    ep = torch.export.export(model, args)
    config = exir.ExecutorchBackendConfig(**(config_kw or {}))
    if kind == "portable":
        manager = exir.to_edge(ep)
    else:
        part = importlib.import_module(pkg + ".backends.xnnpack.partition.xnnpack_partitioner")
        manager = exir.to_edge_transform_and_lower(ep, partitioner=[part.XnnpackPartitioner()])
    return bytes(manager.to_executorch(config=config).buffer)


def main(mode):
    import torch

    report = {"mode": mode, "shim": hasattr(torch._C, "_aten_implemented"),
              "pte_binding": hasattr(torch._C, "_et_serialize_pte"),
              "xnnpack_binding": hasattr(torch._C, "_et_serialize_xnnpack")}
    if mode == "vendored":
        refused = _refuse()
        flatc = _guard_flatc()
        pkg = VENDORED
    else:
        pkg = "executorch"
    lowered = {}
    for name, kind, factory, cfg in LOWERINGS:
        try:
            lowered[name] = {"ok": True, "b64": base64.b64encode(_lower(pkg, kind, factory, cfg)).decode()}
        except Exception as exc:  # noqa: BLE001
            lowered[name] = {"ok": False, **_err(exc)}
    report["lowered"] = lowered
    if mode == "vendored":
        report["refused_imports"] = sorted(set(refused))
        report["executorch_modules"] = sorted(m for m, mod in sys.modules.items() if mod is not None
                                              and (m == "executorch" or m.startswith("executorch.")))
        report["vendored_modules"] = sum(1 for m in sys.modules if m.startswith(VENDORED + "."))
        report["flatc_attempts"] = flatc
    print(MARK + json.dumps(report))


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception as exc:  # noqa: BLE001
        print(MARK + json.dumps({"fatal": _err(exc)}))
