#!/usr/bin/env python3
"""Vendor exir's passes (ExecuTorch v1.4.1's Python lowering path) without its serializer, or check them.

    python scripts/vendor/vendor_exir.py --source ET_CHECKOUT
        write torchnative/python/torchnative/_vendor/executorch/ from an
        ExecuTorch checkout at the pinned commit (needs `ruamel.yaml` in the
        running interpreter, to turn edge.yaml into edge.json)
    python scripts/vendor/vendor_exir.py --check
        verify the tree on disk against its MANIFEST.tsv (no checkout needed):
        every listed file present with its sha256 and size, nothing unlisted
    python scripts/vendor/vendor_exir.py --check --source ET_CHECKOUT
        the same, then regenerate the tree into a scratch directory inside the
        repository and compare every file byte for byte

`--source` may also come from TORCHNATIVE_ET_SRC.

Issue #423, step R5-V. Owner, 2026-11-21 (asked by the leader, option (c) of
decision 4): "Rust로 재구현": the serializer step is re-implemented in Rust
(R5-S1, `torchnative/export/et_serialize.py`) and only exir's passes are
vendored, so no executorch Python package goes into the wheel. The pin is the
one `vendor_executorch.py` uses for the runtime (tag v1.4.1, commit e4d02f41).
How the subset is chosen, where it lives and how it is changed are agent
proposals (ETDEVICE.md section 5), not the owner's words.

What is vendored. The import closure, read from the source with `ast` (nothing
is imported), of three entry modules: `executorch.exir`,
`executorch.backends.xnnpack.partition.xnnpack_partitioner` and
`executorch.backends.xnnpack.xnnpack_preprocess`. Every `import` statement
counts, including those inside functions, so a lazily imported module is in
the tree too. The closure stops at `CUT`: exir's serializer (the step R5-S1
replaced), devtools, the flat-tensor (`.ptd`) serializer, the LLM extension and
the pytree C++ bindings. Each import of a cut module from a vendored one is
either inside a `try/except ImportError` upstream already wrote, or is changed
by a `PATCHES` entry, or is replaced by an `AUTHORED` module; `--check` and
`tests/release/test_exir_vendor.py` hold the tree to that.

How the files are changed, in this order, and nothing else:

  rename   every `executorch.exir`, `executorch.backends`,
           `executorch.extension` and `executorch.devtools` becomes
           `torchnative._vendor.executorch.<same>`, in code and strings
           alike, so the tree never answers `import executorch` and never
           shadows an installed executorch, and the module names exir compares
           against (`serde`) stay consistent with `__module__`.
  patches  `PATCHES`: exact (old, new) replacements, each of which must occur
           the stated number of times, or the script refuses. They remove the
           third-party imports torchnative does not depend on (ruamel.yaml,
           tabulate, torchao, numpy, flatbuffers) and send the serializer to
           the Rust one.
  authored `AUTHORED`: modules written by torchnative in place of upstream's
           (`exir/_serialize/__init__.py`, `_program.py`, `_serialize.py`) and
           the package `__init__.py` files upstream has none of.
  generated `exir/dialects/edge/edge.json`: upstream's edge.yaml loaded with
           ruamel.yaml's safe loader, as exir loads it, and written as JSON.

MANIFEST.tsv lists every file with its kind (`upstream`: byte for byte the
checkout's; `renamed`; `patched`; `authored`; `generated`; `meta`), its
sha256 and size, and for the first three the upstream file's sha256.
"""

import argparse
import ast
import hashlib
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

REPO = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
PKG = REPO / "torchnative" / "python" / "torchnative" / "_vendor"
DEST = PKG / "executorch"
TAG = "v1.4.1"
COMMIT = "e4d02f41f7909e8ed5bf4a14ffc520d733453d9f"
MANIFEST = "MANIFEST.tsv"
PREFIX = "torchnative._vendor.executorch"

ENTRIES = [
    "executorch.exir",
    "executorch.backends.xnnpack.partition.xnnpack_partitioner",
    "executorch.backends.xnnpack.xnnpack_preprocess",
]

#: Modules the closure stops at (a module and everything under it).
CUT = (
    "executorch.devtools",
    "executorch.extension.flat_tensor",
    "executorch.extension.llm",
    "executorch.extension.pytree.pybindings",
    "executorch.exir._serialize._program",
    "executorch.exir._serialize._serialize",
    "executorch.exir._serialize._flatbuffer",
    "executorch.exir._serialize._flatbuffer_program",
    "executorch.exir._serialize.generated",
    "executorch.exir._serialize.padding",
    "executorch.exir._serialize.generate_program",
)

#: Data files the vendored modules read (by `importlib.resources`), beside the
#: closure's `.py` files. edge.yaml is not copied: it becomes edge.json.
DATA = []

RENAME = re.compile(r"(?<![\w.])executorch(?=\.(?:exir|backends|extension|devtools)\b)")

P = PREFIX

#: (path under executorch/, old, new, count). `old` is matched after the rename.
PATCHES = [
    # ruamel.yaml: not a torchnative dependency. edge.yaml is loaded once at
    # import; the vendor step converts it to JSON with ruamel's safe loader.
    ("exir/dialects/edge/_ops.py",
     "from ruamel.yaml import YAML\n",
     "import json  # torchnative: edge.json in place of ruamel.yaml (scripts/vendor/vendor_exir.py)\n",
     1),
    ("exir/dialects/edge/_ops.py",
     '    yaml = YAML(typ="safe")\n'
     '    edge_yaml = resources.read_text(edge_package, "edge.yaml", encoding="utf-8")\n'
     '    edge_dialect_yaml_info = yaml.load(edge_yaml)\n',
     '    edge_dialect_yaml_info = json.loads(\n'
     '        resources.read_text(edge_package, "edge.json", encoding="utf-8")\n'
     '    )\n',
     1),
    # tabulate: not a torchnative dependency; it formats one report string.
    ("exir/backend/utils.py",
     "from tabulate import tabulate\n",
     "try:\n"
     "    from tabulate import tabulate\n"
     "except ImportError:  # torchnative: tabulate is not a dependency\n"
     f"    from {P}._torchnative import tabulate\n",
     1),
    # torchao: not a torchnative dependency. Its two predicates, copied.
    ("backends/xnnpack/utils/utils.py",
     "from torchao.quantization.pt2e.utils import _is_conv_node, _is_conv_transpose_node\n",
     "try:\n"
     "    from torchao.quantization.pt2e.utils import _is_conv_node, _is_conv_transpose_node\n"
     "except ImportError:  # torchnative: torchao is not a dependency\n"
     f"    from {P}._torchnative import _is_conv_node, _is_conv_transpose_node\n",
     1),
    # torchao again: the partitioner asks every config for its original op
    # (`ops_to_not_decompose`), and the three affine q/dq configs import torchao
    # to answer. Without torchao no torchao op can be in the graph, so there is
    # nothing to keep from decomposition: None, the answer of a config with no op.
    ("backends/xnnpack/partition/config/quant_affine_configs.py",
     "        import torchao.quantization.quant_primitives  # noqa\n",
     "        try:\n"
     "            import torchao.quantization.quant_primitives  # noqa\n"
     "        except ImportError:  # torchnative: torchao is not a dependency\n"
     "            return None\n",
     3),
    # numpy: not a torchnative dependency; `np.isclose` on two scalars.
    ("backends/xnnpack/partition/config/generic_node_configs.py",
     "import numpy as np\n",
     "try:\n"
     "    import numpy as np\n"
     "except ImportError:  # torchnative: numpy is not a dependency\n"
     f"    from {P}._torchnative import numpy_isclose_only as np\n",
     1),
    # flatc: never started by torchnative. The payload is serialised by
    # torch._C._et_serialize_xnnpack (R5-S1).
    ("backends/xnnpack/serialization/xnnpack_graph_serialize.py",
     f"from {P}.exir._serialize._flatbuffer import _flatc_compile\n",
     f"from {P}._torchnative import flatc_refused as _flatc_compile\n",
     1),
    ("backends/xnnpack/serialization/xnnpack_graph_serialize.py",
     "def serialize_xnnpack_binary(\n",
     f"from {P}._torchnative import serialize_xnnpack_binary  # noqa: E402,F811\n"
     "\n"
     "\n"
     "def _upstream_serialize_xnnpack_binary(\n",
     1),
    # The .ptd serializer is not vendored: a stand-in that refuses by name.
    ("exir/program/_program.py",
     f"        from {P}.extension.flat_tensor.serialize.serialize import (\n"
     "            FlatTensorConfig,\n"
     "            FlatTensorSerializer,\n"
     "        )\n",
     f"        from {P}.exir._serialize._serialize import (  # torchnative: no .ptd serializer\n"
     "            FlatTensorConfig,\n"
     "            FlatTensorSerializer,\n"
     "        )\n",
     2),
    ("exir/program/_program.py",
     f"    from {P}.devtools.etrecord import ETRecord\n"
     "\n"
     "    return ETRecord()\n",
     f"    from {P}._torchnative import etrecord_refused\n"
     "\n"
     "    return etrecord_refused()\n",
     1),
]

PKG_INIT = '''"""Upstream code vendored into torchnative (see each subpackage's README.md)."""
'''

ROOT_INIT = '''"""exir's passes from ExecuTorch v1.4.1, vendored by scripts/vendor/vendor_exir.py (issue #423, R5-V).

Not an `executorch` package: every module here is named
`torchnative._vendor.executorch.*`, so an installed executorch is never shadowed.
The serializer is torchnative's Rust one (`torchnative.export.et_serialize`).
README.md says what is in the tree and what was changed.
"""
'''

TORCHNATIVE_PY = '''"""What the vendored exir needs from torchnative in place of its third-party imports.

Written by scripts/vendor/vendor_exir.py (issue #423, R5-V). Each name stands in
for one import of a package torchnative does not depend on, or for a step
torchnative does not run; the PATCHES list of the script says which import.
"""

import torch


def tabulate(rows, headers=(), tablefmt=None, **_kwargs):
    """`tabulate.tabulate`'s plain layout, for exir's delegation report text.

    Columns are left-aligned and separated by two spaces, with a dashed rule
    under the headers. The text is not byte-equal to tabulate's own formats;
    it only feeds a human-readable report."""
    table = [[str(c) for c in headers]] if headers else []
    table += [[str(c) for c in row] for row in rows]
    if not table:
        return ""
    width = [max(len(r[i]) if i < len(r) else 0 for r in table) for i in range(max(len(r) for r in table))]
    lines = ["  ".join(c.ljust(width[i]) for i, c in enumerate(r)).rstrip() for r in table]
    if headers:
        lines.insert(1, "  ".join("-" * w for w in width))
    return "\\n".join(lines)


def _is_conv_node(n):
    """torchao 0.18.0 `torchao/quantization/pt2e/utils.py::_is_conv_node`, copied."""
    return n.op == "call_function" and n.target in [
        torch.ops.aten.conv1d.default,
        torch.ops.aten.conv1d.padding,
        torch.ops.aten.conv2d.default,
        torch.ops.aten.conv2d.padding,
        torch.ops.aten.conv3d.default,
        torch.ops.aten.conv3d.padding,
    ]


def _is_conv_transpose_node(n):
    """torchao 0.18.0 `torchao/quantization/pt2e/utils.py::_is_conv_transpose_node`, copied."""
    return n.op == "call_function" and n.target in [
        torch.ops.aten.conv_transpose1d,
        torch.ops.aten.conv_transpose1d.default,
        torch.ops.aten.conv_transpose2d,
        torch.ops.aten.conv_transpose2d.input,
    ]


class numpy_isclose_only:  # noqa: N801 - stands in for the module name `np`
    """`numpy.isclose` for two real scalars, numpy's formula and defaults:
    `abs(a - b) <= atol + rtol * abs(b)`, NaN never close (equal_nan=False),
    equal infinities close."""

    @staticmethod
    def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        a, b = float(a), float(b)
        if a != a or b != b:
            return equal_nan and a != a and b != b
        if a in (float("inf"), float("-inf")) or b in (float("inf"), float("-inf")):
            return a == b
        return abs(a - b) <= atol + rtol * abs(b)


def flatc_refused(*_args, **_kwargs):
    raise NotImplementedError(
        "flatc: torchnative never starts flatc; the XNNPACK payload is serialised by "
        "torch._C._et_serialize_xnnpack (issue #423, R5-S1)")


def serialize_xnnpack_binary(xnnpack_graph, constant_data_bytes):
    """exir's `serialize_xnnpack_binary`, computed by torchnative's Rust serializer."""
    from torchnative.export import et_serialize

    return et_serialize.serialize_xnnpack_binary(xnnpack_graph, constant_data_bytes)


def etrecord_refused():
    raise NotImplementedError(
        "ETRecord: executorch.devtools is not part of torchnative's vendored exir (issue #423, R5-V)")
'''

SERIALIZE_INIT = f'''"""exir's `_serialize` package, as torchnative vendors it (issue #423, R5-V).

Upstream's `__init__.py` re-exports three names from `_program.py`; this one
does the same from torchnative's `_program.py`, whose serializer is the Rust one.
"""

from {P}.exir._serialize._program import (
    deserialize_pte_binary as _deserialize_pte_binary,
    PTEFile as _PTEFile,
    serialize_pte_binary as _serialize_pte_binary,
)

# Internal APIs that should not be used outside of exir.
__all__ = [
    "_deserialize_pte_binary",
    "_serialize_pte_binary",
    "_PTEFile",
]
'''

SERIALIZE_PROGRAM = f'''"""exir's `_serialize/_program.py`, as torchnative vendors it (issue #423, R5-V).

`PTEFile` is upstream's dataclass, copied. `serialize_pte_binary` keeps upstream's
signature and returns upstream's `Cord`; the bytes come from
`torch._C._et_serialize_pte` (R5-S1), which equals upstream's serializer byte for
byte (tests/export/test_et_serialize.py). Reading a program back
(`deserialize_pte_binary`) is not vendored and refuses by name.
"""

from dataclasses import dataclass
from typing import List, Optional

from {P}.exir._serialize._cord import Cord
from {P}.exir._serialize._named_data_store import NamedDataStoreOutput
from {P}.exir.schema import Buffer, Program


@dataclass
class PTEFile:
    """
    Wraps together the data required to serialize into a PTE file.
    """

    program: Program
    # TODO(lfq): add constant data (currently restored in the program)
    # TODO(lfq): update this to List[bytes]
    mutable_data: Optional[List[Buffer]] = None
    named_data: Optional[NamedDataStoreOutput] = None


def serialize_pte_binary(
    pte_file: PTEFile,
    *,
    extract_delegate_segments: bool = False,
    segment_alignment: int = 128,
    constant_tensor_alignment: Optional[int] = None,
    delegate_alignment: Optional[int] = None,
) -> Cord:
    from torchnative.export import et_serialize

    return Cord(
        et_serialize.serialize_pte_bytes(
            pte_file,
            extract_delegate_segments=extract_delegate_segments,
            segment_alignment=segment_alignment,
            constant_tensor_alignment=constant_tensor_alignment,
            delegate_alignment=delegate_alignment,
        )
    )


def deserialize_pte_binary(program_data):
    raise NotImplementedError(
        "deserialize_pte_binary: reading a program back is not part of torchnative's "
        "vendored exir (issue #423, R5-V)")
'''

SERIALIZE_SERIALIZE = f'''"""exir's `_serialize/_serialize.py`, as torchnative vendors it (issue #423, R5-V).

`serialize_for_executorch` keeps upstream's signature and its program half,
transcribed: the PTE file is the program, its mutable data and the named data
destined for it. The other half, external weights written to `.ptd` files by the
flat-tensor serializer, is not vendored: a program that has any refuses by name.
`FlatTensorConfig` and `FlatTensorSerializer` stand in for the names
`exir/program/_program.py` constructs; the serializer refuses if it is used.
"""

from typing import Dict, Optional, Tuple

from {P}.exir._serialize._cord import Cord
from {P}.exir._serialize._named_data_store import NamedDataStoreOutput
from {P}.exir._serialize._program import PTEFile, serialize_pte_binary
from {P}.exir._serialize.data_serializer import DataSerializer

_PTD = ("external weights (.ptd files, the flat-tensor serializer) are not part of "
        "torchnative's vendored exir (issue #423, R5-V)")


class FlatTensorConfig:
    def __init__(self, segment_alignment: int = 128, *args, **kwargs) -> None:
        self.segment_alignment = segment_alignment


class FlatTensorSerializer(DataSerializer):
    def __init__(self, config: Optional[FlatTensorConfig] = None) -> None:
        self.config = config

    def serialize(self, data):
        raise NotImplementedError("FlatTensorSerializer.serialize: " + _PTD)

    def deserialize(self, blob):
        raise NotImplementedError("FlatTensorSerializer.deserialize: " + _PTD)


def serialize_for_executorch(
    emitter_output,
    config,
    data_serializer: DataSerializer,
    named_data_store: Optional[NamedDataStoreOutput] = None,
) -> Tuple[Cord, Dict[str, Cord]]:
    """Serialize the output from Emitter into ExecuTorch artifacts; PTE and PTD files."""

    # Serialize PTE file.
    pte_named_data = None
    if (
        named_data_store is not None
        and len(named_data_store.buffers) > 0
        and len(named_data_store.pte_data) > 0
    ):
        # Create a separate NamedDataStoreOutput with only pte_data; exclude
        # external_data, which shouldn't be serialized with the PTE file.
        if len(named_data_store.external_data) == 0:
            pte_named_data = named_data_store
        else:
            pte_named_data = NamedDataStoreOutput(
                buffers=named_data_store.buffers,
                pte_data=named_data_store.pte_data,
                external_data={{}},
            )
    pte: Cord = serialize_pte_binary(
        pte_file=PTEFile(
            program=emitter_output.program,
            mutable_data=emitter_output.mutable_data,
            named_data=pte_named_data,
        ),
        extract_delegate_segments=config.extract_delegate_segments,
        segment_alignment=config.segment_alignment,
        constant_tensor_alignment=config.constant_tensor_alignment,
        delegate_alignment=config.delegate_alignment,
    )

    # Early exit if no external weights.
    if len(emitter_output.external_constant_map) == 0 and (
        named_data_store is None or len(named_data_store.external_data) == 0
    ):
        return pte, {{}}

    raise NotImplementedError("serialize_for_executorch: " + _PTD)
'''

README = f"""# exir's passes, vendored (torchnative)

ExecuTorch {TAG} (commit {COMMIT}), the Python modules its lowering path
imports, without its serializer. Written by `scripts/vendor/vendor_exir.py`;
never edit it by hand: change the script and re-run it, then
`python scripts/vendor/vendor_exir.py --check`.

Issue #423, step R5-V. The owner's direction (2026-11-21, option (c)): the
serializer step is re-implemented in Rust and only exir's passes are vendored,
so no executorch Python package goes into the wheel. How the subset is chosen and
changed is an agent proposal (docs/devices/ETDEVICE.md section 5).

* Package name: `torchnative._vendor.executorch`. `import executorch` is never
  answered from here.
* Serializer: `exir/_serialize/` holds upstream's `_cord.py`, `_dataclass.py`,
  `_named_data_store.py` and `data_serializer.py`, and torchnative's
  `__init__.py`, `_program.py` and `_serialize.py`, which call the Rust
  serializer (`torchnative/export/et_serialize.py`).
* Not vendored: devtools, the `.ptd` (flat-tensor) serializer, the LLM
  extension, the pytree C++ bindings, flatc. Each refuses by name if reached.
* Third-party imports removed: ruamel.yaml (`edge.yaml` is `edge.json`),
  tabulate, torchao, numpy and flatbuffers (`_torchnative.py` holds what
  replaces each).

MANIFEST.tsv lists every file, its kind, sha256 and size, and for files derived
from upstream the upstream sha256. LICENSE is upstream's.
"""


class Refusal(Exception):
    """A named refusal: printed as-is, exit code 2."""


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git(src, *args):
    done = subprocess.run(["git", "-C", str(src), *args], capture_output=True)
    if done.returncode != 0:
        raise Refusal(f"vendor_exir: refusing: `git {' '.join(args)}` failed in {src}:\n"
                      f"{done.stderr.decode(errors='replace')}")
    return done.stdout.decode()


def verify_source(src):
    if not (src / "exir" / "__init__.py").is_file() or not (src / "backends" / "xnnpack").is_dir():
        raise Refusal(f"vendor_exir: refusing: {src} is not an ExecuTorch checkout (no exir/__init__.py)")
    head = git(src, "rev-parse", "HEAD").strip()
    if head != COMMIT:
        raise Refusal(f"vendor_exir: refusing: {src} is at commit {head}, not ExecuTorch {TAG} "
                      f"({COMMIT}). Nothing was written.")
    dirty = git(src, "status", "--porcelain", "--untracked-files=no").strip()
    if dirty:
        raise Refusal(f"vendor_exir: refusing: {src} has modified tracked files, so it is not "
                      f"{TAG} as published:\n{dirty}")


def is_cut(mod):
    return any(mod == c or mod.startswith(c + ".") for c in CUT)


def module_path(src, mod):
    """`executorch.a.b` -> the checkout's `a/b.py` or `a/b/__init__.py` (relative), or None."""
    parts = mod.split(".")[1:]
    if not parts:
        return None
    for rel in ("/".join(parts) + ".py", "/".join(parts) + "/__init__.py"):
        if (src / rel).is_file():
            return rel
    return None


def imports_of(mod, rel, text):
    """Every module an `import` statement in `text` names (absolute), with its line."""
    tree = ast.parse(text, filename=rel)
    pkg = mod if rel.endswith("__init__.py") else mod.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield node.lineno, a.name, False
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split(".")
                base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                name = ".".join(base + ([node.module] if node.module else []))
            else:
                name = node.module or ""
            yield node.lineno, name, False
            for a in node.names:
                yield node.lineno, name + "." + a.name, True


def closure(src):
    """The relative paths of the closure, and the (file, line, module) edges into CUT."""
    seen, todo, edges = {}, list(ENTRIES), set()
    while todo:
        mod = todo.pop()
        if mod in seen or not mod.startswith("executorch") or is_cut(mod):
            continue
        rel = module_path(src, mod)
        if rel is None:
            continue
        seen[mod] = rel
        parts = mod.split(".")
        todo += [".".join(parts[:i]) for i in range(2, len(parts))]
        for line, name, maybe_attr in imports_of(mod, rel, (src / rel).read_text(encoding="utf-8")):
            if not name.startswith("executorch."):
                continue
            if is_cut(name):
                if not maybe_attr or module_path(src, name) is not None:
                    edges.add((rel, line, name))
            else:
                todo.append(name)
    return sorted(set(seen.values())), sorted(edges)


def rename(text):
    return RENAME.sub(PREFIX, text)


def apply_patches(rel, text):
    for path, old, new, count in PATCHES:
        if path != rel:
            continue
        found = text.count(old)
        if found != count:
            raise Refusal(f"vendor_exir: refusing: a patch for {rel} expects {count} occurrence(s) "
                          f"of {old.splitlines()[0]!r} and found {found}")
        text = text.replace(old, new)
    return text


def edge_json(src):
    try:
        from ruamel.yaml import YAML
    except ImportError as exc:
        raise Refusal("vendor_exir: refusing: ruamel.yaml is not importable in this interpreter; "
                      "edge.json is edge.yaml as exir's ruamel safe loader reads it, so the vendor "
                      "step needs it (the wheel does not)") from exc
    data = YAML(typ="safe").load((src / "exir/dialects/edge/edge.yaml").read_text(encoding="utf-8"))
    return (json.dumps(data, indent=1, ensure_ascii=True) + "\n").encode()


def build(src):
    """{relative path under DEST: (kind, bytes, upstream sha or "")}."""
    verify_source(src)
    files, edges = closure(src)
    authored = {
        "__init__.py": ROOT_INIT,
        "_torchnative.py": TORCHNATIVE_PY,
        "exir/_serialize/__init__.py": SERIALIZE_INIT,
        "exir/_serialize/_program.py": SERIALIZE_PROGRAM,
        "exir/_serialize/_serialize.py": SERIALIZE_SERIALIZE,
    }
    out = {}
    patched_paths = {p[0] for p in PATCHES}
    for rel in files:
        if rel in authored:
            continue
        raw = (src / rel).read_bytes()
        text = raw.decode("utf-8")
        new = rename(text)
        kind = "upstream" if new == text else "renamed"
        if rel in patched_paths:
            new = apply_patches(rel, new)
            kind = "patched"
        out[rel] = (kind, new.encode("utf-8"), sha256(raw))
    unused = patched_paths - set(out)
    if unused:
        raise Refusal(f"vendor_exir: refusing: patches name files outside the closure: {sorted(unused)}")
    # Every edge into CUT must be handled: inside an upstream try/except, a
    # patched file, or a module torchnative authors.
    unhandled = [e for e in edges if e[0] not in authored and e[0] not in patched_paths
                 and not guarded(src, e)]
    if unhandled:
        raise Refusal("vendor_exir: refusing: imports of cut modules nothing handles:\n"
                      + "\n".join(f"  {f}:{ln} {m}" for f, ln, m in unhandled))
    for rel, text in authored.items():
        out[rel] = ("authored", text.encode("utf-8"), "")
    # Package directories upstream leaves without an __init__.py (namespace
    # packages) get an empty one, so setuptools' find_packages keeps them.
    dirs = {str(pathlib.PurePosixPath(r).parent) for r in out}
    for d in sorted(dirs):
        parts = pathlib.PurePosixPath(d).parts
        for i in range(1, len(parts) + 1):
            init = "/".join(parts[:i]) + "/__init__.py"
            if init not in out:
                out[init] = ("authored", b"", "")
    for rel in DATA:
        raw = (src / rel).read_bytes()
        out[rel] = ("upstream", raw, sha256(raw))
    out["exir/dialects/edge/edge.json"] = ("generated", edge_json(src),
                                           sha256((src / "exir/dialects/edge/edge.yaml").read_bytes()))
    lic = (src / "LICENSE").read_bytes()
    out["LICENSE"] = ("upstream", lic, sha256(lic))
    out["README.md"] = ("meta", README.encode("utf-8"), "")
    return out


def guarded(src, edge):
    """Whether the import at `edge` sits inside a `try` whose handler catches ImportError."""
    rel, line, _ = edge
    tree = ast.parse((src / rel).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and node.body[0].lineno <= line <= node.body[-1].end_lineno:
            for h in node.handlers:
                names = []
                if h.type is None:
                    return True
                for t in (h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]):
                    names.append(getattr(t, "id", getattr(t, "attr", "")))
                if {"ImportError", "ModuleNotFoundError", "Exception"} & set(names):
                    return True
    return False


def manifest_bytes(tree):
    lines = ["# kind\tsha256\tsize\tpath\tupstream_sha256"]
    for rel in sorted(tree):
        kind, data, up = tree[rel]
        lines.append(f"{kind}\t{sha256(data)}\t{len(data)}\t{rel}\t{up}")
    return ("\n".join(lines) + "\n").encode()


def write(tree, dest):
    if dest.exists():
        shutil.rmtree(dest)
    for rel, (_, data, _) in tree.items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    (dest / MANIFEST).write_bytes(manifest_bytes(tree))
    init = dest.parent / "__init__.py"
    if not init.is_file() or init.read_text() != PKG_INIT:
        init.write_text(PKG_INIT)


def on_disk(dest):
    out = set()
    for root, dirs, names in os.walk(dest):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for n in names:
            if n.endswith((".pyc", ".pyo")) or n == ".DS_Store":
                continue
            out.add(pathlib.Path(root, n).relative_to(dest).as_posix())
    return out


def check(dest=DEST):
    """Problems with the tree at `dest` against its manifest, as a list of strings."""
    problems = []
    man = dest / MANIFEST
    if not man.is_file():
        return [f"{man} is missing"]
    listed = {}
    for ln in man.read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        kind, digest, size, rel, _up = ln.split("\t")
        listed[rel] = (kind, digest, int(size))
    for rel, (kind, digest, size) in sorted(listed.items()):
        p = dest / rel
        if not p.is_file():
            problems.append(f"missing: {rel}")
            continue
        data = p.read_bytes()
        if len(data) != size or sha256(data) != digest:
            problems.append(f"changed: {rel} (sha256 {sha256(data)[:12]}, {len(data)} bytes; "
                            f"manifest {digest[:12]}, {size} bytes)")
    extra = on_disk(dest) - set(listed) - {MANIFEST}
    problems += [f"not in the manifest: {rel}" for rel in sorted(extra)]
    pkg_init = dest.parent / "__init__.py"
    if not pkg_init.is_file() or pkg_init.read_text() != PKG_INIT:
        problems.append(f"{pkg_init} is missing or not the script's")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="verify instead of write")
    ap.add_argument("--source", default=os.environ.get("TORCHNATIVE_ET_SRC"),
                    help="ExecuTorch checkout at the pinned commit")
    ns = ap.parse_args(argv)
    try:
        if ns.check:
            problems = check()
            if problems:
                print("vendor_exir --check: FAILED\n  " + "\n  ".join(problems))
                return 1
            print(f"vendor_exir --check: {len(on_disk(DEST)) - 1} files match {MANIFEST}")
            if ns.source:
                tree = build(pathlib.Path(ns.source).resolve())
                scratch = REPO / ".scratch" / "vendor_exir_check"
                write(tree, scratch / "executorch")
                diffs = []
                for rel in sorted(on_disk(scratch / "executorch") | on_disk(DEST)):
                    a, b = scratch / "executorch" / rel, DEST / rel
                    if not a.is_file() or not b.is_file() or a.read_bytes() != b.read_bytes():
                        diffs.append(rel)
                shutil.rmtree(scratch)
                if diffs:
                    print("vendor_exir --check --source: regenerated tree differs:\n  " + "\n  ".join(diffs))
                    return 1
                print("vendor_exir --check --source: the regenerated tree is byte-identical")
            return 0
        if not ns.source:
            raise Refusal("vendor_exir: refusing: --source (or TORCHNATIVE_ET_SRC) is required to write")
        tree = build(pathlib.Path(ns.source).resolve())
        write(tree, DEST)
        kinds = {}
        for kind, _, _ in tree.values():
            kinds[kind] = kinds.get(kind, 0) + 1
        print(f"vendor_exir: wrote {len(tree)} files to {DEST.relative_to(REPO)}: "
              + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
        return 0
    except Refusal as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
