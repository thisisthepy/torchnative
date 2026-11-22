# exir's passes, vendored (torchnative)

ExecuTorch v1.4.1 (commit e4d02f41f7909e8ed5bf4a14ffc520d733453d9f), the Python modules its lowering path
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
