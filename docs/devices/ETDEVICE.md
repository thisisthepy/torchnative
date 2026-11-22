# ETDEVICE: ExecuTorch on the device at `.to(device.npu)` (issue #423)

Status: **planned.** This note records the decisions for step R5 of the Android NPU back end
and the plan that follows from them. Nothing below is built yet.

## 1. What is already there

| Part | Issue | Grade (CI run 44853272926) |
|---|---|---|
| `executorch.exir` runs on the shim, XNNPACK lowering included | #306 | structural agreement with upstream (S5.7) |
| `executorch_rt`: runtime core, portable kernels, XNNPACK delegate, macOS arm64 and aarch64 Android | #315, #337, #350 | agrees with upstream's runtime bit for bit on macOS arm64 (S5.9, S5.12, S5.14) |
| exir's `flatc` call replaced by the linked flatbuffers parser | #357 | byte for byte (S5.15) |

What is missing is the step that joins them: `model.to(device.npu)` on Android still raises
`NotImplementedError` in `torchnative/device/_module_to.py::_to_compiled`.

## 2. Decisions

**Owner direction behind the whole path.** "ExecuTorch가 안드로이드 휠 없는건 빌드하면 되고 기기 안 즉석 컴파일이
되게 구현하면 되고 삼성 엑시노스도 지원이 되거든요. 다시 확인해." (owner, 2026-11-08, relayed 2026-11-14);
delegate order XNNPACK, Vulkan, Qualcomm HTP, Samsung/MediaTek ("그래 그 순서로 진행하자", owner,
2026-11-08); no offline or CI per-model `.pte` pipeline (owner, 2026-11-07).

**How exir reaches the device** (owner, 2026-11-21, asked by the leader: "Rust로 재구현"; option (c)):
the serializer step is re-implemented in Rust, and only exir's passes are vendored. No executorch
Python package goes into the wheel. The program is serialised in memory by the Rust serializer,
not by exir's `_serialize`.

**The shape of `.to(device.npu)`** (agent proposal (i) to (iv) of the 2026-11-21 report; the leader
relayed that it stands with the change above, owner 2026-11-21):

1. Android's NPU candidates become `("executorch", <unit>)` ahead of today's bare `("qnn", ...)`.
   The unit is named by the delegate that will run. XNNPACK comes first in the owner's order and
   runs on the **CPU**, so the resolution says "CPU (ExecuTorch XNNPACK delegate)", never "NPU",
   until the HTP delegate is wired.
2. The same module comes back with its forward swapped. The first call captures with `torch.export`,
   with the sequence length as a symbolic `Dim` (owner, 2026-11-13). It lowers with the XNNPACK
   partitioner, serialises in memory through the Rust serializer, and loads the program into
   `executorch_rt`. Later calls run the method; a shape outside the guards recompiles.
3. Step R5a is the forward without a KV cache: logits agree, and greedy `generate` matches token for
   token through full forwards. Step R5b is the StaticCache as mutable buffers for decode.
4. Tested on the macOS arm64 gate, where `executorch_rt` is built, with
   `TORCHNATIVE_DEVICE_HOST=android`. Then on the Galaxy Tab S9 Ultra (#21).

## 3. What "the serializer step" covers, measured from executorch 1.4.1

| Upstream Python (lines) | Does |
|---|---|
| `exir/_serialize/_program.py` (827), `_flatbuffer_program.py` (359), `_flatbuffer.py` (426), `_dataclass.py` (145), `_named_data_store.py` (246), `padding.py`, `_cord.py`, `_serialize.py` (about 210) | `Program` dataclasses to `program.fbs` bytes: extended header, segments, alignment, named data |
| `backends/xnnpack/serialization/xnnpack_graph_serialize.py` (393), `xnnpack_graph_schema.py` (542) | the XNNPACK delegate payload (`runtime_schema.fbs`) and its header |

The schemas are already vendored: `schema/program.fbs` (42 tables, structs, unions and enums) and
`backends/xnnpack/serialization/runtime_schema.fbs` (31). The emitter (`exir/emit/_emitter.py`,
2117 lines) builds the `Program` dataclasses. It is a pass, so it is vendored, not rewritten.

## 4. Plan and size (agent estimate, not a measurement)

| Step | Work | Estimate |
|---|---|---|
| R5-S1 | Rust serializer for `program.fbs` and the XNNPACK payload (flatbuffers builder over code generated once from the vendored schemas and vendored), called from Python with exir's dataclasses; byte-for-byte tests against upstream's serializer on CI | 4 to 6 agent-days |
| R5-V | Vendor exir's pass subset and the pure-Python modules it imports (pinned v1.4.1, sha256 manifest, `--check`, as #312/#319), trimming imports that only the serializer or tooling need | 2 days |
| R5a | `.to(device.npu)` wiring of section 2, host test on the macOS arm64 gate | 2 to 3 days |
| R5-D | Android wheel carries it; run on the Galaxy Tab (#21) | 1 to 2 days |

Total about 9 to 13 agent-days and several CI rounds. **#423 is part of 1.0** (owner, 2026-11-21,
relayed by the leader: "일을 제대로 하면 될거를 쓸데없이 자꾸 미루려고만 하네? 191, 443, 175 전부 1.0에 넣어" and
"다른건 결정해도 되는데 범위를 축소하는건 결정을 해야 할 일이 아니라 해서는 안될 일인거야."). The steps run in the order
of the table; a blocker is reported with what unblocks it.

## 5. R5-V: how exir's passes are vendored (agent proposal, not confirmed)

The owner's direction is the line in section 2: only exir's passes are vendored, and no executorch
Python package goes into the wheel. Everything below is how this round does it, and is an agent
proposal.

- **Where.** `torchnative/python/torchnative/_vendor/executorch`, a subpackage of the wheel's
  `torchnative` package, so pyproject's existing `torchnative*` package list and `**/*` package
  data carry it with no new packaging rule. Every module is renamed to
  `torchnative._vendor.executorch.*`: the tree never answers `import executorch` and never shadows
  an installed executorch.
- **Written by** `scripts/vendor/vendor_exir.py` from an ExecuTorch checkout at the pin the runtime
  uses (tag v1.4.1, commit e4d02f41). It refuses another commit or a modified checkout.
- **What.** The import closure, read with `ast`, of `executorch.exir`, the XNNPACK partitioner and
  `xnnpack_preprocess`. The closure stops at exir's serializer (R5-S1 replaced it), devtools, the
  flat-tensor (`.ptd`) serializer, the LLM extension and the pytree C++ bindings. Measured on the
  pin: 219 files, 37 byte for byte upstream, 153 only renamed, 7 patched, 20 authored, plus the
  generated `edge.json`, README and manifest.
- **Changes.** Exact (old, new, count) patches that the script refuses to apply if the count is
  wrong. They remove the third-party imports torchnative does not depend on (ruamel.yaml: `edge.yaml`
  becomes `edge.json` at vendor time; tabulate; torchao, including the three affine q/dq configs the
  partitioner asks for their op; numpy; flatbuffers) and send both serializers to the Rust ones.
  The authored `exir/_serialize/{__init__,_program,_serialize}.py` keep upstream's signatures and
  call `torchnative.export.et_serialize`; reading a program back and the `.ptd` half refuse by name.
- **Checked by** `vendor_exir.py --check` (every file's sha256 and size against `MANIFEST.tsv`,
  nothing unlisted; with `--source`, regenerated byte for byte) and two gate suites:
  `tests/release/test_exir_vendor.py` (manifest, no un-renamed `executorch` import, every
  third-party import guarded, wheel package data) and `tests/export/test_et_vendored.py` (the eight
  lowerings of `test_et_serialize.py` through the vendored tree, with executorch, flatbuffers,
  ruamel, tabulate and torchao absent and flatc unreachable, equal byte for byte to executorch
  1.4.1's own programs).
- **Upstream text.** The tree keeps upstream's prose, em-dashes included, as `torchnative/rust/vendor/`
  does; `tests/release/test_style.py` exempts both trees.
