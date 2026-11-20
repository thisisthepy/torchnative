# 0.1.0b10: release notes

**The wheels are now built with pypackpack (`ppp`), and the shim and the
CompositeImplicit table moved further toward upstream.** The owner approved this
release on 2026-11-21 and chose the build path: "ppp로 배포". Before this
release the publish workflow built every wheel through setuptools; a published
release now builds seven of the nine wheels with pypackpack 0.4.1
(windows-arm64 and wasm32-emscripten stay on setuptools).

The rest of the range is small: 13 pull requests merged into develop since tag
`v0.1.0b9` (`git log --merges --first-parent v0.1.0b9..origin/develop`, #394 to
#411), three of which also carry earlier branches (#390 and #405 inside #406,
#397 inside #398). They are shim fidelity work (strides, dispatch overloads,
singletons, grad_fn under a mode), more kernels and decomposition bodies from
the #374 list, one crash audit of zero-element CPU kernels, and the ExecuTorch
XNNPACK lowering walls under #306. No benchmark was taken for this release; no
speed claim is made. Every PR below says "CI decides" for its local checks: the
local runs were compile checks only, and the test results are the gate's.

---

## 1. Features added

### Build backend: pypackpack (#53)

- **ppp wheel backend in `scripts/wheel/build.py` (#399).** `pyproject.toml`
  declares `[tool.ppp]` and `[tool.ppp.rust]` beside the unchanged setuptools
  tables. `build.py --backend ppp` runs `ppp build torchnative source single
  --type release --level mixed --overwrite` (with `--target` for a cross target)
  and then the existing post-processing (`upstream_dist_info`,
  `global_deps_stub`, `_repack`, `verify`). `PPP_TARGETS` maps all nine target
  keys to pypackpack triples; `--self-test` checks the map and the pyproject
  entries.
- **`inputs.backend` in `publish-pypi.yml` (#408, stage 2 of #53).** A
  `workflow_dispatch` input and a `preflight` output that every build job reads.
  Under `ppp`, windows-x86_64 builds natively on `windows-latest` (no
  `cargo-xwin`, no `llvm`), the Linux `libtorch_global_deps.so` stub is linked
  with `ziglang`, and the iOS device build is given
  `TORCHNATIVE_PYTHON_FRAMEWORK_DIR`. `build.py` also reads Windows `_C.dll` and
  drive-letter `.d` dep-info correctly. This release changes the default of that
  input to `ppp` (see section 5).
- **`test.yml` installs pypackpack from PyPI (#409).** The `ppp-wheel-linux` job
  uses `uv tool install pypackpack==0.4.1` instead of a checkout of
  `pypackpack@develop` and a Gradle build.

### Kernels and decompositions (part of #374)

- **`aten.minimum.default` (#404)**, mirroring `maximum`, with overload and
  method rows and golden cases.
- **`aten.aminmax.default`, `aten.logical_not_.default`,
  `aten.diagonal_copy.default` (#405, landed through #406).**
- **`aten._adaptive_avg_pool2d.default` and
  `aten.max_pool2d_with_indices.default` (#406).** The second returns
  `(values, indices)` with `int64` flat indices `row * iw + col`.
- **`aten.isinf.default`, `torch.isinf`, `Tensor.isinf` (#411, closes #403).**
  Float32, float64, float16 and bfloat16 compute `abs(x) == inf`; integer and
  bool inputs give all `False` without reading the operand; complex and float8
  refuse by name. `isnan`, `isposinf` and `isneginf` are missing the same way and
  are not part of this release.
- **CompositeImplicit decomposition bodies (#398, with #397).** 18 more bodies
  in `_CIA_TABLE`, matching upstream 2.13.0's logged sequences node for node:
  `clip`, `greater.Scalar/Tensor`, `max.other`, `multiply.Scalar/Tensor`,
  `where.Scalar/ScalarOther/default` (#397), and `view_as`, `argsort`
  (`default`, `stable`), `chunk`, `unflatten.int`, `broadcast_tensors`,
  `is_floating_point`, `std`, `var`, `var_mean` (`default`, `dim`), `real`,
  `imag`, `rms_norm` (#398). `CustomDecompTable` goes from 426 to 444 entries
  (#398 states it was confirmed in CI run 44588395741); 11 ops stay in the
  pending pin.
- **`torch._C._to_functionality_key` (#407, part of #306).** Per-backend keys
  map to their functionality key and alias/composite keys to `Undefined`,
  compared key for key with upstream 2.13.0.

### Shim fidelity

- **Dense `cat` keeps channels-last (#395, closes #301).** When every input
  suggests channels-last the CPU result is laid out channels-last, as upstream's
  `compute_output_memory_format` does.
- **`torch.memory_format`, `torch.layout` and `torch.qscheme` are singletons
  (#394, part of #306).** They are non-instantiable and carry `__copy__`,
  `__deepcopy__`, `__eq__`, `__hash__` and `__reduce__`, so they keep identity
  across `copy.deepcopy` and pickle.
- **Python numbers reach a subclass as upstream's `.Tensor` overload (#396,
  closes #359).** `subclass_facing_overload` runs for every tensor subclass, not
  only wrapper subclasses, scoped to `add`, `sub`, `mul`, `div` and their
  in-place and alias forms. `eq`, `gt`, `remainder`, `fmod`, `fill_` and
  `bitwise_and` keep `.Scalar`.
- **Exir probe names a remaining recursion (#410).** On `RecursionError` the
  probe prints the repeating frame window once with its length and count. The
  shim's FakeTensor dispatch keys gain the Python key.

## 2. Defects fixed

- **A mode-answered op had no `grad_fn` (#400, closes #371).** When a
  `TorchDispatchMode` such as `FakeTensorMode` answered an op, the result came
  back without `from_op`. `mark_from_op_if_unmarked` marks floating and complex
  outputs that carry none and bumps the autograd sequence number only when it
  marks one, so a mode that delegates to an inner op does not count twice.
- **HOP dispatch recursion under exir XNNPACK lowering (#410, part of #306).**
  `torch._C._AutoDispatchBelowAutograd` was a bare counter and excluded nothing,
  so `maybe_run_autograd` re-entered with the same keyset until
  `RecursionError`. It is now a key-state guard adding every Autograd key to the
  TLS exclude set. `DispatchKeySet.highestPriorityTypeId()` follows upstream's
  measured functionality and backend order (16 measured combinations agree), and
  `DispatchKeySet.has()` accepts a key name.
- **`deepcopy` of a graph's `memory_format` made a fresh, unequal object (#394).**
  ExecuTorch's `MemoryFormatOpsPass` deep-copies kwargs, and the copy compared
  unequal to `torch.contiguous_format`.
- **Zero-element CPU kernels sliced storage they did not have (#390, closes
  #385).** Every CPU storage-slicing path in `candle-core` and `torch_c` was
  audited against a zero-element layout with a nonzero offset and against
  non-contiguous strides over a 0-byte storage, and each remaining site now
  returns an empty result or a candle `Error` first (`copy2d_`, reductions,
  upsample, scatter, conv and im2col families, matmul, sort, `sdpa_flash_cpu`,
  `index_copy_slice`, `et.rs::host_bytes`). The vendored candle patch is
  regenerated. One upstream crash candidate is recorded, not fixed here:
  `torch.ops.aten._weight_norm_interface.default` on a `(3, 0)` view with stride
  `(4, 1)` terminates upstream torch with SIGSEGV on macOS arm64 (2.13.0).
- **Dense `cat` of channels-last inputs forced a contiguous result (#395, #301).**
  Five of #301's six rows had already been fixed by #380 and #383; #395 closes
  the remaining one.
- **A subclass saw `.Scalar` where upstream gives `.Tensor` (#396, #359).**

## 3. Test added

- `tests/ops/test_empty_storage_slice.py` (#390): every tensor-taking operator in
  `torch._C._aten_all_implemented()` across both zero-element layouts, one
  subprocess per family, shapes and dtypes compared with upstream, in the gate on
  both lanes. Extended Rust unit tests in `layout.rs` and `aten.rs`.
- `tests/bindings/test_isinf.py` (#411): `torch.isinf` and `Tensor.isinf`
  against upstream 2.13.0 across dtypes and transposed, offset and stepped
  views, plus a `TypicalLogitsWarper`-shaped mask. Golden `isinf_cases`.
- `test_issue_301_output_strides_match_upstream` in
  `tests/ops/test_elemstride.py` (#395): five #301 rows and five channels-last
  `cat` cases.
- `test_clone_node_memory_format_survives_graph_and_deepcopy` in
  `tests/bindings/test_jit_types.py` and
  `test_clone_node_memory_format_gives_upstreams_dim_order` in
  `tests/export/test_exir_shim.py` (#394).
- `test_to_functionality_key_matches_upstream_key_for_key` in
  `tests/export/test_export5.py` (#407).
- CompositeImplicit: `tests/bindings/test_cia_frame.py` cases for every new
  body, `tests/export/test_varmean.py` (all five `var_mean` spellings decompose
  to `aten.var_mean.correction`, as upstream) (#398). Golden cases for
  `minimum`, `aminmax`, `logical_not_`, `diagonal_copy`, `_adaptive_avg_pool2d`
  and `max_pool2d_with_indices` (#404, #405, #406).
- Two known-divergence pins now run as real checks:
  `test_a_mode_answered_op_on_a_grad_path_counts_one_autograd_node` (#400) and
  `test_a_subclass_sees_upstreams_overload_for_a_python_number` (#396).
- Pins moved: schema identities 401 to 402 and core-tag count 138 to 139 (#411),
  tag core count 134 to 135 (#404).

## 4. Build backend: pypackpack 0.4.1

The owner's decision of 2026-11-21 was "ppp로 배포". What it means in
`publish-pypi.yml`:

- A published release (the `release` event) and a `workflow_dispatch` that
  passes no backend build with `ppp`; the dispatch input defaults to `ppp`
  too. `setuptools` stays selectable as an input.
- `ppp` is installed with `uv tool install pypackpack==0.4.1`.
- **Linux through zig.** The two Linux wheels build on `ubuntu-24.04`; the
  `libtorch_global_deps.so` stub is linked with `ziglang`, installed through
  `uv pip install`.
- **Windows native.** `windows-x86_64` builds on `windows-latest` with no
  cross-compiler.
- **Still setuptools, per target.** The matrix flag `ppp: false` keeps
  `windows-arm64` (pypackpack#150: the Cargo backend refuses
  `aarch64-pc-windows-msvc`) and `wasm32-emscripten` (python-multiplatform#175)
  on the setuptools path even when the backend is `ppp`. Those two jobs run on
  their setuptools runner and take every `backend != 'ppp' || !matrix.ppp`
  branch. This release was not built here; the claim is read from the workflow
  file.
- **Platform tags the ppp wheels carry.** Android `android_24_arm64_v8a` (owner
  decision 2026-11-19) and iOS device `ios_13_0_arm64_iphoneos`, the CPython iOS
  floor. The setuptools path keeps `android_21_arm64_v8a` and
  `ios_12_0_arm64_iphoneos`. The "exactly these nine wheels, by name" step of
  the collect job picks the tag set from the backend, so a `ppp` release names
  the new tags and fails on a missing one. The other tags (`macosx_11_0_arm64`,
  `ios_14_0_arm64_iphonesimulator`, the two manylinux 2.17 wheels, `win_amd64`,
  `win_arm64`, `pyemscripten_2026_0_wasm32`) are the same on both paths.
  `tests/release/test_cipub.py` and `test_wheelmatrix.py` still assert the
  `android_21` and `ios_12_0` names; they describe the setuptools path.

## 5. Deleted

- Nothing was removed from the public surface in this range. The one removal is
  the setuptools release default in `publish-pypi.yml` (replaced by `ppp`, see
  section 4); the setuptools path, `setup.py` and its tables remain, and #399
  states that removing `setup.py` is a later step.

## 6. Documentation corrected

- `docs/kernels/STRIDED.md` (§8.4.2, §8.7, §8.9) and `docs/SPEC.md` (S1.13)
  describe the zero-element storage guards (#390).
- The `publish-pypi.yml` `backend` input description names `ppp` as the default,
  pypackpack 0.4.1, and the two targets that stay on setuptools.
- README and `docs/locale/README_ko.md` pin `torchnative==0.1.0b10` in the
  `ppp core add` examples.

## 7. Measured but not implemented

- `isnan`, `isposinf` and `isneginf` are missing as `isinf` was (#411).
- 11 CompositeImplicit ops remain in the pending pin (#374, #398).
- The upstream SIGSEGV on `_weight_norm_interface` for a zero-element view with
  stride `(4, 1)` is recorded in `UPSTREAM_SIGSEGV_ROWS`, not reported upstream
  from here (#390).
- The XNNPACK lowering on the shim (#306) is still open. #394, #407 and #410
  each removed one wall (`memory_format` identity, `_to_functionality_key`, the
  HOP recursion); no claim is made that the lowering now delegates.
- The b9 carried-over items are unchanged and were not re-measured: the 82 of
  297 architectures never numerically judged, Intel NPU never run on hardware
  (#21), the Windows tie order (#81), and the on-device ExecuTorch route.

## 8. Gate at this head

Not stated. The notes were written from the pull request descriptions and the
git history before the release run; no gate result for the head of this branch
is claimed, and the pypackpack build of the nine wheels was not run when this
file was written. The dry run of `publish-pypi.yml` with `backend=ppp` decides.
