# 0.1.0b10: release notes

**The wheels are now built with pypackpack (`ppp`), and the shim and the
CompositeImplicit table moved further toward upstream.** The owner approved this
release on 2026-11-21 and chose the build path: "ppp로 배포". Before this
release the publish workflow built every wheel through setuptools; a `v*` tag
push now builds seven of the nine abi3 wheels with pypackpack 0.4.1.
windows-arm64, wasm32-emscripten and the three free-threaded (cp315t) wheels
stay on setuptools, so a release uploads twelve files: seven built by
pypackpack and five by setuptools.

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
  uses `uv tool install pypackpack==<version>` instead of a checkout of
  `pypackpack@develop` and a Gradle build. #409 pinned 0.4.0; the pin moved to
  0.4.1 in both workflows inside #408 (dbd3a9b7).

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
- **CompositeImplicit decomposition bodies (#398, with #397).** 26 more
  `@_cia_composition` bodies in `bootstrap.py` (13 at `v0.1.0b9`, 39 at this
  head, counted by name), matching upstream 2.13.0's logged sequences node for
  node: `clip`, `greater.Scalar/Tensor`, `max.other`, `multiply.Scalar/Tensor`,
  `reshape_as`, `where.Scalar/ScalarOther/default` (#397), and `view_as`,
  `argsort` (`default`, `stable`), `chunk`, `unflatten.int`,
  `broadcast_tensors`, `is_floating_point`, `std` (`default`, `dim`), `var`
  (`default`, `dim`), `var_mean` (`default`, `dim`), `real`, `imag`,
  `rms_norm` (#398). `min.other`, which #397's description lists, has no body
  at this head. The 26 bodies are not the same number as the
  `CustomDecompTable` delta: that table goes from 426 to 444 entries, 18 more
  (#398 states it was confirmed in CI run 44588395741), because several
  bodies cover overloads the table already counted. 11 ops stay in the pending
  pin.
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
- `tests/release/test_cipub.py`: `PPP_PLATFORMS` and
  `test_the_ppp_wheel_filenames_are_named_in_the_workflow` (this release). The
  nine names a `ppp` release ships are now a second independent copy beside
  the setuptools `NINE_PLATFORMS`, and the workflow must spell both sets;
  deleting either `android_24_arm64_v8a` or `ios_13_0_arm64_iphoneos` from the
  collect step turns the test red. Until this test the ppp names were
  asserted nowhere under `tests/release/`.

## 4. Build backend: pypackpack 0.4.1

The owner's decision of 2026-11-21 was "ppp로 배포". What it means in
`publish-pypi.yml`:

- A `v*` tag push (`on.push.tags`; the workflow has no `release:` trigger and
  nothing in `.github/workflows/` creates a tag) and a `workflow_dispatch`
  that passes no backend build with `ppp`; the dispatch input defaults to
  `ppp` too. `setuptools` stays selectable as an input.
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
- **Still setuptools, the free-threaded wheels.** The `build-ft` job has no
  `ppp` branch: its three cp315t wheels (`macosx_11_0_arm64`,
  `manylinux_2_17_x86_64`, `manylinux_2_17_aarch64`) install setuptools,
  wheel, packaging and build with pip and run `build.py --abi ft` whatever
  the backend. So of the twelve
  files a release uploads, seven are pypackpack-built and five are
  setuptools-built.
- **Platform tags the ppp wheels carry.** Android `android_24_arm64_v8a` and
  iOS device `ios_13_0_arm64_iphoneos`. Both are pypackpack's choice: its
  bundler tags Android at the declared `minSdk`, falling back to API 24, the
  level its Cargo backend links every extension for with the NDK's
  `<triple>24-clang`, and tags iOS devices at 13.0 (pypackpack #118,
  2026-11-13; pypackpack `docs/SPEC.md`). The workflow comment at the collect
  step records the API 24 tag as an owner decision of 2026-11-19; the owner's
  words are not quoted anywhere in this repository, so that line is the
  leader's record of the decision, not a quotation (AGENTS.md rule 7). Nothing
  in this repository checks the ppp Android tag against the CPython the
  extension was built for: under `ppp`, `build.py` skips
  `target.platform_tag(...)`, `verify()` checks only that the tag starts with
  an accepted prefix, and `verify_cross.py` checks that `packaging` would
  generate the name; the by-name list in the collect step is the only pin.
  The setuptools path keeps `android_21_arm64_v8a` and
  `ios_12_0_arm64_iphoneos`, read from the target CPython's
  `ANDROID_API_LEVEL`. The "exactly these nine wheels, by name" step of the
  collect job picks the tag set from the backend, so a `ppp` release names the
  new tags and fails on a missing one. The other tags (`macosx_11_0_arm64`,
  `ios_14_0_arm64_iphonesimulator`, the two manylinux 2.17 wheels, `win_amd64`,
  `win_arm64`, `pyemscripten_2026_0_wasm32`) are the same on both paths.
  `tests/release/test_cipub.py` now carries both sets (`NINE_PLATFORMS` for
  setuptools, `PPP_PLATFORMS` for ppp) and requires the workflow to spell
  both; `test_wheelmatrix.py` still asserts `android_21_*`, which is the
  setuptools derivation in `build.py`.

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
- `docs/platform/PUBLISH_CI.md`, which the workflow header says to read before
  changing it, gains §9 on the `backend` input (which targets pypackpack
  builds, where, and which stay on setuptools), and its §4 runner table now
  names the runners the file actually uses (`macos-15`, and `windows-latest`
  for `windows-x86_64` under `ppp`). Its opening sentence "This workflow has
  never run" is pinned by `test_the_doc_says_the_workflow_has_never_run` and
  is left in place with a dated correction beside it.

## 7. Measured but not implemented

- `isnan`, `isposinf` and `isneginf` are missing as `isinf` was (#411).
- 11 CompositeImplicit ops remain in the pending pin (#374, #398).
- The upstream SIGSEGV on `_weight_norm_interface` for a zero-element view with
  stride `(4, 1)` is recorded in `UPSTREAM_SIGSEGV_ROWS`, not reported upstream
  from here (#390).
- The XNNPACK lowering on the shim (#306) is still open. #394, #407 and #410
  each removed one wall (`memory_format` identity, `_to_functionality_key`, the
  HOP recursion); no claim is made that the lowering now delegates.
- Carried over from b9, not re-measured for this release:
  - **`torch.compile` through the stock path.** The b8 notes called it a
    permanent refusal for the structural reason in
    [`../graph/COMPILE.md`](../graph/COMPILE.md) (PEP 523 frame evaluation is
    not available to a Limited-API extension; measured 2026-09-06). #194
    marked that refusal unconfirmed and #208 added the PEP 523-free engine
    that SPEC S1.8 describes; `COMPILE.md` itself carries no banner and was
    not rewritten, and nothing in this range touched it.
  - **Architecture coverage.** [`../architectures/ARCH100.md`](../architectures/ARCH100.md)
    measured 215 of 297 `transformers` architectures forwarding, with 82 not
    forwarding for want of 31 operators. That document is a frozen baseline;
    its banner points at `docs/numerics/AGREE2.md` (2026-09-12), which
    measured 297 of 297 forwarding and 288 of 290 judgeable architectures
    agreeing numerically, with 7 not judgeable. Neither sweep was re-run for
    this release: 82 of 297 is the ARCH100 count the release test names, and
    288 of 290 is the latest number on file.
  - Intel NPU never ran on hardware (#21), the Windows tie order (#81) still
    has no oracle result recorded in these notes, and the on-device ExecuTorch
    route is unchanged.

## 8. Gate at this head

Not stated. The notes were written from the pull request descriptions and the
git history before the release run; no gate result for the head of this branch
is claimed, and the pypackpack build of the nine wheels was not run when this
file was written. The dry run of `publish-pypi.yml` with `backend=ppp` decides.
