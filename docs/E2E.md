# E2E — 토큰 일치를 회귀 스위트에 박아 넣기

지금까지 "shim 이 상류와 같은 토큰을 낸다"는 여러 번 실측됐지만 전부 캐시 디렉터리 아래의
일회성 프로브 스크립트(`caches/bw-sample-probe/`, 커밋 대상 아님)로만 존재했습니다. 이 문서는
그 성질을 `rust/torch_c/pytests/test_shim.py` 의 테스트 세 개로 회귀 스위트에 박아 넣은 기록입니다.

**한 줄 결론.** 상류와 shim 을 한 프로세스에서 동시에 쓸 수 없다고 알려져 있었는데, 실제로는
**된다** — 다만 `import torch` 로 vendor 트리를 통해서가 아니라, shim 을 `torch._C` 가 아닌 독립
모듈 `_C` 로 파일 경로째 로드할 때만 그렇습니다. 그래서 아래 세 테스트는 (a) 상수 고정도 (b)
서브프로세스도 아닌 **제3의 방법** — 같은 프로세스에서 상류 torch 를 살아 있는 채로 비교 — 을
씁니다. 대가와 이유는 §2 에 적었습니다.

---

## 1. 추가한 테스트와 각각이 잡는 것

세 개 모두 `rust/torch_c/pytests/test_shim.py` 끝, `_main()` 바로 앞 새 절에 있습니다. 헬퍼
(`_e2e_*`, `_E2EBackend`)는 세 테스트가 공유합니다.

### 1.1 `test_two_layer_llama_greedy_matches_upstream_token_for_token`

docs/NN_SURFACE.md §7 · docs/SAMPLING.md §3 이 실측한 것: aten 레벨로 손으로 쓴 2 층
Llama 모양 디코더(RMSNorm · RoPE · flash `sdpa` · SwiGLU)를 결정적 가중치로 채우고, 4 스텝
그리디(`argmax`) 디코딩하면 상류와 같은 토큰이 나오는가.

**오늘 이 기계에서 재측정한 값**: `torch=[7,42,3,88,63,63,63,63]`,
`shim=[7,42,3,88,63,63,63,63]` — 일치. 마지막 스텝 로짓의 최대 절대오차 `~2.3e-06`.

허용오차는 `tools/golden/dtypes.py` 의 `TOLERANCES["float32"]`(`atol=rtol=1e-5`)를 그대로
가져다 썼습니다 — 이 테스트가 새로 지어낸 숫자가 아니라, 골든 하네스가 이미 float32 에 대해
"정상적인 부동소수점 반올림"으로 인정하는 경계입니다.

**잡는 것**: `mm`/`_scaled_dot_product_flash_attention_for_cpu`/`silu`/`rsqrt` 등 이 모델이
쓰는 커널의 값 회귀, 그리고 `argmax` 가 상류와 다른 인덱스를 고르는 회귀.

### 1.2 `test_do_sample_matches_upstream_across_configs_and_reseed_modes`

docs/SAMPLING.md §3 이 실측한 15 개 구성 × 6 스텝 = 90 토큰 전부 일치를 그대로 재현합니다.
9 개는 매 추첨 직전 재시딩(같은 시작점 → 같은 값), 6 개는 시드를 한 번만 주고 스트림을 끝까지
흘려보냅니다(더 강한 판정 — 매 스텝의 소비 워드 수까지 같아야 통과).

**잡는 것**: `topk`/`sort`/`le.Scalar`/`scatter.src`/`squeeze.dim`/`multinomial` 의 값 회귀,
그리고 위 커널들의 **난수 소비량**이 상류와 달라지는 회귀 — 재시딩 없이 흘려보내는 6 개 구성이
이것을 잡습니다(값만 맞고 소비량이 다르면 두 번째 스텝부터 갈립니다).

### 1.3 `test_multinomial_matches_upstream_through_a_second_draw`

docs/SAMPLING.md §2 가 실측한 두 알고리즘 분기(`!replacement || n_sample == 1` 이면 Gumbel
argmax/topk, 아니면 누적합+이분탐색)와 그 워드 소비량을, `(n_cat, n_sample, replacement)` 5
조합 × 시드 3 개에서 직접 대조합니다. **첫 번째 뽑기뿐 아니라 재시딩 없이 이어지는 두 번째
뽑기까지** 대조하는 것이 핵심입니다 — 값만 맞고 소비 워드 수가 다르면 1 회는 우연히 맞고 2 회째
에서 갈립니다.

**잡는 것**: 두 알고리즘 중 하나의 값 회귀, 그리고 두 알고리즘의 워드 소비량 회귀(예: 빠른
경로가 실제로는 `n_sample`/`replacement` 와 무관하게 고정 소비량인데 그 특성이 깨지는 경우).

### 이미 있던 것 — 새로 추가하지 않음

`uniform_`/`normal_` 의 비트 단위 스트림 일치는 이미 이 파일에 있습니다
(`test_uniform_matches_torchs_stream_bit_for_bit`, `test_the_same_seed_gives_the_same_stream`,
`test_normal_takes_a_different_path_at_sixteen_elements`,
`test_normal_caches_the_other_half_of_the_pair_on_the_generator`) — 상류 torch 2.13.0 에서 뽑은
값을 상수로 박아 둔 (a) 방식입니다. 이번 작업 범위와 같은 성질(측정된 것을 회귀로 고정)이라
중복해서 새로 만들지 않았습니다.

---

## 2. 왜 (a)/(b) 가 아니라 제3의 방법인가

지시받은 전제는 "상류와 shim 을 한 프로세스에서 동시에 쓸 수 없다 — 벤더 트리가 `torch` 를
가리므로 둘 중 하나만 보인다"였습니다. **그 전제 자체를 검증했더니 절반만 맞았습니다.**

벤더 트리를 `sys.path` 에 넣고 `import torch` 하면 그것은 vendor 의 `torch/__init__.py` 이고,
그 안의 `_C` 서브모듈 자리에 shim 을 심는 것이 `install_shim.sh` 가 하는 일입니다 — 이 경로에서는
**상류 torch 가 아예 존재하지 않으므로** "동시에 쓴다"는 개념이 성립하지 않습니다. 이것이 원래
전제가 가리키는 상황입니다.

하지만 `tools/golden/loader.py` 의 `load_shim()` 은 그 경로를 쓰지 않습니다. 빌드된 `.dylib` 를
임시 디렉터리에 `_C.so` 로 복사한 뒤 `importlib.util.spec_from_file_location("_C", ...)` 로
**`_C` 라는 이름의 독립 모듈**로 로드합니다 — `torch._C` 가 아닙니다. 이 경로에서 별도로
`sys.path` 에 vendor 를 넣지 않고 `import torch` 하면, 그것은 `spike-venv` 에 실제로 설치된
**상류 torch 2.13.0** 입니다. 둘은 `sys.modules` 에서 서로 다른 키(`"_C"` vs `"torch"`/`"torch._C"`)를
쓰고, 실제로 충돌하지 않습니다 — `caches/bw-sample-probe/e2e_sample.py` 와 `verify_ops.py` 가
이미 이 패턴으로 오늘 실행됐고, 이번 작업에서 이 worktree 의 아티팩트로 다시 돌려 확인했습니다
(§4 참고).

그래서 이 테스트들은 매 실행마다:

1. `import torch` — 상류, `spike-venv` 에 실제 설치된 것 (이 파일은 vendor 를 `sys.path` 에
   넣지 않습니다)
2. `_C` — 이 파일 맨 위에서 이미 `import _C` 로 불러온 shim, `_C.so` 를 통해 로드된 독립 모듈

를 **같은 프로세스에서 직접 대조**합니다. 이것도 결국 "환경에 의존한다"는 (b)의 대가를 그대로
지지만(§3), 서브프로세스 왕복이나 파일로 덤프해 대조하는 비용은 없습니다.

---

## 3. 대가 — 명시

- **상류 torch 가 설치된 인터프리터에서만 의미가 있습니다.** `spike-venv/bin/python` 처럼 실제
  torch 2.13.0 이 깔린 인터프리터로 돌려야 합니다. 그렇지 않은 인터프리터(이 파일의 docstring이
  약속하는 "테스트 의존성 없음" 기본 경로 포함)에서는 세 테스트 모두 **아무것도 검증하지 않고
  조용히 통과**합니다 — `_upstream_torch is None` 이면 즉시 반환합니다. `pytest.skip` 을 안 쓴
  이유는 이 파일이 애초에 pytest 에 의존하지 않기 때문입니다(파일 docstring, `_main()` 참고).
  **이것이 이 접근의 가장 정직해야 할 지점입니다**: 이 세 테스트가 초록인 것이 "shim 이 맞다"를
  증명하는 것과 "상류 torch 가 안 보였다"를 증명하는 것 중 어느 쪽인지, 실행 로그만 봐서는
  구별되지 않습니다. §5 에서 다시 적습니다.
- **torch 버전에 못박히지 않습니다.** (a) 방식과 달리 이 테스트들은 실행 시점에 설치된 상류
  torch 를 그대로 기준으로 삼으므로, `spike-venv` 의 torch 가 업그레이드되면 기준값도 따라
  바뀝니다 — 유지보수 비용이 없는 대신, "정확히 어느 버전에서 측정됐는가"를 실행 로그가 아니라
  별도로 기록해야 합니다. 이번 실측은 **torch 2.13.0** 입니다(`spike-venv` 고정, 2026-08-24).
- **속도**: 상류 torch 의 `import` 자체가 무겁습니다. 아래 §6 참고.

---

## 4. 각 테스트가 실제로 빨간지 확인한 방법

세 테스트 모두, **shim 쪽 입력만** 건드려 상류와 의도적으로 어긋나게 만든 뒤(=커널은 그대로,
`src/` 는 전혀 건드리지 않음 — 다른 두 에이전트가 그 디렉터리에서 동시에 작업 중이었으므로),
`PYTHON=$PY ./pytests/run.sh` 로 다시 돌려 `FAIL` 로 잡히는지 보고, 원래대로 되돌린 뒤
`git diff`(정확히는 되돌린 파일과 되돌리기 전 사본의 `diff`)로 완전히 원상복구됐는지 확인했습니다.

| 테스트 | 무엇을 깼나 | 결과 |
|---|---|---|
| greedy | shim 쪽 프롬프트 마지막 토큰만 `88` → `89` | `FAIL ...: AssertionError: ([7, 42, 3, 88, 63, 63, 63, 63], [7, 42, 3, 89, 68, 63, 48, 63])` |
| do_sample | shim 쪽 재시딩 시드에 `+1` | `FAIL ...: AssertionError: ('reseed', 1.0, 50, 0.95, 0, [7, 42, 3, 88, 58, 63, 41, 78, 83, 78], [7, 42, 3, 88, 63, 41, 85, 13, 47, 91])` |
| multinomial | 두 번째 뽑기 직전 shim 쪽에서만 `multinomial` 을 한 번 더 불러 스트림을 한 칸 밀어놓음 | `FAIL ...: AssertionError: (5, 3, True, 0, 'draw2')` |

세 개 다 깬 뒤 개별적으로 `pytests/run.sh` 를 다시 돌려 `EXIT=1` 과 위 `FAIL` 줄을 직접 봤고,
그때마다 `cp` 로 떠 둔 사본으로 되돌린 뒤 `diff` 로 바이트 단위 동일함을 확인했습니다. 마지막에
`git status --short` 로 이 worktree 전체에서 `rust/torch_c/pytests/test_shim.py` 한 파일만
변경됐음을(= `src/` 무손상) 확인했습니다.

---

## 5. §3 의 정직하지 못한 지점을 메우는 법 — 지금 당장은 안 함

이상적으로는 이 스위트 자체가 "`_upstream_torch is None` 이었다"를 최종 리포트에 드러내야
합니다(현재는 `ok` 로만 찍힙니다 — 통과와 무보증 통과가 구별되지 않습니다). `_main()` 의 출력
형식을 바꾸는 일이라 이 작업의 파일 범위(`test_shim.py` 는 되지만, 판정 기준이 요구하는
`pytests/run.sh` 의 exit 코드 계약을 건드리는 것은 이 작업 하나의 판단으로 정하기엔 큽니다)
밖으로 보고 손대지 않았습니다. **알려진 미해결 사항으로 남깁니다.**

---

## 6. 스위트 실행 시간 변화

`PYTHONPATH=<stage> python3 pytests/test_shim.py` 단독 실행, `spike-venv` 인터프리터:

| | 테스트 수 | wall time |
|---|---|---|
| 이전 | 62 | 0.51s |
| 이후 | 65 | 2.23s |

**약 1.7 초 증가**, 대부분 상류 `import torch`(무거움) + 15 개 구성 × 2 backend × 6 스텝
forward pass 비용입니다. `pytests/run.sh` 전체(cargo 증분 빌드 포함)는 2.3 초로, "몇 분씩"
걸리는 영역과는 자릿수가 다릅니다.

---

## 7. 넣지 않은 것과 이유

- **`torch.nn` 으로 직접 조립한 버전** (docs/NN_SURFACE.md §7 이 측정한 `nn.Linear` +
  `F.scaled_dot_product_attention` + `F.silu` 조립, 상대오차 `~5.8e-07`). 이 shim 은
  `torch.<op>`/`nn.Module` 스펠링이 아니라 `_aten_dispatch` 직접 호출로만 접근 가능하고
  (`overloads.json`/`bootstrap.py` 경유 없이), `nn.Module` 조립을 재현하려면 상류 쪽에서
  `torch.nn.Linear` 등 실제 파이썬 계층을 인스턴스화해야 하는데 이는 이미 §1.1 의
  aten-레벨 버전이 검증하는 것과 같은 커널 집합을 다른 진입점으로 한 번 더 재는 것이라, 새로운
  회귀 포착 범위를 늘리지 않습니다. aten 레벨 버전 하나로 충분하다고 판단했습니다.
- **15 개 구성 전부가 아니라 대표만 넣는 것**은 고려했지만 실측 결과 15 개 전부가 2 초 미만이라
  줄일 이유가 없었습니다. 전부 넣었습니다.
- **`topk` 의 동점 순서 재현**은 넣지 않았습니다. docs/SAMPLING.md §4.2 가 이미 "재현하지
  않기로 했다"고 명시한 범위이고, 이 작업은 구현을 건드리지 않으므로 그 판단을 뒤집을 근거가
  없습니다.
- **fp16/bf16 경로의 `normal_`/`_softmax` 비-마지막-축 오차** (docs/RNG.md §3.3, docs/SAMPLING.md
  §4.3 이 이미 "측정했으나 원인 미확인"으로 남긴 것)는 새 테스트로 옮기지 않았습니다. 그 오차는
  샘플링 경로가 물지 않는다고 이미 측정되어 있고(로짓은 float32, softmax 는 마지막 축), 여기서
  구현하는 3 개 테스트의 범위 밖입니다.
- **`pytests/run.sh`/`_main()` 자체를 고쳐 "상류 torch 없음"을 별도 상태로 보고하는 것**은
  §5 에 적은 대로 일부러 손대지 않았습니다.

---

## 8. 모르는 것 — 명시

- **CI 나 다른 사람의 로컬 환경에 상류 torch 가 없으면 이 세 테스트는 아무것도 검증하지 않고
  조용히 통과합니다.** 이것이 회귀를 실제로 놓치는 상황인지(즉 `spike-venv` 가 아닌 환경에서
  이 스위트가 돌 일이 실제로 있는지)는 확인하지 않았습니다.
- **AVX2/VSX 같은 비-aarch64 호스트**에서 이 세 테스트가 여전히 통과하는지 확인하지 않았습니다
  — docs/RNG.md §3.3 이 이미 "스칼라 경로 밖은 미확인"이라고 적어 둔 것과 같은 범위이고, 이
  작업은 Apple Silicon 에서만 실행했습니다.
- **상류 torch 버전이 2.13.0 에서 바뀌면** 이 세 테스트의 기준값도 함께 바뀝니다 — §3 에 적은
  대로 이것은 (a) 방식과 다른 성질이고, 버전이 바뀐 뒤 실패한다면 "shim 이 깨졌는지 상류가
  바뀌었는지"는 `git log` 로 `spike-venv` 의 torch 버전을 대조해야 답할 수 있습니다. 이 문서
  작성 시점 기준값(torch 2.13.0)은 §3 에 기록해 뒀습니다.
