# Vulkan — 건너뛴 것이 통과로 세어지던 게이트, 그리고 네 커널을 실제로 잰 라운드

**결론부터.** 이전 회차는 `gelu`·`_softmax`·`native_layer_norm`·`bmm` 네 커널을 구현하고 "AGREES 기준을
만족하며 새 테스트가 통과한다"고 보고했습니다. **그 보고는 아무것도 재지 않은 상태에서 나왔습니다.**
게이트는 `0 FAIL` 이었지만 Vulkan 테스트는 전부 건너뛰었고, 건너뛴 테스트는 `ok` 로 세어졌으며, 새
테스트 다섯 개는 **어떤 기계에서도 실행될 수 없는 위치**에 있었습니다.

이번 회차에서 네 커널을 **실제로 돌려 upstream 과 원소 단위로 대조**했습니다. 그 과정에서 넷 중 셋의
결함 다섯 개와, 기존 커널 `div` 가 드라이버에 따라 비트가 어긋나는 문제를 찾아 고쳤습니다.

| 질문 | 답 |
|---|---|
| 네 커널은 **구현됨**인가 | 예. `_vulkan_ops()` 에 있고 셰이더가 돈다 |
| 네 커널은 **일치함**(AGREES)인가 | **예 — float32, 연산자 단위, Apple M1, 두 드라이버(MoltenVK 1.4.2 · kosmickrisp).** `docs/numerics/AGREE.md` §2 의 유도 허용치 안 (§3) |
| 그 판정은 어떤 기계에서든 참인가 | **아니다.** 로더가 없는 기계에서는 **재지 않은 것**이고, 이제 게이트가 그렇게 말한다 (§2) |
| float16 / bfloat16 / float64 는 | **숫자가 없다.** 장치에 올라가는 순간 이름을 대며 거절한다. 일치 주장도 없다 |
| 트랜스포머가 Vulkan 에서 도는가 | **0 개.** `embedding` 이 첫 op 이고 미구현이다 (§4) |
| 성능은 | 재지 않았다. `docs/devices/VULKAN2.md` §4.4 의 이유 그대로 |

<!-- DOCWATCH: count vulkan_tests_ok ge 19 -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/vulkan_coverage.py UNVERIFIED present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/run.sh vulkan_coverage.py present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/vulkan.rs MVK_CONFIG_FAST_MATH_ENABLED present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/vulkan.rs loader_candidates_for present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vulkan4.py test_native_layer_norm_agrees_with_upstream_at_a_derived_tolerance present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vulkan4.py test_bmm_agrees_with_upstream_and_is_the_kernel_it_claims_to_be present -->

위의 `count vulkan_tests_ok` 마커는 **로더가 없는 기계에서는 PASS 가 아니라 SKIP** 으로 보고됩니다
(§2.3). 이 문서의 일치 주장은 그 기계에서 확인된 적이 없다는 뜻이고, DOCWATCH 출력이 그렇게 말합니다.

---

## 1. 로더는 어디 있었나 — "사라졌다"도 "실행된 적 없다"도 아니었다

지시는 둘 중 하나를 물었습니다: *그때는 로더가 있었고 지금은 없다*, 또는 *그 문서의 주장은 실행된 적이
없다*. **둘 다 아니었습니다.**

- `docs/devices/VULKAN2.md` §4 (커밋 `502bb60`) 는 **실제로 실행했습니다.** 시스템 경로(`/usr/local/lib`,
  `/opt/homebrew/lib`, `~/VulkanSDK`)에는 아무것도 없음을 확인하고, **안드로이드 에뮬레이터 번들 안의
  로더**(`~/Library/Android/sdk/emulator/lib64/vulkan/libvulkan.dylib`)를 `DYLD_LIBRARY_PATH` 와
  `VK_DRIVER_FILES` 로 가리켜 `vk_probe` 를 돌렸고, `kosmickrisp` → `"Apple M1"` 에서 `RESULT: PASS` 를
  기록했습니다. §4.5 에 **"아무것도 설치하지 않았다"**고 적었습니다.
- `docs/devices/VULKAN3.md` §4 는 같은 경로로 `_C._vulkan_probe()` 를 돌려 `available: True` 를 봤고, §6.1 은
  macOS SIP 가 `/bin/sh` 를 거치며 `DYLD_*` 를 벗기는 함정을 기록했습니다.
- `docs/devices/VULKAN4.md` §8.1 은 그 함정을 피하는 opt-in 변수 `TORCHNATIVE_VULKAN_DYLD` 를 `run.sh` 에
  넣었습니다. `CLAUDE.md` §2 가 이 변수를 게이트 명령과 함께 적어 둡니다.

**그 로더는 2026-09-14 에도 디스크에 그대로 있습니다.** 조율 세션의 게이트는 그 opt-in 을 주지 않고
돌았고, 그래서 Vulkan 테스트는 **사실대로** "로더 없음"을 말하며 건너뛰었습니다. 문제는 로더가 아니라
**그 건너뜀이 통과로 세어진 것**이었습니다 (§2).

### 1.1 Homebrew 로더를 찾게 한 방법

조율 세션이 이번 회차 중간에 Homebrew 로 `molten-vk` 1.4.2 와 `vulkan-loader` 1.4.357.0 을 설치했습니다.
그런데 셰임은 여전히 찾지 못했습니다 — 실측:

```
ctypes.CDLL("libvulkan.dylib")                   -> no such file
ctypes.CDLL("/opt/homebrew/lib/libvulkan.dylib") -> 로드됨
```

Homebrew 의 접두사는 dyld 기본 탐색 경로 밖입니다. `ash::Entry::load()` 는 맨 이름 하나만 시도하므로,
`vulkan.rs` 에 **`loader_candidates_for`** 를 두어 순서대로 시도합니다:

1. `libvulkan.dylib` — **맨 이름이 여전히 첫째.** `DYLD_LIBRARY_PATH`(곧 `TORCHNATIVE_VULKAN_DYLD`)가
   로더를 고르는 기존 방식이 그대로 이깁니다.
2. `libvulkan.1.dylib`
3. `$VULKAN_SDK/lib/libvulkan.1.dylib` — LunarG SDK 의 관례. 비어 있으면 넣지 않습니다.
4. `/opt/homebrew/lib/libvulkan.1.dylib` — Apple Silicon Homebrew
5. `/usr/local/lib/libvulkan.1.dylib` — Intel Homebrew

**이것들은 가정이 아니라 대체 경로입니다.** 없는 경로는 실패한 `dlopen` 하나이고, 전부 실패하면 오류가
**시도한 모든 경로와 각각의 이유**를 나열합니다. 리눅스는 동적 링커의 soname 탐색(`libvulkan.so.1`,
`libvulkan.so`)만 쓰고 절대 경로를 추측하지 않습니다. 환경변수 대신 이것을 고른 이유: Homebrew 로 설치한
사용자는 아무것도 설정하지 않았고, 설정해야 한다는 사실을 알 방법도 없었습니다.

`_vulkan_probe()` 는 이제 **어느 로더를 열었는지**(`loader`)도 보고합니다. 측정: 환경변수 없이
`/opt/homebrew/lib/libvulkan.1.dylib` → `Apple M1 (INTEGRATED_GPU)`, `TORCHNATIVE_VULKAN_DYLD` 를 주면
`libvulkan.dylib`(에뮬레이터 번들) → `Apple M1`.

MoltenVK 는 **portability 드라이버**라서, 로더가 `VK_KHR_portability_enumeration` 을 요청하지 않는
인스턴스에는 보이지 않습니다 — `docs/devices/VULKAN2.md` §4.2 가 그 거부 문구를 그대로 기록해 두었습니다. 로더가 그
확장을 제공할 때만 켭니다(안드로이드 로더가 받는 요청은 전과 같습니다). 장치가 `VK_KHR_portability_subset`
을 광고하면 그것도 켭니다 — 확장 명세가 요구합니다.

---

## 2. 거짓 초록 — 세 경로, 그리고 그것을 표현할 수 없게 만든 것

### 2.1 무엇이 초록을 만들었나

1. **건너뛴 테스트가 `ok` 로 세어졌다.** Vulkan 테스트는 `(skipped ...)` 를 출력하고 *정상 반환*했고,
   러너는 예외가 없으면 `ok` 를 찍었습니다. `test_shim.py` 와 `test_vulkan4.py` 의 모든 Vulkan 테스트가
   그랬습니다. 게이트 로그에 건너뜀 문구는 있었지만 **아무도 세지 않았습니다.**
2. **새 테스트 다섯 개가 `if __name__ == "__main__": raise SystemExit(_main())` 뒤에 붙어 있었다.**
   스크립트는 그 줄에서 끝나므로, 다섯 함수는 `_main` 이 전역을 훑을 때 **정의조차 되어 있지 않았습니다.**
   로더가 있든 없든 **어떤 기계에서도 실행된 적이 없습니다.** 게이트 로그에 이름이 없던 이유가 이것입니다.
   (그 테스트들의 허용치는 `1e-5`/`1e-4` 로 고른 상수였고 유도된 것이 아니었습니다. 이번에 삭제하고 §3 의
   테스트로 대체했습니다.) 또 기존 `test_a_transformer_still_does_not_forward_and_the_wall_is_named` 은
   네 op 이 **거절됨**을 단언했으므로, Vulkan 이 한 번이라도 돌았다면 네 커널을 넣은 그 회차에서 빨개졌어야
   합니다 — 코드를 읽어 도출한 사실이며, "테스트가 통과한다"는 보고는 아무것도 돌지 않았기 때문에만
   가능했습니다.
3. **모듈 forward 의 하위 프로세스가 로더를 못 찾으면 `(skipped ...)` 후 `ok`.** 이번 회차에서
   **실측**했습니다: 부모 프로세스는 Homebrew 로더를 찾는 새 빌드였고, 벤더 트리의 `_C` 는 아직 옛 빌드라
   하위 프로세스만 로더를 못 찾았습니다. 그 테스트는 `ok` 였습니다.

### 2.2 이제 건너뜀은 세 번째 결과다

`rust/torch_c/pytests/vulkan_coverage.py`:

- 테스트의 건너뜀 도우미가 `vulkan_skip(reason)` 을 기록하고, 러너는 그 테스트에 대해 `ok` 대신
  **`SKIP test_x: <로더 자신의 문구>`** 를 찍습니다.
- 스위트는 끝에 `VULKAN: ran=R ok=K failed=F skipped=S device=...` 한 줄을 찍습니다. 한 번도 Vulkan 을
  건드리지 않은 테스트는 어느 칸에도 들어가지 않습니다.
- `run.sh` 는 스위트마다 출력을 보관하고, 전체를 더해 이렇게 보고합니다:

```
VULKAN COVERAGE: 0 ran (0 ok, 0 failed) / 24 skipped -- device: none
VULKAN COVERAGE: UNVERIFIED -- no Vulkan test executed in this run. A green gate here says nothing about any Vulkan kernel.
```

- **`TORCHNATIVE_REQUIRE_VULKAN=1`** 이면 건너뜀이 하나라도 있거나 아무것도 돌지 않았을 때 게이트가
  실패합니다. **Vulkan 커널에 대한 증거로 내밀 수 있는 실행은 이 형태뿐입니다.** 로더가 없는 CI 는 이것을
  주지 않으므로 여전히 초록이지만, 그 초록은 `UNVERIFIED` 라는 말과 함께 나옵니다.

### 2.3 DOCWATCH — 문서가 재지 않은 일치를 주장할 수 없게

`tools/docwatch/check_docs.py` 에 `count vulkan_tests_ok` 원천을 더했습니다. `test_vulkan4.py` 를 다시
돌려 `VULKAN:` 줄의 `ok` 를 셉니다. **`ran == 0` 이면 값을 돌려주지 않고 `LiveFactsSkip` 을 던져 그 마커를
SKIP 으로, 이유와 함께 보고합니다.** PASS 는 이 회차가 끝내려던 거짓 초록을 되풀이하고, FAIL 은 GPU 가 없는
모든 CI 를 아무도 시험할 수 없었던 주장 때문에 빨갛게 만듭니다. 규칙대로 `ge` 입니다.

이 장치가 **볼 수 없는 것**: `vulkan_skip()`/`vulkan_used()` 를 부르지 않는 Vulkan 테스트는 어느 칸에도
세어지지 않습니다. `test_vulkancov.py` 가 두 스위트의 도우미와 러너를 **행동으로** 확인하지만(소스 grep 이
아님), 세 번째 스위트가 자기만의 건너뜀 경로를 새로 만들면 보이지 않습니다.

---

## 3. 네 커널 — 값

모든 수치는 **upstream 을 float64 로 다시 돌린 값을 진실로 한 텐서 단위 상대오차**이고, 허용치는
`docs/numerics/AGREE.md` §2 의 규칙으로 **op 마다 이 모집단에서 다시 유도**했습니다(upstream float32 자신의 오차의
p90, 8 ulp 바닥). 네 op 모두 p90 이 바닥보다 작아 허용치는 `9.537e-07` 입니다. **허용치를 넓힌 곳은 없습니다.**

측정 환경: Apple M1, 2026-09-14. **MoltenVK** = Homebrew `vulkan-loader` 1.4.357.0 + `molten-vk` 1.4.2(환경변수
없음). **kosmickrisp** = 에뮬레이터 번들 로더 + `libkosmickrisp_icd.json`.

| op (케이스) | 최악 셰임 오차 MoltenVK | 최악 셰임 오차 kosmickrisp | 그 케이스 upstream 자신의 오차 | 원소 최악 (upstream 자신의 원소 오차 대비) MVK / kosmic |
|---|---|---|---|---|
| `gelu` none (4) | 5.791e-08 | 5.791e-08 | 5.791e-08 | 5.94× / 28.07× |
| `gelu` tanh (4) | 4.961e-08 | 5.572e-08 | 4.097e-08 | 165.90× / 164.10× |
| `_softmax` (6) | 2.499e-07 `[2,3,64]` | 2.330e-07 `[2,512]`×8 | 8.305e-08 / 7.744e-08 | 3.48× / 25.47× |
| `native_layer_norm` out (7) | 3.489e-07 | 4.017e-07 | 1.144e-07 | 77.47× / 84.01× |
| `native_layer_norm` mean (7) | 3.187e-07 | 3.187e-07 | 3.041e-08 | 2.67× / 2.67× |
| `native_layer_norm` invstd (7) | 3.116e-07 | 3.731e-07 | 6.540e-08 | 2.61× / 3.13× |
| `bmm` (4) | 3.846e-07 | 3.846e-07 | 2.466e-07 | 7.56× / 7.56× |

`bmm` 은 허용치에 더해 **네 shape 모두, 두 드라이버 모두에서** 행렬곱 커널의 호스트 모델(배치마다 순차
float32 누적 + FMA)과 **비트 단위로 0 개 어긋남**입니다 — 배치 오프셋 결함은 "작은 오차"로 보여 허용치로는
못 잡지만 이 비교는 잡습니다(§5 에서 실제로 잡았습니다).

`native_layer_norm` 은 `weight`/`bias` 의 네 조합과 1-D·2-D `normalized_shape`, 그리고 **세 출력 모두**와
그 shape 를 비교합니다.

### 3.0 숨기지 않는 것 — 원소 단위로는 멀리 떨어진 곳이 있다

위 표의 마지막 열은 **단언하지 않고 출력만 합니다.** `AGREE.md` §2 의 규칙은 텐서 단위이기 때문입니다. 하지만
작은 원소에서 upstream 자신의 오차의 수십~백 배 떨어지는 곳이 있고, 원인을 kosmickrisp 에서 원소 하나씩
확인했습니다:

- **`gelu` none, 28×**: 입력 −2.3616, 진실 −0.0214846, 셰임 오차 7.2e-08 vs upstream 1.1e-09. 셰이더는 정확한
  `erf` 가 아니라 **Abramowitz–Stegun 7.1.26 근사**(|ε| ≤ 1.5e-7)를 씁니다. `docs/devices/VULKAN4.md` §6.1 이
  "근사는 정밀도가 아니라 다른 함수"라고 경고한 그 선택이고, 텐서 단위로는 upstream 과 같은 오차이지만
  **꼬리의 작은 원소에서는 근사 오차가 드러납니다.**
- **`gelu` tanh, 164×**: 입력 −3.2127, 진실 −0.0017890, 셰임 오차 9.5e-08 vs 5.8e-10. `1 + tanh(…)` 가
  tanh ≈ −1 근처에서 상쇄되고, GPU 의 `tanh` 가 upstream libm 보다 몇 ulp 부정확합니다.
- **`native_layer_norm` out `[1,512]`, 84×**: 원소 −0.0044 에서 오차 4.8e-08 vs 5.7e-10. 원인은 평균입니다 —
  셰임 0.42679566, upstream 0.42679581, 진실 0.42679580. **512 항을 float32 로 순차 누적**해 upstream 의 약
  10 배 오차가 나고, 그것이 `x − mean` 이 작은 원소에서 커집니다. `mean[1,512]` 은 텐서 단위로도 upstream
  자신의 오차의 10.5 배입니다 — 허용치 안이라 `AGREE.md` 의 4 배 규칙은 적용되지 않지만, **적용됐다면 넘었을
  숫자**입니다.

### 3.1 기존 커널 `div` 가 MoltenVK 에서 upstream 과 비트가 달랐다

네 커널을 재려고 MoltenVK 로 처음 돌렸을 때 **`test_the_exactly_rounded_ops_are_bit_identical_to_upstream`
이 실패했습니다**: `div[4, 5]` 20 원소 중 3 개. 같은 바이너리가 kosmickrisp 에서는 통과했습니다.
`docs/devices/VULKAN4.md` §4.1 의 "33 케이스 비트 동일"은 **kosmickrisp 에서만 참이었습니다.**

원인은 MoltenVK 가 **Metal 셰이더를 기본으로 fast-math 로 컴파일**하는 것이고, fast-math 는 `a / b` 를
다시 쓰는 것을 허용합니다. 가설을 먼저 확인했습니다: `MVK_CONFIG_FAST_MATH_ENABLED=0` 을 주면 19/19.
고친 방식은 환경변수가 아니라 MoltenVK 가 설정용으로 문서화한 **`VK_EXT_layer_settings`**(레이어 이름
`"MoltenVK"`)입니다 — 이 인스턴스만 IEEE 결과를 요청하고, 사용자의 환경은 건드리지 않습니다. 설정 이름은
**측정으로** 정했습니다:

```
설정 이름                          div 원소 중 upstream 과 비트가 다른 수
MVK_CONFIG_FAST_MATH_ENABLED       0 / 4149
FAST_MATH_ENABLED               1199 / 4149
fastMathEnabled                 1199 / 4149
NOPE_CONTROL (대조군)             1199 / 4149
```

### 3.2 물려받은 커널에서 고친 결함

| 결함 | 이전 동작 | 지금 |
|---|---|---|
| `native_layer_norm` 의 `mean`/`invstd` shape | `input.shape[:axis]` (`[4]`) | upstream 과 같은 `[4, 1]` |
| `weight=None, bias=b` | **`b` 를 조용히 버림** — 한 비트로 "affine 있음"만 전달 | 비트 둘. 네 조합 모두 일치 |
| `weight=w, bias=None` | 이름 없이 거절 | 구현, 일치 |
| `normalized_shape` 가 입력 꼬리와 다름 | **검사 없음** — 엉뚱한 구간을 정규화, 입력보다 길면 `usize` 언더플로 | upstream 문구로 `RuntimeError` |
| `weight`/`bias` shape 불일치 | 검사 없음 | `RuntimeError` |
| `gelu(approximate="foo")` | **정확한 gelu 를 계산해 돌려줌** | `approximate argument must be either none or tanh.` |
| `_softmax` dim 범위 밖 | "마지막 차원만" 이라는 틀린 이유로 거절 | `IndexError: Dimension out of range` |
| `_softmax(half_to_float=True)` | "not supported on CPU" (장치 이름이 틀림) | vulkan 장치 이름으로 거절 |
| `bmm` shape 불일치 | `NotImplementedError: shape mismatch` | upstream 문구의 `RuntimeError` |
| 디스크립터 풀 | 바인딩 수와 무관하게 `3 × 64` | `바인딩 수 × 64` (`native_layer_norm` 은 6) |

모든 거절은 **셰이더를 돌리기 전에** 일어나며, 테스트가 `shader_dispatches` 증가 0 으로 단언합니다.

---

## 4. 남은 벽 — Vulkan 에서 도는 트랜스포머는 0 개

`aten.embedding.default` 는 여전히 `_vulkan_ops()` 에 없습니다. 이유는 구조적입니다: **`vulkan.rs` 는 장치
텐서를 float32 로만 저장**하고, 인덱스 텐서는 int64 입니다. int64 텐서를 올리면 이름을 대며 거절됩니다
(`test_a_transformer_still_does_not_forward_and_the_wall_is_named` 이 단언). `embedding` 은 모든 트랜스포머
forward 의 **첫 op** 이므로, 네 커널이 생겨도 **트랜스포머는 두 번째 op 에 도달하지 못합니다.**
`docs/devices/VULKAN4.md` §2 의 추적에 있던 `expand` 도 여전히 미구현입니다.

이 판정은 `_vulkan_ops()` 와 int64 업로드 거절로 잰 것이지, **이번 회차에 BERT 를 Vulkan 에서 돌려 멈추는
곳을 본 것이 아닙니다.** 네 커널의 일치는 **연산자 단위**이고, 그것들이 모델 안에서 조합되어 도는지는
`embedding` 이 생긴 뒤에야 잴 수 있습니다.

---

## 5. 무력화 — 일부러 깨고 빨개지는지 봤다

`CLAUDE.md` §5.5. Rust/셰이더 쪽은 서로 다른 테스트가 잡는 것끼리 묶어 빌드 세 번으로, 파이썬 쪽은
`/tmp` 의 사본에서 하나씩 했습니다. **초록으로 남은 무력화는 없었습니다.**

| # | 무력화 | 결과 |
|---|---|---|
| A1 | `ci.push_next(&mut layer_settings)` 제거 (fast-math 다시 켜짐) | **FAIL** `div[4, 5] is not bit-identical to upstream: 3 of 20` |
| A2 | 물려받은 결함 재현: `bias` 비트를 `weight` 가 있을 때만 | **FAIL** `out[3, 64] [64] bias: shim 7.579e-01 … exceeds 9.537e-07` |
| A3 | `gelu` 의 `approximate` 검사 제거 | **FAIL** `gelu approximate was computed instead of refused` |
| A4 | `bmm` 셰이더가 항상 `mat2` 의 0 번 배치를 읽게 | **FAIL** `bmm[2,3,4,5]: shim 1.272e+00 … exceeds` |
| B1 | `mean`/`invstd` shape 의 꼬리 1 제거 | **FAIL** `mean has shape [4], upstream [4, 1]` |
| B2 | `normalized_shape` 꼬리 비교 제거 | **FAIL** `layer_norm shape mismatch was computed instead of refused` |
| C1 | Homebrew 대체 경로 두 줄 제거 | 처음엔 **초록** (§5.1). 고친 뒤 **FAIL** `['/opt/homebrew/lib/libvulkan.1.dylib'] exist on disk but the extension never looks there` |
| P1 | 러너가 건너뜀에도 `ok` 를 찍게 | **FAIL** ×2 (`ok   test_needs_vulkan`) |
| P2 | 요약이 `UNVERIFIED` 를 말하지 않게 | **FAIL** ×3 |
| P3 | `TORCHNATIVE_REQUIRE_VULKAN` 무시 | **FAIL** ×3 |
| P4 | DOCWATCH 원천이 `ran == 0` 에도 숫자를 돌려주게 | **FAIL** `count vulkan_tests_ok ge 19 -- vulkan_tests_ok = 0` (SKIP 이어야 함) |
| P5 | `test_vulkan4._vulkan_or_skip` 이 기록하지 않게 | **FAIL** ×4 |
| P6 | `test_shim._vulkan_or_skip` 이 기록하지 않게 | **FAIL** ×4 |
| P7 | `test_shim._main` 을 옛 루프로 되돌림 | **FAIL** `test_both_vulkan_suites_use_the_counting_runner` |

### 5.1 초록으로 남은 무력화 하나 — 그리고 그것이 이 회차 자신의 테스트였다

C1 은 처음에 **`exit=0`** 이었습니다. `test_a_loader_installed_where_this_build_searches_is_found` 의 첫
판은 검사할 경로 목록을 `_C._vulkan_loader_candidates()` — **시험 대상 자신** — 에서 읽었습니다. 대체
경로를 지우면 그 경로가 비교 대상 목록에서도 같은 순간에 사라지므로, 테스트는 "디스크에 있는 후보가
없다"며 반환했습니다. §2 의 거짓 초록과 같은 모양이 이 회차가 쓴 테스트 안에서 다시 나온 것입니다.
그 실행의 `VULKAN:` 줄은 `ran=0 … skipped=19` 였으니 **게이트 요약은 UNVERIFIED 를 말했을 것**이지만,
테스트 자체는 아무것도 증명하지 않았습니다.

고친 판은 알려진 설치 위치(두 Homebrew 접두사, `$VULKAN_SDK/lib`)를 **테스트 쪽에 따로** 적고, 디스크에
있는 것이 후보 목록에 없으면 실패합니다. 같은 무력화로 빨개지는 것을 확인했습니다.

---

## 6. 게이트

---

## 7. 하지 않은 것

- **성능.** 두 드라이버 모두 Metal 번역 계층입니다.
- **실물 Adreno/Mali.** 이번 회차의 모든 숫자는 Apple M1 하나입니다.
- **정확한 `erf`**, **보상 합산(Kahan) 평균**, **마지막이 아닌 차원의 softmax** — §3.0 의 원소 단위 편차와
  §3.2 의 거절이 그 흔적입니다.
- **`embedding`** — 장치에 int64 저장이 필요하고, 그것은 `check_dtype` 의 "float32 only" 를 넓히는 설계
  변경입니다.
- **로더가 없는 기계에서 네 커널이 맞는지** — 정의상 잴 수 없고, 게이트와 DOCWATCH 가 그렇게 말합니다.
