# Vulkan — `embedding` 을 열고, 트랜스포머 블록을 장치 위에서 실제로 돌린 라운드

**결론부터.** `docs/devices/VULKAN5.md` 는 네 커널(`gelu`·`_softmax`·`native_layer_norm`·`bmm`)을
upstream 과 원소 단위로 대조해 **일치**를 확인했지만, 같은 문서의 §4 가 적은 대로 **Vulkan 에서 도는
트랜스포머는 0 개**였습니다. `embedding` 이 모든 트랜스포머 forward 의 첫 op 이고, 그 인덱스 피연산자는
int64 인데 `vulkan.rs` 의 `check_dtype` 은 float32 만 받았기 때문입니다.

이번 회차에 그 벽을 열었고, **트랜스포머 블록 하나가 장치 위에서 끝까지 돌았습니다.**

| 질문 | 답 |
|---|---|
| `embedding` 은 구현됐나 | 예. `_vulkan_ops()` 에 있고 `embedding_i32_f32` 셰이더가 돈다 |
| `embedding` 은 **일치**하는가 | **예 — 허용치가 아니라 비트 단위.** gather 는 산술이 없다 (§4) |
| 트랜스포머가 Vulkan 에서 도는가 | **예 — 블록 하나가 29 개 셰이더로, 읽어오기 0 회.** upstream 과 §5 |
| 사전학습 BERT 가 도는가 | **아니다.** 벽이 `embedding` 에서 `expand`·`slice`·`gather`·`select`·`tanh`·4-D `transpose` 로 옮겨갔다 (§3). **이후 `docs/devices/VULKAN7.md` 에서 닫혔고 `bert-base-uncased` 가 장치 위에서 돈다 — 단 이 목록은 전부가 아니었다** |
| 장치에 int64 저장이 생겼나 | **아니다.** int64 텐서는 int32 로 **범위를 명시해 좁혀** 저장하고, 넘는 값은 이름을 대며 거절한다 (§1) |
| 이 기계는 `shaderInt64` 가 있나 | **있다.** 그래서 이 선택은 능력 보고가 아니라 설계 결정이다 (§1.2) |
| 성능은 | 재지 않았다. `docs/devices/VULKAN2.md` §4.4 의 이유 그대로 |

<!-- DOCWATCH: count vulkan_tests_ok ge 29 -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/vulkan.rs VULKAN_INDEX_MAX present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/vulkan.rs check_storage_dtype present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/vulkan.rs index_range present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/vulkan.rs embedding_vulkan present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/vulkan.rs transpose_batched2d present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vulkan4.py test_an_int64_value_beyond_the_bound_refuses_by_name_and_uploads_nothing present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vulkan4.py test_a_transformer_block_forwards_on_the_gpu_and_agrees_with_upstream present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vulkan4.py test_embedding_is_bit_identical_to_upstream_and_ran_on_the_gpu present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vulkan4.py test_the_transformer_wall_moved_and_the_new_one_is_named present -->

위의 `count vulkan_tests_ok` 마커는 `docs/devices/VULKAN5.md` §2.3 의 장치를 그대로 씁니다 — 로더가
없는 기계에서는 PASS 가 아니라 **SKIP** 으로, 이유와 함께 보고됩니다.

---

## 1. 설계 결정 — 커널이 아니라 저장 클래스

`docs/devices/VULKAN5.md` §7 은 `embedding` 을 미룬 이유를 **"장치에 int64 저장이 필요하고, 그것은
`check_dtype` 의 float32 only 를 넓히는 설계 변경"** 이라고 적었습니다. 맞는 판단이었고, 그래서 이번
회차의 첫 작업은 셰이더가 아니라 **무엇을 저장할 것인가**를 정하는 것이었습니다.

### 1.1 먼저 쟀다 — 그리고 측정이 가정을 뒤집었다

지시받은 대로 이 MoltenVK 가 실제로 무엇을 보고하는지 `_vulkan_probe()` 에 붙여 쟀습니다.
**추측은 "Vulkan 컴퓨트 셰이더는 확장 없이 64 비트 정수를 못 쓴다" 였고, 이 기계는 그 반대였습니다:**

```
Apple M1 via /opt/homebrew/lib/libvulkan.1.dylib
  shaderInt64   = True        <- 추측이 틀린 곳
  shaderInt16   = True
  shaderFloat64 = False
  VK_KHR_8bit_storage  = True
  VK_KHR_16bit_storage = True
```

즉 **이 장치에서는 네이티브 int64 인덱스 버퍼가 가능합니다.** 선택지는 세 개였습니다.

| 선택지 | 무엇이 되나 | 무엇이 문제인가 |
|---|---|---|
| A. 네이티브 int64 저장 | 이 기계에서 인덱스 범위 = torch 의 범위 | `shaderInt64` 는 Vulkan 의 **선택 기능**이다. 모바일 GPU 다수가 없다. 같은 모델이 GPU 에 따라 거절되거나 안 되거나 한다 |
| B. 인덱스를 매 호출 호스트에서 올림 | 저장 클래스를 안 늘려도 된다 | 디코드 루프에서 **토큰당 업로드 1 회**. §2 |
| C. int32 저장 + 명시된 상한 + 이름을 댄 거절 | 모든 장치에서 같은 규칙 | 상한이 2^31−1 이다 |

**C 를 골랐습니다.** 이유는 능력이 아니라 **행동의 이식성**입니다 — A 를 고르면 "이 모델이 돌아가는가"의
답이 드라이버의 기능 비트에 달리게 되고, 그것은 명시된 천장보다 나쁜 실패입니다. 상한 2^31−1 은
GPT-2 의 어휘 50257 의 약 43000 배이므로, **정직한 좁힘이지 절충이 아닙니다.**

`test_the_device_reports_its_integer_capabilities_and_the_index_bound_is_not_one_of_them` 이 이것을
단언합니다. 그리고 이 기계가 `shaderInt64 = True` 이기 때문에 그 단언에는 **이빨이 있습니다**:
장치가 네이티브로 표현할 수 있는 값(2^31)을 **그래도 거절**하는 것을 확인하므로, 상한이 능력 보고가
아니라는 것이 증명됩니다. `shaderInt64` 가 없는 기계에서는 이 구분이 불가능하고, 테스트는 그렇게 말하고
지나갑니다.

### 1.2 좁힘은 잘림이 아니다

`2**31` 을 int32 로 자르면 `-2147483648` 이 되고, 인덱스로서 그것은 **오류가 아니라 그럴듯한 다른 행**
입니다. 그래서 자르지 않고 **올라가는 길목에서 거절**합니다. 메시지는 값·위치·상한을 모두 답니다:

```
aten._to_copy.default: int64 element 1 is 2147483648, which does not fit the int32
index storage every vulkan device in this build uses (bound -2147483648..=2147483647).
64-bit integers in a shader need the optional shaderInt64 feature, so the ceiling is
the same everywhere rather than a property of this GPU -- and a value above it is
refused here rather than truncated into a different, plausible index.
```

거절은 **업로드 전에** 일어납니다: 테스트가 `host_uploads` 증가 0 으로 단언합니다.

### 1.3 저장이 넓어졌지 계산이 넓어진 것이 아니다

`check_dtype`(계산)과 `check_storage_dtype`(저장)을 나눴습니다. **`check_dtype` 은 한 글자도 안 바뀌었고
여전히 float32 만 받습니다.** int64 텐서가 할 수 있는 일은 정확히 셋입니다:

- `aten._to_copy.default` 로 호스트에서 올라오고 내려가기
- `view`/`reshape`(바이트를 안 옮기므로 dtype 과 무관 — 트랜스포머가 `input_ids` 를 reshape 한다)
- `embedding` 의 인덱스 바인딩으로 **읽히기**

정수 산술은 여전히 이 장치에 없습니다. 즉 이 저장 클래스로 미구현 정수 op 을 float 셰이더에 밀어넣는
경로는 존재하지 않습니다.

---

## 2. 인덱스 범위 검사는 어디서 하나 — 그리고 그 비용

upstream 은 테이블 밖 인덱스에 `IndexError` 를 냅니다. 셰이더는 예외를 못 던지고, 대신 할 수 있는 것은
**클램프하거나 쓰레기를 읽는 것**뿐인데 둘 다 이 장치가 존재하는 이유인 "조용한 오답"입니다.

호스트에서 검사해야 하고, 그러려면 인덱스 값을 알아야 합니다. 방법은 둘:

1. **매 호출 인덱스 버퍼를 읽어온다.** 디코드 루프에서 토큰당 호스트 왕복 1 회 — 지연이 지배하는
   루프에서 받아들일 수 없습니다.
2. **업로드 때 `(min, max)` 를 기록해 둔다.** 어차피 값을 한 번 훑으며 좁히는 중이므로 비용이 0 이고,
   `embedding` 은 그 두 수를 테이블 행수와 비교하기만 합니다.

2 를 골랐습니다. 기록은 `VkTensor` 가 아니라 **`VkBuffer`** 에 둡니다 — `view`·`reshape`·`detach`·`alias`
가 `Arc<VkBuffer>` 를 공유하므로 **한 줄도 안 쓰고 따라갑니다.** 이 장치의 어떤 커널도 정수 텐서를
만들지 않으므로, 기록이 없는 정수 버퍼는 존재할 수 없고 그런 것이 오면 추측 대신 거절합니다.

테스트는 범위 거절이 **셰이더 0 회, 읽어오기 0 회**로 일어나는 것을 단언합니다 — 검사가 몰래 왕복하기
시작하면 그 자리에서 빨개집니다.

`padding_idx`·`scale_grad_by_freq`·`sparse` 는 **autograd 전용 인자**라 forward 값을 바꾸지 않습니다.
이름을 대며 거절하는 대신 받아들이는데, 그 근거는 주석이 아니라 테스트입니다 —
`test_embedding_ignores_the_autograd_only_arguments_exactly_as_upstream_does` 가 **upstream 에게 같은
질문을 던져** 값이 같음을 확인하고, 미래의 torch 가 forward 에서 그것을 읽기 시작하면 실패합니다.

---

## 3. 벽은 어디로 옮겨갔나 — 잰 것

`embedding` 만으로는 트랜스포머가 돌지 않았습니다. **한 번에 하나씩 실제로 돌려서** 다음 벽을
찾았습니다(`docs/architectures/VOICE4.md` 가 기록한 실패 — *"단 한 번도 모델을 돌리지 않았다"* — 를
되풀이하지 않는 유일한 방법입니다):

| 회차 | 멈춘 곳 | 무엇을 했나 |
|---|---|---|
| 1 | `aten._to_copy.default` — int64 거절 | §1 의 저장 클래스 |
| 2 | `aten.transpose.int`: 3-D | 마지막 두 차원만 도는 배치 전치 커널 |
| 3 | `aten.div.Scalar` | 스칼라를 푸시 상수에 실어 보내는 커널 (`mul.Scalar` 포함) |
| 4 | — | **블록이 끝까지 돌았다** (§5) |

2 번과 3 번은 **재고를 보고 고른 것이 아니라 forward 가 멈춘 자리**입니다. 배치 전치는 attention 의 두
`bmm` 사이 `k.transpose(1, 2)` 이고, `div.Scalar` 는 `scores / sqrt(d)` 입니다.

### 3.1 사전학습 BERT 는 아직 돈다고 말할 수 없다

`docs/devices/VULKAN4.md` §2 와 같은 shrunk BERT(vocab 64, hidden 32, 2 layer, 4 head, ffn 64, 8 토큰)를
upstream 에서 다시 추적했습니다 — 117 dispatch / 18 종. 이번 회차 뒤에도 남은 것:

```
10  aten.expand.default        1  aten.slice.Tensor
 1  aten.gather.default        1  aten.select.int        1  aten.tanh.default
```

여기에 더해 **BERT 의 `transpose.int` 10 회는 4-D**(batch, head, seq, dim)이고, 이번에 가르친 것은 3-D 의
마지막 두 차원뿐입니다. 실제로 이 빌드에서 BERT 를 돌리면 `aten.slice.Tensor` 에서 이름을 대며 멈춥니다.
`test_the_transformer_wall_moved_and_the_new_one_is_named` 가 이 목록을 고정하므로, 다음 회차가 하나를
가르치면 그 테스트가 실패하며 이 문장을 고치라고 말합니다.

**즉 이 회차가 주장하는 것은 "멀티헤드 트랜스포머가 돈다" 가 아니라 "단일헤드 블록이 돈다" 입니다.**

> **후속 (2026-09-18):** 위 목록은 **전부가 아니었습니다.** 여섯을 모두 가르친 뒤 실제 `bert-base-uncased` 는
> 브로드캐스팅 `aten.add.Tensor`(`[2, S, H] + [1, S, H]`)와 `aten.matmul.default` 에서 차례로 멈췄습니다 — 이
> 추적은 op **이름**을 셌고(`add` 는 이미 "가르친" 이름이었습니다), **upstream** 에서 했기 때문에 셰임이
> 분해하지 않는 `matmul` 이 보이지 않았습니다. 이 8 개를 닫고 사전학습 BERT 가 장치 위에서 돕니다.
> `docs/devices/VULKAN7.md` §3.2, §5.

---

## 4. `embedding` — 값

**허용치가 없습니다.** gather 는 산술을 하지 않습니다 — 출력의 모든 원소는 테이블에서 복사된 float32
이므로, `docs/numerics/AGREE.md` §2 의 유도는 `view`·`t` 와 마찬가지로 **0** 을 냅니다. 그래서 비교는
**upstream 과의 비트 동일성**이고, 허용치를 넓힐 여지가 애초에 없습니다.

| 케이스 (vocab, dim, 인덱스 shape) | 원소 | upstream 과 비트가 다른 수 |
|---|---|---|
| 8, 4, `[3]` | 12 | 0 |
| 8, 4, `[2,3]` | 24 | 0 |
| 50257, 2, `[4]` | 8 | 0 |
| 5, 1, `[1]` | 1 | 0 |
| 6, 3, `[2,2]` | 12 | 0 |
| **합계** | **57** | **0** |

배치 전치와 두 스칼라 op 도 같은 이유로 비트 동일성으로 잽니다(전치는 순수 데이터 이동, 스칼라 op 은
원소당 IEEE 연산 하나).

`div.Scalar` 에는 함정이 하나 있어서 따로 적습니다: **`a / s` 가 `a * (1/s)` 로 바뀌면 안 됩니다.**
역수는 upstream 이 하지 않는 반올림이고, 누가 고르든 어떤 허용치 안에도 들어갑니다 — MoltenVK 의
fast-math 를 끈 것과 같은 종류입니다(`docs/devices/VULKAN5.md` §3.1). 테스트는 비트 동일성을 단언하기
전에 **그 케이스가 둘을 구분할 수 있는지** 먼저 셉니다: 이 스윕에서 역수 형태는 **92 개 원소**를 다르게
만듭니다. 0 이면 그 단언은 아무것도 증명하지 못하므로, 테스트가 그것을 거부합니다.

---

## 5. 이 회차의 산출물 — 트랜스포머 블록이 장치 위에서 돈다

`nn.Embedding → LayerNorm → q,k,v → transpose → bmm → /√d → softmax → bmm → out-proj → residual
→ LayerNorm → fc → gelu → fc → residual`, 단일헤드, `B=2, S=6, D=16, F=32`, vocab 32.
벤더 트리의 셰임을 별도 프로세스에서 돌려 upstream 및 셰임 자신의 CPU 답과 비교했습니다.

```
upstream f32 오차  1.042e-06     (float64 진실 기준)
셰임 cpu           2.341e-06
셰임 vulkan        2.233e-06     = upstream 의 2.14x   (AGREE.md §2 의 4x 규칙 안)
vulkan vs 셰임 cpu 2.70 float32 ulp
셰이더 29 회, 호스트 읽어오기 0 회
```

**세 번째 줄이 이 회차의 실제 주장입니다.** 29 = embedding 1 + layer_norm 2 + Linear 6×(t+matmul+bias)
+ transpose 1 + bmm 2 + div 1 + softmax 1 + gelu 1 + residual add 2. 호스트로 떨어진 forward 도 숫자는
맞게 내놓습니다 — 그래서 값 비교만으로는 구분되지 않고, **셰이더 수와 읽어오기 0** 이 구분합니다.

---

## 6. 무력화 — 일부러 깨고 빨개지는지 봤다

`CLAUDE.md` §5.5. 먼저 **구현 전체를 되돌린 빌드**에서 새 테스트를 돌렸고(이 회차의 red 단계),
그다음 개별 보증을 하나씩 깼습니다. **초록으로 남은 무력화는 없었습니다.**

| # | 무력화 | 결과 |
|---|---|---|
| R | 구현 전체를 이전 커밋 상태로 되돌림 | **FAIL ×14** — 새 테스트 전부 + 셰이더 도달성 |
| D1 | `embedding` 의 인덱스 범위 검사 제거 | **FAIL** `index 5 was gathered instead of refused` |
| D2 | int64 업로드가 거절 대신 `as i32` 로 자르게 | **FAIL ×2** `2147483648 was narrowed instead of refused`, 그리고 §1.1 의 상한 테스트 |
| D3 | 배치 전치가 항상 0 번 배치를 읽게 | **FAIL** `transpose[2,3,4](1,2): 12 of 24 elements differ from upstream's bits` |
| D4 | `div.Scalar` 가 `a * (1/s)` 를 계산하게 | **FAIL** `div.Scalar by 3.0: 24 of 64 elements differ` |
| D5 | **`embedding` 이 호스트에서 gather 하고 vulkan 라벨을 달게** | **FAIL ×2** `{'shader_dispatches': 0, 'host_uploads': 1, 'host_downloads': 2}`, 그리고 op 별 셰이더 수 단언 |

**D5 가 이 문서에서 가장 중요한 줄입니다.** 값은 전부 맞았고(호스트 gather 는 비트까지 같습니다),
`.device` 도 `vulkan` 이었습니다. 그것을 잡은 것은 값이 아니라 **카운터**입니다 — 조용한 CPU 폴백은
답으로는 구분되지 않고 계측으로만 구분됩니다.

---

## 6.1 게이트 — 네 번 돌렸고, 그중 둘이 이 회차의 증거다

`TORCHNATIVE_REQUIRE_VULKAN=1` 로 돌렸습니다. **건너뛴 Vulkan 테스트가 하나라도 있으면 실패하는
형태**이고, `docs/devices/VULKAN5.md` §2.2 가 말한 대로 Vulkan 커널의 증거로 내밀 수 있는 실행은
이것뿐입니다.

| 회차 | GATE_EXIT | ok | FAIL | DOCWATCH | VULKAN COVERAGE |
|---|---|---|---|---|---|
| 기준선 (`develop`, 변경 전 같은 워크트리) | 0 | 1629 | 0 | 1190/1190 | 23 ran / 0 skipped |
| 1 | **1** | 1634 | 5 | (도달 못 함) | 33 ran / 0 skipped |
| 2 | **1** | 1638 | 1 | (도달 못 함) | 33 ran / 0 skipped |
| 3 | 0 | **1639** | **0** | **1200/1200** | **33 ran / 0 skipped** |
| 4 | 0 | **1639** | **0** | **1200/1200** | **33 ran / 0 skipped** |

1 회차의 다섯 실패 중 **하나만 이 회차의 것**이었습니다 —
`test_vulkan_probe_answers_the_same_question_the_device_does` 가 `_vulkan_probe()` 의 키 집합을 정확히
단언하는데, §1.1 의 여섯 키가 늘었기 때문입니다. 고쳤습니다(양방향 단언은 그대로 유지 — 키가 사라지는
것도 실패여야 합니다).

**나머지 넷은 CoreML/ANE 쪽이고, 이 회차와 인과 경로가 없습니다.** 근거 셋:

1. 이 회차가 만진 것은 `vulkan.rs`, `test_vulkan4.py`, 셰이더 셋, 문서, 그리고 `test_shim.py` 의
   단언 한 줄뿐입니다. CoreML 스위트는 `test_shim` 에서 픽스처를 가져오지만 그 단언은 함수 본문 안입니다.
2. 3·4 회차에서 아무것도 안 고쳤는데 사라졌습니다.
3. **`test_bf16ane.py` 만 단독으로 세 번 돌려 봤더니 통과 1, 실패 2 였고, 두 실패의 모양이 서로
   달랐습니다** (`survived: 3` / `naive-path fixture raised` ×4). 즉 이 스위트는 이 트리에서 원래
   간헐적입니다. `test_compiling_leaves_no_compiled_bundle_behind_in_the_system_temp` 의 독스트링 자신이
   *"another CoreML process was running concurrently"* 를 가능성으로 적어 두었고, 이 기계에서는 다른
   워크트리의 에이전트가 같은 시스템 임시 디렉터리를 씁니다 — 실측으로 그곳에 `*.mlmodelc` 533 개가
   쌓여 있었습니다.

**고치지 않았습니다.** 이 회차의 범위가 아니고, 남의 회차 결함을 고치면서 그 근거를 여기에 적는 것은
`CLAUDE.md` §5.3 이 경고한 "같은 칸에 섞기" 입니다. **미해결로 남겨 두고 여기에 적습니다.**

> **후속 (2026-09-17):** 원인은 이 문단이 짐작한 대로 동시 실행이었고, 측정으로 확인됐습니다 — `survived` 는 사용자 공용 디렉터리의
> 스냅샷 차이라서 다른 프로세스가 살려 둔 `tmp*.mlmodelc` 를 우리 것으로 셌습니다. 픽스처는 이제 자기가 쓴 번들을 기록해서
> 판정합니다. `docs/graph/NPU2.md` §8.6.

---

## 7. 하지 않은 것

- **멀티헤드 / 사전학습 BERT.** §3.1 의 다섯 op 과 4-D 전치가 남아 있습니다. *(후속: `docs/devices/VULKAN7.md`
  에서 닫혔고, 목록에 없던 두 op 까지 닫아 `bert-base-uncased` 가 돕니다.)*
- **성능.** 아무것도 재지 않았습니다. 한 커널당 한 번의 `vkQueueSubmit` + fence 대기는 그대로입니다.
- **정수 산술.** 저장만 생겼고 계산은 없습니다 (§1.3).
- **float16 / bfloat16 / float64** — `docs/devices/VULKAN5.md` 그대로, 장치에 올라가는 순간 거절합니다.
- **`embedding` 의 backward / sparse grad** — 이 장치에 backward 가 없습니다.
- **실물 Adreno/Mali.** 이번 회차의 모든 숫자도 Apple M1 하나입니다. §1.1 의 선택이 특히 그 기계들을
  염두에 둔 것이므로, 그곳에서 확인되지 않았다는 사실을 함께 적습니다.
