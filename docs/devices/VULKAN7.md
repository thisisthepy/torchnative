# Vulkan — 사전학습 BERT 가 장치 위에서 돈 라운드, 그리고 VULKAN6 의 벽 목록이 틀렸던 두 곳

**결론부터.** `docs/devices/VULKAN6.md` §3.1 은 단일헤드 블록이 돈 뒤 사전학습 BERT 를 막는 것을
`expand`·`slice`·`gather`·`select`·`tanh`·4-D `transpose` 로 적었습니다. 이 여섯을 가르쳤고, 실제
`bert-base-uncased` 를 돌려 보니 **목록에 없던 벽이 두 개 더** 나왔습니다 — 브로드캐스팅
`add.Tensor` 와 `aten.matmul.default` (§3.2). 그 둘까지 닫고 나서:

| 질문 | 답 |
|---|---|
| 사전학습 BERT 가 Vulkan 에서 도는가 | **예 — `bert-base-uncased`, 12 레이어·12 헤드, `from_pretrained` 그대로, `m.to("vulkan")`.** 셰이더 398 회, 호스트 읽어오기 0, 업로드 0 (§5) |
| upstream 과 **일치**하는가 | **예.** `last_hidden_state` 는 upstream 자신의 float32 오차의 **0.49 배**, `pooler_output` 은 **0.94 배** — `docs/numerics/AGREE.md` §2 의 4 배 규칙 안 (§5) |
| 어떤 조건에서 | `attn_implementation="eager"`, `input_ids` 만, B=2, S=12. **기본값(`sdpa`)과 `attention_mask` 를 주면 멈춘다** — 이름을 대며 (§6) |
| VULKAN6 의 벽 목록은 전부였나 | **아니다.** 두 개가 빠져 있었고, 빠진 이유가 둘 다 **측정 방법**에 있었다 (§3.2) |
| 인덱스 범위 정책은 | **그대로.** `VULKAN_INDEX_MAX` 는 손대지 않았다. 뷰가 만든 인덱스 텐서는 **상속된 상계**를 들고 다니고, 그것이 표에 안 맞으면 이름을 대며 거절한다 (§3) |
| 새 커널이 호스트에서 계산해도 잡히나 | **예.** `slice` 를 호스트 쌍둥이로 바꾸면 값은 맞고 카운터가 빨개진다 (§8 N6) |
| 성능은 | 재지 않았다. `docs/devices/VULKAN2.md` §4.4 의 이유 그대로 |

<!-- DOCWATCH: count vulkan_tests_ok ge 38 -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/vulkan.rs VIEW_RANK_MAX present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/vulkan.rs materialise_view present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/vulkan.rs check_index_range present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/vulkan.rs broadcast_binary present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/vulkan.rs matmul_vulkan present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/vulkan.rs gather_vulkan present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/vulkan.rs strided_gather_u32 present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/vulkan.rs VULKAN_INDEX_MAX present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vulkan4.py test_a_pretrained_bert_forwards_on_the_gpu_and_agrees_with_upstream present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vulkan4.py test_the_view_ops_are_bit_identical_to_upstream_and_ran_on_the_gpu present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vulkan4.py test_an_index_range_is_inherited_through_views_and_a_loose_one_refuses_by_name present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vulkan4.py test_broadcasting_arithmetic_is_bit_identical_to_upstream_and_one_shader present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vulkan4.py test_matmul_agrees_with_upstream_and_ran_as_one_batched_product present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vulkan4.py test_gather_refuses_what_upstream_refuses_before_any_gpu_work present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vulkan4.py test_tanh_agrees_with_upstream_at_a_derived_tolerance present -->

`count vulkan_tests_ok` 는 `docs/devices/VULKAN5.md` §2.3 의 장치 그대로입니다 — 로더가 없는 기계에서는
PASS 가 아니라 SKIP 입니다. **사전학습 BERT 테스트는 로컬 가중치가 없으면 SKIP 으로 세어집니다**
(내려받지 않습니다 — 게이트가 네트워크에 닿으면 안 됩니다). 이 기계에서 가중치는
`/Volumes/macMini/caches/hf-home` 에 있고, 다른 곳이면 `TORCHNATIVE_BERT_DIR` 로 가리킵니다.

---

## 1. 시작점 — 벽을 다시 쟀다

VULKAN6 의 문장을 인용하기 전에 **실제 `bert-base-uncased` 로 재측정**했습니다(`CLAUDE.md` §6):

```
AutoModel.from_pretrained(<hf-home>/bert-base-uncased, attn_implementation="eager")
m.to("vulkan"); m(input_ids=ids.to("vulkan"))
  -> NotImplementedError: aten.slice.Tensor: not implemented for the vulkan device ...
     at modeling_bert.py:84  position_ids = self.position_ids[:, past : seq + past]
  counters: shader_dispatches 0, host_uploads 0, host_downloads 0
```

VULKAN6 §3.1 의 "`aten.slice.Tensor` 에서 이름을 대며 멈춘다" 는 **참이었습니다.** 그 뒤의 목록이
전부였는지는 §3.2 가 답합니다.

---

## 2. 설계 — 뷰는 stride 가 아니라 한 번의 복사로

VULKAN6 의 여섯 중 넷(`slice`·`select`·`expand`·`transpose`)은 upstream 에서 **뷰**입니다. 그런데
`VkTensor` 는 shape 만 있고 stride 가 없습니다(`docs/devices/VULKAN4.md` §6). 선택지는 둘이었습니다:

| 선택지 | 무엇이 되나 | 무엇이 문제인가 |
|---|---|---|
| A. `VkTensor` 에 stride·offset 추가 | 뷰가 공짜 | **기존 셰이더 17 개를 쓰는 커널 전부**가 비연속 입력을 알아야 한다. 하나라도 잊으면 조용한 오답 |
| B. 뷰를 **그 자리에서 물질화** | 모든 텐서가 계속 연속 버퍼. 기존 커널 무변경 | 뷰 한 번에 디스패치 한 번 |

**B 를 골랐습니다.** 이 장치의 설계는 "잊으면 틀린다" 를 "잊으면 거절한다" 로 바꾸는 것이고, A 는 그
반대 방향입니다. 비용(뷰당 디스패치 1 회)은 숨기지 않고 op 별 디스패치 표에 적었습니다.

### 2.1 셰이더 하나로 넷을 — `strided_gather_u32`

```
out[i] = src[base + Σ_d coord_d(i) · stride[d]]
```

뷰 op 넷은 모두 이 식에 `(shape, stride, base)` 만 다르게 넣은 것입니다 — `slice` 는 base 를 옮기고
stride 하나를 곱하고, `select` 는 base 를 옮기고 축 하나를 지우고, `expand` 는 늘어난 축의 stride 를
0 으로, `transpose` 는 두 항목을 바꿉니다. upstream 의 뷰 함수들이 자기 stride 를 유도하는 방식
그대로입니다.

**페이로드를 `float` 가 아니라 `uint` 로 선언했습니다.** 산술이 없고 복사만 하므로, float32 워드와
int32 인덱스 워드가 이 커널에게는 같은 4 바이트입니다. BERT 의 int64 `position_ids` 를 장치 위에서
자를 수 있는 것이 이것 덕분이고, float 복사가 NaN 페이로드를 정규화할 여지도 없습니다. 결과는
**구성상 upstream 과 비트 동일**이고 테스트가 그렇게 단언합니다.

### 2.2 명시된 두 상한 — 능력 보고가 아니다

- **`VIEW_RANK_MAX = 8`.** shape 과 stride 를 push constant 에 싣습니다: 4 + 8 + 8 = 20 워드 = 80
  바이트. Vulkan 은 모든 장치에 **최소 128 바이트**를 요구하므로, 이 상한은 GPU 의 성질이 아니라
  라이브러리의 성질입니다 — `VULKAN_INDEX_MAX` 와 같은 모양의 결정입니다. 브로드캐스트 커널(§3.2)은
  4 + 8 × 3 = 28 워드 = 112 바이트로 역시 128 안입니다. 랭크 9 는 GPU 작업 전에 이름을 대며
  거절합니다.
- **셰이더 인덱스는 `uint`.** 원소 수·오프셋·stride 가 `u32::MAX` 를 넘으면 다른 그럴듯한 원소로
  감깁니다. 그래서 **할당 전에** 값과 상한을 대며 거절합니다(`shader_u32`). `expand` 가 특히 그렇습니다:
  upstream 은 `expand([1], [2**32 + 1])` 에 공짜 뷰로 답하지만 이 장치는 그것을 물질화해야 합니다.
  테스트가 그 거절을 `4294967297` 이라는 값과 함께 확인합니다.

뷰가 **항등**이면(출력이 입력과 같은 shape·stride·base) 디스패치하지 않고 버퍼를 공유합니다. 이
장치의 어떤 op 도 제자리 쓰기를 하지 않으므로 안전하고, upstream 의 답도 같은 저장소의 뷰입니다.
BERT 의 풀러가 쓰는 `hidden[:, 0]` 의 `[:, ...]` 가 이 경우입니다 — 무력화 N11 이 이 분기를 지우면
BERT 의 셰이더 수가 398 → 399 가 되어 빨개집니다.

---

## 3. 인덱스 범위 — 뷰를 지나면 상계가 된다

VULKAN6 §2 는 인덱스 버퍼의 `(min, max)` 를 **업로드 때** 기록해서 `embedding` 이 읽어오기 없이
범위를 검사하게 했습니다. 그런데 이제 정수 텐서가 업로드 말고도 생깁니다 — `slice`·`select`·
`expand`·`gather` 의 출력입니다.

그 출력의 값은 **원본 값의 부분집합**이므로 원본의 범위는 여전히 참인 상계입니다. 하지만 **느슨할 수
있고**, 정확한 범위는 읽어오기 없이는 알 수 없습니다. 그래서:

- `IndexRange { lo, hi, exact }` 로 **정확한지 상속된 것인지**를 함께 기록합니다.
- 상계가 표 안에 들면 그대로 진행합니다. **BERT 가 정확히 이 경우입니다**: `position_ids[:, :12]` 는
  원본 `arange(512)` 에서 `[0, 511]` 을 상속하고, 위치 임베딩 표는 512 행이므로 통과합니다.
  `token_type_ids` 는 `zeros` 에서 `[0, 0]` 을 상속하고 `gather` 를 지나도 그대로입니다.
- 상계가 표 밖이면 **실제 값이 범위 안일 수도 있으므로** 값을 지어내지 않고, 상속된 상계라는 사실과
  그 값을 대며 거절합니다:

```
aten.embedding.default: this index tensor is a view of another, and the only bound on its values
this device holds without reading it back is the one inherited from that source: [0, 49].
49 is not in [0, 8), so the indices cannot be shown to be in range and are not gathered. Its own
values may all be in range; upload them directly with .to("vulkan") to record their exact range
(docs/devices/VULKAN7.md §3).
```

**이것은 upstream 이 답하는 곳에서의 거절입니다** (`arange(50)[2:6]` 로 8 행 표를 조회하면 upstream 은
답합니다). 좁힘이고, 테스트
`test_an_index_range_is_inherited_through_views_and_a_loose_one_refuses_by_name` 가 그것을 좁힘으로
고정합니다. `VULKAN_INDEX_MAX` 는 한 글자도 바뀌지 않았고, 범위를 GPU 에서 읽어오는 경로도 만들지
않았습니다.

`gather` 는 인덱스 버퍼의 **두 번째 독자**이므로 같은 검사(`check_index_range`)를 거칩니다. 정확한
범위가 표 밖이면 upstream 의 문구 그대로(`index 5 is out of bounds for dimension 1 with size 5`) 셰이더
0 회로 거절합니다.

### 3.2 VULKAN6 의 목록에 없던 두 벽 — 그리고 왜 빠졌나

여섯을 가르치고 BERT 를 돌리자 **다음에 멈춘 곳은 목록에 없는 op** 이었습니다:

| 차례 | 멈춘 곳 | 그때까지의 셰이더 | 원인 |
|---|---|---|---|
| 1 | `aten.slice.Tensor` | 0 | VULKAN6 의 목록 그대로 |
| 2 | `aten.add.Tensor`: `[2, 7, 768]` + `[1, 7, 768]` | 7 | **브로드캐스트.** 위치 임베딩 덧셈 (`modeling_bert.py:104`) |
| 3 | `aten.matmul.default` | 22 | **셰임은 matmul 을 분해하지 않는다** (`eager_attention_forward`, `modeling_bert.py:125`) |
| 4 | — | 398 | **BERT 가 끝까지 돌았다** (§5) |

1·2 는 그 순서대로 가르치며 잰 것입니다. 3 은 브로드캐스트와 함께 구현했기 때문에 **따로 멈춘 것을 본
적이 없어서**, `matmul` 의 디스패치 팔만 뺀 빌드로 다시 돌려 확인했습니다. 셰이더 수 7 과 22 는 §5 의
유도식(임베딩에서 `add` 앞까지 7; 임베딩 9 + q,k,v 9 + 전치 4 = 22)과 맞습니다.

두 누락은 모두 **측정 방법**에서 왔습니다:

1. **추적이 op 이름을 셌다.** `add.Tensor` 는 VULKAN3 부터 "가르친 op" 이었으므로 벽으로 보이지
   않았습니다 — 하지만 동일 shape 만 됐습니다. 이름 목록은 **어떤 shape 으로** 불리는지를 담지
   않으므로(VULKAN4 §2 는 추적의 배치 크기조차 기록하지 않았습니다), 위치 임베딩 덧셈이
   브로드캐스트라는 사실을 볼 수 없었습니다. 배치 1 이면 `[1, S, H] + [1, S, H]` 라 브로드캐스트도
   아닙니다. `CLAUDE.md` §5.4 가 BigVGAN 의 meta 커널로 기록한 함정과 같은 모양입니다 — **목록은
   "커널이 있는가" 에 답했고 질문은 "이 모델이 도는가" 였습니다.**
2. **추적은 upstream 에서 했고, 셰임은 matmul 을 분해하지 않는다.** VULKAN4 §2.1 이 바로 이 차이를
   적어 두었습니다: *"shim 은 `aten.matmul.default` 를 하나의 op 으로 유지합니다 … upstream 은 그것을
   `expand` + `bmm` + `_unsafe_view` 로 분해합니다."* 그러니 upstream 추적의 `expand` 10 회 중 8 회는
   **셰임에서는 한 번도 디스패치되지 않는** matmul 의 내부였고, 셰임이 실제로 부르는 `matmul` 은
   목록에 없었습니다. **두 추적이 "같은 순위를 준다" 는 것은 참이었지만, 같은 벽을 준다는 뜻은
   아니었습니다.**

`test_the_transformer_wall_moved_and_the_new_one_is_named` 는 이제 여섯에 `matmul` 을 더해 **가르쳐져
있음**을 고정합니다. 그러나 그 목록을 전부라고 믿게 해주는 것은 그 테스트가 아니라 §5 의 실제
forward 입니다.

**브로드캐스트** 는 `add`/`sub`/`mul`/`div` `.Tensor` 에 대해 **디스패치 한 번**으로 합니다 — 두 피연산자의
stride(브로드캐스트 축은 0)를 push constant 에 실은 `broadcast_binary_f32` 입니다. 원소당 IEEE 연산
하나이므로 비트 동일성으로 잽니다. 동일 shape 는 기존 커널을 그대로 씁니다.

**`matmul`** 은 upstream 의 랭크 규칙(1-D 는 행/열로 올렸다가 결과에서 지움, 배치 차원은
브로드캐스트)을 따르고, 곱셈 자체는 평탄화한 배치 위의 **기존 `bmm` 커널** 한 번입니다 — 모든 텐서가
연속이므로 평탄화는 바이트를 옮기지 않습니다. 배치가 브로드캐스트되어야 하는 피연산자만
`strided_gather_u32` 로 한 번씩 물질화합니다. BERT 의 두 matmul 은 배치가 같으므로 각각 1 회입니다.

---

## 4. 값 — 커널별 대조

모두 upstream 의 `torch.ops.aten.<op>` 에 **같은 질문**을 던져 비교했습니다(`slice` 의 클램프,
`expand` 의 `-1`, 음수 dim·index 를 이 문서가 해석하지 않도록).

| op | 케이스 | 기준 | 결과 |
|---|---|---|---|
| `slice`·`select`·`expand`·`transpose` | 24 (f32 20, int64 4; 랭크 0–5) | **비트 동일** | 24 / 24. 그중 **19** 는 "뷰를 무시하고 앞에서부터 복사" 결함을 구분할 수 있음 (테스트가 ≥ 15 를 요구) |
| `gather` | 7 × `sparse_grad` 2 (f32 6, int64 2; 랭크 1–4) | **비트 동일** | 14 / 14 |
| `add`/`sub`/`mul`/`div` 브로드캐스트 | 7 shape 쌍 × 4 op × 양 순서 = 56 | **비트 동일** | 56 / 56 |
| `tanh` | 5 shape, 스케일 0.5–12.5 | 유도 허용치 `max(p90 5.739e-08, 8 ulp 9.537e-07)` = 9.537e-07 | 최악 **1.161e-07** (upstream 자신 5.698e-08); 3196 원소 중 716 개가 upstream f32 와 비트가 다름 |
| `matmul` | 9 (BERT 의 두 모양, 2-D, 3-D, 배치 브로드캐스트 2, 벡터 3) | 유도 허용치 = 9.537e-07 | 최악 **8.337e-08** (upstream 자신 8.337e-08); 1613 원소 중 59 개가 비트가 다름 |

`tanh` 와 `matmul` 의 유도 허용치는 두 경우 모두 **8 ulp 바닥**이 결정했습니다(p90 이 그보다
작았습니다) — `docs/numerics/AGREE.md` §2 의 규칙 그대로이고, 새로 고른 수가 아닙니다. 원소 단위
비율로 보면 `matmul` 의 최악 원소는 upstream 자신의 오차의 **4.72 배**입니다. 이 값은 단언이 아니라
출력입니다(AGREE 의 단언은 텐서 스케일입니다). 합의 누적 순서가 다르기 때문이며,
`docs/devices/VULKAN4.md` §4 의 FMA 분석과 같은 종류입니다.

---

## 5. 산출물 — 사전학습 BERT 가 장치 위에서 돈다

`test_a_pretrained_bert_forwards_on_the_gpu_and_agrees_with_upstream`. 진짜 `transformers` 5.15.1,
진짜 `AutoModel.from_pretrained`, 진짜 토크나이저, 두 문장(길이가 달라 패딩됨), B=2·S=12.
벤더 트리의 셰임을 별도 프로세스에서 돌리고, 같은 체크포인트를 upstream 에서 float32 와 float64 로
돌려 비교했습니다.

```
last_hidden_state   upstream f32 오차 2.263e-05   셰임 cpu 2.315e-05   셰임 vulkan 1.119e-05  (0.49x)
pooler_output       upstream f32 오차 9.400e-06   셰임 cpu 4.792e-06   셰임 vulkan 8.881e-06  (0.94x)
12 레이어, 셰이더 398 회, 호스트 읽어오기 0, forward 중 업로드 0
모든 파라미터와 버퍼: vulkan / 두 출력: vulkan
```

**398 은 측정에서 베낀 수가 아니라 모델 코드에서 유도한 수입니다** (`modeling_bert.py`, eager,
`input_ids` 만, 위 디스패치 표):

| 부분 | 셰이더 | 내역 |
|---|---|---|
| 임베딩 | 9 | `slice`(position_ids) + `expand`(token_type_ids, B=2 라 항등 아님) + `gather` + `embedding` 3 + `add` 2 + `layer_norm` |
| 레이어당 | 32 | Linear q,k,v 3 × (t + matmul + bias) = 9 · view 뒤 `transpose(1,2)` 3 · `transpose(k,2,3)` 1 · `matmul` + `mul.Scalar` + `softmax` + `matmul` 4 · 문맥 `transpose` 1 · attention 출력 Linear + add + layer_norm 5 · intermediate Linear + gelu 4 · output Linear + add + layer_norm 5 |
| 풀러 | 5 | `[:, 0]` = 항등 slice 0 + `select` 1 + 2-D Linear 3 + `tanh` 1 |
| **합계** | **398** | 9 + 32 × 12 + 5 |

측정이 이 수와 **첫 실행에서 일치**했습니다. 값 비교만으로는 호스트로 떨어진 forward 를 구분할 수
없으므로(§8 N6 이 실제로 그렇습니다), 이 테스트의 무게는 **셰이더 398 과 읽어오기 0** 에 있습니다.

**이 주장의 크기.** "Vulkan 에서 사전학습 BERT 가 **eager attention, 마스크 없이** 돈다" 입니다.
"`from_pretrained` 의 기본값 그대로 돈다" 가 아닙니다 — §6.

---

## 6. 지금 멈추는 곳 — 잰 것

| 조건 | 멈춘 곳 | 그때까지의 셰이더 | 이유 |
|---|---|---|---|
| `from_pretrained` 기본값 (셰임에서 `sdpa` 로 잡힘) | `aten._scaled_dot_product_flash_attention_for_cpu.default` | 21 | 융합 attention op. 이 장치에 커널이 없다 |
| eager + `attention_mask` | `aten._to_copy.default`: int64 → bool | 9 | `masking_utils.py:824` 가 마스크를 `bool` 로 바꾼다. 이 장치에는 **bool 저장 클래스가 없고** 변환 셰이더도 없다 |

둘 다 **이름을 대며 거절**했고 읽어오기는 0 이었습니다. 두 번째는 커널 하나가 아니라 **저장 클래스
설계**입니다 — VULKAN6 §1 이 int64 에 대해 한 결정과 같은 무게이므로, 이번 회차에서 커널로 밀어넣지
않았습니다.

---

## 7. 이번 회차에 고친 결함

**`aten.rs::broadcast_shape` 가 upstream 과 다른 축을 보고했다.** 왼쪽부터 검사했기 때문에, 여러 축이
어긋나면 upstream(오른쪽부터 검사하는 `infer_size`)과 다른 축을 댔습니다:

```
[2, 3] + [3, 2]
  upstream : The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1
  이전     : The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0
```

이 도우미는 **meta** 커널들(비교 op, `add`/`sub`/`mul`/`div`, `where`, `pow`, `matmul`)이 씁니다. 발견 경위: 새 브로드캐스트
테스트가 부분 문자열이 아니라 **upstream 의 문장 전체**를 비교했기 때문입니다. 기존 테스트 셋은 모두
`"must match the size of tensor b" in str(e)` 만 봤으므로 이 차이를 볼 수 없었습니다. 고쳤고,
meta 경로에서 세 쌍으로 고정했습니다.

**고치지 않은 것:** **dense(CPU) 경로의 같은 거절은 candle 의 문구**(`candle: shape mismatch in
broadcast_add, lhs: [2, 3], rhs: [3, 2]`)를 냅니다. upstream 의 문구가 아닙니다. 이번 범위 밖이라
적어만 둡니다.

---

## 8. 무력화 — 일부러 깨고 빨개지는지 봤다

`CLAUDE.md` §5.5. 먼저 **구현 전의 빌드**에서 새 테스트·수정 테스트를 돌려 **FAIL 11** 을 확인했고
(red 단계), 그다음 보증을 하나씩 깼습니다. **16 개 전부 빨개졌고, 초록으로 남은 무력화는 없습니다.**

| # | 무력화 | 빨개진 테스트 (대표 메시지) |
|---|---|---|
| N1 | 뷰 커널이 `base` 를 무시 | 뷰 스윕 `slice step 2 neg: 40 of 40 elements differ`, 상속 범위 |
| N2 | `expand` 가 늘어난 축에 원본 stride 유지 | 내부 오프셋 가드 `offset 1023 is outside a source of 512`, BERT, 상속 범위 |
| N3 | 뷰가 인덱스 범위를 상속하지 않음 | BERT (embedding 이 "no recorded range" 로 거절), 상속 범위 |
| N4 | 상속된 범위를 `exact` 로 표시 | 상속 범위 — 메시지가 "49 is not in [0, 8)" 로, **읽지 않은 값을 주장** |
| N5 | `gather` 의 범위 검사 제거 | `gather index == size was computed instead of refused` |
| N6 | **`slice`/`select`/`expand` 가 호스트에서 계산하고 vulkan 라벨** | **값은 전부 맞았다.** 잡은 것은 카운터: `0 shaders, expected 1 … host_downloads 1`, BERT `read 63 buffer(s) back` |
| N7 | 브로드캐스트의 b 가 a 의 stride 사용 | 브로드캐스트 `112 of 224 elements differ`, BERT 5.149 |
| N8 | `matmul` 이 배치 브로드캐스트를 건너뜀 | matmul 스윕 (셰이더 수) |
| N9 | `tanh` 가 `relu` 셰이더를 돌림 | tanh 스윕 8.305e-01, BERT 풀러 6.398 |
| N10 | `matmul` 이 항상 두 피연산자를 복사 | BERT `446 compute shaders, expected 398`, op 표, matmul 스윕 |
| N11 | 항등 뷰 지름길 제거 | 뷰 스윕 `slice identity: 1 shaders, expected 0`, BERT 399 |
| N12 | `broadcast_shape` 를 왼쪽부터 (옛 동작) | 브로드캐스트 테스트의 meta 문구 단언 |
| N15 | `VIEW_RANK_MAX` 를 9 로 (셰이더는 그대로) | 9 개 테스트 — push 레이아웃이 어긋나 모든 뷰가 틀림, 랭크 9 가 거절되지 않음 |
| N16 | `embedding` 의 범위 검사 제거 | 기존 `index 5 was gathered`, 상속 범위 |
| S1 | `gather` 셰이더가 인덱스를 무시 | gather 스윕 `14 of 14 elements differ` |
| S2 | 뷰 셰이더가 축 순서를 뒤집음 | 뷰 스윕, matmul 브로드캐스트, BERT 7.939 |
| S3 | 브로드캐스트 `div` 를 `x * (1/y)` 로 | 브로드캐스트 `div … 67 of 224 elements differ` |

**N6 이 이 문서에서 가장 중요한 줄입니다** — VULKAN6 §6 D5 를 새 커널에서 다시 확인한 것입니다.
호스트 쌍둥이의 비트는 upstream 과 같고 `.device` 는 `vulkan` 입니다. 구분한 것은 카운터뿐입니다.

**하지 않은 무력화: `shader_u32` 제거.** 그 거절을 없애면 테스트의 `expand([2**32 + 1])` 가 **17 GB
할당**을 시도합니다. 16 GB 기계를 다른 회차의 에이전트와 공유하는 상황에서 돌릴 수 없는 실험이라
하지 않았습니다. 이 가드에는 **테스트가 있지만 무력화로 이빨이 확인되지는 않았습니다.**

**하네스 결함 하나를 적습니다.** S1–S3 은 셰이더를 고치고 되돌린 뒤 `.comp` 의 mtime 만 갱신했고,
그래서 S2·S3 실행에서 `test_the_checked_in_spirv_is_not_stale_for_any_shader` 가 **S1 의 흔적**으로
빨개졌습니다(그 테스트가 제 일을 한 것입니다). 무력화 뒤 `.comp`·`.spv` 를 원본과 바이트 비교해 복원을
확인하고 `.spv` 의 mtime 을 갱신했습니다.

---

## 9. 게이트

`TORCHNATIVE_REQUIRE_VULKAN=1` 로 돌렸습니다 — 건너뛴 Vulkan 테스트가 하나라도 있으면 실패합니다.

| 회차 | GATE_EXIT | ok | FAIL | DOCWATCH | VULKAN COVERAGE |
|---|---|---|---|---|---|
| (§9.1 에 채움) | | | | | |

### 9.1 측정 조건

(게이트 실행 뒤 채움.)

---

## 10. 하지 않은 것

- **`sdpa` (기본값) 과 `attention_mask`.** §6. 마스크는 bool 저장 클래스라는 설계 결정이 먼저입니다.
- **디스패치 그리드 상한 검사.** `dispatch_kernel` 은 `ceil(n / 64)` 그룹을 한 번에 보내고,
  `maxComputeWorkGroupCount[0]` 의 Vulkan 최소 보장은 65535 입니다 — **원소 4,194,240 개**. 이 장치의
  어떤 커널도 그것을 검사하지 않습니다(이번 회차 이전부터). BERT-base 의 가장 큰 출력은 여기서
  B × S × 3072 = 73,728 개라 닿지 않지만, 긴 시퀀스·큰 배치는 닿을 수 있습니다. 넘으면 Vulkan 명세상
  잘못된 사용이고 결과는 정의되지 않습니다. **잰 적이 없는 위험으로 적습니다.**
- **dense 경로의 브로드캐스트 거절 문구** (§7).
- **성능.** 아무것도 재지 않았습니다. 뷰 하나당 디스패치 하나, 커널 하나당 `vkQueueSubmit` + fence
  대기 한 번은 그대로입니다.
- **정수 산술.** 여전히 없습니다. 뷰 커널은 정수 워드를 **옮길** 뿐입니다.
- **실물 Adreno/Mali.** 이번 회차의 모든 숫자도 Apple M1 + MoltenVK 1.4.2 하나입니다. §2.2 의 두
  상한이 특히 그 기계들을 염두에 둔 것이므로, 그곳에서 확인되지 않았다는 사실을 함께 적습니다.
