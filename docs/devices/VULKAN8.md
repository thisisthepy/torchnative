# Vulkan — sdpa 의 `logsumexp` 를 계산했고, 그것이 backward 를 막던 것이 아니었다

**결론부터.** `docs/devices/VULKAN7.md` 를 낸 회차는 이렇게 적어 두었습니다:

> Vulkan sdpa 는 `logsumexp` 로 `None` 을 돌려주므로 그것을 지나는 backward 는 실패한다.
> 기존부터 그랬고, 소비하는 곳이 없으며(`bootstrap.py` 의 `sdpa` 는 `[0]` 을 취한다),
> 계산하려면 새 커널이 필요하다.

새 커널을 썼습니다. 그리고 **그 문장에서 자연스럽게 읽히는 결론 — "채우면 backward 가 된다" — 은
틀렸습니다.** `tape.rs` 의 `sdpa_backward` 는 `logsumexp` 를 **읽지 않습니다.** 저장된 q·k·v 에서
확률을 다시 계산해 교과서 형태를 미분하고, `logsumexp` 슬롯에 대해 하는 일이라고는 그쪽으로
들어온 gradient 를 **거절**하는 것뿐입니다.

| 질문 | 답 |
|---|---|
| `logsumexp` 가 이제 나오는가 | **예.** 셰이더 `logsumexp_lastdim_f32` 한 번, 디스패치 1 회 (§2) |
| upstream 과 일치하는가 | **예.** 유도 허용치 9.537e-07 에 대해 최악 **9.702e-08** — upstream 자신의 오차의 **1.00 배** (§3) |
| 그 허용치에 이빨이 있는가 | **예.** 틀린 값 두 개를 넣어 거절되는 것을 확인했다 (§3.1) |
| 두 번째 패스가 꼭 필요했나 | **예. 근거는 §2** — 최댓값과 지수합은 softmax 셰이더 안에 이미 있지만 레지스터에서 죽는다 |
| **backward 가 이제 되는가** | 이 회차에는 **아니오** — `aten.sum.dim_IntList` 에서 멈췄다 (§4). `logsumexp` 는 막던 것이 아니었다. **그 벽은 `docs/devices/VULKAN9.md` 가 치웠고, backward 는 이제 돈다** |
| 호스트로 떨어지지 않았음은 무엇이 보증하나 | 디스패치 카운터. 호스트 쌍둥이로 바꾸면 **값은 전부 맞고 카운터만 빨개진다** (§6 N4) |
| `VULKAN_INDEX_MAX` · `check_dtype` 을 건드렸나 | **아니오.** 한 글자도 바뀌지 않았다 |

<!-- DOCWATCH: symbol-in-file crates/torch_c/src/vulkan.rs logsumexp_lastdim present -->
<!-- DOCWATCH: symbol-in-file crates/torch_c/src/vulkan.rs LOGSUMEXP_LASTDIM_F32_SPV present -->
<!-- DOCWATCH: symbol-in-file tests/test_vulkan4.py test_the_sdpa_logsumexp_agrees_with_upstream_at_a_derived_tolerance present -->
<!-- DOCWATCH: symbol-in-file tests/test_vulkan4.py test_a_wrong_logsumexp_is_rejected_by_this_tolerance present -->
<!-- DOCWATCH: symbol-in-file tests/test_vulkan4.py test_the_logsumexp_ran_on_the_gpu_and_cost_exactly_one_more_shader present -->
<!-- DOCWATCH: symbol-in-file tests/test_vulkan4.py test_a_fully_masked_row_reports_upstreams_logsumexp_convention present -->
<!-- DOCWATCH: symbol-in-file tests/test_vulkan4.py test_a_backward_through_the_vulkan_sdpa_now_runs_and_agrees_with_upstream present -->
<!-- DOCWATCH: symbol-in-file tests/test_vulkan4.py test_every_test_in_this_file_is_actually_collected present -->

---

## 1. 그 슬롯에 무엇이 들어가는가 — 이름이 아니라 측정

`_scaled_dot_product_flash_attention_for_cpu` 의 두 번째 결과를 **별도 프로세스에서 직접 쟀습니다.**
이름에서 유추하지 않았습니다.

```
aten::_scaled_dot_product_flash_attention_for_cpu(
    Tensor query, Tensor key, Tensor value, float dropout_p=0., bool is_causal=False,
    *, Tensor? attn_mask=None, float? scale=None) -> (Tensor output, Tensor logsumexp)
```

| 항목 | 측정값 |
|---|---|
| shape | `query.shape[:-1]` — `(B, H, T)`. 질의 행마다 값 하나 |
| dtype | `float32` (`float16` 질의에도 `float32`. `test_metaemb.py` 가 meta 장치에서 이미 고정) |
| 값 | **마스크·스케일이 적용된** 점수를 키 축으로 자연로그 log-sum-exp. `torch.logsumexp(scores.double(), -1)` 대비 최악 **2.28e-07** — float32 자신의 반올림 |
| 레이아웃 | **비연속.** shape `(2,3,5)` 에 stride `(15,1,3)` — upstream 은 `(B,T,H)` 를 잡고 `transpose(1,2)` 를 돌려준다 |
| 전부 마스크된 행 | **0.0.** `torch.logsumexp` 의 `-inf` 가 **아니다** (§3.2) |

**마지막 두 줄이 이 문서가 유추로 얻을 수 없었던 것들입니다.**

*`attn_mask` 와 `scale` 은 키워드 전용*이라는 것도 여기서 나왔습니다 — 위치 인자로 7 개를 주면
`takes 5 positional argument(s) but 7 was/were given` 입니다.

### 1.1 stride 는 따라가지 않는다 — 그리고 그렇게 적는다

`VkTensor` 에는 shape 만 있고 stride 가 없습니다(`docs/devices/VULKAN4.md` §6). 그러므로 이 장치는
upstream 의 전치된 레이아웃을 **재현할 수 없고, 재현한다고 주장하지 않습니다.** 원소 단위로 값이
일치하고, stride 는 이 장치 자신의 것입니다. 누가 나중에 발견하도록 두지 않고 여기에 적습니다.

---

## 2. 두 번째 패스가 필요한가 — 근거

`sdpa_vulkan` 은 바로 윗줄에서 `aten._softmax.default` 를 돌리고, 그 셰이더
(`softmax_lastdim_f32.comp`)는 행 최댓값 `m` 과 지수합 `sum` 을 **이미 계산합니다.**
`logsumexp = m + log(sum)` 이므로 필요한 것은 전부 그 안에 있습니다.

**그런데 그것들은 그 셰이더의 레지스터에서 죽습니다.** 꺼내오려면 `aten._softmax.default` 가
텐서 둘을 돌려주어야 하고, `_softmax` 는 이 장치에서 **네 군데가 공유하는 단일 출력 op** 입니다.
선택지는 둘이었습니다:

| 선택지 | 비용 |
|---|---|
| A. `_softmax` 의 계약을 바꿔 `(out, lse)` 를 돌려준다 | 공유 op 의 계약 변경. 모든 호출부가 두 번째 값을 무시해야 하고, 그 op 은 더 이상 upstream 의 `_softmax` 가 아니다 |
| B. 별도 리덕션 커널 1 회 | 이미 장치에 있는 텐서를 한 번 더 읽는다. 디스패치 +1, 출력 `rows` 워드 |

**B 를 골랐습니다.** 이것은 두 번째 **패스**이지 두 번째 **forward** 가 아닙니다 — matmul 은 다시
돌지 않고, 호스트로 읽어오는 것도 없습니다. 비용은 숨기지 않고 §5 의 표에 적었습니다.

`logsumexp_lastdim` 은 **디스패치 표에 올리지 않았습니다.** `aten.logsumexp.default` 에는 `dim`
리스트·`keepdim`·정수 승격 규칙(`aten.rs`)이 있고 이 커널은 그중 아무것도 하지 않습니다.
`vulkan_ops()` 에 이름을 올리면 그 전부를 주장하는 것이 됩니다. **op 목록은 온전히 구현된 것들의
목록으로 남습니다.**

---

## 3. 값 — 유도된 허용치, 그리고 그 허용치의 이빨

오라클은 **upstream 자신의 커널**입니다(§1 의 rank-4 5 케이스, 마스크 있는 것 2 개 포함).
허용치는 `docs/numerics/AGREE.md` §2 의 규칙대로 **이 모집단의 upstream float32-대-float64 오차**에서
다시 유도합니다 — 이 파일에 적힌 수가 아닙니다.

```
sdpa logsumexp: 5 cases, tolerance max(p90 8.651e-08, 8 ulp 9.537e-07) = 9.537e-07;
  worst shim 9.702e-08 at lse[1, 2, 3, 16, 3, True] (upstream's own 9.702e-08);
  19/84 elements differ in bits from upstream f32; worst element 1.00x upstream's own error on it
```

최악 원소가 upstream 자신의 오차의 **1.00 배**입니다. `tanh`·`matmul` 과 마찬가지로 허용치를
결정한 것은 **8 ulp 바닥**이었습니다(p90 이 그보다 작았습니다).

### 3.1 허용치에 이빨이 있는지 확인했다

`docs/architectures/VOICE4.md` §6 은 허용치를 **100 배 넓히고도 전부 초록**이던 회차를 적어 두었습니다.
그것을 막는 것은 더 작은 수가 아니라, **틀린 답을 넣고 거절되는지 보는 것**입니다.
`test_a_wrong_logsumexp_is_rejected_by_this_tolerance` 가 그렇게 합니다:

| 넣은 값 | 유도 허용치 9.537e-07 에 대해 |
|---|---|
| 정답 (대조군) | 8.651e-08 — 통과 |
| **스케일 안 한** 점수의 logsumexp | 8.782e-01 — 거절 (원소 단위 7,367,036 배) |
| `log` 를 빼먹은 값 | 9.978e+00 — 거절 (83,700,776 배) |

### 3.2 전부 마스크된 행 — upstream 의 규약이지 `torch.logsumexp` 의 것이 아니다

```
_scaled_dot_product_flash_attention_for_cpu, 행 전체가 -inf  ->  logsumexp 0.0,  출력 전부 0
torch.logsumexp([-inf, -inf, -inf], dim=-1)                  ->  -inf
```

구현 대상은 앞의 것이므로 셰이더가 그 분기를 가집니다. 분기가 없으면 `exp(-inf - -inf)` 가 NaN 이
되어 **둘 중 어느 답도 아닙니다** (§6 N2 가 그것을 확인합니다).

**옆에서 고치지 않은 것:** 같은 행의 **출력**은 이 장치에서 NaN 입니다. `sdpa_vulkan` 이
`aten._softmax.default` 를 돌리고 그 셰이더에 같은 무방비 shift 가 있기 때문입니다.
`aten._safe_softmax.default` 가 그것을 다루는 op 이고 이 장치에 **이미 있습니다.** 그러나 어느
softmax 를 쓰는지 바꾸면 이 장치의 **모든 sdpa forward 의 산술이 바뀌므로**, logsumexp 회차에
끼워 넣지 않았습니다. §5 에 남깁니다.

---

## 4. backward 는 되는가 — **아니오**, 그리고 무엇이 더 필요한지

`logsumexp` 가 정확한 지금, 별도 인터프리터에서 실제로 돌려 봤습니다:

```
q, k, v = randn(1,2,3,4).to("vulkan").requires_grad_(True)
o = F.scaled_dot_product_attention(q, k, v)
torch.autograd.grad(o, [q,k,v], grad_outputs=ones(1,2,3,4).to("vulkan"))

  -> NotImplementedError: aten.sum.dim_IntList: not implemented for the vulkan device
     counters: shader_dispatches ..., host_downloads 0
```

**`logsumexp` 는 막던 것이 아니었습니다.** `tape.rs::sdpa_backward` 의 주석이 그것을 직접 말합니다 —
그 규칙은 저장된 q·k·v 에서 확률을 다시 계산합니다:

```
P  = softmax(scale * q k^T + mask)        out = P v
dv = P^T dout                             dP  = dout v^T
dS = P * (dP - rowsum(dP * P))
dq = scale * dS k                         dk  = scale * dS^T q
```

`logsumexp` 는 이 식 어디에도 없습니다. 규칙이 그 슬롯에 대해 하는 유일한 일은
`gouts[1].is_some()` 일 때 **거절**하는 것입니다.

멈춘 곳은 `dS` 줄의 `rowsum(dP * P)` 입니다. 이 장치가 **아직 갖고 있지 않은** 것들:

| op | 무엇 때문에 필요한가 |
|---|---|
| `aten.sum.dim_IntList` | **규칙 자신의 `rowsum(dP * P)`. 지금 멈추는 곳** |
| `aten.sum.default` | 손실을 만드는 데 필요. 규칙에 닿기도 전에 걸린다 |
| `aten.ones_like.default`, `aten.zeros_like.default` | `grad_outputs`, gradient 누적 |
| `aten.ones.default`, `aten.tril.default`, `aten.eq.Scalar`, `aten.masked_fill.Scalar` | `is_causal=True` 분기 (Vulkan forward 는 `is_causal` 을 아예 거절하므로 그것을 푼 뒤에나 닿는다) |

규칙이 쓰는 나머지(`matmul`, `mul.Scalar`, `add.Tensor`, `_safe_softmax`, `mul.Tensor`, `sub.Tensor`,
`transpose.int`, `reshape`, `expand`)는 **전부 있습니다.** 즉 남은 것은 **리덕션 하나 계열**입니다.

`test_a_backward_through_the_vulkan_sdpa_still_does_not_work_and_names_what_is_missing` 이 이 상태를
이름으로 고정하고, **목록이 바뀌는 날 빨개지도록** 쓰여 있었습니다 — 누가 `sum.dim_IntList` 를
가르치면 그 테스트가 실패하고, 없어진 벽을 계속 주장할 수 없습니다.

**그리고 실제로 그렇게 됐습니다.** 다음 회차가 `sum.dim_IntList` 를 가르치자 그 테스트가
"a backward through the vulkan sdpa succeeded" 로 빨개졌고, 그 자리에
`test_a_backward_through_the_vulkan_sdpa_now_runs_and_agrees_with_upstream` 이 들어갔습니다.
**위 표의 그 줄과 §7 의 첫 항목은 이 문서가 낸 회차의 상태이며 더 이상 현재 상태가 아닙니다** —
현재 상태는 `docs/devices/VULKAN9.md` 입니다.

---

## 5. VULKAN7 §5 의 `sdpa` + `attention_mask` 주장은 **검증된 적이 없다**

이 회차에서 나온 것 중 가장 무거운 것이고, 이 회차의 작업이 아니라 **직전 회차의 것**입니다.

커밋 `716d16e` ("BERT on Vulkan takes an attention_mask") 는 이렇게 적었습니다:

> `bert-base-uncased` 가 **기본값 `sdpa`** 와 **진짜 패딩된 `attention_mask`** 로 Vulkan 에서
> forward 한다 … 셰이더 410 회, 호스트 업로드 1 · 읽어오기 0, CPU 와 약 8.6e-06 에서 일치.

그 주장을 담당하는 테스트
`test_the_pretrained_bert_sdpa_mask_forward_builds_mask_on_host_and_agrees_with_upstream` 은
**한 번도 실행된 적이 없습니다.**

원인은 파일 구조입니다. 그 테스트는 `test_vulkan4.py` 의 `raise SystemExit(_main())` **아래**에
정의되어 있었고, 모듈은 위에서 아래로 실행되므로 **`def` 에 닿기 전에 인터프리터가 나갔습니다.**
러너는 41 개가 든 파일에서 40 개를 수집했고, **그 두 숫자를 비교하는 것이 아무것도 없었습니다.**
`AGENTS.md` §17.5 의 "실패할 수 없는 검증" 의 가장 순수한 형태입니다 — 약한 단언이 아니라
**도달하지 않는** 단언입니다.

러너를 파일 맨 끝으로 옮기자 처음으로 실행됐고, **실패합니다:**

```
masking_utils.py:507  sdpa_mask:  batch_arange = torch.arange(batch_size, device=device)
  -> NotImplementedError: aten.arange.default: the vulkan device in this build stores float32 only, not int64
```

그 테스트의 방식은 `BertModel.get_extended_attention_mask` 를 몽키패치해 마스크를 호스트에서
만들어 한 번 업로드하는 것이었는데, **transformers 5.15.1 의 BERT 는 그 메서드를 부르지 않습니다.**
`BertModel.forward` → `_create_attention_masks` → `create_bidirectional_mask` →
`masking_utils.sdpa_mask` 로 가고, `sdpa_mask` 가 맨 먼저 하는 일이 모델 장치 위의 `arange` 입니다.
이 백엔드에는 정수 산술이 **의도적으로** 없으므로(`docs/devices/VULKAN6.md` §1) 이름을 대며 거절하고,
패치된 메서드는 그것을 막을 기회를 얻지 못합니다.

**그러므로 "기본값 sdpa + 진짜 attention_mask 로 BERT 가 Vulkan 에서 돈다" 는 검증되지 않았고,
그 안의 숫자(410 셰이더, 업로드 1, 8.6e-06)는 이 트리의 무엇도 만들어낸 적이 없습니다.**

이번 회차가 한 일은 그 테스트를 **측정된 것을 고정하도록** 다시 쓴 것입니다
(`..._refuses_at_arange_and_the_host_mask_hook_is_never_called`). 단언 둘이고, 두 번째가 이것을
증상이 아니라 증거로 만듭니다:

- forward 가 `aten.arange.default` 에서 **이름을 대며** 거절한다;
- `get_extended_attention_mask` 호출 횟수가 **0** 이다 — **그것이 이유다.** 이 단언이 없으면
  실패를 읽는 사람은 호스트 마스크 경로가 도달됐는데 깨진 것이라고 합리적으로 오해합니다.

`test_every_test_in_this_file_is_actually_collected` 가 같은 일이 다시 일어나는 것을 막습니다 —
소스의 `def test_` 개수와 러너가 수집한 개수를 비교합니다.

### 5.1 §5 에서 함께 드러난, 감사하지 않은 관찰

거절 시점의 카운터는 `{shader_dispatches: 10, host_uploads: 1, host_downloads: 1}` 입니다.
**읽어오기가 0 이 아닙니다.** 임베딩 프롤로그가 먼저 돌고 그 안의 무언가가 버퍼 하나를 호스트로
읽어옵니다. 테스트가 이 숫자를 **정확히** 고정하지만, 이번 회차는 **그것이 무엇인지 조사하지
않았습니다.** forward 중의 설명되지 않은 읽어오기는 `docs/devices/VULKAN6.md` 가 "값으로는 잡을 수
없다" 고 한 바로 그 모양이므로, 축복하지 않고 **미해결로 적습니다.**

---

## 6. 무력화 — 일부러 깨고 빨개지는지 봤다

구현 전 빌드에서 새 테스트가 **FAIL 5** 였고(적색 단계), 그다음 보증을 하나씩 깼습니다.

| # | 무력화 | 빨개진 테스트 |
|---|---|---|
| N1 | 셰이더가 최댓값을 되더하지 않음 (`log(sum)`) | 일치 스윕 `shim 7.448e-01 … exceeds 9.537e-07`, 마스크된 행 |
| N2 | 전부 마스크된 행의 분기 제거 | 마스크된 행 — `reported logsumexp nan; … NaN is neither answer` |
| N3 | 스케일·마스크 **전**의 점수로 계산 | 일치 스윕 `6.024e-01`, 마스크된 행 `6.445986` |
| N4 | **호스트 쌍둥이** — 같은 산술을 CPU 에서 하고 업로드 | **값은 전부 맞았다.** 잡은 것은 카운터: `ran 5 compute shaders, expected 6` (두 테스트) |
| N5 | 러너 아래에 테스트를 정의 | 수집 가드 — `1 test(s) are defined in this file but were not collected` |

**N4 가 이 문서에서 가장 중요한 줄입니다** — `docs/devices/VULKAN7.md` §8 N6 을 새 커널에서 다시
확인한 것입니다. 호스트 쌍둥이의 값은 일치 스윕과 마스크된 행 테스트를 **전부 통과**했고,
`.device` 는 `vulkan` 이었습니다. 구분한 것은 디스패치 카운터뿐입니다.

---

## 7. 하지 않은 것

- **backward.** §4. `aten.sum.dim_IntList` 가 첫 벽이고, 목록은 §4 의 표입니다.
  **(이 회차에서는. `docs/devices/VULKAN9.md` 가 그 리덕션을 구현했고 backward 는 이제 돕니다.)**
- **`sdpa` 가 `_safe_softmax` 를 쓰도록 바꾸는 것.** §3.2. 모든 sdpa forward 의 산술이 바뀝니다.
- **`sdpa` + `attention_mask` 로 BERT 를 실제로 돌리는 것.** §5. `masking_utils.sdpa_mask` 에서
  가로채거나 정수 산술을 가르쳐야 하고, 후자는 `check_dtype` 의 정책을 넓히는 일이라
  **조용히 하지 않습니다.**
- **§5.1 의 읽어오기 1 회가 무엇인지.**
- **`logsumexp` 의 stride.** §1.1 — 이 장치가 재현할 수 없습니다.
- **성능.** 아무것도 재지 않았습니다. `docs/devices/VULKAN2.md` §4.4 의 이유 그대로.
- **실물 Adreno/Mali.** 이 회차의 모든 숫자도 Apple M1 + MoltenVK 1.4.2 하나입니다.
