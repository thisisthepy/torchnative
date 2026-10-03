# Vulkan — `sdpa_backward` 가 이 장치에서 돈다. 벽은 리덕션 하나였다

**결론부터.** `docs/devices/VULKAN8.md` §4 는 이렇게 적어 두었습니다:

> `logsumexp` 는 막던 것이 아니었습니다. 멈춘 곳은 `dS` 줄의 `rowsum(dP * P)` 입니다.
> `aten.sum.dim_IntList` 가 첫 벽이고, 목록은 §4 의 표입니다.

그 벽을 실측으로 다시 확인하고(§1), 그 리덕션을 구현했습니다. **그것 하나로 backward 가
끝까지 돕니다** — 업스트림과 원소 단위로 일치하고, 호스트로 한 워드도 내려가지 않습니다.

| 질문 | 답 |
|---|---|
| 첫 벽이 정말 `sum.dim_IntList` 였나 | **예.** 별도 프로세스 실측 (§1). 추론이 아니다 |
| `sum.dim_IntList` 가 이제 도는가 | **예.** 셰이더 `sum_dims_f32`, 임의의 `dim` 리스트에 디스패치 **1 회** (§2) |
| upstream 과 일치하는가 | **예.** 유도 허용치 9.537e-07 에 대해 최악 **5.003e-08** — upstream 자신의 오차의 **0.48 배** (§3) |
| `sum.default` 도 | **예.** 최악 **3.535e-08**, upstream 자신의 오차의 **0.20 배** (§3) |
| **backward 가 이제 되는가** | **예.** dq·dk·dv 전부. 최악 **1.441e-07** (upstream 자신 1.417e-07) (§4) |
| 호스트로 떨어지지 않았음은 무엇이 보증하나 | 디스패치 카운터. backward 는 셰이더 **17 회**, 업로드 0, 읽어오기 **0** (§4) |
| 그 카운터에 이빨이 있나 | **예.** 호스트 쌍둥이로 바꾸면 **값은 전부 맞고 카운터만 빨개진다** (§6 N2) |
| 다음 벽은 | **`aten.ones_like.default`** — `o.sum().backward()` 의 gradient 시드. 실측 (§5). **`docs/devices/VULKAN10.md` 가 그것과 `mean` 을 치웠고, 훈련 한 스텝이 이제 이 장치에서 돕니다** |
| `VULKAN_INDEX_MAX` · `check_dtype` 을 건드렸나 | **아니오.** 한 글자도 바뀌지 않았다 |

<!-- DOCWATCH: count vulkan_tests_ok ge 52 -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/vulkan.rs sum_vulkan present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/vulkan.rs SUM_DIMS_F32_SPV present -->
<!-- DOCWATCH: op-implemented aten.sum.dim_IntList -->
<!-- DOCWATCH: op-implemented aten.sum.default -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vulkan4.py test_sum_over_dims_agrees_with_upstream_at_a_derived_tolerance present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vulkan4.py test_sum_default_agrees_with_upstream_at_a_derived_tolerance present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vulkan4.py test_a_wrong_sum_reduction_is_rejected_by_this_tolerance present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vulkan4.py test_the_sum_reduction_ran_on_the_gpu_in_exactly_one_dispatch present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vulkan4.py test_sum_refuses_what_this_device_does_not_do_rather_than_reaching_for_the_cpu present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vulkan4.py test_a_backward_through_the_vulkan_sdpa_now_runs_and_agrees_with_upstream present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vulkan4.py test_the_vulkan_sdpa_backward_never_left_the_gpu present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vulkan4.py test_what_the_sdpa_backward_still_cannot_reach_is_unreachable_for_a_reason present -->

---

## 1. 벽을 다시 쟀다 — 표를 믿지 않았다

`docs/devices/VULKAN8.md` §4 의 표를 그대로 받아 구현을 시작하지 않았습니다. 직전 회차가
`logsumexp` 에서 **엉뚱한 op 에 회차를 거의 다 쓸 뻔했던** 전례가 있으므로, 별도 인터프리터에서
먼저 돌렸습니다:

```
q, k, v = randn(1,2,3,4).to("vulkan").requires_grad_(True)
o = F.scaled_dot_product_attention(q, k, v)
torch.autograd.grad(o, [q,k,v], grad_outputs=ones(1,2,3,4).to("vulkan"))

  -> NotImplementedError: aten.sum.dim_IntList: not implemented for the vulkan device
     counters: {shader_dispatches: 9, host_uploads: 1, host_downloads: 0}
```

벽은 움직이지 않았습니다. 그래서 그 리덕션 하나만 구현했습니다.

---

## 2. stride 가 없는 장치에서 임의의 축을 줄이는 법

`VkTensor` 에는 shape 만 있고 stride 가 없습니다(`docs/devices/VULKAN4.md` §6). 그래서 CPU 의
상투적인 수단 — 줄일 축을 뒤로 permute 하고 마지막 축 리덕션 커널을 부르는 것 — 이 **이 장치에는
없습니다.** permute 를 하려면 전치된 사본을 실제로 만들어야 하고, 그것은 리덕션 한 번마다 할당
하나와 디스패치 하나가 더 붙는다는 뜻입니다.

대신 **모든 축의 소스 stride 를 호스트에서 shape 로부터 유도해** (`stride[d] = prod(shape[d+1..])`,
이것이 "contiguous" 의 정의입니다) 셰이더에 넘깁니다. 축은 이미 **나뉘어** 도착합니다 — 살아남는
축이 먼저(출력 순서), 줄일 축이 그 뒤. 셰이더는 두 인덱스 공간을 각각 걸어갑니다:

```
out[i] = sum_j src[base(i) + offset(j)]
```

그래서 **임의의 `dim` 리스트가 디스패치 1 회**입니다. 인접하지 않은 두 축(`[0, 2]`)도, 빈
리스트(torch 는 "모든 축"으로 읽습니다)도, `dim=None` 도 같은 커널 한 번입니다.
푸시 상수는 20 워드 = 80 바이트로, `strided_gather_u32` 와 같은 예산 안입니다.

### 2.1 보정 합 — 이것은 고른 것이 아니라 재고 나서 바꾼 것

처음에는 여덟 개의 누적 레인으로 upstream 의 벡터 리덕션 모양을 흉내 냈습니다. **부족했습니다:**

```
sum_all[2, 512]:  shim 1.139e-06   유도 허용치 9.537e-07   (upstream 자신 9.181e-07)
```

upstream 의 CPU 리덕션은 벡터 레인에 대한 cascade 인데 이 셰이더의 한 invocation 에는 그럴 트리가
없습니다. **허용치를 넓히는 것은 선택지가 아니므로**(그것이 이 저장소가 반복해서 당한 실패입니다)
산술을 고쳤습니다 — Kahan-Babuška-Neumaier 보정합입니다. 각 덧셈이 버린 반올림을 보정항이
이어받으므로 결과는 upstream **보다 정확**해집니다:

```
sum_all[2, 512]:  shim 3.535e-08   (upstream 자신의 오차의 0.20 배)
```

§6 N1 이 보정항을 지우면 무슨 일이 일어나는지 다시 확인합니다.

---

## 3. 값 — 유도된 허용치, 그리고 그 허용치의 이빨

오라클은 **upstream 자신의 커널**입니다. 허용치는 `docs/numerics/AGREE.md` §2 의 규칙대로 이
모집단의 upstream float32-대-float64 오차에서 **다시 유도**합니다 — 이 파일에 적힌 수가 아닙니다.

```
sum.dim_IntList: 10 cases, tolerance max(p90 3.022e-07, 8 ulp 9.537e-07) = 9.537e-07;
  worst shim 5.003e-08 at sum[2, 512] dim=[-1] keepdim=False (upstream's own 3.022e-07);
  21/39 elements differ in bits from upstream f32; worst element 0.48x upstream's own error

sum.default: 4 cases, tolerance max(p90 1.543e-07, 8 ulp 9.537e-07) = 9.537e-07;
  worst shim 3.535e-08 at sum_all[2, 512] (upstream's own 9.181e-07);
  3/4 elements differ in bits from upstream f32; worst element 0.20x upstream's own error
```

스윕은 `[-1] keepdim=True`(규칙 자신의 `rowsum` 모양), 선두 축, 인접하지 않은 두 축, 빈 리스트,
`dim=None`, 512 길이 축, 랭크 1~4 를 포함합니다.

### 3.1 허용치에 이빨이 있는지 확인했다

`docs/architectures/VOICE4.md` §6 은 허용치를 100 배 넓히고도 전부 초록이던 회차를 적어 두었습니다.
그것을 막는 것은 더 작은 수가 아니라 **틀린 답을 넣고 거절되는지 보는 것**입니다:

| 넣은 값 | 유도 허용치 9.537e-07 에 대해 |
|---|---|
| 정답 (대조군) | 4.309e-08 — 통과 |
| **다른 축**을 줄인 값 | 5.608e-01 — 거절 (원소 단위 **9,212,466 배**) |
| 각 행의 **마지막 항을 빠뜨린** 값 | 3.599e-01 — 거절 (원소 단위 **13,099,890 배**) |

---

## 4. backward — **된다**, 그리고 무엇이 그것을 보증하나

`test_a_backward_through_the_vulkan_sdpa_now_runs_and_agrees_with_upstream` 은 별도 인터프리터에서
vendored shim 을 돌리고, q·k·v·grad_output 을 **평평한 리스트로 건네** 양쪽이 같은 숫자를
미분하게 합니다(같은 시드가 같은 값을 준다고 가정하지 않습니다). 돌려받은 세 gradient 를
이 프로세스의 upstream autograd 와 원소 단위로 비교합니다:

```
sdpa backward (dq, dk, dv): 3 cases, tolerance max(p90 1.417e-07, 8 ulp 9.537e-07) = 9.537e-07;
  worst shim 1.441e-07 at grad_q (upstream's own 1.417e-07);
  64/72 elements differ in bits from upstream f32
```

**그리고 카운터.** 이것이 값으로는 절대 할 수 없는 말을 합니다 — 저장된 q·k·v 를 내려받아
호스트에서 미분하고 답 세 개를 올려보내는 구현은 위 일치 테스트를 **정확히 통과**하고
`.device` 도 `vulkan` 이라고 말합니다. 구분하는 것은 이것뿐입니다:

```
sdpa backward on vulkan: 17 compute shaders, 0 uploads, 0 downloads
```

`test_the_vulkan_sdpa_backward_never_left_the_gpu` 가 그 **0** 을 단언합니다. 디스패치 수는
`>= 10` 으로 두었습니다 — 정확한 수는 `tape.rs` 가 고를 일이고, 여기서 못 박으면 관측 가능한
것을 아무것도 바꾸지 않는 리팩터에 이 테스트가 깨집니다. 움직여서는 안 되는 것은 **0** 입니다.

---

## 5. 다음 벽 — 읽은 것이 아니라 실측한 것

`grad_outputs` 를 직접 주면 규칙이 끝까지 돕니다. 손실 모양의 backward 는 그 전에 gradient 를
심어야 하고, 거기서 멈춥니다:

```
o.sum().backward()      -> NotImplementedError: aten.ones_like.default
o.mean().backward()     -> NotImplementedError: aten.mean.default
is_causal=True (forward) -> NotImplementedError: ... implemented only for is_causal=False
```

`aten.sum.default` 는 **더 이상 그 벽이 아닙니다** — 이 회차가 가르쳤습니다.
`test_what_the_sdpa_backward_still_cannot_reach_is_unreachable_for_a_reason` 이 이 세 줄을
실측으로 고정합니다. 거절 메시지가 **가르쳐진 op 을 전부 나열**하므로, 그 안에서
`aten.sum.default` 를 부분 문자열로 찾으면 그것을 벽이라고 오독합니다 — 그래서 테스트는
메시지 맨 앞의 op 이름만 읽습니다.

### 5.1 `is_causal` 분기의 네 op 은 왜 구현하지 않았나

`docs/devices/VULKAN8.md` §4 의 표에서 `aten.ones.default` · `aten.tril.default` ·
`aten.eq.Scalar` · `aten.masked_fill.Scalar` 는 전부 `is_causal=True` 분기의 것입니다.
**이 장치의 forward 가 `is_causal` 을 이름을 대며 거절하므로 그 분기는 진입할 수 없습니다.**
지금 `tril` 커널을 쓰면 어떤 테스트도 그것을 실행시킬 수 없고, **초록 표시가 붙은 검증되지 않은
코드**가 됩니다. 그래서 쓰지 않았고, 테스트는 그 op 들이 아직 없다는 것과 **그 이유가 아직
참이라는 것**을 둘 다 고정합니다 — forward 가 `is_causal` 을 받게 되는 날 이 테스트가 빨개집니다.

---

## 6. 무력화 — 일부러 깨고 빨개지는지 봤다

구현 전 빌드에서 새 테스트가 **FAIL 4** 였고(적색 단계), 그다음 보증을 하나씩 깼습니다.

| # | 무력화 | 빨개진 테스트 |
|---|---|---|
| N1 | 보정항을 버린다 (`c[i] = sum`) | 일치 스윕 둘 — `sum_all[2,512] 1.841e-06`, `sum[3,7] dim=[] 3.843e-06`, 허용치 9.537e-07 |
| N2 | **호스트 쌍둥이** — 같은 보정합을 CPU 에서 하고 업로드 | **값은 전부 맞았다.** 잡은 것은 카운터: `ran 0 compute shaders, expected exactly 1` · `{dispatches 0, uploads 1, downloads 1}` (두 테스트) |
| N3 | 각 리덕션의 **마지막 항을 빠뜨린다** (`j + 1 < total`) | 일치 스윕 둘 **그리고 sdpa backward 일치** — `grad_q 5.993e-01` |
| — | (셰이더를 컴파일하지 않고 두었을 때) | `.spv` 신선도 검사와 "모든 셰이더가 디스패처에서 도달 가능한가" 검사가 둘 다 빨갰다 |

**N2 가 이 문서에서 가장 중요한 줄입니다.** 호스트 쌍둥이의 값은 일치 스윕을 **전부 통과**했고
`.device` 는 `vulkan` 이었습니다. 구분한 것은 디스패치 카운터뿐입니다.

**N3 은 backward 테스트에 이빨이 있는지의 증거입니다.** 리덕션을 한 항만큼 틀리게 하자
gradient 일치가 `5.993e-01` 로 깨졌습니다 — 즉 그 테스트는 통과할 수밖에 없는 테스트가 아닙니다.

### 6.1 N3 이 처음에는 backward 를 빨갛게 만들지 않았다 — 그것이 함정이었다

N3 을 처음 돌렸을 때 일치 스윕 둘만 빨갛고 **backward 테스트는 초록**이었습니다. 무력화가
약해서가 아니라, backward 프로브가 **vendored 트리의 `_C.abi3.so` 를 쓰는 별도 프로세스**이기
때문입니다 — 변형된 빌드를 스테이지에만 복사하고 vendored 트리에는 복사하지 않았으므로,
그 프로세스는 멀쩡한 빌드를 미분하고 있었습니다.

**서브프로세스로 측정하는 테스트를 무력화할 때는 그 서브프로세스가 읽는 아티팩트를 바꿔야
합니다.** 이것을 적어 두는 이유는, 이 함정이 "무력화했는데 초록이다 → 테스트가 무의미하다" 라는
**반대 방향의 오진**을 만들기 때문입니다.

---

## 7. 하지 않은 것

- **`aten.ones_like.default` · `aten.zeros_like.default`.** §5 의 다음 벽. 이것을 가르치면
  `o.sum().backward()` 가 끝까지 돌지만, 이 회차는 `sdpa_backward` 의 벽 하나를 끝까지
  증명하는 데 썼습니다. **(`docs/devices/VULKAN10.md` 가 구현했습니다 — 시드는 업로드가
  아니라 셰이더입니다.)**
- **`aten.mean.default` · `aten.mean.dim`.** 같은 커널에 나눗셈 하나지만 §5 의 세 번째 줄입니다.
  **(`docs/devices/VULKAN10.md` 가 구현했습니다 — 같은 커널, divisor 하나, 디스패치 1 회.)**
- **`is_causal` 분기의 네 op.** §5.1 — 실행시킬 수 없으므로 증명할 수 없습니다.
- **`dim=None` + `keepdim=True`.** 이 장치는 `aten.rs` 의 dense 커널과 **같게** 랭크 0 으로
  접습니다(`sum_all`). upstream 의 `torch.sum(x, dim=None, keepdim=True)` 는 전부 1 인 shape 을
  주므로 둘이 갈라지지만, shim 안에서 두 커널이 서로 다르게 답하는 것이 더 나쁜 결함이므로
  dense 쪽에 맞췄습니다. 스윕에 넣지 않았고, 여기에 적습니다.
- **성능.** 아무것도 재지 않았습니다. 이 리덕션은 출력 원소당 invocation 하나이므로
  `sum.default` 는 **단일 invocation** 입니다 — 정확하지만 병렬이 아닙니다.
  `docs/devices/VULKAN2.md` §4.4 의 이유 그대로, 이 회차는 값만 주장합니다.
- **실물 Adreno/Mali.** 이 회차의 모든 숫자도 Apple M1 + MoltenVK 하나입니다.
