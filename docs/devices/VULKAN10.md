# Vulkan — 한 커널의 gradient 에서 **훈련 한 스텝**으로. 벽은 둘이었고 셋째는 결함이었다

**결론부터.** `docs/devices/VULKAN9.md` §5 는 다음 벽을 추론이 아니라 실측으로 남겨 두었습니다:

> `o.sum().backward()` -> `aten.ones_like.default`
> `o.mean().backward()` -> `aten.mean.default`

그 둘을 다시 실측으로 확인하고(§1), 둘 다 구현했습니다. **그리고 그것만으로는 스텝이 완성되지
않았습니다** — 셋째 벽은 op 이 아니라 `bootstrap.py` 의 gradient 누적 경로에 있던 결함이었고
(§4), 그것까지 고친 뒤에 **forward · loss · backward · 파라미터 갱신이 전부 이 장치에서 돕니다.**

| 질문 | 답 |
|---|---|
| VULKAN9 이 적은 벽이 아직 그 벽이었나 | **예.** 별도 프로세스 실측 (§1). 추론이 아니다 |
| `mean.default` · `mean.dim` 이 도는가 | **예.** `sum_dims_f32` 에 divisor 하나, 디스패치 **1 회** (§2) |
| upstream 과 일치하는가 | **예.** 유도 허용치 9.537e-07 에 대해 `mean.dim` 최악 **4.865e-08**, `mean.default` 최악 **6.533e-08** (§2.1) |
| `ones_like` · `zeros_like` 가 도는가 | **예. 그리고 업로드가 아니라 셰이더다** — 그 이유가 §3 |
| **훈련 한 스텝이 도는가** | **예.** 셰이더 **18 회**, 업로드 0, 읽어오기 0. 갱신된 가중치가 upstream CPU 의 같은 스텝과 원소 단위로 **비트까지 일치** (§5) |
| 손실이 실제로 내려가는가 | **예.** 3 스텝 `2.746104 -> 2.359784 -> 2.062743`, 셰이더 54 회, 업로드 0, 읽어오기 0 (§5.1) |
| 그 0 에 이빨이 있나 | **예.** 호스트 쌍둥이로 바꾸면 **값은 전부 맞고 카운터만 빨개진다** (§6 N3) |
| 다음 벽은 | **`aten.add_.Tensor`** — 이미 있는 `.grad` 에 두 번째로 누적할 때. 실측 (§7) |
| `VULKAN_INDEX_MAX` · `check_dtype` 을 건드렸나 | **아니오.** 한 글자도 바뀌지 않았다 |

<!-- DOCWATCH: count vulkan_tests_ok ge 64 -->
<!-- DOCWATCH: symbol-in-file crates/torch_c/src/vulkan.rs like_fill present -->
<!-- DOCWATCH: symbol-in-file crates/torch_c/src/vulkan.rs FILL_F32_SPV present -->
<!-- DOCWATCH: symbol-in-file crates/torch_c/shaders/fill_f32.comp uintBitsToFloat present -->
<!-- DOCWATCH: symbol-in-file crates/torch_c/shaders/sum_dims_f32.comp do_mean present -->
<!-- DOCWATCH: op-implemented aten.mean.default -->
<!-- DOCWATCH: op-implemented aten.mean.dim -->
<!-- DOCWATCH: op-implemented aten.ones_like.default -->
<!-- DOCWATCH: op-implemented aten.zeros_like.default -->
<!-- DOCWATCH: symbol-in-file tests/test_vulkan4.py test_ones_like_and_zeros_like_are_filled_by_a_shader_not_by_an_upload present -->
<!-- DOCWATCH: symbol-in-file tests/test_vulkan4.py test_ones_like_refuses_a_dtype_this_device_cannot_hold present -->
<!-- DOCWATCH: symbol-in-file tests/test_vulkan4.py test_ones_like_refuses_a_device_it_would_have_to_leave_to_honour present -->
<!-- DOCWATCH: symbol-in-file tests/test_vulkan4.py test_mean_over_dims_agrees_with_upstream_at_a_derived_tolerance present -->
<!-- DOCWATCH: symbol-in-file tests/test_vulkan4.py test_mean_default_agrees_with_upstream_at_a_derived_tolerance present -->
<!-- DOCWATCH: symbol-in-file tests/test_vulkan4.py test_a_wrong_mean_reduction_is_rejected_by_this_tolerance present -->
<!-- DOCWATCH: symbol-in-file tests/test_vulkan4.py test_the_mean_reduction_ran_on_the_gpu_in_exactly_one_dispatch present -->
<!-- DOCWATCH: symbol-in-file tests/test_vulkan4.py test_a_training_step_runs_end_to_end_on_the_vulkan_device_and_agrees_with_upstream present -->
<!-- DOCWATCH: symbol-in-file tests/test_vulkan4.py test_the_training_step_never_left_the_gpu present -->
<!-- DOCWATCH: symbol-in-file tests/test_vulkan4.py test_three_training_steps_drive_the_loss_down_and_track_upstream present -->

---

## 1. 벽을 다시 쟀다 — 앞 문서의 표를 그대로 받지 않았다

`docs/devices/VULKAN8.md` 회차가 `logsumexp` 에 회차를 거의 다 쓸 뻔했던 전례가 있으므로,
별도 인터프리터에서 vendored shim 을 돌려 먼저 확인했습니다. 벽은 움직이지 않았습니다:

```
o.sum().backward()   -> NotImplementedError: aten.ones_like.default
                        counters: {shader_dispatches: 7, host_uploads: 0, host_downloads: 0}
o.mean().backward()  -> NotImplementedError: aten.mean.default
                        counters: {shader_dispatches: 6, host_uploads: 0, host_downloads: 0}
```

VULKAN9 §1 이 적은 `{dispatches 9, uploads 1, downloads 0}` 과 숫자가 다른 것은 **회귀가
아니라 브래킷의 차이**입니다 — 이 프로브는 세 입력을 카운터를 읽기 **전에** 올려두므로
`host_uploads` 가 0 이고, forward 만 세므로 디스패치가 적습니다. 숫자를 나란히 놓고 비교할
때 브래킷이 같은지 보지 않으면 이런 차이가 회귀로 읽힙니다.

같은 프로브로 **훈련 스텝에 필요한 나머지도 한꺼번에** 물었습니다. 그 답이 이 회차의 범위를
정했습니다:

```
mean.dim · empty_like · fill_.Scalar · add_.Tensor   -> 거절
div.Scalar · mul.Scalar · sub.Tensor · clone · expand -> 이미 돈다 (각 1 디스패치, 0 업로드)
```

마지막 줄이 중요합니다. **파라미터 갱신에 새 op 이 필요 없습니다** — `w - lr * w.grad` 는
`mul.Scalar` 와 `sub.Tensor` 이고 둘 다 이미 있습니다. 그래서 이 회차는 in-place `add_` 를
쓰지 않는 쪽을 택했고, 그것은 취향이 아니라 §7 의 이유 때문입니다.

---

## 2. `mean` 은 두 번째 커널이 아니라 divisor 하나다

`sum_dims_f32` 는 이미 임의의 축 집합을 한 번의 디스패치로 줄입니다(VULKAN9 §2). `mean` 은
그 보정합을 항 수로 나눈 것뿐이므로, 푸시 상수의 **쓰이지 않던 네 번째 워드**를 `do_mean`
플래그로 쓰고 셰이더 안에서 한 번 나눕니다.

**두 가지를 하지 않았고, 둘 다 이유가 있습니다.**

- `sum` 뒤에 `mul.Scalar` 로 역수를 곱하지 않았습니다. 역수는 upstream 이 하지 않는 반올림이고,
  `docs/devices/VULKAN5.md` §3.1 이 MoltenVK 의 fast-math 를 끈 것이 정확히 그 부류의 재작성
  때문입니다.
- `sum` 뒤에 `div.Scalar` 를 붙이지도 않았습니다. 그러면 디스패치가 2 회가 되는데, 그것도
  GPU 위이고 값도 맞을 것입니다 — **카운터만이 둘을 구분합니다.** 그래서 테스트는 `>= 1` 이
  아니라 **`== 1`** 을 단언합니다.

### 2.1 값 — 유도된 허용치

오라클은 upstream 자신의 커널이고, 허용치는 `docs/numerics/AGREE.md` §2 의 규칙대로 이 모집단의
upstream float32-대-float64 오차에서 **다시 유도**합니다.

```
mean.dim: 10 cases, tolerance max(p90 8.059e-08, 8 ulp 9.537e-07) = 9.537e-07;
  worst shim 4.865e-08 at mean[2, 3, 4] dim=[0, 2] keepdim=False (upstream's own 5.650e-08);
  17/39 elements differ in bits from upstream f32; worst element 0.59x upstream's own error

mean.default: 4 cases, tolerance max(p90 6.847e-08, 8 ulp 9.537e-07) = 9.537e-07;
  worst shim 6.533e-08 at mean_all[2, 3, 4] (upstream's own 6.847e-08);
  3/4 elements differ in bits from upstream f32; worst element 0.55x upstream's own error
```

스윕은 `sum` 과 같은 형상 집합입니다 — 선두 축, 인접하지 않은 두 축, 빈 리스트, `dim=None`,
512 길이 축, 랭크 1~4. 보정합 덕분에 두 경우 모두 **upstream 자신의 오차보다 작습니다.**

### 2.2 허용치에 이빨이 있는지 확인했다

`sum` 테스트가 절대 볼 수 없는 두 가지 틀린 답을 넣었습니다 — divisor 를 **살아남는 축의
길이**로 잘못 잡은 것과, 아예 나누지 않은 것:

| 넣은 값 | 유도 허용치 9.537e-07 에 대해 |
|---|---|
| 정답 (대조군) | 7.888e-08 — 통과 |
| **kept 축의 길이**를 divisor 로 | 1.667e-01 — 거절 (원소 단위 **1,398,102 배**) |
| **나누지 않음** | 6.000e+00 — 거절 (원소 단위 **50,331,653 배**) |

---

## 3. gradient 시드는 **셰이더**다 — `torch.ones(device="vulkan")` 은 아직 업로드인데도

`vulkan.rs` 의 `factory` 는 `torch.ones(2, 2, device="vulkan")` 을 호스트에서 만들어 올리고,
그 자리에 *"fill 셰이더는 배선에 대해 더 증명하는 것이 없다"* 고 적어 두었습니다. **그 말은
프로그램 가장자리의 생성자에 대해서는 참이고, 이 op 에 대해서는 거짓입니다.**

`torch/autograd/__init__.py` 의 `_make_grads` 는 모든 `loss.backward()` 의 시드를
`torch.ones_like(out, memory_format=preserve_format)` 으로 만듭니다. 업로드하는 `ones_like` 는
**모든 훈련 스텝 한가운데에 호스트 전송을 하나씩 넣습니다.** 그러면 `host_uploads == 0` 이
더 이상 "이 스텝이 조용히 CPU 에서 돌지 않았다" 는 증거가 아니게 됩니다 — 그리고 VULKAN9 §6 N2 가
기록한 대로, **값과 `.device` 로는 그 구분을 할 수 없습니다.** 그래서 시드는 커널입니다.

`fill_f32.comp` 은 채울 값을 푸시 상수에 **비트 패턴**으로 받습니다(`scalar_f32.comp` 과 같은
방식). 스테이징 버퍼도, 반올림도 없습니다.

`dtype=` 을 float32 아닌 것으로 주면 **이름을 대며 거절**합니다. int64 요청에 float32 를 조용히
돌려주는 것은 `check_dtype` 정책(`docs/devices/VULKAN6.md` §1)이 존재하는 이유 그 자체입니다.

`device=` 는 **읽습니다, 무시하지 않습니다.** `aten_dispatch` 는 *텐서 인자*를 보고 이 백엔드를
고르므로 `zeros_like(vk, device="cpu")` 가 여기까지 옵니다 — upstream 은 그 호출에 진짜 CPU
텐서로 답하고, `aten.rs` 의 `zeros_or_empty_like` 가 그 non-meta 절반을 "더 흥미로운 쪽" 이라고
적어 두었습니다. 인자를 무시하면 `VkTensor` 가 돌아가는데, 그것은 **형상도 맞고 값도 맞고 디스패치
수도 맞는 틀린 답**입니다. 일치 스윕도 카운터도 그것을 볼 수 없으므로 이름을 대며 거절합니다.
`test_ones_like_refuses_a_device_it_would_have_to_leave_to_honour` 가 그것입니다.

`memory_format=preserve_format` 은 **받습니다.** 거절하면 사용자가 만나는 벽이 "없는 op" 이
아니라 "인자" 를 가리키게 되고, 그것은 `aten.rs` 의 `zeros_or_empty_like` 가 이미 한 번 하고
기록해 둔 실수입니다.

---

## 4. 셋째 벽은 op 이 아니었다 — 결함이었다

두 op 을 넣고 나서 스텝은 **backward 를 끝까지 통과한 뒤**(디스패치 15 회) 이렇게 멈췄습니다:

```
File "torch_c_bootstrap.py", line 7212, in _dense_copy_of
NotImplementedError: torch._C shim: this tensor is on the vulkan device; its storage
is a VkBuffer, not something a CPU kernel can read.
```

`_dense_copy_of` 는 `AccumulateGrad` 가 `.grad` 에 처음 저장할 때 부르는 것으로,
`gradient.contiguous()` 를 한 뒤 **`data_ptr()` 을 비교해** 복사가 실제로 일어났는지 봅니다.
이 장치에는 그 질문에 줄 답이 없습니다:

- **stride 가 없습니다**(`docs/devices/VULKAN4.md` §6). `contiguous()` 는 같은 `VkBuffer` 를
  공유해서 돌려줍니다 — 언제나 "같은 저장소" 입니다.
- `data_ptr()` 은 호스트 주소가 없으므로 **거절**합니다.

즉 이것은 "가르치지 않은 op" 이 아니라 **stride 가 있는 장치를 전제한 코드가 stride 없는 장치를
만난 것**입니다. 고친 방식은 그 전제를 명시적으로 나누는 것입니다: vulkan 이면 질문을 하지 않고
무조건 `clone()` 합니다. 그 `clone` 은 이 장치의 실제 on-device 복사 한 번이고,
`data_ptr()` 을 vulkan 에 가르쳐 답을 지어내는 쪽은 **선택하지 않았습니다** — 없는 주소를
있다고 말하는 것이기 때문입니다.

§6 N4 가 이 한 줄을 되돌리면 무슨 일이 일어나는지 다시 확인합니다.

---

## 5. 훈련 한 스텝 — **된다**, 그리고 무엇이 그것을 보증하나

4->3 선형층, 평균제곱오차, SGD 한 스텝. 별도 인터프리터에서 vendored shim 을 돌리고, `x`·`y`·`w`
를 **평평한 리스트로 건네** 양쪽이 같은 숫자를 미분하게 합니다.

```
pred = x @ w ;  diff = pred - y ;  loss = (diff * diff).mean()
loss.backward()
with torch.no_grad():  updated = w - lr * w.grad
```

```
training step (loss, grad, updated weights): 3 cases,
  tolerance max(p90 0.000e+00, 8 ulp 9.537e-07) = 9.537e-07;
  worst shim 0.000e+00 at loss (upstream's own 0.000e+00);
  0/25 elements differ in bits from upstream f32
```

**0/25.** 손실·gradient·갱신된 가중치가 upstream CPU 의 같은 스텝과 비트까지 같습니다.

그리고 세 가지를 따로 단언합니다. 같은 주장이 아닙니다:

- **가중치가 실제로 움직였다** — 최악 좌표가 `1.335e-01` (lr=0.1). gradient 가 전부 0 이어도
  "upstream 과 일치" 는 만족됩니다. 그것이 증명하는 것은 아무것도 없습니다.
- **손실과 gradient 도 일치한다** — 두 오차가 상쇄되어 우연히 맞는 가중치에 도달한 스텝을
  통과로 읽지 않기 위해서입니다.
- **카운터.** `18 compute shaders, 0 uploads, 0 downloads`. 저장된 값을 내려받아 호스트에서
  전부 계산하고 답을 올려보내는 구현은 위 일치 테스트를 **정확히 통과**하고 모든 텐서의
  `.device` 가 `vulkan` 이라고 말합니다. 구분하는 것은 이 두 개의 0 뿐입니다.

파라미터 갱신을 in-place 로 쓰지 않았습니다. **이 장치의 어떤 op 도 in-place 로 쓰지
않습니다** — `detach`/`alias` 가 버퍼를 공유하므로(`vulkan.rs` `dispatch`), `w -= lr * w.grad`
는 미구현이 아니라 **불건전**합니다. §7.

### 5.1 한 스텝은 부호를 증명하지 않는다 — 그래서 세 스텝

부호가 뒤집힌 gradient 도 "가중치를 움직" 이고, 잘못 스케일된 리덕션도 그렇습니다. 그리고
둘 다 "upstream 과 같은 스텝" 과 비교하는 것만으로는 잡히지 않는 경우가 있습니다. 손실이
내려가는 것은 **비교가 아니라 산술 자체의 성질**입니다:

```
training loop on vulkan: losses 2.746104 -> 2.359784 -> 2.062743,
  54 compute shaders, 0 uploads, 0 downloads
training loop (losses, final weights): 2 cases, tolerance 9.537e-07;
  worst shim 8.682e-08 at losses (upstream's own 0.000e+00)
```

세 스텝 내내 호스트 경계를 **한 번도** 넘지 않습니다. 스텝 사이에 호스트에서 다시 심어지는
것이 없다는 뜻이고, 그것이 `0 uploads` 가 하는 말입니다.

---

## 6. 무력화 — 일부러 깨고 빨개지는지 봤다

구현 전 빌드에서 새 테스트 8 개 중 **FAIL 7** 이었습니다(적색 단계). 여덟 번째는
§2.2 의 이빨 검사로, upstream 자신의 값만 쓰므로 구현 전에도 통과합니다 — 그 테스트가
검사하는 것은 이 커널이 아니라 **허용치**이고, 그래서 적색 단계에서 초록인 것이 맞습니다.
§3 의 `device=` 거절은 구현 뒤에 실측으로 드러난 결함이고(무시되어 `VkTensor` 가
돌아오고 있었습니다), 프로브가 `cpu -> vulkan` 을 찍은 것이 그 적색 증거입니다.
그다음 보증을 하나씩 깼습니다.
**매번 stage 와 vendored 트리에 둘 다 복사했습니다** — VULKAN9 §6.1 이 기록한 함정입니다.

| # | 무력화 | 빨개진 테스트 |
|---|---|---|
| N1 | `fill_f32` 이 값 대신 0 을 쓴다 | `ones_like` 값 · **"가중치가 움직였다"** (`0.000e+00`) · **"손실이 내려간다"** (`2.746104` 세 번) |
| N2 | `mean` 이 나누지 않는다 | `mean.dim` · `mean.default` 일치 스윕 **그리고** 훈련 스텝 (`loss 2.300e+01`) 과 루프 |
| N3 | **호스트 쌍둥이** — 같은 fill 을 CPU 에서 만들어 업로드 | **값은 전부 맞았다.** 잡은 것은 카운터: `ran 0 compute shaders, expected exactly 1` · 스텝 `{dispatches 17, uploads 1}` · 루프 `3 up` |
| N4 | §4 의 vulkan 분기를 되돌린다 | 훈련 스텝 · 루프 · `o.sum().backward()` 가 전부 `data_ptr()` 에서 멈춘다 |

**N3 이 이 문서에서 가장 중요한 줄입니다.** 호스트 쌍둥이에서
`test_a_training_step_..._agrees_with_upstream` 은 **초록으로 남았습니다** — 값이 전부 맞으니까
당연합니다. 빨개진 것은 카운터를 보는 테스트들뿐이었습니다. 값과 `.device` 라벨로는 조용한
CPU 폴백을 탐지할 수 없고 카운터로는 할 수 있다는 것의, 이 회차의 증거입니다.

**N1 이 두 번째로 중요합니다.** 시드가 0 이 되면 gradient 가 전부 0 이 되는데, 그것은
**일치 테스트를 통과할 수도 있는 고장**입니다(upstream 도 같은 것을 계산한다면). 잡은 것은
"가중치가 움직였다" 와 "손실이 내려간다" 이고, 둘 다 비교가 아니라 성질을 보는 단언입니다.

---

## 7. 다음 벽 — 읽은 것이 아니라 실측한 것

```
o.sum().backward()                    -> ok
o.mean().backward()                   -> ok
훈련 스텝 세 번 (매번 새 leaf)         -> ok
같은 .grad 에 두 번째 누적              -> NotImplementedError: aten.add_.Tensor
is_causal=True (forward)              -> NotImplementedError: ... only for is_causal=False
```

> **`docs/devices/VULKAN11.md` 가 이 벽을 치웠습니다.** 아래 진단(불건전하다)은 옳았고,
> 그 뒤의 전제 — *"이 장치에 없는 공유 분석이 필요하다"* — 가 틀렸습니다. 분석은 이미
> 표현되어 있었습니다(`Arc<VkBuffer>` 의 strong count, 부분 겹침이 없으므로 정확함).
> 지금은 배타적 소유가 증명될 때만 in-place 로 쓰고, 공유일 때는 이름을 대며 거절합니다.
> **`torch.optim.SGD` 가 이 장치에서 돕니다.**

`aten.add_.Tensor` 는 `AccumulateGrad` 가 두 번째 backward 부터 쓰는 in-place 덧셈입니다.
**커널이 없어서가 아니라 이 장치에서 건전하지 않아서 쓰지 않았습니다** — `detach`/`alias` 가
`VkBuffer` 를 공유하므로, in-place 쓰기는 다른 텐서가 읽고 있는 별칭을 통해 쓰게 되고 그것을
**조용히** 합니다. 가르치려면 커널이 아니라 이 장치에 없는 **공유 분석**이 필요합니다.

실용적으로 이것이 막는 것은 `zero_grad(set_to_none=False)` 와, `.grad` 를 비우지 않고 두 번
backward 하는 경우입니다. `set_to_none=True`(현재 torch 의 기본값)와 스텝마다 새 leaf 를 만드는
방식은 §5.1 이 보인 대로 **이미 돕니다.**

`is_causal` 분기의 네 op (`ones.default`·`tril`·`eq.Scalar`·`masked_fill.Scalar`) 은
VULKAN9 §5.1 의 이유 그대로 여전히 구현하지 않았습니다 — forward 가 `is_causal` 을 이름을 대며
거절하므로 실행시킬 수 없고, 실행시킬 수 없는 커널은 **초록 표시가 붙은 검증되지 않은 코드**입니다.
`test_what_the_sdpa_backward_still_cannot_reach_is_unreachable_for_a_reason` 이 그 op 들의 부재와
**그 이유가 아직 참이라는 것**을 둘 다 고정하고, 이제 `add_.Tensor` 벽도 같은 방식으로 고정합니다.

---

## 8. 하지 않은 것

- **`aten.add_.Tensor` 와 모든 in-place op.** §7 — 커널의 문제가 아니라 별칭 분석의 문제입니다.
- **`aten.empty_like.default` · `aten.fill_.Scalar`.** §1 의 프로브가 거절한다고 보고했지만
  훈련 스텝 경로에 없습니다. `empty_like` 는 `like_fill` 에 인자 하나 차이지만, 이 회차에
  그것을 실행시키는 테스트가 없으므로 쓰지 않았습니다.
- **`torch.optim` 의 옵티마이저.** `foreach` SGD 는 `p.grad.add_` 와 `p.add_` 를 쓰므로
  §7 의 벽 뒤에 있습니다. 이 회차의 스텝은 손으로 쓴 SGD 입니다.
  > **정정 (`docs/devices/VULKAN11.md` §7).** 두 가지가 틀렸습니다. (1) 기본 경로
  > `torch.optim.SGD` 는 이제 **돕니다** — momentum 과 `zero_grad(set_to_none=False)` 포함,
  > 업로드 0 · 읽어오기 0. (2) `foreach` 는 §7 의 벽 뒤에 있지 **않았습니다**: vulkan op 에
  > 닿기도 전에 `torch._C._group_tensors_by_device_and_dtype` 에서 멈추며, 이것은 별칭 문제와
  > 무관하고 따라서 별칭 문제를 답해도 풀리지 않습니다.
- **`torch.nn.Module` · `Linear`.** 파라미터가 `torch.nn.Parameter` 이고 `.to("vulkan")` 이
  거기서 무엇을 하는지 재지 않았습니다. 스텝은 맨 텐서로 썼습니다.
- **성능.** 아무것도 재지 않았습니다. `fill_f32` 과 `mean` 은 출력 원소당 invocation 하나이므로
  `mean.default` 는 **단일 invocation** 입니다 — 정확하지만 병렬이 아닙니다.
  `docs/devices/VULKAN2.md` §4.4 의 이유 그대로, 이 회차는 값만 주장합니다.
- **실물 Adreno/Mali.** 이 회차의 모든 숫자도 Apple M1 + MoltenVK 하나입니다.

---

## 9. 부수적으로 드러난 것 — `DYLD_LIBRARY_PATH` 가 `rustc` 를 죽인다

Vulkan 로더를 가리키려고 셸에 `DYLD_LIBRARY_PATH` 를 export 한 채로 `cargo build` 를 돌리자
**`rustc` 가 `initialize_available_targets` 에서 SIGSEGV** 했습니다. 한 번이 아니라 매번이고,
`RUST_MIN_STACK` 을 올려도 `-j 1` 로 낮춰도 같았습니다.

처음에는 다른 회차가 함께 돌아 생긴 메모리 압력으로 오진했습니다(load 5.8). 그 가설이 틀린
것은 load 3.4 에서도 같았기 때문이고, 진짜 원인은 그 환경 변수가 **`rustc` 자신의 dyld 에도
상속된다**는 것입니다. 에뮬레이터의 `lib64/vulkan` 에 있는 라이브러리가 `rustc` 의 것보다
먼저 잡힙니다.

    빌드와 실행의 환경을 같은 셸 스크립트에 넣지 마십시오.
    `TORCHNATIVE_VULKAN_DYLD` 만으로 테스트는 로더를 찾습니다 — `DYLD_LIBRARY_PATH` 는
    이 저장소의 어떤 테스트에도 필요하지 않았습니다.

이것은 `docs/devices/VULKAN3.md` §6.1 의 뒤집힌 짝입니다. 거기서는 macOS 가 `DYLD_*` 를
**지워서** 테스트가 조용히 스킵됐고, 여기서는 **지우지 않아서** 컴파일러가 죽었습니다.
