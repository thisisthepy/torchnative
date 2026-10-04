# Vulkan: in place. 벽은 커널이 아니라 **별칭 분석**이었고, 그 분석은 근사가 아니라 정확하다

**결론부터.** `docs/devices/VULKAN10.md` §7 은 다음 벽을 `aten.add_.Tensor` 로 적고, 그것이
미구현이 아니라 **불건전(unsound)** 하다고 적었습니다:

> no op on this device writes in place, and `detach`/`alias` share the buffer,
> so `-=` would be **unsound**, not merely unimplemented.

그 진단은 옳았습니다. 틀린 것은 그 뒤에 붙은 전제, *"가르치려면 이 장치에 없는 공유 분석이
필요하다"*. 였습니다. **그 분석은 이미 거기 있었습니다.** `VkTensor` 는 `shape` 와
`Arc<VkBuffer>` 이고, 이 모듈이 쓰는 모든 디스크립터는 `offset 0, VK_WHOLE_SIZE` 입니다.
즉 부분 겹침이 존재할 수 없고, 두 텐서가 같은 바이트를 만지는 것은 **두 `Arc` 가 서로의
복제일 때 그리고 그때뿐**입니다. `Arc::strong_count == 1` 은 배타성의 휴리스틱이 아니라
**증명**입니다.

| 질문 | 답 |
|---|---|
| VULKAN10 의 벽이 아직 그 벽이었나 | **예.** 별도 프로세스 실측, 브래킷 명시 (§1) |
| 별칭 문제를 어떻게 처리했나 | (a) **진짜 소유권 분석.** (b) copy-on-write 와 (c) 영구 거절은 기각했고, 기각 이유가 둘 다 실측이다 (§2) |
| copy-on-write 가 왜 안 되나 | **틀린 답을 내놓기 때문.** 무력화 M4 에서 `torch.optim.SGD` 가 **가중치를 하나도 움직이지 않았다** (§6 M4) |
| 불건전이 실제로 일어나는가 | **예.** `y = x.detach()` 뒤의 `x.add_(z)` 를 만들었고, 분석을 걷어내면 **실제로 `y` 가 바뀐다** (§6 M1) |
| 공유 버퍼에 쓰면 어떻게 되나 | **디스패치 전에 이름을 대며 거절.** 셰이더 0 회, 별칭의 값이 그대로 (§3) |
| upstream 과 일치하는가 | **예.** 24 케이스, 유도 허용치 9.537e-07 에 대해 최악 **6.602e-08**, **696 원소 전부 비트 동일** (§4) |
| **`torch.optim.SGD` 를 쓸 수 있는가** | **예.** upstream 자신의 옵티마이저가 그대로 돈다. momentum 포함, 업로드 0, 읽어오기 0 (§5) |
| `zero_grad(set_to_none=False)` 는 | **예.** 그리고 그다음 backward 가 기존 `.grad` 에 누적한다 (§5) |
| 카운터에 이빨이 있나 | **예.** 호스트 쌍둥이는 **값이 전부 맞고 일치 테스트가 초록으로 남았다**; 잡은 것은 카운터뿐 (§6 M3) |
| `VULKAN_INDEX_MAX` · `check_dtype` 을 건드렸나 | **아니오.** 한 글자도 바뀌지 않았다 |

<!-- DOCWATCH: count vulkan_tests_ok ge 71 -->
<!-- DOCWATCH: symbol-in-file torchnative/rust/torch_c/src/vulkan.rs require_exclusive present -->
<!-- DOCWATCH: symbol-in-file torchnative/rust/torch_c/src/vulkan.rs vulkan_storage present -->
<!-- DOCWATCH: symbol-in-file torchnative/rust/torch_c/src/vulkan.rs INPLACE_ADD_F32_SPV present -->
<!-- DOCWATCH: symbol-in-file torchnative/rust/torch_c/src/vulkan.rs INPLACE_SCALAR_F32_SPV present -->
<!-- DOCWATCH: symbol-in-file torchnative/rust/torch_c/shaders/inplace_add_f32.comp uintBitsToFloat present -->
<!-- DOCWATCH: symbol-in-file torchnative/rust/torch_c/shaders/inplace_scalar_f32.comp uintBitsToFloat present -->
<!-- DOCWATCH: op-implemented aten.add_.Tensor -->
<!-- DOCWATCH: op-implemented aten.mul_.Scalar -->
<!-- DOCWATCH: op-implemented aten.fill_.Scalar -->
<!-- DOCWATCH: op-implemented aten.zero_.default -->
<!-- DOCWATCH: symbol-in-file tests/devices/vulkan/test_vulkan4.py test_the_device_can_say_which_tensors_share_a_buffer_and_that_is_the_analysis present -->
<!-- DOCWATCH: symbol-in-file tests/devices/vulkan/test_vulkan4.py test_the_storage_report_refuses_a_tensor_that_is_not_on_this_device present -->
<!-- DOCWATCH: symbol-in-file tests/devices/vulkan/test_vulkan4.py test_the_in_place_ops_agree_with_upstream_at_a_derived_tolerance present -->
<!-- DOCWATCH: symbol-in-file tests/devices/vulkan/test_vulkan4.py test_an_in_place_op_wrote_into_the_buffer_it_was_given_and_cost_one_shader present -->
<!-- DOCWATCH: symbol-in-file tests/devices/vulkan/test_vulkan4.py test_an_in_place_write_through_a_shared_buffer_refuses_and_leaves_the_alias_intact present -->
<!-- DOCWATCH: symbol-in-file tests/devices/vulkan/test_vulkan4.py test_in_place_refuses_the_operands_it_has_no_kernel_for_rather_than_reaching_for_the_cpu present -->
<!-- DOCWATCH: symbol-in-file tests/devices/vulkan/test_vulkan4.py test_torch_optim_sgd_drives_a_parameter_on_this_device_and_agrees_with_upstream present -->

---

## 1. 벽을 다시 쟀다: 그리고 브래킷을 적는다

`docs/devices/VULKAN8.md` 의 전례대로, 앞 문서의 표를 받지 않고 별도 인터프리터에서 vendored
shim 으로 먼저 확인했습니다. 벽은 움직이지 않았습니다.

```
같은 .grad 에 두 번째 누적   -> NotImplementedError: aten.add_.Tensor
                              counters: {shader_dispatches: 8, host_uploads: 0, host_downloads: 0}
w.grad.zero_()              -> NotImplementedError: aten.zero_.default
                              counters: {shader_dispatches: 0, host_uploads: 0, host_downloads: 0}
torch.optim.SGD(...).step() -> NotImplementedError: aten.add_.Tensor
torch.optim.SGD(foreach=True) -> NotImplementedError: torch._C._group_tensors_by_device_and_dtype
add_.Tensor · add_.Scalar · mul_.Scalar · zero_.default
  · fill_.Scalar · copy_.default · empty_like.default  -> 전부 거절
```

**브래킷을 명시합니다.** 위 `{8, 0, 0}` 은 **두 번째 `backward()` 호출만** 감싼 것입니다.
첫 backward 와 입력 업로드는 카운터를 읽기 전에 끝나 있습니다. VULKAN10 §1 의 `{7,0,0}` 은
forward 만, VULKAN9 §1 의 `{9,1,0}` 은 입력 업로드까지 포함한 것이었습니다. **세 숫자는 서로
비교 가능하지 않습니다.** 브래킷을 적지 않은 카운터는 다음 회차에서 회귀로 읽힙니다.

마지막 줄이 이 회차의 범위를 정했습니다. `foreach` SGD 는 **별칭 문제 뒤에 있지 않습니다**.
vulkan op 에 닿기도 전에 `torch._C` 의 다른 미구현 함수에서 멈춥니다. VULKAN10 §8 은 그것을
"§7 의 벽 뒤에 있다" 고 적었는데, **그 문장은 틀렸고 §7 에서 정정합니다.**

---

## 2. 결정: (a) 진짜 소유권 분석. 그리고 (b)·(c) 를 기각한 근거

세 선택지를 받았고, 취향이 아니라 근거로 골랐습니다.

### 2.1 (b) copy-on-write 는 건전한 것이 아니라 **틀린 답**이다

"공유되어 있으면 새 버퍼에 쓴다" 는 메모리 안전성 측면에서는 흠이 없습니다. 그러나 이것은
메모리 안전성 문제가 아니라 **의미론** 문제입니다. torch 의 in-place op 은 *모든 별칭이 변화를
본다* 는 것이 계약의 일부입니다:

```python
y = x.detach()
x.add_(z)          # upstream: y 도 바뀐다
```

copy-on-write 는 `y` 를 그대로 두고 **성공을 보고합니다.** 거절은 사용자가 보지만 틀린 답은
보지 못합니다. 그리고 이것은 논증이 아니라 실측입니다. §6 의 M4 에서 copy-on-write 구현은
`torch.optim.SGD` 에서 **가중치를 하나도 움직이지 않았습니다** (`SGD moved no weight: 0.000e+00`).
옵티마이저가 조용히 아무 일도 하지 않는 것, 그것이 (b) 의 실제 모습입니다.

### 2.2 (c) 영구 거절은 정직하지만 **목표를 만족하지 않는다**

"이름을 대며 영구히 거절하고 functional path 로 `torch.optim` 을 돌린다" 가 정당한 답이라는
것에 동의합니다. 문제는 그 functional path 가 **존재하지 않는다**는 것입니다.
`torch.optim.SGD` 는 upstream 자신의 파일이고, 그 갱신은 한 줄입니다:

```python
param.add_(d_p, alpha=-lr)          # torch/optim/sgd.py, _single_tensor_sgd
```

이것을 우회하려면 옵티마이저를 다시 쓰는 수밖에 없고, 그러면 답한 질문이
*"사람이 이 장치에서 `torch.optim.SGD` 를 쓸 수 있는가"* 가 아니라
*"우리가 SGD 를 다시 쓸 수 있는가"* 가 됩니다. VULKAN10 §5 의 손으로 쓴 스텝이 이미
후자를 답했고, 그것으로는 부족합니다. (functional 스펠링이 **여전히 한 스텝을 옮긴다**는 것은
`test_a_training_step_runs_end_to_end_...` 가 그대로 남아 계속 증명합니다. 분석을 철회해야 할
날이 오면 돌아갈 자리가 있다는 뜻입니다.)

### 2.3 (a) 를 고른 이유: 분석이 이미 있었다

이 장치에 별칭 분석이 없다는 VULKAN10 의 진술은 **분석을 *만들어야 한다*는 뜻으로 읽혔지만,
실제로는 그 정보가 이미 표현되어 있었습니다.**

```rust
pub struct VkTensor { pub buffer: Arc<VkBuffer>, pub shape: Vec<usize> }
```

* 이 모듈의 모든 디스크립터는 `offset 0, VK_WHOLE_SIZE` 입니다 → **부분 겹침이 없습니다.**
* 별칭을 만드는 경로는 `Arc` 복제뿐입니다 (`detach`/`alias`/`contiguous`/`view`/`reshape`).
* 따라서 두 텐서가 같은 바이트를 만진다 ⟺ 같은 `Arc` 를 공유한다.
* **`Arc::strong_count == 1` 은 배타적 소유의 증명**이고, 보수적으로 틀릴 수 있는 방향은
  "공유가 아닌데 공유라고 말하는" 쪽 하나뿐입니다. 즉 과잉 거절이지 조용한 오답이 아닙니다.

GIL 이 이 읽기를 결론적으로 만듭니다. 새 별칭을 만드는 모든 경로가 `dispatch` 를 지나고,
`dispatch` 는 GIL 아래에서 돕니다. 검사 시점부터 제출 시점까지 이 스레드가 GIL 을 쥐고 있으므로
그 사이에 별칭이 생길 수 없습니다.

`require_exclusive` 가 이 한 가지를 하고, `_C._vulkan_storage(t)` 가 그 판정을 파이썬에 노출합니다.
**노출이 필요한 이유는 테스트가 공유 상태를 *만들어* 보여야 하기 때문**입니다. 이 장치에는
그 외에 두 텐서가 같은 버퍼 위에 있다고 말해 주는 것이 없습니다.

### 2.4 `tensor_arg` 를 쓸 수 없었다: 분석을 자기 손으로 망가뜨린다

`crate::aten::tensor_arg` 는 `PyTensorBase` 를 **값으로** 추출합니다. 즉 래퍼를 복제하고,
그러면 `Arc` 도 복제됩니다. 그 헬퍼로 수신자를 집으면 **배타적으로 소유된 텐서가 전부
`shares == 2` 로 읽히고**, 그 2 는 디스패처 자신이 만든 유령입니다. in-place 경로는 `vk_arg`
로 `Bound<PyTensorBase>` 를 받아 `borrow()` 합니다. `_vulkan_storage` 도 같은 이유로
`borrow` 입니다.

---

## 3. 거절: 그리고 그것이 **디스패치 전**이라는 것

공유 버퍼에 in-place 쓰기가 오면 이름을 대며 거절합니다. 테스트가 요구하는 것은 세 가지이고,
셋은 같은 주장이 아닙니다:

1. **op 이름을 댄다**. 메시지 맨 앞이 `aten.add_.Tensor:` 입니다. 거절 메시지는 가르친 op 을
   전부 나열하므로 부분 문자열 검색은 목록 안의 `aten.add.Tensor` 를 벽으로 오인합니다.
2. **디스패치가 0 이다**. 쏘고 나서 raise 하는 구현은 메시지만 읽는 테스트를 통과합니다.
3. **별칭의 값이 그대로다**. 이것이 이 테스트를 "메시지 테스트" 가 아니라 "오염 테스트" 로
   만듭니다.

그리고 **반쪽이 하나 더 있습니다: 별칭이 사라지면 같은 쓰기가 통과해야 합니다.** 이것이 없으면
"공유일 때 거절한다" 는 "항상 거절한다" 로 퇴화하고 테스트는 그것을 알아채지 못합니다.

```
add_/mul_/zero_/fill_ refuse a detach/view/contiguous alias before dispatching,
and run once it is dropped
sharing is Arc-exact: detach/alias/contiguous/view/reshape share, clone does not;
a dropped alias restores exclusivity
```

별칭이 사라지면 배타성이 **복구된다**는 것도 따로 단언합니다. 올라가기만 하는 카운트는
어떤 버퍼든 한 번 view 를 뜨면 영원히 in-place 를 거절하게 만들고, 그것은 조용한 기능 상실입니다.

---

## 4. 값: 유도된 허용치, 그리고 696/696 비트 동일

오라클은 upstream 자신의 in-place 커널이고, 허용치는 `docs/numerics/AGREE.md` §2 의 규칙대로
이 모집단의 upstream float32-대-float64 오차에서 다시 유도합니다. 측정은 별도 서브프로세스입니다.

```
in-place add_/mul_/fill_/zero_: 24 cases,
  tolerance max(p90 5.843e-08, 8 ulp 9.537e-07) = 9.537e-07;
  worst shim 6.602e-08 at mul_ [3, 5, 7] (upstream's own 6.602e-08);
  0/696 elements differ in bits from upstream f32
```

**`alpha` 를 일부러 쓸었습니다** (`1.0`, `-0.1`, `2.5`). `torch.optim.SGD` 의 갱신이
`p.add_(d_p, alpha=-lr)` 이므로, `alpha` 를 무시하는 `add_` 는 평범한 `add_` 스윕을 통과한 뒤
**모든 가중치를 틀린 거리만큼 옮깁니다.** §6 M2 가 그것을 실제로 보입니다.

산술 선택 하나를 적어 둡니다. 셰이더는 `a[i] + s * b[i]` 로, **별도로 반올림되는 두 번의 IEEE
연산**이고 `fma(s, b[i], a[i])` 가 아닙니다. 이 저장소의 CPU 팔(`aten.rs` 의
`arith_inplace_tensor`)이 alpha 로 스케일한 뒤 더하므로, 융합 곱셈-덧셈은 **같은 op 의 CPU
스펠링과 다른 함수를 계산하게 됩니다.** 둘 다 유도 허용치 안이고, 자기 자신과 일치하는 쪽을
택했습니다.

---

## 5. 이 회차가 존재하는 이유: **`torch.optim.SGD` 가 돈다**

질문은 *"`add_` 에 커널이 있는가"* 가 아니라 *"사람이 이 장치에서 `torch.optim.SGD` 를 쓸 수
있는가"* 였습니다. upstream 자신의 옵티마이저를 손대지 않고 그대로 돌립니다.

```
SGD plain:                       losses [2.746104]
                                 {dispatches 17, uploads 0, downloads 0}
SGD three steps:                 losses [2.746104, 2.359784, 2.062743]
                                 {dispatches 51, uploads 0, downloads 0}
SGD momentum (0.9):              losses [2.746104, 2.359784, 1.799582]
                                 {dispatches 56, uploads 0, downloads 0}
SGD zero_grad(set_to_none=False): losses [2.746104, 2.359784, 2.062743]
                                 {dispatches 53, uploads 0, downloads 0}

torch.optim.SGD (losses, weights): 8 cases, tolerance 9.537e-07;
  worst shim 1.048e-07 at SGD plain: losses (upstream's own 1.048e-07);
  4/58 elements differ in bits from upstream f32
```

네 케이스 모두 **업로드 0, 읽어오기 0**. 손실 텐서는 카운터 브래킷 **안에서 읽지 않고 보관**한
뒤 나중에 내려받습니다. 스텝마다 `.cpu()` 를 하면 `host_downloads` 가 이 프로브 자신의 보고를
세게 되고 카운터가 증거이기를 그만둡니다.

세 가지를 따로 단언합니다. 같은 주장이 아닙니다:

- **가중치가 움직였다**. Gradient 가 전부 0 이어도 "upstream 과 일치" 는 만족됩니다. M4 에서
  이 단언 하나가 copy-on-write 를 잡았습니다.
- **손실이 내려간다**. 비교가 아니라 산술의 성질입니다. M2 에서 `2.75 → 9.48 → 48.74` 로
  이 단언이 alpha 결손을 잡았습니다.
- **카운터**: 값과 `.device` 로는 조용한 CPU 폴백을 구분할 수 없습니다 (§6 M3).

`momentum=0.9` 는 다른 손실 궤적(`1.799582`)을 냅니다. 같았다면 momentum 인자가 무시되고 있다는
뜻이고, 나머지 단언은 전부 통과했을 것입니다. momentum 경로는
`buf.mul_(momentum).add_(d_p)` 이므로 **`mul_.Scalar` 를 실행시키는 유일한 테스트**이기도 합니다.

`zero_grad(set_to_none=False)` 는 `p.grad.zero_()` 를 부르고, 그다음 backward 가 그 `.grad` 에
`add_` 로 누적합니다. VULKAN10 §7 이 막혀 있다고 적은 바로 그 경로입니다.

---

## 6. 무력화: 일부러 깨고 빨개지는지 봤다

구현 전 빌드에서 새 테스트 7 개 중 **FAIL 5** 였습니다(적색 단계). 초록이던 둘은
`_vulkan_storage` 의 두 테스트인데, **그 함수는 설계 결정을 내리기 위한 측정 도구로 먼저
만들었기 때문**입니다. 분석이 성립하는지 재 보지 않고 커널을 쓸 수는 없었습니다. 그래서
그 둘에는 M5 를 따로 붙였습니다.

**다섯 번 모두 `scripts/vendor/install_shim.sh` 로 빌드했습니다.** 그 스크립트가 아티팩트를
**vendored 트리에 직접** 설치하고, 같은 파일을 stage 로 복사했습니다. 즉 서브프로세스 프로브가
읽는 `.so` 와 인프로세스 테스트가 읽는 `.so` 가 매번 같은 변이체였습니다. `docs/devices/VULKAN9.md`
§6.1 이 기록한 함정의 반대 오진(변이가 무해해 보이는 것)을 막는 것이 이것입니다.

| # | 무력화 | 빨개진 테스트 |
|---|---|---|
| M1 | `require_exclusive` 를 `shares >= 1` 로, **분석을 걷어낸다** | **1 개.** 공유 버퍼 거절 테스트가 `aten.add_.Tensor wrote through a buffer shared with a detach alias` 로 실패, 즉 **별칭이 실제로 오염된다** |
| M2 | `add_` 가 `alpha` 를 무시한다 | 일치 스윕 (`9.416e-01` vs 허용치 `9.537e-07`) **그리고** SGD 의 손실 궤적 `2.75 → 9.48 → 48.74` |
| M3 | **호스트 쌍둥이**, 두 피연산자를 내려받아 CPU 에서 더하고 같은 버퍼에 올린다 | **값은 전부 맞았고 일치 테스트는 초록이었다.** 잡은 것은 카운터 셋: `{dispatches 0, uploads 1, downloads 2}` · `ran 0 compute shaders, expected 1` · `SGD plain crossed the host boundary` |
| M4 | **copy-on-write**, 새 버퍼에 답을 쓰고 새 텐서를 돌려준다 | 5 개. 결정적인 것은 **`SGD moved no weight: 0.000e+00`**, 옵티마이저가 조용히 아무 일도 하지 않는다 |
| M5 | `_vulkan_storage` 가 언제나 `shares=1` 이라고 답한다 | **1 개.** 공유 보고 테스트가 `detach` 에서 실패 |

**M1 이 이 문서에서 가장 중요합니다.** 지시받은 것은 *"두 텐서가 버퍼를 공유하고 in-place 쓰기가
다른 쪽을 오염시키는 경우를 만들어라"* 였고, M1 이 정확히 그 상태입니다: 분석만 걷어내면
같은 코드가 같은 입력에 대해 **별칭을 조용히 바꿉니다.** 불건전은 논증이 아니라 재현입니다.

**M3 이 두 번째입니다.** 호스트 쌍둥이에서 일치 테스트는 초록으로 남았습니다. 값이 전부
맞으니 당연합니다. `.device` 도 전부 `vulkan` 이라고 말합니다. 구분한 것은 두 개의 0 뿐입니다.

**M4 가 §2.1 의 실측 근거입니다.** copy-on-write 가 "건전하지만 in-place 는 아닌" 타협이
아니라 **틀린 답**이라는 것을, 옵티마이저가 아무것도 안 하는 것으로 보입니다.

---

## 7. 아직 거절하는 것: 그리고 그것들이 **같은 종류가 아니라는 것**

VULKAN10 §8 은 `foreach` SGD 가 *"§7 의 벽 뒤에 있다"* 고 적었습니다. **그 문장은 틀렸습니다.**
실측하면 vulkan op 에 닿기도 전에 멈춥니다:

```
torch.optim.SGD(foreach=True) -> NotImplementedError:
    not implemented in torch._C shim: torch._C._group_tensors_by_device_and_dtype
```

이것은 별칭 문제가 아니었고, 따라서 별칭 문제를 답한다고 해서 풀리지 않습니다. 테스트가 그
**멈추는 지점의 이름**까지 고정하므로, 나중에 다른 곳에서 멈추게 되면 그 주장을 다시 재야 합니다.

나머지 거절은 **쓰지 않은 커널**이고, 소유권 분석은 그대로 쓸 수 있습니다:

```
copy_ · sub_ · div_ · empty_like  -> NotImplementedError (이름을 대며)
```

쓰지 않은 이유는 VULKAN9 §5.1 과 같습니다. **실행시킬 테스트가 없는 커널은 초록 표시가 붙은
검증되지 않은 코드**입니다. `is_causal` 네 op (`ones.default`·`tril`·`eq.Scalar`·
`masked_fill.Scalar`) 도 forward 가 `is_causal` 을 거절하는 한 그대로입니다.

in-place 를 **네 개만** 가르친 것도 같은 규율입니다. 네 개는 `torch.optim.SGD` 와
`zero_grad(set_to_none=False)` 가 실제로 부르는 것 전부이고, 그 이상은 아닙니다.

`add_` 는 **브로드캐스트하지 않습니다.** out-of-place `aten.add.Tensor` 는 브로드캐스트하지만
(`broadcast_binary`), 그 모양을 고정된 목적지에 쓰는 것은 다른 커널이고 측정된 것 중 필요로 한
것이 없습니다. CPU 로 흘러내리는 대신 이름을 대며 거절합니다. 그러지 않으면 위의 모든 카운터가
읽을 수 없게 됩니다.

---

## 8. 하지 않은 것

- **`aten.copy_.default` · `sub_` · `div_` · `empty_like`.** §7. 분석은 준비돼 있고 커널이 없습니다.
- **`foreach` 경로.** `torch._C._group_tensors_by_device_and_dtype` 는 이 회차의 범위가 아닙니다.
- **`Adam` 등 다른 옵티마이저.** `torch.optim.Adam` 은 `addcmul_`·`addcdiv_`·`sqrt` 를 더 부릅니다.
  재지 않았고, 주장하지 않습니다.
- **`torch.nn.Module` · `Linear`.** 파라미터는 `torch.nn.Parameter` 로 만들었지만 모듈을 통째로
  `.to("vulkan")` 한 학습은 재지 않았습니다.
- **성능.** 아무것도 재지 않았습니다. in-place 가 할당 하나를 없애지만, 그것을 재지 않았으므로
  숫자로 말하지 않습니다. `docs/devices/VULKAN2.md` §4.4 의 이유 그대로 값만 주장합니다.
- **실물 Adreno/Mali.** 이 회차의 모든 숫자도 Apple M1 + MoltenVK 하나입니다.
- **다중 스레드.** §2.3 의 GIL 논증은 이 저장소가 GIL 아래에서 돈다는 것에 의존합니다.
  free-threaded 빌드에서 이 분석이 성립하는지는 재지 않았습니다.
