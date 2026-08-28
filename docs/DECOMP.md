# 분해 패스 — 캡처한 ATen 을 Core ATen 으로 낮춘다

## 0. 이 문서가 답하는 것

- **분해 패스가 섰는가.** 섰습니다. `torchnative.export.decompose` 이고, 규칙은 한 줄도
  여기서 쓰지 않았습니다 — 상류의 `torch/_decomp` 를 *실행*합니다 (§1).
- **무엇이 덮이고 무엇이 안 덮이는가.** §4 가 실측 표이고, **덮이지 않는 것 전부에 이름과
  원인이 붙어 있습니다.** 원인은 네 가지로 갈리고 각각 고치는 자리가 다릅니다.
- **분해 후 값이 달라지는가.** §5. 이 트레이스에서는 **비트 단위로 안 달라졌습니다.**
  달라질 수 있다는 것을 기대했고, 안 달라진 것을 측정으로 적습니다.
- **ExecuTorch Edge 까지 얼마나 남았는가.** §7. 분해는 필요조건이고 충분조건이 아닙니다 —
  남은 것 중 가장 큰 것은 **functionalization** 이고 이 패스가 하는 일이 아닙니다.
- **이 작업이 찾은 결함.** §6. 패스가 "낮추지 못했다" 가 아니라 **"낮추면 답이 달라진다"**
  로 거절한 것이 둘 있고, 둘 다 이 저장소의 진짜 버그입니다.

구현은 `torchnative/src/main/torchnative/export/decompose.py`, 트레이스가 상수 텐서를 내주도록
`rust/torch_c/src/capture.rs` 에 게터 하나(`constant_values`)가 늘었고, 테스트는
`rust/torch_c/pytests/test_shim.py` 의 decompose 절 6 개와 capture 절에 1 개입니다.

---

## 1. 왜 지금, 그리고 여기에 무엇을 쓰지 *않았는가*

docs/CAPTURE.md §5 가 이 문제를 이렇게 적었습니다 — **기록되는 방언이 ATen 이지 Core ATen 이
아닙니다.** ExecuTorch 의 Edge dialect 는 Core ATen 위에 정의되어 있고, 같은 문서의 가장 작은
예제(`nn.Sequential(Linear, ReLU, Linear)`)가 이미 `aten.t.default` 를 기록합니다. 이론적 우려가
아니라 첫 모델에서 일어나는 일입니다.

그래서 캡처와 델리게이트 사이에 패스가 하나 서야 하고, 이것이 그 패스입니다.

**이 파일에 없는 것 두 가지가 이 패스의 설계입니다.**

| 없는 것 | 어디서 오는가 |
|---|---|
| 분해 규칙 | `torch/_decomp/decompositions.py` · `torch/_refs` — 벤더 트리에 이미 있습니다 |
| Core ATen op 목록 | `torchgen/packaged/ATen/native/native_functions.yaml` 의 `tags: core` — 같은 트리 |

분해표를 손으로 옮겨 적는 것은 이 프로젝트의 전제("재구현이 아니라 진짜 파이썬 생태계를 돌린다")를
정면으로 어깁니다. 193 개의 op 이름을 옮겨 적는 것도 한 치수 작은 같은 실수입니다. 둘 다 상류가
고치면 우리 쪽이 조용히 낡습니다.

### 한 노드를 낮추는 방법 — 기록기를 트레이서로 쓴다

분해 규칙은 텐서를 받는 파이썬 함수입니다. 그러므로 규칙을 적용한다는 것은 그것을 **실행**한다는
뜻이고, 실행이 어떤 op 을 내는지 알아내는 도구는 이미 있습니다 — 기록기입니다.

```
1. 노드의 텐서 피연산자마다 기록된 shape/dtype/device 로 플레이스홀더를 만든다
2. _capture_begin(플레이스홀더) → 상류 규칙 호출 → _capture_end
3. 나온 서브 트레이스를 부모의 그 자리에 끼워 넣는다
4. 고정점까지 반복 (분해 결과가 또 non-core 일 수 있다)
```

**캡처 층의 거절이 이 패스를 공짜로 지켜 줍니다** (docs/CAPTURE.md §4). 텐서 *값*으로 분기하는
분해는 `aten._local_scalar_dense.default` 에 닿아 자기 서브 기록을 오염시키고, 패스는 그것을
이름을 대고 거절합니다 — 플레이스홀더 값에 대해서만 맞는 그래프를 내놓는 대신에. *형태*나
*dtype* 으로 분기하는 것은 가드가 둘 다 고정하므로 건전합니다.

플레이스홀더를 `zeros` 로 채우는 것은 그래서입니다. 값에 의존하는 분해는 위 경로로 잡히므로,
남은 선택은 "재현 가능한가" 뿐이고 `empty` 의 쓰레기 값은 재현 가능하지 않습니다.

### 거절

고정점 뒤에 남은 non-core op 은 **이름을 대고 거절**합니다. 조용히 통과시키면 ExecuTorch 가
나중에 프로그램을 거부하는데 **어느 op 때문인지, 어느 패스의 책임인지 아무것도 가리키지
않습니다.** 거절은 네 가지 벽 중 어디에 부딪혔는지까지 말합니다 (§4 의 분류가 그 네 가지입니다).

---

## 2. Core ATen 을 무엇으로 판정하는가 — `torch.Tag.core` 는 이 빌드에서 못 씁니다

docs/CAPTURE.md §5 의 108/70/38 은 **상류 torch 위에서** 잰 것입니다. 벤더 트리 위에서 같은 것을
재면 이렇게 나옵니다:

```
_aten_all_implemented()                     120
그중 torch.Tag.core 가 붙은 것                0     ← 전부
```

원인은 `bootstrap.py` 의 `_get_operation_overload` 가 태그 목록으로 항상 `[]` 를 돌려주는 것입니다.
즉 이 빌드에서 태그로 판정하면 **"아무것도 core 가 아니다"** 가 나오고, 패스는 프로그램 전체를
거절합니다.

그래서 판정은 벤더 트리의 `native_functions.yaml` 을 읽습니다. **그 파일이 상류가
`torch.Tag.core` 를 생성하는 원본입니다.**

| | 개수 |
|---|---|
| yaml `tags: core` | **193** |
| 상류 `torch.Tag.core` (설치된 torch 2.13.0 실측) | 189 |
| 차이 | 4 — `adaptive_avg_pool1d.default` · `avg_pool1d.default` · `resize_.default` · `sym_is_contiguous.default` |

넷은 파일에 core 로 적혀 있는데 상류의 `OpOverload.tags` 로는 나타나지 않습니다. **넓은 쪽을
택했습니다** — 이 차이는 패스가 노드를 *받아들이게* 만들 수는 있어도 거절하게 만들 수는 없고,
넷 다 상류 자신의 파일이 core 라고 부르는 것입니다.

읽는 방법은 YAML 파서가 아니라 줄 스캐너입니다. `pyyaml` 이 이 배포판의 선언된 의존성이 아니기
때문입니다(pyproject.toml 은 상류 torch 의 순수 파이썬 의존성만 싣고, 거기에 pyyaml 은 없습니다) —
여기서 `import yaml` 하면 올바르게 설치된 휠에서 이 모듈이 임포트 실패합니다. 대신
`test_decompose_reads_core_aten_out_of_the_vendored_tree` 가 **스캔 결과를 진짜 YAML 파스와
diff** 합니다. 상류가 형식을 바꾸면 조용히 짧아진 목록이 아니라 빨간 테스트가 됩니다.

---

## 3. 어느 분해표에 닿을 수 있는가 — 세 개의 숫자가 다르다

| | 항목 | 크기 |
|---|---|---:|
| A | 상류 `core_aten_decompositions()` | **940** |
| B | 상류 `_core_aten_decompositions_post_autograd()` | **435** |
| C | **벤더 트리에서 실제로 얻은 것** (= B 를 여기서 부른 것) | **224** |

패스는 A 를 **매번 먼저 시도**하고, 실패하면 C 로 내려가면서 `decomposition_table_source()` 에
그 이유를 남깁니다. 하드코딩하지 않은 것은 의도입니다 — shim 이 아래의 빠진 함수를 구현하는 날
이 코드는 수정 없이 더 큰 표를 집습니다. 그리고 **조용한 폴백은 조용한 커버리지 손실**이므로
테스트가 현재 값을 못박고 있습니다.

### A → B 를 막는 것

```
NotImplementedError: not implemented in torch._C shim:
    torch._C._dispatch_get_registrations_for_dispatch_key
```

`core_aten_decompositions()` 는 `CustomDecompTable` 이고, 그 생성자가 C++ 디스패처에 등록된
`CompositeImplicitAutograd` op 을 전부 열거합니다. 이 `_C` 는 파이썬 셰임이라 열거할 C++ 디스패처가
없습니다. **이것이 표의 CIA 절반(940 − 435 = 505 개 규칙)을 통째로 못 쓰게 만듭니다.**

### B → C 를 막는 것 — 이것이 더 근본적입니다

상류와 벤더 트리에서 `torch._decomp.global_decomposition_table["post_autograd"]` 를 세면:

| | 레지스트리 항목 | 그중 `.default` |
|---|---:|---:|
| 상류 | 1097 | 456 |
| 벤더 트리 | **592** | **525** |

원인은 `bootstrap.py` 의 `_jit_get_operation` 이 **모든 패킷의 오버로드 목록으로 `["default"]` 를
돌려주는 것**입니다. 그래서 `torch.ops.aten.transpose.op_overloads()` 가 여기서는
`[aten.transpose.default]` 를 내는데, 상류에서는 `[aten.transpose.int]` 입니다 — `transpose` 에는
`default` 오버로드가 **존재하지 않습니다.**

`@register_decomposition(aten.transpose)` 처럼 **패킷 단위**로 등록된 규칙은 `_add_op_to_registry`
가 `op_overloads()` 로 펼치므로, 전부 실재하지 않는 `.default` 키에 내려앉습니다. 규칙은 트리
안에 있는데 **아무도 찾을 수 없는 이름 아래 있습니다.**

§4 에서 "규칙 없음(상류에는 있음)" 으로 분류된 7 개 중 **5 개가 정확히 이 현상**입니다:

```
aten.transpose.int        규칙이 aten.transpose.default 에 있음
aten.rsub.Scalar          규칙이 aten.rsub.default 에 있음
aten.masked_fill.Scalar   규칙이 aten.masked_fill.default 에 있음
aten.masked_fill.Tensor   같음
aten.isin.Tensor_Tensor   규칙이 aten.isin.default 에 있음
```

**`.default` 키를 폴백으로 읽는 우회는 하지 않았습니다.** 패킷 등록과 진짜 `.default` 오버로드
등록이 구분되지 않으므로, 그 우회는 **어떤 오버로드에 다른 오버로드의 규칙을 조용히 적용**할 수
있습니다. 결과 메타 검사(§6)가 대부분 잡겠지만 "대부분" 은 이 자리에서 쓸 수 있는 단어가
아닙니다. 고쳐야 할 곳은 `_jit_get_operation` 이고, 그것은 이번 작업의 범위가 아니라 **이름을 댄
항목**입니다.

---

## 4. 실측 — 무엇이 덮이고 무엇이 안 덮이는가

### 모집단

| | 개수 |
|---|---:|
| `_aten_all_implemented()` | **120** |
| └ Core ATen | 73 |
| └ Core ATen 밖 | **47** |
| &nbsp;&nbsp;&nbsp;└ 캡처가 애초에 거절하는 것 (변이 12 · 난수 3) | 15 |
| &nbsp;&nbsp;&nbsp;└ **트레이스에 나타날 수 있는 non-core op** | **32** |

변이·난수 15 개를 뺀 것은 편의가 아닙니다 — docs/CAPTURE.md §4 가 그것들을 이름을 대고
거절하므로 **어떤 트레이스에도 들어올 수 없습니다.** 분해 대상이 아니라 캡처 대상이 아닙니다.

### 그 32 개에 패스를 돌린 결과

| 판정 | 개수 |
|---|---:|
| **Core ATen 으로 낮춰짐** | **4** |
| 규칙이 이 빌드에서 실행되지 않음 | 13 |
| 닿을 수 있는 표에 규칙 없음 — **상류 전체 표에는 있음** | 7 |
| 규칙이 상류에도 없음 | 5 |
| 규칙이 기록과 다른 결과를 냄 (→ 거절) | 2 |
| 캡처가 구간 출력으로 거절 | 1 |

각 항목의 이름을 전부 답니다.

#### 낮춰짐 (4)

```
aten._unsafe_view.default   →  aten.view.default
aten.detach.default         →  aten.alias.default
aten.split.Tensor           →  aten.split_with_sizes.default
aten.stack.default          →  aten.cat.default + aten.view.default
```

#### 규칙은 있는데 이 빌드에서 실행되지 않음 (13) — 벽이 shim 쪽에 있습니다

| op | 무엇에 막혔나 |
|---|---|
| `aten.t.default` | `torch.transpose` — `overloads.json` 에 항목 없음 |
| `aten.silu.default` | `torch.sigmoid` — 같음 |
| `aten._safe_softmax.default` | `torch.softmax` — 같음 |
| `aten.zeros_like.default` | `torch.full_like` — 같음 |
| `aten.ones.default` · `aten.zeros.default` | `aten.full.default` 의 `layout` 인자 미구현 |
| `aten.arange.default` · `aten.arange.start` | `aten.arange.start_step` 의 `layout` 인자 미구현 |
| `aten.empty_like.default` · `aten.new_ones.default` | `TensorBase.layout` 미구현 |
| `aten.floor_divide.default` | `aten.div.Tensor_mode` 커널 없음 |
| `aten._scaled_dot_product_flash_attention_for_cpu.default` | `aten._scaled_dot_product_attention_math` 커널 없음 |
| `aten.softplus.default` | `torch._C._dynamo.eval_frame.set_eval_frame` 미구현 |

넷으로 갈립니다 — **`torch.<fn>` 오버로드 표의 구멍 4 개**, **`layout` 인자/속성 5 개**,
**커널 2 개**, **dynamo 1 개**. `layout` 다섯 개는 하나의 원인이므로 가장 값싼 다음 걸음입니다.

#### 닿을 수 있는 표에 규칙이 없음 — 상류 전체 표에는 있음 (7)

```
aten.transpose.int        §3 의 패킷 붕괴
aten.rsub.Scalar          §3 의 패킷 붕괴
aten.masked_fill.Scalar   §3 의 패킷 붕괴
aten.masked_fill.Tensor   §3 의 패킷 붕괴
aten.isin.Tensor_Tensor   §3 의 패킷 붕괴
aten.matmul.default       CIA 절반에만 있음 (§3 의 A→B). py_kernel 은 실재함
aten.max.other            상류 규칙이 "C++ CIA 커널을 돌려라" 자체 — 파이썬 규칙이 없음
```

마지막 줄이 별개의 벽입니다. 상류의 그 항목은
`functools.partial(_special_op_to_decompose_cia)` 이고, 그것이 하는 일은 **C++ 의
CompositeImplicitAutograd 커널을 호출하는 것**입니다. 파이썬 트리만 벤더링한 이 빌드에는 그
커널이 없으므로, §3 의 A→B 를 고쳐도 이 항목은 열리지 않습니다.

#### 규칙이 상류에도 없음 (5)

```
aten.contiguous.default
aten.histc.default
aten.lift_fresh.default
aten.max.default
aten.reshape.default
```

`lift_fresh.default` 와 `max.default` 는 docs/CORE_ATEN.md §0 이 이미 "분해로도 안 풀림" 으로
재 둔 둘입니다 — **그 측정이 재확인됐고, 목록이 셋 늘었습니다.** 이 다섯은 §7 의 "네 번째 층"
(어느 쪽에도 안 걸리는 잔여) 이고, 직접 구현하거나 상류에 분해를 넣는 것 외에 길이 없습니다.

#### 캡처가 거절 (1)

`aten.is_floating_point.default` 는 텐서가 아닌 결과를 냅니다. 구간의 *출력*으로는 캡처가 이름을
대고 거절하고, 구간 *안의* 노드로는 기록될 수 있습니다(docs/CAPTURE.md §4 의 메타데이터 허용목록).
후자로 들어오면 이 패스는 "non-core 인데 규칙이 없다" 로 거절합니다.

---

## 5. 판정 — 분해 전후로 재생 결과가 일치하는가

**일치합니다. 비트 단위로.**

측정한 프로그램(테스트 `test_decompose_lowers_a_trace_to_core_aten` ·
`test_decomposed_replay_matches_eager_bit_for_bit` 가 도는 것과 같은 것):

```python
def program(x, w):
    h = aten.stack.default([x, x * 2.0])   # ← Core ATen 밖
    h = h.view(4, 4)
    y = torch.relu(torch.mm(h, w))
    lo, hi = aten.split.Tensor(y, 2)       # ← Core ATen 밖
    return lo + hi
```

```
기록      7 노드   mul.Scalar  stack.default  view  mm  relu  split.Tensor  add.Tensor
분해 후   8 노드   mul.Scalar  cat.default  view  view  mm  relu  split_with_sizes  add.Tensor

Core ATen 밖:  분해 전 [split.Tensor, stack.default]  →  분해 후 []
```

노드가 **늘어난 것**이 요점입니다. 분해는 op 하나를 여럿으로 바꾸므로, 개수가 그대로인 패스는
아무 일도 안 한 것입니다.

| 입력 | 분해된 트레이스 재생 | 기록된 트레이스 재생 | eager |
|---|---|---|---|
| `ones(2,4)` (기록에 쓰인 것) | 같음 | 같음 | 기준 |
| `× 0.5` · `× -2.0` · `× 7.25` (처음 보는 것) | **세 번 모두 완전 일치** | 완전 일치 | 기준 |

**비트 동일을 기대하지 않았던 것을 적어 둡니다.** 분해는 같은 수학이지만 연산 순서가 달라지므로
부동소수점의 마지막 비트가 달라질 자격이 있습니다. 이 트레이스에서는 달라지지 않았고, 이유도
구조적입니다 — 여기서 일어난 세 치환(`stack`→`cat`+`view`, `split`→`split_with_sizes`,
`_unsafe_view`→`view`)은 전부 **데이터 이동이지 산술이 아닙니다.** 즉 이 결과는 "분해는 비트
동일하다" 의 증거가 아니라 **"이 분해들은 산술을 건드리지 않는다"** 의 증거입니다. 산술을
재배열하는 규칙(`silu`, `_safe_softmax`, `baddbmm`)이 열리는 날 이 표는 다시 재야 하고, 그때
차이가 나오면 **허용치를 넓히는 것이 아니라 크기를 재서 여기에 적어야 합니다**
(docs/DEVICE.md §5 — 허용치보다 이름 붙인 예외).

가드는 낮추기를 통과해도 가드로 남습니다: 분해된 트레이스에 `ones(3,4)` 를 주면 `[2, 4]` 를
이름에 담아 거절합니다. **낮추기는 트레이스를 일반화하지 않습니다.**

---

## 6. 이 패스가 찾은 것 — 미구현이 아니라 결함 둘

패스는 분해 결과의 shape·dtype·device 를 **기록된 것과 대조**합니다. 분해는 같은 함수여야
하므로, 다르면 무언가가 틀린 것이고 그대로 내보내면 아무도 다시 보지 않는 그래프가 아래로
내려갑니다. 그 검사가 두 개를 잡았고, **둘 다 이 저장소의 진짜 버그입니다.**

### 6.1 `aten.sum.dim_IntList` 의 빈 `dim` 목록 — **고쳐짐**

> **2026-08-28 수정.** `rust/torch_c/src/aten.rs::sum_or_mean` 이 빈 `dim` 목록을 모든 축으로
> 확장하도록 고쳐졌습니다 (`dims.map(|d| if d.is_empty() { (0..rank).collect() } else { d })`).
> 아래는 그 전까지의 상태를 남긴 기록입니다.

상류 규칙은 `sum(x)` 를 `sum(x, dim=[], dtype=None)` 으로 다시 씁니다. **빈 `dim` 목록은 "모든
차원을 축약하라" 는 뜻입니다.**

```
                        shape        값
상류  sum(ones(3,4), [])   []        12.0
여기  sum(ones(3,4), [])   [3, 4]    입력 그대로   ← 고치기 전
```

고치기 전 커널은 입력을 그대로 돌려줬습니다. `aten.sum.default` 는 골든 하네스가 검사했지만
(당시 2383/2383 통과), **`aten.sum.dim_IntList` 에 빈 리스트를 주는 케이스는 아무도 만들지
않았습니다.** 분해 패스가 그 인자를 만들어내는 첫 호출자였습니다. `dim=[0,1]`(명시적 전체
축약)·`keepdim=True` 조합·중복 `dim`(양쪽 다 거절, candle 이 이미 잡고 있었음)도 상류 2.13.0
과 대조해 `tools/golden/cases.py`의 `sum_dim_cases`/`mean_dim_cases`에 케이스로 남겼습니다 —
같은 함정이 `mean.dim`에도 있었는지 확인하는 것이 목적이었고, 있었습니다(같은 커널을 공유).

이 결함을 잡았던 테스트 `test_decompose_refuses_a_rule_that_disagrees_with_the_recording` 는
`test_decompose_lowers_sum_default_now_that_the_kernel_agrees` 로 이름이 바뀌었습니다 — 커널이
고쳐졌으므로 이제 거절이 아니라 **낮추기가 성공하는 것**을 못박습니다. "규칙이 recording과
불일치하면 거절한다"는 세 번째 벽 자체는 `test_decompose_refuses_by_name_what_it_cannot_lower`
가 `aten.baddbmm.default`(§6.2, 아직 안 고쳐짐)로 커버리지를 이어받았습니다.

### 6.2 `aten.baddbmm.default` 분해의 dtype 승격

```
                       입력           분해 결과 dtype
상류  baddbmm(f32, f32, f32)          float32
여기  baddbmm(f32, f32, f32)          float64
```

상류의 `baddbmm` 분해는 `beta`/`alpha` 를 파이썬 float 로 곱합니다. 상류에서는 파이썬 스칼라가
텐서의 dtype 을 끌어올리지 않는데(type promotion 규칙), 여기서는 float64 로 승격됩니다.
**이것은 분해 경로만의 문제가 아니라 스칼라 승격 규칙의 발산**이므로 다른 경로에서도 나올 수
있습니다. 아직 어느 op 에서 나오는지 좁히지 않았습니다 — 여기 적어 두는 것이 지금 할 수 있는
전부입니다.

---

## 7. ExecuTorch Edge 까지 남은 거리

**분해는 필요조건이고 충분조건이 아닙니다.** 상류에게 같은 모델을 물어보면 답이 이렇습니다:

```python
ep = torch.export.export(nn.Sequential(Linear(4,8), ReLU(), Linear(8,3)), (x,))
ep.run_decompositions(core_aten_decompositions())
# → aten.permute.default, aten.addmm.default, aten.relu.default,
#   aten.permute.default, aten.addmm.default
```

`aten.t.default` 가 `permute` 가 됩니다 — `t` → `transpose.int` → `permute` 의 두 단계이고,
둘째 단계는 §3 이 막고 있는 CIA 절반에 있습니다. **즉 docs/CAPTURE.md §5 가 지목한 바로 그
op 은 아직 안 됩니다.** 이 패스가 여는 것은 그 옆의 4 개이고, `t` 는 §3 · §4 의 벽 두 개가
동시에 걸려 있습니다.

그리고 Core ATen 에 닿아도 Edge 는 아직입니다.

| 남은 것 | 왜 이 패스가 아닌가 |
|---|---|
| **functionalization (view → `_copy`)** | Edge 는 `view_copy` · `permute_copy` · `transpose_copy` 를 씁니다. 그것들은 `tags: view_copy` 이지 `core` 가 아니고, **Core ATen 은 `view`/`permute` 를 그대로 둡니다.** 변환은 `to_edge` 의 일이고 분해표에 없습니다 |
| **stride / dim order** | `TensorBase` 에 `.stride()` 가 없습니다 (docs/META.md §6). 텐서 표현이 바뀌는 날의 문제 (docs/CAPTURE.md §5-2) |
| **`graph_signature` 의 이름** | 상수가 인덱스로만 식별됩니다. `constant_values` 가 생겨서 값은 나오지만 FQN 은 위층에서 붙여야 합니다 (§5-3) |
| **직렬화** | 트레이스가 프로세스 밖으로 못 나갑니다. `.pte` 로 가는 길이 여기를 지납니다 |
| **변이 · 별칭** | 캡처가 거절합니다. KV 캐시가 먼저 부딪힙니다 |

DESIGN.md §5 의 3 층 구조에서 **2 층("분해 테이블을 벤더링 — 롱테일이 자동으로 core op 으로
분해됨")이 이제 배선됐습니다.** 다만 "자동으로" 는 과했습니다 — docs/GAP.md §0 이 이미
"분해는 eager 폴백이 아니라 트레이싱 시점의 변환" 이라고 정정했고, 이번 작업이 그 트레이싱
시점을 만들었습니다. 그리고 그 위에서 재보니 **32 개 중 4 개**입니다. CORE_ATEN.md §4.2 가
요구한 "네 번째 층 — 어느 쪽에도 안 걸리는 잔여" 는 이제 이름이 5 개입니다 (§4).

---

## 8. 판정 — 전부 종료 코드로

| 검사 | 결과 |
|---|---|
| `cargo build --release` | 0 |
| `PYTHON=... sh rust/torch_c/pytests/run.sh` | 0 — **136/136 통과** (decompose 6 개 + capture 상수 1 개 포함) |
| `python tools/golden/compare.py` | 0 — **2383/2383**, ops=109, KNOWN DIVERGENCE 0 |
| `python rust/torch_c/pytests/verify_schemas.py` | 0 — **255/255** |

### 테스트가 실패할 수 있는지 확인했다

초록을 받았으므로 무력화해 보고 빨개지는지 두 번 확인했습니다.

- **`is_core()` 가 항상 True 를 돌려주게 했더니** — 패스가 아무것도 낮추지 않고 고정점 검사에
  걸려 `gave up after 8 rounds with these ops still outside Core ATen: aten.split.Tensor,
  aten.stack.default` 로 죽었습니다. `test_decompose_lowers_a_trace_to_core_aten` 과
  `test_decomposed_replay_matches_eager_bit_for_bit` 이 그것으로 빨개집니다.
- **결과 메타 대조(`_meta_matches`) 한 줄을 끊었더니** — §6.1 의 `aten.sum.default` 가
  **통과했습니다.** 8 노드짜리 도로는 그대로 초록이고 오직 그 테스트만 빨개지는데, 그것이
  이 검사가 무엇을 위해 있는지 그대로 보여줍니다: 낮추기가 되는지가 아니라 **낮춘 것이 같은
  함수인지**를 보는 검사입니다.

무력화 뒤에는 `git status --short` 로 원상복구를 확인했습니다.

---

## 9. 미완으로 남긴 것

| | 상태 |
|---|---|
| `t` · `silu` · `softmax` 등 산술을 재배열하는 분해 | **안 됨.** §4 의 벽 두 종류가 걸려 있습니다. 이것들이 열려야 §5 의 "비트 동일" 이 진짜 시험을 받습니다 |
| `overloads.json` 의 구멍 4 개 (`transpose` · `sigmoid` · `softmax` · `full_like`) | 안 채웠습니다. 채우는 것은 기계적이고 `verify_schemas.py` 가 상류와 대조해 줍니다 |
| `layout` 인자/속성 5 개 | 안 채웠습니다. **가장 값싼 다음 걸음** — 하나의 원인이 5 개 op 을 막고 있습니다 |
| `_jit_get_operation` 의 오버로드 목록 (§3) | 안 고쳤습니다. `bootstrap.py` 는 이번 회차에 다른 작업이 열려 있어 건드리지 않았습니다. **규칙 505 개가 이것 하나에 걸려 있습니다** |
| `_dispatch_get_registrations_for_dispatch_key` (§3) | 안 구현했습니다. 구현해도 CIA 항목의 상당수는 C++ 커널을 부르므로 다 열리지는 않습니다 |
| §6.1 `sum.dim_IntList([])` | **2026-08-28 고쳐졌습니다** — 빈 `dim` 목록이 모든 축으로 확장됩니다. `mean.dim` 이 같은 커널을 공유해 같은 수정으로 같이 고쳐졌습니다 |
| §6.2 `baddbmm` dtype | **고치지 않았습니다.** 스칼라 승격 규칙의 문제이고, 테스트가 현재 동작을 못박아 두었습니다 |
| functionalization (view → `_copy`) | 없음 (§7). 이 패스가 아닙니다 |
| 여러 트레이스에 걸친 측정 | 없음. §4 의 32 개는 **구현된 op 을 하나씩 단독 트레이스로** 돌린 것이지, 실제 모델 트레이스의 분포가 아닙니다 |

마지막 줄이 이 측정의 가장 큰 한계입니다. §4 의 "32 개 중 4 개" 는 **op 개수 비율**이지
"모델의 몇 %가 낮춰진다" 가 아닙니다. 실제 모델에서는 소수의 op 이 대부분의 노드를 차지하므로
두 숫자는 전혀 다를 수 있고, 어느 쪽으로 다를지는 재지 않았습니다.

---

## 10. 재현

```sh
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-decomp
PY=/Volumes/macMini/caches/spike-venv/bin/python

bash vendor/vendor_torch.sh                  # 새 worktree 만
PYTHON=$PY bash vendor/install_shim.sh       # 도로 테스트는 벤더 트리의 산출물을 읽는다
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib

PYTHON=$PY sh rust/torch_c/pytests/run.sh    # decompose 6 개 포함
$PY tools/golden/compare.py                  # 벤더 트리를 PYTHONPATH 에 넣지 말 것
$PY rust/torch_c/pytests/verify_schemas.py
```

§2 · §3 · §4 의 실측:

```sh
# 표 크기 셋, Core ATen 판정, 그리고 32 개 스윕
PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 $PY -c '
import torch, torch._decomp as D
from torchnative.export import core_ops, decomposition_table, decomposition_table_source
print("core:", len(core_ops()))
print("table:", decomposition_table_source(), len(decomposition_table()))
print("registry:", len(D.global_decomposition_table["post_autograd"]))
print("overloads of aten.transpose:", torch.ops.aten.transpose.op_overloads())
print("tags of aten.addmm.default:", torch.ops.aten.addmm.default.tags)'
```

같은 것을 상류 torch 로 (벤더 트리를 `PYTHONPATH` 에서 빼고) 돌리면 §3 의 비교 열이 나옵니다.
