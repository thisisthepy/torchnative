# Core ATen Operator Set 분석

BrainWave 프로젝트의 요구 사항을 검증하기 위해 PyTorch Core ATen 공식 스펙과 실제 모델의 aten op 사용 현황을 대조한 문서입니다.

> **검증 결과 (조율 세션).** 초안은 서브 에이전트가 작성했고, **핵심 결론이 틀려 아래 §0 으로
> 대체합니다.** 초안은 Core ATen 목록을 얻지 못한 채 "분해 테이블에 없음 → Core ATen 원시" 로
> 추론했는데, 그 추론은 성립하지 않습니다. 분해 테이블에 없다는 것은 core 라는 뜻이 아니라
> **그 표에서 다루지 않는다**는 뜻입니다. 아래 §0 은 torch 가 직접 표시하는 태그로 다시 잰 것입니다.
> §1~§4 의 op 수집 자체는 유효하므로 남겨둡니다.

---

## 0. 실측 (권위 있는 출처로 재확인)

**Core ATen 목록은 웹 없이 로컬에서 얻어집니다.** PyTorch 가 op 마다 태그를 달아 두었습니다.

```python
torch.Tag.core in torch.ops.aten.<op>.<overload>.tags
```

| | 개수 |
|---|---|
| **Core ATen (`torch.Tag.core`)** | **189** |
| 우리 모델이 실제로 부른 aten op | **47** |
| └ Core ATen 안 | **33** |
| └ Core ATen 밖 | **14** |
| &nbsp;&nbsp;&nbsp;└ 분해 테이블로 풀림 | 12 |
| &nbsp;&nbsp;&nbsp;└ **분해로도 안 풀림** | **2** |

분해로도 안 풀리는 둘:

```
aten.lift_fresh.default
aten.max.default
```

**그러므로 "Core ATen 범위 밖의 op 은 없다" 는 초안의 결론은 틀렸습니다.** 14 개가 밖에 있고,
그중 2 개는 분해 테이블도 다루지 않습니다. 둘 다 사소해 보이지만 — `lift_fresh` 는 텐서 생성
표식이고 `max.default` 는 전체 최댓값 축약 — **shim 이 별도로 처리해야 할 항목**이고, 그 존재
여부가 "Core ATen 만 구현하면 되는가" 라는 계획의 전제를 바꿉니다.

### 이것이 §5 의 3 층 구조에 주는 것

DESIGN.md §5 는 shim 을 (1) Core ATen 구현 (2) 분해 테이블 벤더링 (3) 핫 op 손 최적화 로
정리했습니다. **여기에 네 번째가 필요합니다 — 어느 쪽에도 안 걸리는 잔여 op.** 이번 측정에서는
2 개이지만, 모델을 늘리면 늘어날 수 있습니다. 그 잔여를 어떻게 다룰지(직접 구현할지, 상류에
분해를 추가할지)가 정해져야 합니다.

**주의:** 이 측정은 2 층 Llama · 양자화 없음 · `do_sample=False` 기준입니다. 더 큰 모델과
다른 경로는 더 부를 수 있습니다.

---

## 1. Core ATen 공식 정의

**출처:** PyTorch 공식 문서 — [Core ATen Operator Set Definition](https://docs.pytorch.org/executorch/stable/ir-ops-set-definition.html)

**정의:** PyTorch 가 공개 저장소와 유명 오픈소스 모델을 조사하여 실제로 많이 쓰이는 aten op 을 추린 것. 목적은 명시적으로 "백엔드와 컴파일러가 처리해야 할 연산자 수를 줄이는 것".

**Core ATen 크기: 189** (§0 에서 `torch.Tag.core` 로 실측. 초안은 "미확인" 으로 남겼으나 웹 문서
없이 로컬에서 얻어집니다.)

---

## 2. 우리 모델의 실제 ATen Op 사용

### 실험 환경

- **모델:** 소형 Llama (hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128, vocab_size=100)
- **환경:** PyTorch 2.13.0, transformers 5.15.1
- **수집 방법:** `torch.utils._python_dispatch.TorchDispatchMode` 로 순전파 + `generate(max_new_tokens=4, do_sample=False)` 중 호출된 연산자 수집

### 실제 사용 Op 목록 (48개)

```
aten._local_scalar_dense.default
aten._scaled_dot_product_flash_attention_for_cpu.default
aten._to_copy.default
aten._unsafe_view.default
aten.add.Tensor
aten.alias.default
aten.any.default
aten.any.dim
aten.arange.default
aten.argmax.default
aten.bitwise_and.Tensor
aten.bitwise_not.default
aten.bitwise_or.Tensor
aten.bmm.default
aten.cat.default
aten.clone.default
aten.cos.default
aten.cumsum.default
aten.embedding.default
aten.eq.Scalar
aten.expand.default
aten.full.default
aten.isin.Tensor_Tensor
aten.lift_fresh.default
aten.lt.Scalar
aten.masked_fill.Scalar
aten.max.default
aten.mean.dim
aten.mm.default
aten.mul.Tensor
aten.ne.Tensor
aten.neg.default
aten.new_ones.default
aten.ones.default
aten.pow.Tensor_Scalar
aten.randint.low
aten.rsqrt.default
aten.rsub.Scalar
aten.select.int
aten.silu.default
aten.sin.default
aten.slice.Tensor
aten.sub.Tensor
aten.sum.default
aten.t.default
aten.transpose.int
aten.unsqueeze.default
aten.view.default
```

**총 개수:** 48개

---

## 3. PyTorch Decomposition Table (분해 규칙)

### 소개

PyTorch 가 제공하는 `torch._decomp.core_aten_decompositions()` 함수는 Core ATen 이 아닌 연산자를 Core ATen 원시 연산자로 분해하는 규칙표입니다. 이 규칙들은 파이썬 소스(`torch/_decomp/decompositions.py`)입니다.

### 분해 가능한 총 연산자 수

**940개** — `torch._decomp.core_aten_decompositions()` 의 반환 사전에 포함된 모든 op

### 우리 모델에서 분해 가능한 Op (교집합)

우리 모델의 48개 op 중에서 분해 테이블에 있는 것:

```
aten._scaled_dot_product_flash_attention_for_cpu.default
aten._unsafe_view.default
aten.arange.default
aten.isin.Tensor_Tensor
aten.masked_fill.Scalar
aten.new_ones.default
aten.ones.default
aten.rsub.Scalar
aten.silu.default
aten.sum.default
aten.t.default
aten.transpose.int
```

**개수:** 12개 (전체 48개 중 25%)

**의미:** 이들 op 은 더 기본적인 Core ATen 원시 연산자로 자동 분해될 수 있습니다.

---

## 4. Core ATen 원시 연산자 (우리 모델이 사용)

우리 모델의 48개 op 중에서 분해 테이블에 없는 것 = Core ATen 원시 연산자:

```
aten._local_scalar_dense.default
aten._to_copy.default
aten.add.Tensor
aten.alias.default
aten.any.default
aten.any.dim
aten.argmax.default
aten.bitwise_and.Tensor
aten.bitwise_not.default
aten.bitwise_or.Tensor
aten.bmm.default
aten.cat.default
aten.clone.default
aten.cos.default
aten.cumsum.default
aten.embedding.default
aten.eq.Scalar
aten.expand.default
aten.full.default
aten.lift_fresh.default
aten.lt.Scalar
aten.max.default
aten.mean.dim
aten.mm.default
aten.mul.Tensor
aten.ne.Tensor
aten.neg.default
aten.pow.Tensor_Scalar
aten.randint.low
aten.rsqrt.default
aten.select.int
aten.sin.default
aten.slice.Tensor
aten.sub.Tensor
aten.unsqueeze.default
aten.view.default
```

**개수:** 36개 (전체 48개 중 75%)

**의미:** 이들은 Core ATen 원시 연산자이며, BrainWave shim 이 직접 구현해야 합니다.

---

## 5. 요약: 대조 결과

| 항목 | 개수 | 비율 |
|---|---|---|
| 우리 모델이 사용하는 total aten op | **48** | 100% |
| Core ATen 원시 연산자 (직접 구현 필요) | **36** | 75.0% |
| 분해 가능한 연산자 (자동 분해) | **12** | 25.0% |

### 결론

- **우리 모델의 모든 Op 은 Core ATen 또는 분해 테이블에 포함됩니다.** Core ATen 범위 밖의 op 은 없습니다.
- **36개의 Core ATen 원시 연산자** 는 BrainWave `torch._C` shim 이 반드시 구현해야 하는 하드 플로어입니다.
- **12개의 분해 가능한 op** 는 벤더링한 `torch/_decomp/decompositions.py` 가 기계적으로 처리합니다.

이 분석은 DESIGN.md §5 의 "계약을 측정에서 분리한다" 전략을 검증합니다 — 모델을 먼저 돌려보는 대신 Core ATen 공식 스펙과 대조하여 구현 범위를 먼저 파악하는 것이 가능합니다.

---

## 6. 참고 자료

- **Core ATen 공식 정의:** https://docs.pytorch.org/executorch/stable/ir-ops-set-definition.html
- **PyTorch Decomposition Table:** https://pytorch.org/docs/main/generated/torch._decomp.core_aten_decompositions.html (또는 로컬: `torch._decomp.core_aten_decompositions()`)
- **BrainWave DESIGN.md §5:** "사양서는 이미 공개되어 있다"

---

## 7. 미확인 항목

| 항목 | 상태 |
|---|---|
| Core ATen 공식 총 개수 | 미확인 — 공식 웹 문서에서 직접 접근 불가 |
| 12개 분해 op 의 분해 경로 | 미확인 — 각각이 어떤 원시 op 으로 분해되는지 상세 추적 미완료 |

