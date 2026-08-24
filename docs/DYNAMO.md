# `_C._dynamo` — 다음 임계 경로의 실제 크기

OVERLOAD.md §8 이 멈춘 자리(`torch._C._dynamo.eval_frame.set_guard_error_hook`,
`torch/_dynamo/guards.py:5457`, 모듈 스코프)를 이어서 잰 기록입니다. **코드는 고치지 않았습니다.**
`git status --short`로 확인 가능한 변경 범위는 이 문서 하나뿐입니다.

판정은 두 가지로 나눕니다 — 표면 크기는 `dir()` 로 직접 센 것, 실제 요구사항은 계측기가 잡은
것. 계측기가 진짜로 접근을 잡는지 자체 검증(self-check)을 먼저 통과시킨 뒤에 측정했습니다.

환경: `/Volumes/macMini/caches/spike-venv/bin/python`, torch 2.13.0, transformers 5.15.1.

---

## 0. 한눈에

| | 값 |
|---|---|
| `_C._dynamo` 전체 표면 (직접 셈, `dir()` 재귀) | **137개** — 서브모듈 4개(`eval_frame` 37 · `guards` 86 · `utils` 1 · `compiled_autograd` 5) + 최상위 직속 8개 |
| 기존 `surface.json`이 잡고 있는 것 | **8개** (최상위만, 서브모듈 미탐지 — §5) |
| `GenerationMixin` 임포트가 실제로 접근하는 이름 | **52개** (137개 중 38%) |
| 그중 실제로 **호출**되는 이름 | **2개** — `set_guard_error_hook`, `set_code_exec_strategy` |
| 모델 생성 + forward + `generate()` 단계에서 추가로 접근되는 이름 | **0개** (import 시점에서 전부 끝남) |
| `.pyi` 스텁에 이미 선언되어 있는가 (접근된 52개 기준) | **52/52** (완전) |
| off-switch(이름 부재로 끄는 방법) 가능성 | **없음** — 전부 무조건 임포트문, `try/except` 없음 |

**결론을 먼저 적습니다.** "`_C._dynamo` 서브시스템 전체를 구현해야 한다"는 처음 우려보다 범위가
훨씬 좁습니다. `GenerationMixin` 을 통과시키는 데 필요한 것은 137개 중 52개이고, 그중 **동작까지
필요한 것은 2개뿐**입니다. 나머지 50개는 임포트 문이 성공하기만 하면 되는 자리표입니다
(§3). `torch.compile` 을 쓰지 않는다는 전제가 성립하는 한 이 결론은 안정적입니다 — 실제로
forward·`generate()` 를 돌려도 추가 접근이 0이었습니다(§4).

---

## 1. 표면 — `dir()` 로 직접 센 것

```
>>> torch._C._dynamo
서브모듈 4개: eval_frame(37) · guards(86) · utils(1) · compiled_autograd(5)
최상위 직속 8개: PyMappingSlots · PyNumberSlots · PySequenceSlots · PyTypeSlots ·
                get_type_slots · has_slot · is_valid_var_name · strip_function_call
합계: 4 + 8 + 37 + 86 + 1 + 5 = 137 (서브모듈 자체 4개를 이름으로 셀지는 관점 차이 —
     "속성"으로 보면 141, "말단 심볼"로 보면 137. 이 문서는 후자를 씁니다)
```

`eval_frame` 은 프레임 평가 후킹(`set_eval_frame`), `guards` 는 가드 트리 빌드/평가 C++ 타입
(`GuardManager`, `RootGuardManager`, 각종 `*GuardAccessor`), `utils` 는 이름 하나
(`is_instancemethod`), `compiled_autograd` 는 컴파일된 backward 관련 5개입니다. `eval_frame`
말고도 서브시스템 표면이 크다는 배경 설명(OVERLOAD.md §8)이 정확했습니다 — 오히려 `guards` 가
`eval_frame` 보다 2배 이상 큽니다.

---

## 2. 계측 방법과 작동 증거

### 2.1 왜 검증부터 했는가

이 프로젝트에서 계측 방법 자체가 틀렸던 전례가 넷 있다고 지시받았습니다(C_SURFACE.md §5의
"접근 99.7%" 오염, IMPORT_WALLS.md의 `grep -q MODEL_OK` 오탐 등). 그래서 **측정 전에 계측기가
실제로 접근을 잡는지부터 자체 검증**했습니다.

### 2.2 방법

`torch._C._dynamo` 의 4개 서브모듈과 최상위 모듈 자체를, 각각 `types.ModuleType` 프록시로
교체했습니다. 설치 지점을 **둘 다** 바꿨습니다 — `sys.modules["torch._C._dynamo.eval_frame"]`
(→ `from torch._C._dynamo.eval_frame import X` 스타일을 잡음)과 부모 모듈의 속성
`torch._C._dynamo.eval_frame = proxy`(→ `torch._C._dynamo.eval_frame.X` 스타일을 잡음) 둘 다.
프록시의 `__getattr__` 에서 이름과 스테이지를 기록하고, 값이 콜러블이면 호출까지 감싸는
wrapper 를 돌려줘 실제 호출도 별도로 기록합니다.

### 2.3 자체 검증 — 실행 로그

```
$ /Volumes/macMini/caches/spike-venv/bin/python /tmp/dynamo_probe2.py
SELF-CHECK PASSED (top-level + submodule, both getattr styles, call tracking, negative control).
```

자체 검증이 확인한 것 4가지:

1. `from torch._C._dynamo.eval_frame import set_guard_error_hook` 스타일(대부분의 실제 코드가
   쓰는 형태)이 잡히는가 — 잡힘.
2. `torch._C._dynamo.eval_frame.set_guard_error_hook` 속성 접근 스타일이 **별개로** 잡히는가 —
   잡힘 (import 시 1회, 이후 속성 접근 시 카운트가 늘어남을 직접 확인).
3. 실제 호출(`sgeh(dummy)`)이 `CALLS` 에 기록되는가 — 기록됨.
4. **음성 대조군**: 건드리지 않은 이름(`has_slot`, `profile_guard_manager`)이 0으로 남는가 —
   남음. (이게 없으면 "전부 접근됨"처럼 보이는 §5의 함정에 다시 빠집니다.)

같은 프로세스에서 `import torch` 는 이미 끝난 상태(즉 `_C._dynamo` 서브모듈들은 이미 존재)에서
프록시를 설치했습니다 — 이는 §5에서 확인하듯 `import torch` 자체는 `torch._dynamo` 파이썬
패키지를 아직 끌어오지 않으므로, 이후 `GenerationMixin` 경로에서 벌어지는 최초의 전체 임포트를
그대로 잡을 수 있습니다.

---

## 3. `GenerationMixin` 이 실제로 요구하는 것

### 3.1 측정

```python
from transformers.generation.utils import GenerationMixin
```

이 한 줄로 `torch._dynamo` 파이썬 패키지(93개 모듈, IMPORT_WALLS.md 3차)가 최초로 통째로
임포트됩니다 (`masking_utils.py:42` 의 `from torch._dynamo._trace_wrapped_higher_order_op import
...`, `torch >= 2.6` 이면 무조건 실행 — §6). 그 안에서 `_C._dynamo` 의 이름 **52개**가
`getattr` 됩니다.

- `_C._dynamo` 최상위: `PyMappingSlots` · `PyNumberSlots` · `PySequenceSlots` · `PyTypeSlots` ·
  `get_type_slots` · `has_slot` · `strip_function_call` (7개) — 출처
  `torch/_dynamo/variables/object_protocol.py:20` 의 일괄 임포트,
  `torch/_dynamo/guards.py:798` 의 `strip_function_call = torch._C._dynamo.strip_function_call`.
  (`is_valid_var_name` 은 표면에 있지만 **이 경로에서는 접근되지 않았습니다** — 사용처가
  `guards.py:2073` 인데 메서드 본문 안이라 실제 컴파일이 일어날 때만 실행됩니다.)
- `eval_frame` 17개: `torch/_dynamo/eval_frame.py:59-70` 의 임포트 블록(11개) +
  `torch/_dynamo/guards.py:49` 의 `code_framelocals_names`(1개) +
  `torch/_dynamo/bytecode_debugger.py:74` 의 `NULL_STACK_VALUE` 등 산발적 임포트(5개).
- `guards` 28개: `torch/_dynamo/guards.py:50-77` 의 단일 임포트 블록에서 26개 +
  `torch/_dynamo/convert_frame.py:59` 의 `GlobalStateGuard`(1개) +
  `compute_overlapping_tensors`(1개, 사용처 미추적).
- `utils` · `compiled_autograd` — **0개.** 이 경로에서 전혀 건드리지 않습니다.

### 3.2 그중 실제로 호출되는 것 — 2개뿐

```
[A_import_GenerationMixin] eval_frame.set_code_exec_strategy(
    args=(<code object nonrecursive_disable_wrapper ...>,), kwargs={}) -> None
[A_import_GenerationMixin] eval_frame.set_guard_error_hook(
    args=(<function guard_error_hook at 0x...>,), kwargs={}) -> None
```

두 호출 다 **모듈 스코프의 부기(bookkeeping) 등록**이지, 결과를 쓰는 계산이 아닙니다.

| 이름 | 호출 지점 | 인자 | 반환값 사용 | 성격 |
|---|---|---|---|---|
| `set_guard_error_hook` | `torch/_dynamo/guards.py:5457`, 모듈 스코프 | 콜백 함수 1개 | **아니오** (대입 없는 문장) | 가드 평가 중 예외가 났을 때 부를 훅을 등록. 훅은 **eval-frame 후킹이 실제로 설치돼 있을 때만** 트리거됨 — `torch.compile` 을 안 쓰면 절대 안 불림 |
| `set_code_exec_strategy` | `torch/_dynamo/decorators.py:125`, 모듈 스코프 (`skip_code()` 경유) | 코드 객체 1개 + `FrameExecStrategy` | **아니오** | 특정 함수의 바이트코드에 "추적하지 말고 건너뛰라"는 표식을 다는 것. 이 표식도 **eval-frame 후킹이 실제로 도는 동안에만** 참조됨 |

**둘 다 "이름이 있으면 되는" 것과 "동작해야 하는" 것의 중간입니다** — `hasattr` 로 존재만
확인하고 넘어가는 자리(off-switch)가 아니라 **무조건 호출되지만**, 그 호출의 효과는
`torch.compile` 을 쓰지 않는 한 아무도 관측하지 않습니다. 그러므로 시그니처(인자 개수·타입)만
맞는 아무 동작 없는 no-op 이면 충분합니다 — 예외만 던지지 않으면 됩니다.

`.pyi` 스텁(`vendor/torch/_C/_dynamo/eval_frame.pyi:12-18`)의 실제 시그니처:

```python
def set_guard_error_hook(hook: DynamoGuardHook) -> None: ...
def set_code_exec_strategy(code: types.CodeType, strategy: _FrameExecStrategy) -> None: ...
```

둘 다 반환형이 `None` 으로 선언돼 있습니다 — 상류 자신도 반환값을 안 쓰는 API 라는 뜻입니다.

### 3.3 나머지 50개 — 존재만 필요

`from X import (...)` 문에 이름이 나열돼 있으면, 파이썬은 그 문장이 실행되는 시점에
**그 이름들이 존재하는지만** 확인합니다. 클래스(`GuardManager`, `RootGuardManager`,
`LeafGuard`, 각종 `*GuardAccessor` 등 19개)는 `isinstance`/서브클래싱에 쓰이지만, **그 사용은
전부 함수·메서드 본문 안**이라 그 함수가 실제로 호출될 때만 실행됩니다 — 그리고 그 함수들은
가드 트리를 빌드/순회하는 코드라서 **`torch.compile` 이 실제로 컴파일을 시도할 때만** 실행됩니다.

직접 확인한 예:

```
torch/_dynamo/convert_frame.py:332  (함수 _fn 본문 안)
    guards = GlobalStateGuard()
```

`GlobalStateGuard()` 인스턴스화는 `_fn` 이 실제로 호출될 때만 일어나고, `_fn` 은
`torch._dynamo.eval_frame` 이 프레임 평가 후킹으로 설치하는 콜백입니다 — 후킹 자체가
`torch.compile()` 호출 없이는 설치되지 않습니다.

`GlobalStateGuard` 타입 힌트로도 한 번 더 나오는데(`guards.py:4392`,
`self.global_state: torch._C._dynamo.guards.GlobalStateGuard | None = None`), `guards.py:18` 에
`from __future__ import annotations` 가 있어 **이 어노테이션은 런타임에 평가되지 않습니다**
(문자열로만 저장). 그래서 이 자리는 접근 카운트에 안 잡혔습니다 — 이것도 자체 검증이 맞았다는
방증입니다(잡을 게 없는데 안 잡힘).

---

## 4. forward + `generate()` — 추가 접근 0

Llama 2층 hidden 32 모델을 만들어 `model(input_ids)` 와 `model.generate(input_ids,
max_new_tokens=3)` 까지 같은 계측기로 돌렸습니다.

```
[A_import_GenerationMixin] OK
[B_build_and_forward] OK logits shape=(1, 5, 64)
[C_generate] OK gen shape=(1, 8)

Total distinct names accessed: 61   (52개 실이름 + __file__/__path__ 등 더더 9개)
Total distinct names CALLED: 2
```

B·C 단계에서 **`_C._dynamo` 이름이 단 하나도 추가로 접근되지 않았습니다.** import 시점에
전부 끝난다는 뜻입니다 — `torch.compile` 을 쓰지 않는 한 실행 경로는 이 서브시스템을 다시
건드리지 않습니다. 이것이 "얼마나 들어가야 하는가"라는 질문에 대한 가장 중요한 답입니다:
**런타임 동작이 아니라 순수하게 import-time 부기입니다.**

---

## 5. 부수 발견 — 기존 `surface.json` 이 이 표면을 놓치고 있었다

`vendor/gen_surface.py::submodule_stubs()` 는 `torch/_C/` 아래에서 `.pyi` 파일 또는
`__init__.pyi` 가 있는 디렉터리를 **한 항목**으로 등록합니다. `_dynamo/` 는 디렉터리이고
`__init__.pyi` 가 있으므로 **"_dynamo" 라는 이름 하나**로 등록되고, 그 안의
`eval_frame.pyi`(105줄) · `guards.pyi`(509줄) · `compiled_autograd.pyi`(13줄) 는
**재귀적으로 파싱되지 않습니다.**

```
$ python3 -c "import json; d=json.load(open('rust/torch_c/src/surface.json'));
              print(d['submodules']['_dynamo'])"
{'functions': ['strip_function_call', 'is_valid_var_name', 'get_type_slots', 'has_slot'],
 'types': {'PyMappingSlots': ..., 'PyNumberSlots': ..., 'PySequenceSlots': ..., 'PyTypeSlots': ...},
 'values': []}
```

**8개.** 실제 137개의 6%입니다. `_export/` 도 같은 구조(중첩 `.pyi` 1개)를 갖고 있어 같은 버그의
영향을 받을 가능성이 있습니다(확인은 `_dynamo` 만 했습니다 — `_export` 는 이 작업 범위 밖).

**좋은 소식은 `.pyi` 데이터 자체는 이미 있다는 것입니다.** §3.1에서 접근된 52개 전부
(`eval_frame.pyi` · `guards.pyi` 기준) 스텁에 **선언돼 있습니다** — 시그니처까지 포함해서.
빠진 건 `gen_surface.py` 가 그 파일들을 찾아 읽는 로직이지, 원본 데이터가 아닙니다.

`utils.pyi` 는 벤더링 트리에 **아예 없습니다**(런타임에는 `is_instancemethod` 하나가 있는데도).
이 경로에서 안 쓰이므로 우선순위는 낮지만, `_C._dynamo` 를 패키지로 완성하려면 언젠가 채워야
합니다 — **미확인.**

---

## 6. off-switch 가능성 — 없음

IMPORT_TORCH.md 가 찾은 32개 off-switch(`hasattr(torch._C, "...")` 패턴)는 **`torch._C` 최상위에
대한 질문**입니다. `_dynamo` 내부의 임포트문들은 전부 무조건 실행되는 `from X import (...)` 이지,
`try/except ImportError` 로 감싸여 있지 않습니다 — `torch/_dynamo/eval_frame.py:59`,
`torch/_dynamo/guards.py:49-77` 둘 다 확인했습니다.

한 단계 위, `transformers.masking_utils` 자체에도 조건이 있긴 합니다.

```python
# masking_utils.py:38-42
_is_torch_greater_or_equal_than_2_6 = is_torch_greater_or_equal("2.6", accept_dev=True)
if _is_torch_greater_or_equal_than_2_6:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex
```

이론적으로는 "torch 버전을 2.6 미만으로 자칭하면 이 임포트를 피할 수 있다"는 레버가
존재하지만, **DESIGN.md §1 의 전제(파사드 금지, 벤더링한 소스는 한 줄도 안 고침)와
IMPORT_TORCH.md 의 관문 조건(`transformers` 가 요구하는 `torch >= 2.5.0`, 그리고 최신 API 표면
전반이 2.6+ 를 가정)에 부딪힙니다.** 버전을 낮춰 자칭하는 것은 이 우회로가 막는 것보다 더 많은
것을 깰 가능성이 높습니다 — **실측하지 않았고, 이 문서의 판단으로는 시도할 가치가 낮습니다
(미확인으로 남깁니다).**

**결론: off-switch 없음.** `_C._dynamo` 표면 자체를 채우는 것 외에 다른 경로가 안 보입니다.

---

## 7. 최소 구현 제안

`torch.compile` 을 절대 쓰지 않는다는 전제 위에서:

1. **`_C._dynamo` 를 패키지로 등록**하고 그 아래 4개 서브모듈(`eval_frame` · `guards` ·
   `utils` · `compiled_autograd`)도 패키지로 등록합니다 — VENDOR.md 벽 8과 같은 메커니즘
   (`_SubmoduleFinder`, `rust/torch_c/src/bootstrap.py:260` 부근)을 그대로 재사용하면 됩니다.
   이미 있는 인프라입니다.
2. **52개 이름을 채웁니다** — 소스는 이미 벤더링된 `.pyi` 3개
   (`vendor/torch/_C/_dynamo/{eval_frame,guards,compiled_autograd}.pyi`)와 §3.1의 접근 목록의
   교집합입니다. 클래스(19개)는 빈 몸체의 자리표 타입으로, 함수(29개, 호출 2개 제외)는
   시그니처만 맞춘 no-op 으로 충분합니다 — **호출되지 않으므로 동작을 구현할 필요가 없습니다.**
3. **`set_guard_error_hook` · `set_code_exec_strategy` 2개만 신경 씁니다** — 인자 개수와
   타입(콜러블 1개 / 코드객체+전략객체)만 받아들이고 `None` 을 돌려주는 no-op. 예외만 던지지
   않으면 정확성에 영향이 없습니다(§3.2 — 훅이 실제로 트리거되려면 eval-frame 후킹이 설치돼
   있어야 하는데, `torch.compile` 을 안 쓰므로 설치되지 않습니다).
4. **`utils`(`is_instancemethod` 1개) · `compiled_autograd`(5개)는 이번 경로에서 필요 없습니다**
   — 빈 패키지로만 존재하면 됩니다. 나중에 다른 임포트 경로가 건드릴 수 있으니 완전히
   생략하지는 말되, 이번 최소 구현의 우선순위에서는 뒤로 둡니다.
5. 기존 `overloads.json`/aten 작업과 같은 패턴으로, **표를 상류 `.pyi` 와 대조하는 검증기**
   (`verify_schemas.py` 류)를 하나 두는 것을 권합니다 — §5에서 `.pyi` 자체는 신뢰할 수 있음을
   확인했으니, 표를 손으로 옮겨 적을 때 생기는 오타/누락만 잡으면 됩니다.

**작업량 추정(코드 작성 없이, 이미 있는 인프라와 비교한 상대적 크기):** `TensorBase` 694개
멤버를 채운 작업(C_SURFACE.md §3, 이미 완료됨)에 비하면 **1/13 규모**(52 vs 694)이고, 그마저
50개는 no-op 자리표, 실제 동작이 필요한 건 2개뿐입니다. "서브시스템 표면"이라는 처음 우려보다
훨씬 작은 작업입니다.

---

## 8. 미확인

| # | 항목 | 상태 |
|---|---|---|
| 1 | `utils.pyi` 가 벤더링 트리에 없는 이유, 필요해지는 경로가 있는지 | 미확인. 이번 경로에선 0회 접근 |
| 2 | `_export/` 도 `gen_surface.py` 의 같은 미탐지 버그 영향을 받는지 | 존재는 확인(§5), 내용은 미조사 — 이 작업 범위 밖 |
| 3 | `transformers.masking_utils` 의 `torch >= 2.6` 조건을 낮춰 dynamo 임포트를 우회하는 것이 실제로 더 적은 피해로 가능한지 | 미실측. DESIGN.md §1 전제와 충돌 가능성이 커서 시도하지 않았음 |
| 4 | `torch._dynamo` **파이썬 패키지**(93개 `.py`) 쪽에서 벤더링한 소스 자체가 추가로 요구하는 것(순수 파이썬 의존성, 예: `sympy` — `eval_frame.py:9` 에서 확인됨)이 전부 충족 가능한지 | 이번 측정 환경(spike-venv)엔 이미 있어 막히지 않았음. torchnative 최종 벤더링 tree 기준으로는 미확인 |
| 5 | `GenerationMixin` 이외의 다른 transformers 진입점(비-생성형 모델, 트레이닝 경로)이 `_C._dynamo` 의 다른 이름을 추가로 요구하는지 | 미측정. 이 문서는 `from_config`/`generate` 경로만 봄(OVERLOAD.md §8과 동일 범위) |
| 6 | `torch.compile` 을 실수로라도 트리거하는 transformers 내부 경로가 있는지(예: 일부 최신 모델의 `attn_implementation="sdpa"` 선택 로직이 조건부로 `torch.compile` 래핑을 시도하는 경우) | 미확인. 이번 측정은 `do_sample=False`, 기본 config 만 사용 |

---

## 9. 재현

```bash
PY=/Volumes/macMini/caches/spike-venv/bin/python
$PY /tmp/dynamo_probe2.py
```

핵심 부분만 발췌:

```python
proxy = types.ModuleType(qualname)
def tracked_getattr(name, ...):
    ACCESS.setdefault(key, []).append(STAGE)          # 접근 기록
    val = getattr(real_mod, name)
    if callable(val) and not isinstance(val, type):
        def wrapper(*a, **k):
            ret = val(*a, **k)
            CALLS.setdefault(key, []).append((...))    # 호출 기록
            return ret
        return wrapper
    return val
proxy.__getattr__ = tracked_getattr
sys.modules[f"torch._C._dynamo.{sub}"] = proxy   # style 1: from-import
setattr(dynamo_c, sub, proxy)                    # style 2: attribute access
```

자체 검증(양성 3종 + 음성 대조군 1종)을 통과한 뒤에만 실측값을 신뢰했습니다(§2.3).
