# 이름 붙이기 — 커널은 있는데 철자가 없던 19 개

조율 세션이 실측한 `Tensor` 메서드 25 개 중 19 개 거부 목록:

    argmax softmax flatten le topk sort masked_fill repeat split chunk
    narrow squeeze permute clamp tril triu flip gather index_select

이 문서는 그중 몇 개가 **커널이 이미 있어서 철자만 채우면 되는지**, 몇 개가 **커널부터 없어서
손댈 수 없는지**를 가르고, 채운 철자가 상류와 같은 값을 내는지 실측한 기록이다.

판정은 종료 코드로만 한다. grep 으로 성공을 판단하지 않는다.

---

## 0. 한눈에

| | 이전 | 지금 |
|---|---|---|
| `verify_schemas.py` | 154/154 | **170/170** (+16) |
| 골든 하네스 | 1616/1616, ops covered=78 | **1616/1616, ops covered=78** (무회귀) |
| `--inject-fault value/shape/dtype` | 1/1/1 | **1/1/1** (그대로) |
| 호스트 스모크 (`pytests/run.sh`) | — | **62/62, exit 0** |
| 3 타깃 (host / androidNdk arm64-v8a / aarch64-apple-ios) | — | **전부 exit 0** |

19 개 중 **커널이 있어서 실제로 고칠 수 있었던 것은 7 개** (`argmax`, `softmax`, `le`,
`topk`, `sort`, `masked_fill`, `squeeze` — 뒤의 세 개는 부분적으로), 나머지 12 개는 커널
자체가 `aten.rs` 에 없어 이 작업 범위(스펠링)로는 손댈 수 없다.

---

## 1. 채운 철자와 상류 대조 결과

전부 `/Volumes/macMini/caches/spike-venv/bin/python`(torch 2.13.0)을 상류로 두고,
`tools/golden/loader.py::load_shim()` 로 이 빌드의 `_C.so` 를 직접 로드해 `TensorBase.<method>`
와 `_VariableFunctions.<fn>` 양쪽을 실측했다. 재현 스크립트는 이 보고서 이후 폐기했으므로
숫자만 남긴다.

### `argmax` — 새 스펠링, 완전히 동작

`methods.json`(`argmax`) 과 `overloads.json`(기존 `argmax` 항목에 `.out` 은 이미 있었으나
`.default` 사용례를 재확인)을 채웠다. 커널: `aten.argmax.default`.

    x.argmax()                 -> 상류와 동일 (스칼라 인덱스)
    x.argmax(dim=1)             -> 상류와 동일 (shape/value)
    x.argmax(dim=1, keepdim=True) -> 상류와 동일
    torch.argmax(x, dim=1)       -> (`_VariableFunctions.argmax`) 상류와 동일

### `softmax` — 새 스펠링, 테이블이 아니라 Python 합성으로

`methods.json` 의 `_README` 가 이미 적어 둔 이유대로, `Tensor.softmax` 의 파서 레벨 키는
`aten.softmax.int` 인데 상류 디스패처가 실제로 보는 키는 `aten._softmax.default` 다
(`TorchDispatchMode` 로거로 재확인, `docs/NN_SURFACE.md` §6). 테이블은 스키마 문자열 그대로
디스패치 키를 만들기 때문에 이 어긋남을 테이블로는 표현할 수 없다 — 그래서 `bootstrap.py` 에
`_install_tensor_softmax` 를 새로 추가해 `to`/`item` 과 같은 "Python 레벨" 그룹에 넣었다.

측정한 실제 분기 (`dtype=None`, `dtype=self.dtype`, `dtype=다른 타입` 셋 다 재현):

    x.softmax(dim)                        -> aten._softmax.default(x, dim, False)
    x.softmax(dim, dtype=x.dtype)          -> aten._softmax.default(x, dim, False)  (변환 없음)
    x.softmax(dim, dtype=other)            -> aten._to_copy.default(x, dtype=other)
                                               다음 aten._softmax.default(_, dim, False)

    softmax(dim=-1)                 -> 상류와 float32 로 일치 (atol 1e-6)
    softmax(dim=-1, dtype=float64)  -> 상류와 float64 로 일치 (atol 1e-10)

`methods.json` 의 README 코멘트도 갱신해 "테이블에는 없지만 Python 합성으로는 있다"는 것을
명시했다.

### `topk` — 새 스펠링, 완전히 동작

`methods.json`/`overloads.json` 모두에 추가. 커널: `aten.topk.default` (유일한 오버로드 —
`.values` out 변형은 커널이 없고, `Tensor.topk` 의 벤더 `.pyi` 에도 out= 이 없어 아예 넣지
않았다). `values`/`indices` 둘 다 상류와 일치 확인(dim=2, k=2, largest 기본값).

### `sort` — 새 스펠링, 완전히 동작

`methods.json`/`overloads.json` 모두에 추가. 순서는 벤더 `__init__.pyi` 의 `@overload` 순서
그대로 — `stable=` 키워드 전용 변형(`aten.sort.stable`, 커널 없음)이 먼저, 일반형
(`aten.sort.default`, 커널 있음)이 나중. `stable` 인자에 기본값 텍스트가 없어(`bool? stable`
에 `=` 없음) `has_default_value()` 가 거짓이 되고, `stable=` 을 안 주면 첫 스키마가 바인딩에
실패해 두 번째로 정확히 넘어간다 — 상류 디스패치 로그와 일치 실측함.

    x.sort(dim=1)                 -> 상류와 일치 (values, indices 둘 다)
    x.sort(dim=1, descending=True) -> 상류와 일치
    torch.sort(x, dim=1)           -> (`_VariableFunctions.sort`) 상류와 일치

out 변형(`sort.values`, `sort.values_stable`)은 넣지 않았다 — 커널이 없고, `max.dim` 등 이미
있는 다중 반환 op 들도 이 표에서 out 변형을 넣지 않는 것이 기존 관례였다(값 두 개를 `out=` 튜플로
받는 바인딩은 이 `_Overloads` 기계가 아예 표현하지 못한다).

### `masked_fill` — 이미 있었다. 손대지 않음

`methods.json` 에 `masked_fill.Tensor`/`masked_fill.Scalar` 가 **이미** 있었고, `aten.rs` 도
둘 다 커널이 있다(`Scalar` 는 `IMPLEMENTED`, `Tensor` 는 `IMPLEMENTED_AWAITING_GOLDEN` — 골든
집계에서만 빠져 있을 뿐 디스패치는 정상). 실제로 두 형태 다 이 빌드에서 상류와 일치하는 값을
낸다:

    x.masked_fill(mask, -9.0)          -> 상류와 일치 (Scalar)
    x.masked_fill(mask, torch.tensor(3.5)) -> 상류와 일치 (Tensor)
    x.masked_fill(non_bool_mask, ...)   -> 상류처럼 거부(런타임 에러) — 정확히 그 자리에서

19 개 거부 목록에 있었던 이유는 **이 작업에서 알아낼 수 없었다** — 이번 빌드에서는 재현되지
않았다. 가능한 설명은 (a) 측정이 이 커밋 이전의 낡은 아티팩트에 대고 이루어졌거나,
(b) 측정에 쓰인 인자 모양이 여기서 시도한 것과 달랐거나(예: dtype 이 다른 mask, 브로드캐스트가
필요한 shape 등) 둘 중 하나로 보인다. 조율 세션이 원래 측정 스크립트로 다시 돌려 보는 편이
확실하다.

### `le` — 이미 있었다. 절반만 커널이 있다

`methods.json` 에 `le.Tensor`/`le.Scalar` 가 **이미** 있었다. 그러나 `aten.rs` 에는
`aten.le.Scalar` 커널만 있고 `aten.le.Tensor` 는 없다:

    x.le(0.5)   -> 상류와 일치 (Scalar, 커널 있음)
    x.le(other_tensor) -> "aten op not implemented in torch._C shim: aten.le.Tensor"

스펠링은 정확하게 해석되고(두 후보 중 맞는 것을 고름), 커널이 없는 쪽만 그 자리에서 거부한다 —
이것이 `methods.json` 의 README 가 말하는 "테이블 항목이 커널을 보장하지 않는다"는 바로 그
사례다. `le.Tensor` 커널이 이 작업의 파일 범위(`aten.rs` 금지) 밖이라 채울 수 없다. masked_fill
과 마찬가지로, 19 개 목록에 `le` 가 오른 것이 `le.Scalar` 쪽 실패였는지 `le.Tensor` 쪽
실패였는지는 알 수 없다 — 전자라면 이 빌드에서는 이미 고쳐져 있고, 후자라면 여전히 커널이
없어서 막혀 있다.

### `squeeze` — 부분적으로 새 스펠링

`methods.json`/`overloads.json` 모두에 세 오버로드를 벤더 `.pyi` 순서(무인자 → `dim:int` →
`dim:int[]`) 그대로 추가했다. 커널은 `aten.squeeze.dim` 하나뿐이다:

    x.squeeze(1)     -> 상류와 일치 (커널 있음)
    x.squeeze()      -> "aten op not implemented ...: aten.squeeze.default" (커널 없음)
    x.squeeze((1,3)) -> "aten op not implemented ...: aten.squeeze.dims" (커널 없음)

무인자·리스트 형태는 스펠링만 채워 두었다 — 정확한 키로 거부하므로 다음 작업의 작업 큐에
그대로 잡힌다.

### `multinomial` — 목록 밖이지만 요청받은 대로 채움

19 개 목록에는 없지만, 작업 지시가 "`torch.multinomial` 같은 함수 스펠링도 없다"고 직접
지목했고 커널(`aten.multinomial.default`)이 있으므로 `methods.json`/`overloads.json` 양쪽에
추가했다. `.out` 변형은 벤더 `.pyi` 의 함수 레벨에는 있지만 메서드 레벨에는 없어 `overloads.json`
에만 넣었다(`argmax` 와 같은 패턴).

    x.multinomial(3, replacement=True)      -> shape/dtype 상류와 일치, 인덱스 범위 내 (RNG 스트림은
    torch.multinomial(x, 3, replacement=True) -> 다르므로 값 자체는 비교 대상이 아님 — 다른 uniform_/
                                                   normal_ 항목들과 같은 근거)

---

## 2. 커널이 없어서 손대지 않은 것 (12 개 + `torch.randn`)

`aten.rs` 의 `IMPLEMENTED`/`IMPLEMENTED_AWAITING_GOLDEN` 어디에도 다음 이름의 커널이 없다.
`_aten_dispatch` 가 유일한 문이므로, 스펠링만 채워 봤자 정확한 이름으로 거부할 뿐 아무것도
동작시키지 못한다 — 그리고 이 작업은 커널 추가가 금지되어 있다. 손대지 않고 그대로 남겼다:

    flatten repeat split chunk narrow permute clamp tril triu flip gather index_select

추가로, 지시에서 예로 든 `torch.randn` 도 커널이 없다(`aten.rs` 에 `randn` 계열 op 자체가
없음 — `normal_`/`uniform_`/`randint` 만 있다). 이것도 목록으로만 남긴다.

이 12+1 개는 커널을 만드는 다른 에이전트에게 넘긴다.

---

## 3. 표 변경 요약

`overloads.json`: `argmax`(기존 항목 재확인) 뒤에 `topk`, `sort`, `squeeze`, `multinomial`
네 항목 신설. `methods.json`: `masked_fill` 뒤에 `argmax`, `topk`, `sort`, `squeeze`,
`multinomial` 다섯 항목 신설, `_README` 의 `softmax` 코멘트를 Python 합성 위치를 가리키도록
갱신. `bootstrap.py`: `_install_tensor_softmax` 신설, `_install_tensor_methods` 에서 호출.

모든 스키마 문자열은 `str(torch.ops.aten.<op>.<ov>._schema)` (torch 2.13.0)에서 그대로
전사했다 — 지어낸 것이 없다. `pytests/verify_schemas.py` 가 그것을 재확인한다(170/170).

---

## 4. 두 번째 회차 — 커널이 82 개로 늘어난 뒤 (`addmm`/`native_layer_norm`/`split.Tensor`/`tanh`)

`docs/GPT2.md`가 이 넷을 구현해 커널이 78 → 82 개가 됐고, §5에 "커널은 있는데 스펠링이 없다"는
표를 명시적으로 남겼다. 이 회차는 그 표를 메우는 작업이다. 근거는 두 가지뿐이었다 — `_aten_implemented()`가
돌려주는 82개 목록(권위 있는 출처, `aten.rs`를 grep하지 않았다)과, 상류 `torch.ops.aten.<op>.overloads()` /
벤더링된 `.pyi`.

### 4.0 82개 전부 감사

지난 회차가 열어 둔 14개(`flatten masked_fill repeat split chunk narrow squeeze permute clamp
tril triu flip gather index_select`) 중 커널이 **새로 생긴 것은 `split` 하나뿐**이다
(`aten.split.Tensor`). `masked_fill`/`squeeze`는 커널도 스펠링도 지난 회차 그대로(부분)이고,
나머지 11개는 여전히 커널 자체가 없다 — 이번에도 건드리지 않았다.

그래서 판단 기준을 "14개 목록"에서 "82개 커널 전부"로 넓혔다. `_aten_implemented()`의 82개
각각에 대해 `overloads.json`/`methods.json`의 스키마를 파싱해 `(op, overload)` 키를 뽑고,
82개와 대조하는 감사 스크립트를 돌렸다. 결과: **82개 전부가 어떤 형태로든 스펠링을 갖고
있다** — 표 항목(대다수), Python 레벨 합성(`softmax`/`to`/`item`/`__getitem__`/`linear`/
`silu`/`sdpa`/`layer_norm`, 기존 + 이번에 추가한 `layer_norm`), 또는 상류 자체에 공개
스펠링이 없는 내부 전용 op(`alias.default` — `hasattr(torch.Tensor, "alias")` /
`hasattr(torch, "alias")` 둘 다 실측 결과 `False`, 감쌀 스펠링이 존재하지 않는다).

### 4.1 채운 것

| 이름 | 파일 | 상류와 값 대조 |
|---|---|---|
| `tanh` (함수+메서드) | `overloads.json`/`methods.json` | 일치 (`torch.tanh(x)` == `x.tanh()`) |
| `addmm` (함수+메서드) | `overloads.json`/`methods.json` | 일치 |
| `mm` (메서드 — 함수는 이미 있었다) | `methods.json` | 일치 |
| `select` (함수+메서드) | `overloads.json`/`methods.json` | 일치 |
| `scatter` (함수+메서드) | `overloads.json`/`methods.json` | `.src` 일치. `.value`/`.reduce`/`.value_reduce`는 커널이 없어 정확한 이름으로 거부 (실측) |
| `split` (함수+메서드, int 형태) | `overloads.json`/`methods.json` | 일치 (`torch.split(x,1,0)` == `x.split(1,0)`) |
| `split_with_sizes` (함수+메서드) | `overloads.json`/`methods.json` | 커널 없음 — 정확한 이름으로 거부 (실측, 아래 4.2) |
| `native_layer_norm` (함수) | `overloads.json` | 3-튜플(out, mean, rstd) 전부 일치 |
| `layer_norm` (함수, Python 합성) | `bootstrap.py` `_install_composites` | 일치 (`F.layer_norm`/`torch.layer_norm`/`nn.LayerNorm`이 전부 도달하는 지점) |
| `_get_cudnn_enabled`/`_set_cudnn_enabled` | `bootstrap.py` `_install_behaviour` | 상태 게터/세터, `F.layer_norm`이 읽는 값을 실제로 반환 |

전부 `tools/golden/loader.py::load_shim()`으로 이 빌드의 `_C.so`를 직접 로드해
`TensorBase.<method>`와 `_VariableFunctions.<fn>` 양쪽을 실측했다 (이전 회차와 같은 방법).
`torch 2.13.0`을 상류로 두고 동일 입력에 `torch.allclose`로 비교했으며, 전부 일치했다
(`tanh`/`addmm`/`mm`/`select`/`scatter.src`/`split`/`layer_norm`/`native_layer_norm`의
세 출력 전부).

**전체 vendor 트리(`import torch`)로도 시도했으나 판단 근거로 쓰지 않았다.** `TORCH_USE_RTLD_GLOBAL=1`로
`vendor/`를 얹어 실제로 실행해 보면, `x.argmax()`처럼 이미 동작이 확인된 기존 스펠링조차
`torch._C._functorch.is_functorch_wrapped_tensor`에서 막힌다 — 이번 회차가 만든 문제가
아니라 vendor 트리를 통한 메서드 호출 전반에 걸친 기존 벽이다(범위 밖의 `_functorch` 서브모듈
배선 문제로 보인다, 원인은 추적하지 않았다). 그래서 지난 회차와 같은 방법(`loader.load_shim()`로
`_C`를 직접 로드해 `TensorBase`/`_VariableFunctions`를 부르는 것)으로 판단했다 — `docs/GPT2.md` §4가
판정을 aten 레벨로 내린 것과 같은 이유다.

### 4.2 `split`의 스키마 두 개가 이름이 다른 이유

`methods.json`의 `split` 항목은 `aten::split.Tensor`와 `aten::split_with_sizes`를 **한 키
아래** 담고 있다 — 이 표에서 유일하게 스키마들이 op 이름을 공유하지 않는 항목이다. 상류
`torch/_tensor.py:Tensor.split`을 직접 읽으면(`TorchDispatchMode`로도 재확인)
`isinstance(split_size, int)`로 분기해서 `torch._VF.split`과 `torch._VF.split_with_sizes`
중 하나를 호출한다 — 진짜로 다른 두 aten op다. `_Overloads.resolve()`는 키와 스키마의 op
이름이 같은지 검사하지 않고 스키마를 순서대로 시도하다 바인딩되는 것을 쓰므로, 이 방식으로도
정확히 상류와 같은 분기를 재현한다: int 인자는 `split.Tensor`(커널 있음)에 바인딩되고
list/tuple 인자는 `split_with_sizes`(커널 없음, `aten.split_with_sizes.default`로 정확히
거부)에 바인딩된다. `overloads.json`은 다르다 — `_VariableFunctions.split`은 실측상 int
형태로만 불리므로(벤더 트리가 리스트를 항상 `split_with_sizes`로 직접 보낸다)
`split`/`split_with_sizes`를 별개 키로 깔끔하게 나눴다.

### 4.3 `F.layer_norm`이 실제로 가는 키

`TorchDispatchMode` 로거로 재확인(torch 2.13.0): `F.layer_norm(...)`, `torch.layer_norm(...)`,
`nn.LayerNorm`의 forward가 **전부 `aten.native_layer_norm.default` 하나로만** 기록되고,
`aten.layer_norm.default`는 한 번도 찍히지 않는다. `torch/nn/functional.py:2966`의
`F.layer_norm`은 `torch.layer_norm(input, normalized_shape, weight, bias, eps,
torch.backends.cudnn.enabled)`을 그대로 부르고, `aten::layer_norm`은 `Composite
ImplicitAutograd`라 실제 커널을 갖지 않는다 — C++ 본체가 `std::get<0>(native_layer_norm(...))`
그 자체다. 그래서 `layer_norm`을 `overloads.json`에 원본 스키마로 넣지 않고
(`softmax`가 `methods.json`에서 빠진 것과 정확히 같은 이유 — 파서 키와 실제 디스패치 키가
다르면 표 항목은 절대 구현되지 않을 작업 항목의 이름만 짓는다), `bootstrap.py`
`_install_composites`에 `native_layer_norm.default`를 부르고 첫 번째 결과만 돌려주는
Python 합성으로 심었다. `native_layer_norm` 자체는 진짜 leaf 커널이라(디스패치 키가 그대로)
`overloads.json`에 평범한 표 항목으로 넣었다.

`F.layer_norm`은 이 합성에 닿기 전에 `torch.backends.cudnn.enabled`부터 읽는데, 이건
`torch._C._get_cudnn_enabled()`를 부르는 프로퍼티다. 이 함수가 지난 회차까지 raising 스텁이라
`F.layer_norm`이 그 자리에서 막혔다(`docs/GPT2.md` §5가 정확히 이 지점을 지목했다). 진짜
cuDNN 백엔드는 없어도(`_has_cudnn=False`, 안 바뀜) 게터/세터 자체는 평범한 불리언 상태
셀이면 되므로 `_install_behaviour`에 실제 게터/세터를 심었다(기본값 `True`, 상류 기본값과
동일).

**`nn.LayerNorm(...)`은 여전히 안 된다 — `TensorBase.zero_`가 다른 에이전트 담당이라 손대지
않았다.** `reset_parameters`가 `init.zeros_(self.bias)`를 부르는 자리에서 정확히 그 이름으로
막힌다(실측, `docs/GPT2.md`가 예고한 그대로). `F.layer_norm(x, [C], weight, bias)`처럼 이미
만들어진 파라미터 텐서를 직접 넘기는 경로는 `zero_`를 거치지 않으므로 이번 수정만으로 된다.

### 4.4 스키마 숫자

`verify_schemas.py`: **170/170 → 199/199** (+29 — `overloads.json` 72→90, `methods.json`
98→109). 골든 하네스는 **1781/1781, ops covered=82 그대로**(무회귀 — 스펠링 추가는 커널
집합을 바꾸지 않는다, `_aten_implemented()`가 유일한 출처이기 때문). `--inject-fault
value/shape/dtype` 전부 그대로 exit 1. 호스트 스모크(`pytests/run.sh`) exit 0. 3 타깃
(host / androidNdk arm64-v8a / aarch64-apple-ios) 전부 exit 0.

### 4.5 손대지 않은 것

`flatten repeat chunk narrow permute clamp tril triu flip gather index_select` — 여전히
`aten.rs`에 커널이 없다(`_aten_implemented()`에 없음). `masked_fill`/`squeeze`도 커널·스펠링
모두 지난 회차 상태 그대로 다시 확인만 했다. `torch.softmax`/`F.softmax`(함수 레벨)는
`Tensor.softmax`와 별개로 여전히 스펠링이 없다 — 지난 회차가 메서드 폼만 고쳤고 이번 회차도
지시 범위(14개 목록 + GPT2.md의 넷) 밖이라 손대지 않았다, 실측만 남긴다
(`_VariableFunctions.softmax`가 여전히 "overload resolution has no table entry" 로 거부).

---

## 5. 세 번째 회차 — `gelu`/`gather`/`zero_`, 그리고 `nn.LayerNorm(...)`이 실제로 열리는지

`docs/ARCH.md`가 이 셋의 커널을 넣고 나서(82 → 85개) 남긴 항목을 메우는 회차다. §0이 지목한 대로
`zero_`의 스펠링이 없으면 `nn.LayerNorm`의 `reset_parameters`가 `init.zeros_(self.bias)`에서
막힌다. 그 벽이 실제로 걷히는지가 이 회차의 진짜 판정이다.

### 5.0 세 이름 중 하나는 표(table) 항목이 아니다 — 측정으로 갈랐다

셋을 지시받은 대로 `overloads.json`/`methods.json`에 다 넣기 전에, 각각이 실제로 상류에서
어느 표면으로 도달하는지부터 쟀다(`TorchDispatchMode` 로거, torch 2.13.0):

    x.gather(dim, idx)        -> aten.gather.default   (Tensor 메서드로 존재)
    torch.gather(x, dim, idx) -> aten.gather.default    (_VariableFunctions 로 존재)
    x.zero_()                 -> aten.zero_.default     (Tensor 메서드로 존재)
    torch.zero_(x)            -> aten.zero_.default     (_VariableFunctions 로도 존재 -- 벤더 .pyi 에 실제로 있다)
    F.gelu(x, approximate=..) -> aten.gelu.default

`gather`와 `zero_`은 파서 키와 디스패치 키가 일치하는 평범한 leaf라서 표 항목으로 충분하다.
`gelu`는 다르다 — **`hasattr(torch.Tensor, "gelu")`가 `False`다.** `Tensor.gelu()`라는 메서드
자체가 상류에 없다. `torch/nn/functional.py:2054`가 `gelu = _add_docstr(torch._C._nn.gelu, ...)`이므로
`F.gelu`는 `_C._nn` 서브모듈의 이름이지 `TensorBase`도 `_VariableFunctions`도 아니다 — `linear`/
`silu`가 표가 아니라 `bootstrap.py`의 `_install_nn`에 있는 것과 정확히 같은 이유이고 같은 자리다.
그래서 `gelu`는 `overloads.json`/`methods.json`에 넣지 않았다 — 넣었다면 상류가 절대 보지 않는
파서 키(`aten.gelu.default`를 `Tensor.gelu`나 `torch.gelu`로 노출하는 것 자체가 상류에 없는 이름을
짓는 일)를 만드는 셈이었다.

### 5.1 채운 것

| 이름 | 파일 | 위치 |
|---|---|---|
| `gather` (메서드 `aten::gather`, 함수 `aten::gather`/`.out`) | `methods.json`/`overloads.json` | 평범한 표 항목. 메서드 쪽 벤더 `.pyi`(`__init__.pyi:4767`)에는 `out=`이 없어 `.out` 스키마를 안 넣었고, 함수 쪽 `.pyi`(`_VariableFunctions.pyi:13255`)에는 있어 `.out`을 먼저(관례대로) 넣었다 — 커널이 없어(`aten.gather.out`은 `_aten_implemented()`에 없다) `out=`을 주는 호출은 정확한 이름으로 거부된다(실측) |
| `zero_` (메서드 `aten::zero_`, 함수 `aten::zero_`) | `methods.json`/`overloads.json` | 평범한 표 항목. `torch.ops.aten.zero_.overloads()`가 `['default']` 하나뿐이라 `.out` 걱정이 없다 |
| `gelu` (`_C._nn.gelu`) | `bootstrap.py` `_install_nn` | `silu`/`linear`와 같은 자리의 Python 합성. `dispatch("aten.gelu.default", input, approximate=approximate)`로 넘긴다 |

셋 다 `_C.so`를 재빌드해 `tools/golden/loader.py`로 직접 로드하고 상류 torch 2.13.0과 값을
대조했다 — `gather`(메서드/함수 둘 다), `zero_`(메서드/함수 둘 다), `gelu`(`approximate='none'`/
`'tanh'` 둘 다)가 **전부 비트까지 일치**했다(`torch.tensor(...).tolist()`를 나란히 찍어 비교).
`gelu`의 두 근사식 값은 `docs/ARCH.md` §1이 실측한 상류 수치와도 그대로 일치한다:

    none [-3,-1,-0.5,0,0.5,1,3] -> [-0.00405, -0.158655, -0.154269, 0.0, 0.345731, 0.841345, 2.995950]
    tanh [-3,-1,-0.5,0,0.5,1,3] -> [-0.00364, -0.158808, -0.154286, 0.0, 0.345714, 0.841192, 2.996363]

### 5.2 처음 쓴 `gelu` 합성은 틀렸다 — `approximate`가 위치 인자로도 통과했다

첫 구현은 `def gelu(input, approximate="none")`이었다. `dispatch()`에는 항상 `approximate=`를
키워드로 넘기니 `aten.rs`의 키워드 전용 검사(`args.len() > 1`이면 거부)를 통과할 것이라고
**추론했는데, 틀렸다.** 그 검사는 *`dispatch`가 받는 튜플의 길이*만 보고, 이 합성은 자신의
호출자가 `approximate`를 위치로 줬든 키워드로 줬든 항상 `dispatch(..., approximate=approximate)`
한 개의 위치 인자로 재포장해서 넘긴다 — 그래서 **`F.gelu(x, "tanh")`(위치 인자)가 조용히
통과했다.** 상류는 이걸 `TypeError`로 거부한다(측정, `torch._C._nn.gelu(x, "tanh")` ->
`gelu() takes 1 positional argument but 2 were given`).

`vendor/probe.py --mode strict --target torch`로 벤더 트리 전체를 이 빌드의 `_C` 위에 얹어
`F.gelu(x, "tanh")`를 직접 불러보고서야 잡았다 — 추론이 아니라 실행해서 발견했다. 고친 것은
`aten.rs`가 아니라 **이 합성 자신의 Python 시그니처**다: `def gelu(input, *, approximate="none")`.
`*`가 없으면 `dispatch` 쪽 키워드 전용 검사가 아무리 정확해도 그 앞에서 이미 잘못된 값을
받아들인 뒤이므로 소용이 없다. 고친 뒤 같은 호출이 상류와 같은 자리에서 거부되는 것을
재확인했다:

    torch.nn.functional.gelu(x, "tanh")  ->  TypeError: gelu() takes 1 positional argument but 2 were given
    (상류와 셰임 양쪽 동일 메시지, 재빌드 후 재측정)

### 5.3 `nn.LayerNorm(...)`이 이제 열린다 — `vendor/probe.py`로 끝까지 확인했다

`docs/ARCH.md` §0/§4.3이 지목한 벽은 `reset_parameters`의 `init.zeros_(self.bias)` ->
`TensorBase.zero_`였다. `zero_` 스펠링을 채운 뒤 `TORCH_USE_RTLD_GLOBAL=1`로 벤더 트리 전체를
이 `_C` 위에서 `import torch`시키고(`vendor/probe.py --mode strict --target torch`, exit 0,
`torch.__version__ == 2.13.0`까지 확인), 그 프로세스 안에서 직접 실행했다:

    torch.nn.LayerNorm(8)                 -> 생성자 성공
      .weight.tolist()  == [1.0]*8        (fill_.Scalar 경로, 그대로)
      .bias.tolist()    == [0.0]*8        (zero_ 경로 -- 이번에 열림)
    ln(torch.ones(2, 8))                  -> forward 성공, shape (2, 8)
      결과 전부 0.0  (상수 입력의 LayerNorm은 분산이 0이라 정확히 0이 나오는 것이 맞다)

**생성자와 순전파 둘 다 통과했다.** `docs/ARCH.md` §0이 예고한 두 겹(`_get_cudnn_enabled`는
이미 해결, `zero_`가 남은 것) 뒤에 **셋째 겹은 없었다** — `zero_` 스펠링을 채우는 것만으로
`nn.LayerNorm(...)`이 끝까지 갔다.

주의할 것 하나: `repr(ln)`이나 `print(ln.weight)`(값이 아니라 텐서 객체 자체를 출력하는 경로)는
여전히 `torch._C._functorch.is_functorch_wrapped_tensor`에서 막힌다 — §4.1이 이미 기록한,
이 작업과 무관한 기존 벽이다(`_tensor_str.py`가 모든 텐서 repr에서 그 이름을 부른다). `.tolist()`로
값만 읽으면 이 벽을 타지 않는다. `torch.randn`도 여전히 커널이 없어 거부된다(스펠링 문제가 아니라
`aten.rs`에 `randn` 계열 자체가 없음, §2에서 이미 남긴 목록 그대로) — 그래서 위 forward 검증은
`torch.ones`로 입력을 만들었다.

### 5.4 11개 커널 없는 목록 재확인 — 새로 열린 것 없음

`docs/ARCH.md`가 이번에 커널을 넣은 것은 `gelu`/`gather`/`zero_` 셋뿐이다. 남은 11개
(`flatten repeat chunk narrow permute clamp tril triu flip index_select`)를 이번 빌드의
`_aten_implemented()`에 대고 다시 확인했다 — **전부 여전히 커널이 없다**(`permute`는 다른
에이전트가 지금 작업 중이라는 지시를 받아 건드리지 않았다). 커널이 없는 스펠링은 정확한 이름으로
거부되는 것 말고 할 수 있는 일이 없으므로, 이 11개는 그대로 다음 회차로 넘긴다.

### 5.5 숫자

`verify_schemas.py`: **199/199 → 204/204** (+5 — `overloads.json` 90→93 [`gather`+`.out`,
`zero_`], `methods.json` 109→111 [`gather`, `zero_`]; `gelu`는 `_install_nn` 합성이라 이 표에
없다). 골든 하네스는 **1934/1934, ops covered=85 그대로**(무회귀 — 스펠링은 `_aten_implemented()`가
답하는 커널 집합을 바꾸지 않는다). `--inject-fault value/shape/dtype` 전부 그대로 exit 1. 호스트
스모크(`pytests/run.sh`, `test_shim.py` 65개 + `compare.py --self-test`) exit 0. 3 타깃
(host / androidNdk arm64-v8a / aarch64-apple-ios) 전부 exit 0.

### 5.6 손대지 않은 것 / 이 회차 밖

`aten.rs`, `tools/golden/cases.py`, `tools/golden/compare.py`, `rust/torch_c/pytests/test_shim.py`는
이번 회차의 파일 범위 밖이라 한 줄도 고치지 않았다 — 커널 추가도 하지 않았다. `nn.LayerNorm`에
대한 회귀 테스트를 `test_shim.py`에 박아 두는 것은 다음 회차의 작업 항목이다(§4가 남긴 것과 같은
이유 — 이번에 손으로 확인한 것을 자동화하지 못했다).
