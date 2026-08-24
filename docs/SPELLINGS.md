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
