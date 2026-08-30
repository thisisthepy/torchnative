# 분해 패스 — 캡처한 ATen 을 Core ATen 으로 낮춘다

## 0. 이 문서가 답하는 것

- **분해 패스가 섰는가.** 섰습니다. `torchnative.export.decompose` 이고, 규칙은 한 줄도
  여기서 쓰지 않았습니다 — 상류의 `torch/_decomp` 를 *실행*합니다 (§1).
- **`aten.t.default` 이 낮아지는가.** **낮아집니다.** docs/CAPTURE.md §5 가 "가장 작은 예제에서
  이미 일어나는 일" 로 지목한 그 op 이고, 이제 `aten.permute.default` 가 됩니다. **그 예제
  전체**(`nn.Sequential(Linear, ReLU, Linear)`)가 Core ATen 으로 완전히 내려가고, 결과는
  상류가 같은 모듈에 내놓는 것과 op 단위로 같습니다 (§5).
- **몇 개가 낮아지는가.** **37 개 중 9 개.** 이전 회차의 32 개 중 4 개에서 올랐고, 무엇이
  올렸는지와 남은 27 개가 왜 남았는지는 §4 가 전부 이름으로 적습니다.
- **무엇이 막고 있었는가.** 두 가지였고 둘 다 열렸습니다 — `_jit_get_operation` 이 모든 패킷의
  오버로드를 `["default"]` 로 답하던 것(§3.1)과 `_dispatch_get_registrations_for_dispatch_key`
  가 없던 것(§3.2). 세 번째는 이 작업이 찾은 것입니다: **패스가 규칙을 기록된 kwarg 이름으로
  부르고 있었습니다**(§3.4).
- **분해 후 값이 달라지는가.** §6. 이번에 열린 규칙 중 **산술을 하는 것 두 개**(`matmul`,
  `isin`)까지 포함해 여전히 비트 단위로 일치합니다.
- **패스가 거절하는 세 벽 중 세 번째에 예제가 있는가.** **없습니다** (2026-08-30). 마지막
  예제였던 `aten.baddbmm.default` 의 dtype 승격이 고쳐졌고(§7.2, 원인은 docs/BIND.md §9),
  훑어본 결과 그 자리를 물려받는 op 이 없습니다. 벽은 그대로이고 그것을 건드리는 op 이
  없는 것이며, "없다" 를 확인한 방법과 그 확인이 실패할 수 있다는 것은 §7.2.1 입니다.
- **ExecuTorch Edge 까지 얼마나 남았는가.** §8. 분해는 필요조건이고 충분조건이 아닙니다.

바뀐 파일: `rust/torch_c/src/bootstrap.py`, `rust/torch_c/src/overloads.json`,
`torchnative/src/main/torchnative/export/decompose.py`,
`rust/torch_c/pytests/test_shim.py`, `rust/torch_c/pytests/verify_schemas.py`,
그리고 §4 의 표를 만드는 `rust/torch_c/pytests/decomp_sweep.py`.

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

**이번 회차도 같은 규칙을 지켰습니다.** 열린 5 개는 규칙을 새로 써서 열린 것이 아니라,
상류의 규칙이 *찾아지고* *실행되게* 만들어서 열렸습니다. §3 의 넷은 전부 배선 결함입니다.

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
않습니다.**

---

## 2. Core ATen 을 무엇으로 판정하는가

판정은 벤더 트리의 `native_functions.yaml` 을 읽습니다. **그 파일이 상류가 `torch.Tag.core` 를
생성하는 원본입니다.**

| | 개수 |
|---|---|
| yaml `tags: core` | **193** |
| 상류 `torch.Tag.core` (설치된 torch 2.13.0 실측) | 189 |
| 차이 | 4 — `adaptive_avg_pool1d.default` · `avg_pool1d.default` · `resize_.default` · `sym_is_contiguous.default` |

넷은 파일에 core 로 적혀 있는데 상류의 `OpOverload.tags` 로는 나타나지 않습니다. **넓은 쪽을
택했습니다** — 이 차이는 패스가 노드를 *받아들이게* 만들 수는 있어도 거절하게 만들 수는 없습니다.

### 이전 판본의 이유는 사라졌고, 결정은 남았습니다

이전 판본은 파일을 읽는 이유를 이렇게 적었습니다: **이 빌드에서는 `OpOverload.tags` 가 전부
비어 있으므로 태그로 판정하면 "아무것도 core 가 아니다" 가 나온다.** 그것은 사실이었고,
이번에 고쳤습니다 (§3.3). 지금은:

```
구현된 op                       129
그중 torch.Tag.core 가 붙은 것   77
파일이 core 라고 부르는 것        77      ← 같음
2584 개 전체에 대해 상류와 대조   2584/2584
```

**그래도 `core_ops()` 는 계속 파일을 읽습니다.** 위 4 개 때문입니다 — 태그 기반 판정은 상류에서도
좁은 쪽이고, 좁은 쪽은 이 패스를 더 많이 거절하게 만듭니다. 두 판정은
`test_core_ops_and_op_tags_agree` 가 **서로 대조**합니다. 한 출처를 두 번 읽으면 자기 자신과
일치할 뿐이므로, 두 스캔이 두 소비자에게 도달하는 것을 확인하는 것이 검사의 내용입니다.

읽는 방법은 YAML 파서가 아니라 줄 스캐너입니다. `pyyaml` 이 이 배포판의 선언된 의존성이 아니기
때문입니다 — 여기서 `import yaml` 하면 올바르게 설치된 휠에서 이 모듈이 임포트 실패합니다. 대신
`test_decompose_reads_core_aten_out_of_the_vendored_tree` 가 **스캔 결과를 진짜 YAML 파스와
diff** 합니다.

---

## 3. 무엇이 막고 있었는가 — 네 개의 배선 결함

### 3.1 `_jit_get_operation` 이 모든 패킷의 오버로드로 `["default"]` 를 답했다

`torch/_decomp/__init__.py:82` 는 패킷 단위 등록(`@register_decomposition(aten.transpose)`)을
`packet.op_overloads()` 로 펼칩니다. 그 목록이 `["default"]` 이면 상류의 `transpose` 규칙이
**`aten.transpose.default`** 에 내려앉는데, 그런 오버로드는 어느 torch 에도 없습니다. 규칙은
트리 안에 있었고, **아무도 찾을 수 없는 이름 아래** 있었습니다.

| | 레지스트리 항목 | 그중 `.default` |
|---|---:|---:|
| 상류 | 1097 | 456 |
| 고치기 전 | 592 | 525 |
| **고친 뒤** | **1004** | **461** |

고친 방법: 오버로드 이름을 `native_functions.yaml` 에서 읽습니다. 세 출처의 합집합이고
(파일 · `Library.define()` 이 만든 스키마 · `overloads.json`/`methods.json`), 셋 다 침묵하면
`["default"]` 로 떨어져 레지스트리를 열어 둡니다 (docs/SCHEMA.md §12). **거꾸로는 하지
않습니다** — 오버로드를 아는 패킷에 `default` 를 덧붙이지 않습니다. 그것이 고치려는 결함입니다.

`verify_schemas.py` 의 `check_overload_names` 가 구현된 op 의 패킷 99 개를 상류와 대조합니다
(99/99). 방향이 둘이고 **같은 주장이 아닙니다**:

- 상류에 없는 오버로드를 여기서 내놓는 것 → **실패.** 없는 키에 규칙을 얹는 것이 곧 결함입니다.
- 상류에 있고 여기 없는 것 → 대개 **바람직합니다.** 175 개이고 전부 torchgen 이 생성하는
  `.out` 변형과 TorchScript 의 숫자 빌트인(`aten::sub.float_int`)입니다. 상류 자신이
  `_dispatch_has_kernel` 로 후자를 버립니다. 그중 규칙을 가진 18 개는 전부 `.out` 변형이고,
  이 빌드는 `.out` 키를 하나도 구현하지 않으므로 **어떤 기록에도 들어올 수 없습니다** —
  그 전제까지 같은 검사가 확인합니다.

**이 결함은 저장소 안에서는 보이지 않았습니다.** 스키마 검사들은 `(이름, 오버로드)` 하나씩을
묻고 *목록*이 무엇이든 옳으며, 저장소 안 테스트에는 대조할 상류가 없습니다. 보이는 증상은
전혀 다른 자리의 숫자 둘(592/525)뿐이었습니다.

### 3.2 `_dispatch_get_registrations_for_dispatch_key` 가 없었다

`core_aten_decompositions()` 는 `CustomDecompTable` 이고, 그 생성자가
`CompositeImplicitAutograd` 등록을 전부 열거합니다. 이 `_C` 에는 열거할 C++ 디스패처가
없어서 예외를 던졌고, 패스는 더 작은 표로 떨어졌습니다.

같은 파일이 답을 가지고 있습니다. `torchgen/model.py:872` 가 **`dispatch:` 블록이 없고
structured 도 structured_delegate 도 아닌 모든 항목**에 CIA 를 부여합니다 — 목록의 대부분이
거기서 나오므로, 글자 `CompositeImplicitAutograd` 만 찾는 스캔은 3 분의 1 만 찾습니다.

```
파일에서 파생한 aten CIA 이름     743
상류 디스패처가 답하는 것          744
여기 있고 상류에 없는 것             0
상류에 있고 파일에 없는 것           1   aten::get_gradients (TorchScript 빌트인)
```

**백엔드 키는 답하지 않고 이름을 대고 거절합니다.** 같은 파일이 `CPU` 에 무엇이 등록되는지도
적어 두었지만, 그것으로 답하면 이 빌드가 갖고 있지 않은 커널 1500 개를 주장하게 됩니다.
답하는 것은 파일이 권위를 갖는 **별칭 키 넷**뿐입니다.

이 목록은 *구체화*에만 쓰입니다 — `_materialize_cpp_cia_ops` 가 이름을 훑어
`torch.ops.aten.<name>.<overload>` 를 존재하게 만들 뿐이고, 여기서 진짜 CIA 인지 판정하는 것은
`op.py_kernels` 를 보는 `_is_cia_op` 입니다. 그러므로 넉넉한 목록이 무언가를 몰래 들여올 수는
없고, 무언가를 빠뜨릴 수만 있습니다.

| | 표 크기 |
|---|---:|
| 상류 `core_aten_decompositions()` | 940 |
| 상류 `_core_aten_decompositions_post_autograd()` | 435 |
| 고치기 전 (여기) | 224 |
| **고친 뒤 (여기)** | **414** |

414 와 940 의 차이는 §7 이 셉니다.

### 3.3 `OpOverload.tags` 가 전부 `[]` 였다

두 곳이 이것을 읽습니다. 하나는 §2 가 이미 잰 Core ATen 판정이고, 다른 하나는
`torch/_decomp/__init__.py:57` 의 `maybe_aliasing_or_mutating` 입니다 — 그것이 항상 거짓이어서
`_collect_all_valid_cia_ops` 가 `aten.dropout.default` 와 `aten.unsafe_chunk.default` 를
수집했습니다. 상류는 정확히 그 태그로 둘을 제외합니다.

태그도 같은 파일에서 읽습니다. 옮겨 적는 것이 아니라 **torchgen 의 규칙을 따라가는 것**이고,
그 규칙에는 파일에 안 적힌 것이 셋 있습니다 (`torchgen/model.py:756-765`):

```
pt2_compliant_tag   모든 aten 항목에 무조건
out                 out 인자를 가진 항목   (이름이 `out` 인 인자가 아니라 "변경 가능한 kwarg-only 인자")
inplace             이름이 `_` 로 끝나는 항목 (그리고 `__iand__` 류 — `_` 하나로 안 끝납니다)
```

셋을 빼면 **118 개 중 0 개**가 상류와 일치합니다. `pt2_compliant_tag` 하나가 전부에 붙기
때문입니다.

**여기서 결함이 하나 더 나왔습니다.** 셋을 넣고도 2584 개 중 3 개가 남았습니다 —
`sinh.out` · `rsub.Scalar` · `tanh_backward.default` 가 `pointwise` 를 잃었습니다. 원인은
파일 형식이었습니다: 그 셋은 항목과 자기 `tags:` 줄 사이에 **0 번째 칸에서 시작하는 주석**이
끼어 있고, 스캐너가 그것을 항목의 끝으로 읽었습니다. YAML 은 주석을 보지 않습니다.
`decompose.py` 의 `_scan_core_tags` 도 같은 결함을 가지고 있었습니다(`core` 태그는 하나도
그 뒤에 없어서 결과는 같았습니다). 둘 다 고쳤습니다.

```
OpOverload.tags, 파일이 선언한 2584 개 전부:   2584/2584 상류와 일치
```

### 3.4 패스가 규칙을 기록된 kwarg 이름으로 불렀다

§3.1 을 고치고 `overloads.json` 에 `transpose`/`permute` 를 넣었더니 `aten.t.default` 이
**둘째 회차**에서 죽었습니다:

```
TypeError: transpose() got an unexpected keyword argument 'self'
```

기록기는 shim 의 `torch.<fn>` 해석기가 넘긴 것을 그대로 적습니다. 그 해석기는 aten 스키마에
바인딩하고 **스키마의 이름**으로 넘기므로, `torch.transpose(x, 0, 1)` 은
`aten.transpose.int(self=%0, dim0=0, dim1=1)` 로 기록됩니다. 그런데 상류의 규칙은
`torch._refs.transpose(a, dim0, dim1)` 이고 첫 인자 이름이 `a` 입니다.

**상류에는 이 문제가 없습니다** — C++ 디스패처는 커널에 인자를 위치로 넘기고, 분해 함수의
파라미터 이름은 그 함수의 사정입니다. 그래서 패스가 호출을 그 모양으로 되돌립니다:
스키마가 `*` 앞에 선언한 인자는 전부 위치로, 진짜 kwarg-only 인 것만 이름으로. 스키마 텍스트가
없거나 스키마가 모르는 kwarg 이 있으면 **기록 그대로 넘깁니다** — 그러면 이 함수가 지어낸
재작성이 아니라 규칙 자신의 오류로 실패합니다.

이것은 §3.1 · §3.2 와 성격이 다릅니다. 저 둘은 shim 의 결함이고, **이것은 이 패스의
결함**이었습니다. 앞의 둘을 고치기 전에는 도달할 수 없어서 보이지 않았습니다.

### 3.5 `overloads.json` 의 구멍 셋

§3.4 를 고칠 수 있게 만든 것들입니다. 이것들은 **모델 경로가 부르는 이름이 아니라,
분해 규칙이 도는 동안 *상류 자신의 규칙*이 부르는 이름**입니다:

```
torch.transpose   aten.t.default 의 규칙이 부른다      → aten::transpose.int
torch.permute     aten.transpose.int 의 규칙이 부른다  → aten::permute
torch.sub         aten.rsub.Scalar 의 규칙이 부른다    → aten::sub.{out,Tensor,Scalar}
```

셋 다 **이 빌드가 이미 구현한 aten 키**를 가리키므로, 표 항목이 수정의 전부입니다 — 커널을
지어내지 않았습니다. `verify_schemas.py` 가 스키마 문자열을 상류와 대조합니다 (131/131).

---

## 4. 실측 — 무엇이 덮이고 무엇이 안 덮이는가

`rust/torch_c/pytests/decomp_sweep.py` 가 이 표를 만듭니다.

### 모집단

| | 개수 |
|---|---:|
| `_aten_all_implemented()` | **129** |
| └ Core ATen | 77 |
| └ Core ATen 밖 | **52** |
| &nbsp;&nbsp;&nbsp;└ 캡처가 애초에 거절하는 것 (변이 12 · 난수 3) | 15 |
| &nbsp;&nbsp;&nbsp;└ **트레이스에 나타날 수 있는 non-core op** | **37** |

변이·난수 15 개를 뺀 것은 편의가 아닙니다 — docs/CAPTURE.md §4 가 그것들을 이름을 대고
거절하므로 **어떤 트레이스에도 들어올 수 없습니다.**

> **이전 회차의 32 와 지금의 37 은 같은 모집단이 아닙니다.** 그 사이에 구현된 op 이
> 120 → 129 로 늘었고, 늘어난 것 중 다섯(`masked_select` · `min.default` · `unbind.int` ·
> `view.dtype` · `where.ScalarOther`)이 non-core 입니다. **이 회차 시작 시점의 판정은
> 37 개 중 5 개**였습니다(`sum.default` 가 커널 수정으로 §7.1 에서 이미 열려 있었음).
> 이 작업이 올린 것은 **5 → 9** 입니다.

### 판정

| 판정 | 이번 | 이 회차 시작 시점 |
|---|---:|---:|
| **Core ATen 으로 낮춰짐** | **9** | 5 |
| 규칙을 찾았고, 이 빌드에서 실행되지 않음 | 15 | 13 |
| 규칙이 상류에 있는데 여기서 **닿을 수 없음** | **2** | 9 |
| 규칙이 어디에도 없음 (상류 포함) | 9 | 8 |
| 규칙이 기록과 다른 결과를 냄 (→ 거절) | 1 | 1 |
| 캡처가 구간 출력으로 거절 | 1 | 1 |

**세 번째 줄이 이 작업의 숫자입니다: 9 → 2.** 닿을 수 없던 9 개 중 6 개는 §3.1 의 패킷 붕괴
(`isin` · `transpose.int` · `rsub.Scalar` · `masked_fill.Scalar` · `masked_fill.Tensor` ·
`unbind.int`), 3 개는 §3.2 의 CIA 절반(`matmul` · `max.other` · `where.ScalarOther`)이었습니다.
남은 2 개는 상류 규칙 자체가 "C++ CIA 커널을 불러라" 인 것들입니다.

**두 번째 줄이 늘어난 것은 후퇴가 아닙니다.** 규칙이 *찾아졌기* 때문에 벽이 한 칸 아래로
내려간 것이고, 새 벽은 전부 이름이 있습니다 (아래 표). 네 번째 줄이 하나 는 것도 같은 이유로,
`masked_fill.Tensor` 의 규칙이 이제 돌아서 `aten.contiguous.default` 를 내는데 그것에 규칙이
없습니다.

#### 낮춰짐 (9)

```
aten.t.default         →  aten.permute.default              ← CAPTURE.md §5 가 지목한 op
aten.transpose.int     →  aten.permute.default
aten.matmul.default    →  aten.mm.default
aten.isin.Tensor_Tensor→  aten.view.default + eq.Tensor + any.dims
aten.sum.default       →  aten.sum.dim_IntList
aten._unsafe_view.default → aten.view.default
aten.detach.default    →  aten.alias.default
aten.split.Tensor      →  aten.split_with_sizes.default
aten.stack.default     →  aten.cat.default + aten.view.default
```

앞의 넷이 이 작업이 연 것입니다. `t` · `transpose` 는 §3.1 · §3.4 · §3.5 셋이 동시에 걸려
있었고, `matmul` 은 §3.2 (CIA 절반), `isin` 은 §3.1 (패킷 붕괴) 이었습니다.

#### 규칙을 찾았고 실행되지 않음 (15) — 벽이 shim 쪽에 있습니다

| op | 무엇에 막혔나 | 어느 파일 |
|---|---|---|
| `aten.ones.default` · `aten.zeros.default` · `aten.new_ones.default` | `aten.full.default` 의 `layout` 인자 미구현 | `aten.rs` |
| `aten.arange.default` · `aten.arange.start` | `aten.arange.start_step` 의 `layout` 인자 미구현 | `aten.rs` |
| `aten.empty_like.default` | `TensorBase.stride` 미구현 | `tensor.rs` |
| `aten.floor_divide.default` | `aten.div.Tensor_mode` 커널 없음 | `aten.rs` |
| `aten.masked_fill.Scalar` | `aten.where.ScalarSelf` 커널 없음 | `aten.rs` |
| `aten._scaled_dot_product_flash_attention_for_cpu.default` | `aten._scaled_dot_product_attention_math` 커널 없음 | `aten.rs` |
| `aten.silu.default` | `torch.sigmoid` — `overloads.json` 에 항목 없음, **그리고 커널도 없음** | 둘 다 |
| `aten._safe_softmax.default` | `torch.softmax` — 같음. `softmax.int` 는 상류에서 CIA 이므로 자리는 `_install_composites` 입니다 | `bootstrap.py` |
| `aten.zeros_like.default` | `torch.full_like` — 같음 | 둘 다 |
| `aten.unbind.int` | `torch.tensor_split` — 같음 | 둘 다 |
| `aten.softplus.default` | `torch._C._dynamo.eval_frame.set_eval_frame` 미구현 | `bootstrap.py` |
| `aten.rsub.Scalar` | `torch.sub(스칼라, 텐서)` 를 해석기가 바인딩하지 못함 (§7.3) | `bootstrap.py` |

**`layout` 다섯 개가 여전히 가장 값싼 다음 걸음입니다** — 하나의 원인이 다섯을 막고 있고,
그 원인은 `aten.rs` 안의 인자 하나입니다.

`transpose`/`permute`/`sub` 와 달리 `sigmoid` · `full_like` · `tensor_split` 은
`overloads.json` 항목만으로 열리지 않습니다. **그 aten 키에 커널이 없기 때문입니다** —
표만 넣으면 에러 메시지가 "표에 없다" 에서 "커널이 없다" 로 바뀔 뿐입니다. 그래서 넣지
않았습니다.

#### 규칙이 상류에도 없음 (9)

```
aten.contiguous.default     aten.max.default            aten.reshape.default
aten.histc.default          aten.min.default            aten.view.dtype
aten.lift_fresh.default     aten.masked_select.default  aten.masked_fill.Tensor
```

`lift_fresh.default` 와 `max.default` 는 docs/CORE_ATEN.md §0 이 이미 "분해로도 안 풀림" 으로
재 둔 둘입니다. 이 열은 직접 구현하거나 상류에 분해를 넣는 것 외에 길이 없습니다.

`aten.masked_fill.Tensor` 는 이 열에 있지만 **자기 규칙 때문이 아닙니다.** 규칙이 있고 돌고,
그것이 `aten.contiguous.default` 를 내는데 그것에 규칙이 없습니다. 즉 `contiguous` 하나가
둘을 막습니다.

#### 규칙이 상류에 있는데 여기서 닿을 수 없음 (2)

```
aten.max.other          상류 규칙이 functools.partial(_special_op_to_decompose_cia)
aten.where.ScalarOther  같음
```

상류의 그 항목이 하는 일은 **C++ 의 CIA 커널을 호출하는 것**입니다. 파이썬 트리만 벤더링한 이
빌드에는 그 커널이 없으므로, §3.2 를 고쳐도 이 둘은 열리지 않습니다 — `_is_cia_op` 가
`py_kernels` 를 보고 애초에 수집하지 않습니다. **이것이 940 과 414 의 차이를 만드는 벽이고,
§7 이 그 크기를 셉니다.**

#### 캡처가 거절 (1)

`aten.is_floating_point.default` 는 텐서가 아닌 결과를 냅니다. 구간의 *출력*으로는 캡처가 이름을
대고 거절하고, 구간 *안의* 노드로는 기록될 수 있습니다.

#### 규칙이 기록과 다른 결과를 냄 (0) — 2026-08-30 기준

`aten.baddbmm.default` 이 여기 있었고, 지금은 없습니다. dtype 항등성이 고쳐져 낮아집니다
(§7.2). **이 칸을 채우는 op 이 지금은 하나도 없습니다** — 훑어서 확인한 결과이고, 어떻게
확인했는지와 그 확인이 실패할 수 있다는 것을 §7.2.1 이 적습니다.

---

## 5. `aten.t.default`, 그리고 docs/CAPTURE.md §5 의 예제 전체

CAPTURE.md §5 는 이 op 을 "가장 작은 예제에서 이미 일어나는 일" 로 지목했습니다. 이제 그
예제 전체가 내려갑니다:

```python
m = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))

기록      aten.t.default  aten.addmm.default  aten.relu.default  aten.t.default  aten.addmm.default
분해 후   aten.permute.default  aten.addmm.default  aten.relu.default  aten.permute.default  aten.addmm.default

Core ATen 밖:  분해 전 [aten.t.default]  →  분해 후 []
```

**상류에게 같은 모듈을 물었을 때의 답과 op 단위로 같습니다** (이전 판본의 §7 이 인용해 둔 것):

```python
ep = torch.export.export(m, (x,))
ep.run_decompositions(core_aten_decompositions())
# → aten.permute.default, aten.addmm.default, aten.relu.default,
#   aten.permute.default, aten.addmm.default
```

두 회차가 필요합니다: `t` → `transpose.int` → `permute`. 둘째 회차가 §3.1 · §3.4 두 벽에
동시에 걸려 있었습니다.

`test_the_smallest_model_lowers_the_way_upstream_lowers_it` 이 이 목록을 순서까지 못박고,
기록에 쓰이지 않은 입력 셋으로 재생해 eager 와 대조합니다.

---

## 6. 판정 — 분해 전후로 재생 결과가 일치하는가

**일치합니다. 비트 단위로.** 이전 회차의 경고가 여기서 시험을 받았습니다.

이전 판본은 이렇게 적었습니다 — 그때 열린 세 치환(`stack`→`cat`+`view`,
`split`→`split_with_sizes`, `_unsafe_view`→`view`)은 전부 **데이터 이동이지 산술이 아니므로**,
비트 동일은 "분해는 비트 동일하다" 의 증거가 아니라 "이 분해들은 산술을 건드리지 않는다" 의
증거라고. **그러니 산술을 하는 규칙이 열리는 날 다시 재라고.**

이번에 둘이 열렸습니다:

| 규칙 | 무엇이 되는가 | 산술인가 | 결과 |
|---|---|---|---|
| `aten.matmul.default` | `aten.mm.default` | **예** — 곱셈 누산 | 완전 일치 |
| `aten.isin.Tensor_Tensor` | `view` + `eq.Tensor` + `any.dims` | **예** — 비교와 축약 | 완전 일치 |
| `aten.t.default` | `aten.permute.default` | 아니오 | 완전 일치 |

`matmul` → `mm` 이 비트 동일한 것에는 구조적 이유가 있습니다: 이 shim 에서 두 이름이 같은
커널에 닿습니다. 그러므로 **이것도 "분해 일반" 의 증거가 아닙니다.** 진짜 시험은 산술 순서를
바꾸는 규칙(`silu`, `_safe_softmax`, `baddbmm`)이고, 그것들은 §4 의 두 번째 열에 있습니다.
그때 차이가 나오면 **허용치를 넓히는 것이 아니라 크기를 재서 여기에 적어야 합니다**
(docs/DEVICE.md §5 — 허용치보다 이름 붙인 예외).

셋 중 `baddbmm` 이 2026-08-30 에 열렸습니다 (§7.2). `bmm` + `add.Tensor` 가 되고, 재생 결과가
eager 와 값·shape·dtype 모두 일치합니다 — 이번에도 커널이 겹쳐서가 아니라, 분해가 실제로 같은
함수이기 때문입니다. `silu` 와 `_safe_softmax` 는 여전히 §4 의 두 번째 열에 있습니다.

§5 의 프로그램과 이전 판본의 7→8 노드 프로그램 둘 다, 기록에 쓰이지 않은 입력 셋
(`× 0.5` · `× -2.0` · `× 7.25`)에서 낮춘 트레이스 · 기록된 트레이스 · eager 가 전부 일치합니다.

가드는 낮추기를 통과해도 가드로 남습니다: 분해된 트레이스에 `ones(3,4)` 를 주면 `[2, 4]` 를
이름에 담아 거절합니다. **낮추기는 트레이스를 일반화하지 않습니다.**

---

## 7. 이 패스가 찾은 것 — 미구현이 아니라 결함

패스는 분해 결과의 shape·dtype·device 를 **기록된 것과 대조**합니다. 분해는 같은 함수여야
하므로, 다르면 무언가가 틀린 것입니다.

### 7.1 `aten.sum.dim_IntList` 의 빈 `dim` 목록 — **고쳐짐** (2026-08-28)

`rust/torch_c/src/aten.rs::sum_or_mean` 이 빈 `dim` 목록을 모든 축으로 확장하도록 고쳐졌습니다.
그 전에는 입력을 그대로 돌려줬고, `aten.sum.default` 의 상류 규칙이
`sum(x, dim=[], dtype=None)` 을 만들어 그 경로의 첫 호출자가 되었습니다. `mean.dim` 이 같은
커널을 공유해 같은 수정으로 함께 고쳐졌습니다.

### 7.2 `aten.baddbmm.default` 분해의 dtype 승격 — **고쳐짐** (2026-08-30)

```
                       입력           분해 결과 dtype
상류  baddbmm(f32, f32, f32)          float32
전    baddbmm(f32, f32, f32)          float64
지금  baddbmm(f32, f32, f32)          float32   → aten.bmm.default, aten.add.Tensor 로 낮아집니다
```

**원인은 스칼라 승격 규칙이 아니라 `TensorBase.dtype` 의 반환값이었습니다.** 이전 판본은 "아직
어느 op 에서 나오는지 좁히지 않았습니다" 라고 적어 두었는데, docs/BIND.md §9 가 그것을 찾아냈고
`b8c3ea1` 이 고쳤습니다.

`tensor.rs` 의 getter 가 읽을 때마다 새 `PyDtype` 를 만들었습니다. `__eq__` 는 태그를 비교하므로
`t.dtype == torch.float32` 는 언제나 옳았지만, **`t.dtype is torch.float32` 는 어떤 텐서에
대해서도, 어떤 dtype 에 대해서도, 매번 `False`** 였습니다 — `t.dtype is t.dtype` 조차
`False` 였습니다.

상류의 `torch/_prims_common/__init__.py::get_higher_dtype` 는 이렇게 시작합니다:

```python
    a, b = _extract_dtype(a), _extract_dtype(b)
    if a is b:
        return a
```

이 한 줄이 `ordered_datatypes` 표로 떨어지지 않게 막는 가드입니다. `baddbmm` 의 상류 규칙에는
`@pw_cast_for_opmath` 가 붙어 있어 `elementwise_dtypes` → `get_higher_dtype` 를 지나가는데,
**float32 두 개가 `is` 로 같지 않으니 가드를 지나쳐 표로 떨어졌고, 거기서 float64 가 나왔습니다.**
분해 결과가 기록과 다른 dtype 을 내놓았고 패스가 그것을 잡아 거절했습니다 — 정확히 옳은
동작이었고, 진짜 결함이었습니다.

dtype 이 이제 인턴되므로 규칙과 기록이 일치하고, 트레이스는 거절되는 대신 낮아집니다.
`test_decompose_lowers_baddbmm_default_now_that_the_dtype_is_a_singleton` 이 세 고리를 전부
못박습니다: 항등성 계약 자체, 낮아진 op 목록, 그리고 재생 결과가 eager 와 float32 로 같다는 것.

**그리고 이것이 §7 세 번째 벽의 마지막 예제였습니다.** 아래를 보십시오.

### 7.2.1 벽 3 에는 지금 예제가 없습니다 — 그리고 그것은 발견이지 누락이 아닙니다

`decompose` 의 세 벽은 그대로입니다:

1. 규칙이 아예 없음 — `aten.reshape.default`
2. 규칙이 있고, 실행하다 이 shim 에 없는 것을 만남 — `aten.zeros_like.default`
3. 규칙이 있고, 실행되고, **기록과 다른 결과를 냄** — **지금 이것을 하는 op 이 없습니다**

벽 3 자체가 사라진 것이 아닙니다. `decompose` 는 여전히 모든 분해 결과의 meta 를 기록과
대조하고 여전히 불일치에 거절합니다. 사라진 것은 **그것을 건드리는 op** 입니다.

없다는 것은 찾아보고 내린 결론입니다. 두 가지로 훑었습니다:

- `pytests/decomp_sweep.py` — 구현된 비 Core op 전체(모집단 38 개). 결과: LOWERED 10,
  REFUSED 26(전부 벽 1 또는 벽 2), CAPTURE_RAISED 1, NO_CASE 2. **DISAGREES 0.**
- 낮아지는 그 10 개를 dtype 8 종 · 여러 shape · `beta`/`alpha` 조합 · 다중 op 트레이스로 넓혀
  **188 개 트레이스**를 분해. **DISAGREES 0.**

`aten.sum.default` 가 §7.1 로, `aten.t.default` 가 §3 으로 옮겨간 것과 같은 일이 세 번째로
일어났고, 이번에는 자리를 물려받을 op 이 없었습니다. **없는 것을 지어내지 않았습니다** — 꾸며낸
불일치는 이 저장소가 반복해서 잡아내는 "실패할 수 없는 검사" 그 자체입니다.

**없다고 단언하는 검사는 그 자체로 실패할 수 있어야 합니다.** 그래서
`test_decompose_refuses_by_name_what_it_cannot_lower` 는 인구조사와 **양성 대조**를 함께
단언합니다: `_meta_matches` 를 강제로 불일치시키면 같은 프로브가 `DISAGREES` 를 보고하고,
되돌리면 `LOWERED` 로 돌아옵니다. 이 대조가 없으면 "아무 op 도 불일치하지 않는다" 는
**대조 자체가 조용히 멈춰도 똑같이 통과**합니다 — CLAUDE.md §5.5 가 세 번 잡아낸 그 실패
양식입니다.

벽 3 에 새 예제가 생기면 그 단언이 이름을 대고 빨개지고, 그때가 이 절을 되돌릴 때입니다.

### 7.3 스칼라가 앞에 오는 호출 — 새로 드러남

`aten.rsub.Scalar` 의 상류 규칙은 `torch.sub(other, self, alpha=alpha)` 이고, `other` 가
스칼라입니다. 상류의 `PythonArgParser` 는 그것을 0 차원 텐서로 감싸 `sub.Tensor` 로 보냅니다.
이 shim 의 해석기는 그렇게 하지 않아 거절합니다:

```
TypeError: torch.sub(): no matching overload in torch._C shim for (float, Tensor, alpha=int)
```

`overloads.json` 에 `sub` 를 넣은 뒤에야 보이게 된 벽입니다. **표의 문제가 아니라 해석기의
문제**이고, `torch.add(1.0, t)` 같은 모든 호출에 같은 벽이 있습니다. 이 회차에서는 고치지
않았습니다 — 해석기의 바인딩 규칙을 건드리는 변경이라 영향 범위가 이 패스보다 넓습니다.

### 7.4 파일의 0 번째 칸 주석 — §3.3

세 개의 `pointwise` 태그를 잃고 있었습니다. 스캐너 둘이 같은 결함을 가지고 있었습니다.

---

## 8. ExecuTorch Edge 까지 남은 거리

**분해는 필요조건이고 충분조건이 아닙니다.**

### 표가 아직 940 이 아니다

| | 표 | 여기 |
|---|---:|---:|
| `_core_aten_decompositions_post_autograd()` | 435 | **383** |
| `_collect_all_valid_cia_ops()` 로 더해지는 것 | 469 | **33** |
| 합계 (`core_aten_decompositions()`) | 940 | **414** |

두 줄의 원인이 다릅니다.

- **post_autograd 에서 빠진 52 개는 전부 `.out` 변형**입니다. torchgen 이 빌드 타임에
  생성하고 파일에는 없는 1148 개(docs/SCHEMA.md §12)의 일부이고, `pyyaml` 을 의존성으로
  받아들이면 전부 채워집니다. **이 빌드는 `.out` 키를 하나도 구현하지 않으므로 어떤 기록에도
  들어올 수 없습니다** — `verify_schemas.py` 가 그 전제를 매번 확인합니다.
- **CIA 에서 빠진 436 개는 C++ 커널입니다.** 상류의 `_is_cia_op` 는 C++ 디스패처에 물어보고,
  여기서는 `op.py_kernels` 만 답할 수 있습니다. 파이썬 CIA 커널이 있는 것 33 개는 열렸고,
  나머지는 **파이썬 트리만 벤더링한다는 이 프로젝트의 전제 자체에 걸려 있습니다.**
  `aten.max.other` · `aten.where.ScalarOther` 가 그 벽의 이름입니다 (§4).

### 그리고 Core ATen 에 닿아도 Edge 는 아직입니다

| 남은 것 | 왜 이 패스가 아닌가 |
|---|---|
| **functionalization (view → `_copy`)** | Edge 는 `view_copy` · `permute_copy` · `transpose_copy` 를 씁니다. 그것들은 `tags: view_copy` 이지 `core` 가 아니고, **Core ATen 은 `view`/`permute` 를 그대로 둡니다.** 변환은 `to_edge` 의 일이고 분해표에 없습니다. **§5 의 결과가 `permute` 인 것이 곧 이 항목의 크기입니다** |
| **stride / dim order** | `TensorBase` 에 `.stride()` 가 없습니다 (docs/META.md §6). `empty_like` 이 §4 에서 여기 걸립니다 |
| **`graph_signature` 의 이름** | 상수가 인덱스로만 식별됩니다. 값은 `constant_values` 로 나오지만 FQN 은 위층에서 붙여야 합니다 |
| **직렬화** | 트레이스가 프로세스 밖으로 못 나갑니다. `.pte` 로 가는 길이 여기를 지납니다 |
| **변이 · 별칭** | 캡처가 거절합니다. KV 캐시가 먼저 부딪힙니다 |

DESIGN.md §5 의 3 층 구조에서 **2 층("분해 테이블을 벤더링 — 롱테일이 자동으로 core op 으로
분해됨")이 배선됐고, 이번 회차에 그 배선의 네 결함이 제거됐습니다.** 다만 "자동으로" 는
여전히 과합니다 — 37 개 중 9 개입니다.

---

## 9. 판정 — 전부 종료 코드로

| 검사 | 결과 |
|---|---|
| `cargo build --release` | 0 |
| `PYTHON=... sh rust/torch_c/pytests/run.sh` | 0 — **176/176 통과** (이전 169) |
| `python tools/golden/compare.py` | 0 — **2744/2744**, ops=118 |
| `python rust/torch_c/pytests/verify_schemas.py` | 0 — **4200/4200** (이전 3076) |

`verify_schemas.py` 가 새로 확인하는 것 셋:

```
packet overload lists:                     99/99    상류와 대조 (§3.1)
OpOverload.tags:                         118/118    상류와 대조 (§3.3)
CompositeImplicitAutograd registrations: 744/744    상류와 대조 (§3.2)
```

### 검사가 실패할 수 있는지 확인했다

초록을 받았으므로 결함을 주입해 각 검사가 실제로 빨개지는지 확인했습니다.

| 주입한 결함 | 잡은 검사 | 결과 |
|---|---|---|
| `_overload_names` 를 `["default"]` 로 되돌림 | `verify_schemas check_overload_names` | 86 개 패킷 지목, exit 1 |
| CIA 의 암묵 기본값(`torchgen/model.py:872`) 삭제 | `verify_schemas check_cia_registrations` | 653 개 지목, exit 1 |
| `_as_the_dispatcher_would_call_it` 무력화 | `test_decompose_lowers_the_op_capture_md_named` 외 8 개 | exit 1 |
| 태그의 암묵 규칙 셋 삭제 (개발 중 실측) | `verify_schemas check_tags` | **118 개 중 0 개 일치**, exit 1 |
| 0 번째 칸 주석 처리 누락 (개발 중 실측) | 같은 검사 | 3 개 지목 (`sinh.out` · `rsub.Scalar` · `tanh_backward.default`) |

마지막 두 줄은 주입이 아니라 **개발 중에 실제로 났던 상태**입니다. 검사가 그것을 잡아서
§3.3 이 존재합니다.

주입 뒤에는 `cp` 백업에서 복구하고 `git status --short` 로 확인했습니다.

---

## 10. 미완으로 남긴 것

| | 상태 |
|---|---|
| `layout` 인자 5 개 (`aten.rs`) | **가장 값싼 다음 걸음.** 하나의 원인이 5 개 op 을 막습니다 |
| `contiguous` 의 규칙 | 상류에 없습니다. 하나가 `masked_fill.Tensor` 까지 둘을 막습니다 |
| `sigmoid` · `full_like` · `tensor_split` 커널 | 표만으로는 안 열립니다 (§4). `aten.rs` |
| `softmax` 합성 | `softmax.int` 는 상류에서 CIA 이므로 `overloads.json` 이 아니라 `_install_composites` 가 자리입니다. `_safe_softmax` 하나가 걸려 있습니다 |
| 스칼라 선행 인자 (§7.3) | 해석기의 바인딩 규칙 문제. `rsub.Scalar` 가 여기 걸립니다 |
| ~~§7.2 `baddbmm` dtype~~ | **고쳐졌습니다** (2026-08-30). 원인은 `TensorBase.dtype` 의 비인턴 반환값이었습니다 — docs/BIND.md §9. 벽 3 에는 이제 예제가 없고, 그것을 확인한 방법은 §7.2.1 |
| C++ CIA 커널 436 개 (§8) | 프로젝트의 전제에 걸려 있습니다. 열 방법이 있는지부터 미정입니다 |
| `.out` 변형 1148 개 | `pyyaml` 을 의존성으로 받아들이면 전부. 지금은 도달 불가라는 것이 확인됩니다 (§8) |
| functionalization (view → `_copy`) | 없음 (§8). 이 패스가 아닙니다 |
| **여러 트레이스에 걸친 측정** | 여전히 없음. §4 의 "37 개 중 9 개" 는 **op 개수 비율**이지 "모델의 몇 %가 낮춰진다" 가 아닙니다 |

마지막 줄이 이 측정의 가장 큰 한계입니다 — 다만 §5 가 그 방향으로 한 걸음입니다. 실제 모듈
하나에 대해서는 **5 노드 중 5 노드**가 낮춰졌고, 두 숫자가 얼마나 다른지를 처음으로 보여줍니다.

---

## 11. 재현

```sh
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-accel
PY=/Volumes/macMini/caches/spike-venv/bin/python

bash vendor/vendor_torch.sh                  # 새 worktree 만
PYTHON=$PY bash vendor/install_shim.sh
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib

PYTHON=$PY sh rust/torch_c/pytests/run.sh    # 176
$PY tools/golden/compare.py                  # 2744/2744 ops=118
$PY rust/torch_c/pytests/verify_schemas.py   # 4200/4200
```

§4 의 표:

```sh
PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 \
    $PY rust/torch_c/pytests/decomp_sweep.py          # --json 이면 거절문까지
```

§3 의 숫자:

```sh
PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 $PY -c '
import torch, torch._decomp as D
from torchnative.export import core_ops, decomposition_table, decomposition_table_source
print("core:", len(core_ops()))
print("table:", decomposition_table_source(), len(decomposition_table()))
print("registry:", len(D.global_decomposition_table["post_autograd"]))
print("overloads of aten.transpose:", torch.ops.aten.transpose.overloads())
print("tags of aten.addmm.default:", torch.ops.aten.addmm.default.tags)
print("CIA:", len(torch._C._dispatch_get_registrations_for_dispatch_key(
    "CompositeImplicitAutograd")))'
```

같은 것을 상류 torch 로 (벤더 트리를 `PYTHONPATH` 에서 빼고) 돌리면 §3 의 비교 열이 나옵니다.
