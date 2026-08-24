# CKPT — 훈련된 가중치를 읽는다

이 shim 은 **체크포인트를 한 번도 읽은 적이 없었습니다.** 지금까지의 모든 검증 — 골든 2095 케이스,
E2E 토큰 일치, 샘플링 90 토큰 — 이 전부 **난수로 초기화한 가중치**로 한 것입니다. 모델을 조립하는
것과 훈련된 모델을 불러오는 것은 다른 경로이고, 후자는 한 번도 실행된 적이 없었습니다.

이 문서는 그 경로를 재고, 어디까지 뚫었고, 어디서 멈췄는지를 적습니다.

---

## 1. 판정 요약

**상류가 저장한 체크포인트를 shim 이 읽어, 그 가중치로 순전파를 돌리고 상류와 대조한 결과:**

| 경로 | 결과 | 로짓 최대 절대오차 |
|---|---|---|
| `torch.load` (zip, `weights_only=True`) | **동작** | **2.98e-08** |
| `torch.load` (zip, `weights_only=False`) | **동작** | 2.98e-08 |
| `safetensors.torch.load(bytes)` | **동작** | 2.98e-08 |
| `safetensors.torch.load_file(backend="pread")` | **동작** | 2.98e-08 |
| `safetensors.torch.load_file()` — 기본 `mmap` 백엔드 | 거부 | — |
| `torch.load` (legacy 비-zip 포맷) | 거부 | — |
| `Module.state_dict()` / `load_state_dict()` | **동작** | — |

정상 범위는 2.3e-09~5.2e-06 이고, 2.98e-08 은 그 안입니다.

그리고 **두 리더가 같은 답을 냅니다** — 같은 가중치를 `torch.load` 로 읽은 것과 safetensors 로
읽은 것의 차이는 `0.0` 입니다. 근사가 아니라 비트 단위로 같습니다.

### 회귀 없음

```
골든              2095/2095, ops=91, pending 0            exit 0   (develop 기준과 동일)
--inject-fault    value / shape / dtype                   전부 exit 1
pytests run.sh    (자가검사 11×11 포함)                     exit 0
verify_schemas    204/204                                  exit 0
3 타깃            host / android-arm64 / ios-arm64          전부 exit 0
```

---

## 2. `636a3cc` 가 무엇을 했나

이 작업은 인터럽트로 죽은 에이전트의 커밋(`636a3cc`, 4 파일 269 줄)을 이어받은 것입니다. 보고가
없었으므로 먼저 판정했습니다. **결론: 그 커밋은 safetensors 경로를 끝까지 뚫어 놓았고, 코드는
옳습니다.** 다시 하지 않았습니다.

들어 있던 것 세 가지:

| 위치 | 무엇 | 판정 |
|---|---|---|
| `lib.rs` `_frombuffer` | `torch.frombuffer` 의 구현 | **옳음.** safetensors 로드의 전부 |
| `dtype.rs` | `uint1..7` · `int1..7` 14 개 태그 | **옳음.** 아래 설명 |
| `tensor.rs` `is_meta` | `load_state_dict` 가 파라미터마다 읽음 | **옳음** |
| `bootstrap.py` | `varfns.frombuffer` 배선 | **옳음** |

**14 개 서브바이트 dtype 이 왜 필요한가**를 재확인했습니다. 커밋 주석의 주장이 맞습니다 —
`torch/_weights_only_unpickler.py` 가 허용목록을 만들 때

```python
for t in [getattr(torch, f"uint{x}") for x in range(1, 8)]: ...
```

를 **체크포인트를 한 바이트도 읽기 전에** 무조건 실행하므로, 14 개 중 하나만 없어도 모든
`torch.load` 가 `AttributeError` 로 죽습니다. 이 문서의 §3 이 그 다음 벽을 재는 것은 이 14 개가
있었기 때문에 가능했습니다.

**주석에 적힌 미검증 주장 하나를 실측으로 확인했습니다.** `_frombuffer` 의 doc 은 "이 함수 하나로
safetensors 경로가 첫 벽에서 완전한 state dict 까지 간다" 고 적었는데, 근거가 남아 있지 않았습니다.
재측정 결과 **사실입니다** (§3 표 참조).

**그 커밋이 하지 않은 것**: `torch.load` 는 전혀 손대지 않았고, 어떤 경로도 테스트되지 않았으며,
`docs/CKPT.md` 를 네 군데에서 참조하는데 그 파일이 없었습니다(이 문서가 그것입니다).

---

## 3. 각 경로가 어디서 막혔나 — 실측

측정 방법: 각 경로를 독립된 `try/except` 로 돌려 **첫 벽이 다음 벽을 가리지 않게** 했습니다.
`vendor/probe.py` 의 record 모드와 같은 발상이되, 대상이 import 가 아니라 로드 경로입니다.

### 3.1 safetensors

```
load_file()                → torch._C.StorageBase.from_file      (mmap 백엔드)
load_file(backend="pread") → 통과
load(bytes)                → 통과
```

**기본 백엔드가 `mmap` 이고, 그것만 막힙니다.** `safe_open(..., backend="mmap")` 은 safetensors
자신의 Rust 확장 안에서 `torch._C.StorageBase.from_file` 로 mmap 스토리지를 만들고, 그것은
"텐서가 파일에 별칭으로 붙는다"는 개념이라 candle 에 대응물이 없습니다. `backend="pread"` 는
같은 컨테이너를 읽고 `torch.frombuffer` 로 나오므로 통과합니다.

**즉 safetensors 는 인자 하나로 오늘 동작합니다.** 이것이 이 조사에서 나온 가장 실용적인 결과입니다.

### 3.2 `torch.load`, zip 컨테이너 (기본)

벽을 하나씩 뚫으며 잰 **완전한 순서**입니다. 각 줄이 앞줄을 메운 뒤에야 보였습니다.

| # | 벽 | 어디서 | 메운 방법 |
|---|---|---|---|
| 1 | `PyTorchFileReader.get_all_records` | `serialization.py:2232` | `_ZipRecords` (bootstrap.py) |
| 2 | `TypedStorage(wrap_storage=...)` 가 진짜 `UntypedStorage` 를 요구 | `storage.py:836` | `StorageBase` (storage.rs) |
| 3 | `TensorBase.set_` | `_utils.py:198` | tensor.rs |
| 4 | `TensorBase.element_size` | `set_` 내부 | tensor.rs |
| 5 | `tensor._backward_hooks = ...` 에 setter 없음 | `_utils.py:246` | tensor.rs, 실슬롯 |
| 6 | `torch._C._check_sparse_tensor_invariants` | `_utils.py:290` | bootstrap.py, `False` |

여섯 개를 메우면 zip 포맷이 끝까지 갑니다. `weights_only` 는 `True`/`False` 둘 다 통과합니다.

### 3.3 `torch.load`, legacy 포맷 — **막혔고, 그것이 옳습니다**

`torch.save(..., _use_new_zipfile_serialization=False)` 로 저장한 것은 **거부합니다.** 이것은
못 뚫은 것이 아니라 **뚫으면 안 되는 것**이며, §4 가 그 이유입니다.

---

## 4. 이 조사에서 나온 제일 중요한 것 — 조용히 0 이 되는 경로

**측정된 사실.** legacy 포맷을 소박하게 구현하면 `torch.load` 가 성공하고,
`load_state_dict` 가 `<All keys matched successfully>` 를 돌려주고, **모든 가중치가 `0.0` 입니다.**
예외는 한 곳에서도 나지 않습니다.

```
LOADED legacy ['down.bias', 'down.weight', 'embed.weight', ...]
  every loaded tensor is all-zero: True
  down.bias[:4] = [0.0, 0.0, 0.0, 0.0]
  truth         = [0.06125, 0.14750, 0.23375, -0.18000]
```

**원인은 별칭과 복사의 차이이고, 버그가 아니라 구조입니다.**

상류에서 스토리지는 텐서 메모리의 *소유자*이고 텐서는 그 위의 뷰입니다 — `set_` 은 별칭을
만듭니다. candle 은 자기 메모리를 소유하므로 `set_` 이 **복사**할 수밖에 없습니다. 그런데 두
컨테이너 포맷은 스토리지를 채우는 시점이 `set_` 기준으로 반대입니다.

```
zip     스토리지를 레코드에서 채움  →  _rebuild_tensor → set_      복사해도 옳다
legacy  _rebuild_tensor → set_      →  스토리지를 파일에서 채움     복사하면 0 이 된다
```

별칭 의미론에서는 두 순서가 같은 답을 내므로 상류는 한 경로로 둘 다 처리합니다. 복사
의미론에서는 두 번째 순서가 **`set_` 시점의 스토리지 내용** — 즉 갓 할당된 0 — 을 집습니다.

### 규약이 아니라 불변식으로 막았다

이것을 "legacy 를 구현하지 않는다" 는 합의로 두면, 나중에 `_set_from_file` 이 사소해 보여서
구현되는 순간 되살아납니다. 그래서 **`StorageBase` 가 `filled` 를 들고 있고, `set_` 은 한 번도
채워진 적 없는 스토리지를 거부합니다.**

```
NotImplementedError: TensorBase.set_: the storage has never been filled. This shim's
set_ copies out of the storage instead of aliasing it, so a tensor built from an empty
storage would be silently zero. The caller must deliver the bytes before set_, not
after (see storage.rs and docs/CKPT.md §4).
```

`filled` 를 세우는 문은 `_shim_fill` 하나뿐이고, 그것은 바이트를 실제로 받은 경우에만 불립니다.
**legacy 순서를 우연히 타는 것이 불가능해집니다.**

---

## 5. 뷰로 저장된 텐서 — 두 번째 조용한 오답을 닫았다

체크포인트의 텐서가 자기 스토리지의 **연속 구간이라는 보장이 없습니다.** `torch.save` 는 발견한
그대로 stride 와 storage_offset 을 기록하므로, `state_dict` 가 `w.t()` 를 들고 있으면
비연속으로 도착합니다.

처음 구현은 앞에서부터 `numel` 개를 읽었고, 그러면 전치 텐서가 **dtype 도 shape 도 맞는데 숫자만
틀린 채로** 나옵니다. 역시 예외가 없습니다.

지금은 `gather_strided` 가 `(storage_offset, size, stride)` 를 실제로 걸어 행 우선으로 모읍니다.
바이트 단위 gather 라 dtype 과 무관하고, 근사가 아니라 정의 그대로입니다.

**실측 — 상류가 저장한 14 개 항목 전부 비트 단위 일치:**

```
  [ok ] transposed     torch.float32    stride=(1, 4)      worst=0
  [ok ] slice_offset   torch.float32    storage_offset=5   worst=0
  [ok ] tied_a/tied_b  가중치 공유(한 스토리지, 두 키)        worst=0
  [ok ] w_f16 / w_bf16 / w_f64 / buf_i64 / buf_i32 / buf_bool / scalar / empty / rank3
```

**남은 차이 하나**: 가중치 공유(weight tying)는 **값으로는 보존되지만 동일성으로는 보존되지
않습니다.** 상류에서 `tied_a is tied_b` 의 스토리지는 하나인데, 여기서는 복사이므로 둘이
독립입니다. 읽기 전용 추론에서는 보이지 않고, 한쪽을 제자리 수정하면 갈립니다.

---

## 6. 구현한 것 / 때운 것 / 못 한 것

### 구현한 것

| | 어디 | |
|---|---|---|
| `StorageBase` | `rust/torch_c/src/storage.rs` (신규) | 바이트 버퍼 + `filled` 불변식 |
| `TensorBase.set_` | `tensor.rs` | strided gather, 네 가지 거부 |
| `TensorBase.element_size` | `tensor.rs` | dtype 태그 기준 |
| `gather_strided` | `tensor.rs` | 뷰를 정의대로 읽음 |
| `from_le_bytes` | `tensor.rs` | `_frombuffer` 와 `set_` 의 **공통** 바이트→텐서 |
| `PyTorchFileReader` | `bootstrap.py` `_ZipRecords` | zip 컨테이너 |

`from_le_bytes` 를 공유로 뽑은 것은 정리가 아니라 판정 근거입니다 — 두 리더가 dtype 처리와
`torch.bool` 정규화를 따로 갖고 있으면 §1 의 "두 리더가 비트 단위로 같다" 가 우연이 됩니다.

**`_aten_dispatch` 는 그대로 단일 출입구입니다.** 여기 추가된 것 중 aten op 은 하나도 없고
(`_aten_implemented()` 는 91 개 그대로, 골든 pending 0), 따라서 `tools/golden/cases.py` 에
붙일 케이스도 없습니다.

### 때운 것 (papered over — 구현이 아님)

| | 무엇을 했나 | 왜 무해한가 |
|---|---|---|
| `_backward_hooks` | 저장하되 아무도 읽지 않는 실슬롯 | autograd 가 없어 훅이 발화할 일이 없고, 이 경로가 쓰는 값은 항상 빈 `OrderedDict()` |
| `_check_sparse_tensor_invariants` | 항상 `False` | 상류 기본값이 `False`, 그리고 이 shim 에 sparse 텐서가 없어 검사 대상이 공집합 |
| `_RecordHolder` | 상류는 Tensor 를 돌려주는데 여기서는 두 메서드만 답하는 객체 | 호출자가 `._typed_storage()._untyped_storage` 만 함. Tensor 를 흉내내면 체크포인트를 통째로 한 번 더 복사 |
| `PyTorchFileReader` 가 Rust 가 아니라 bootstrap.py | 컨테이너 파싱은 텐서 연산이 아니고, CPython 이 검증된 zip 리더를 이미 들고 있음 | 대안은 손으로 쓴 zip 파서나 3 타깃에 새 크레이트 의존성 — 둘 다 더 나쁜 거래 |

### 못 한 것 (거부하고 이름을 답함)

| | 무엇이 필요한가 |
|---|---|
| `torch.load` legacy 포맷 | 진짜 별칭 스토리지. §4 |
| `torch.load(mmap=True)` | `UntypedStorage.from_file` + 스토리지 슬라이싱 |
| `safetensors` 기본 `mmap` 백엔드 | 같은 것 |
| `get_record_offset_no_read` | torch 의 레코드 정렬 산술 재현. **틀린 오프셋은 예외가 아니라 옆 텐서의 바이트**라 추측하지 않음 |
| `int8` · `uint16` · `uint64` · complex 로 저장된 체크포인트 | candle 이 못 담음. dtype 이름을 대며 거부 |
| 음수 stride | torch 가 만들지 않으므로 추측하지 않음 |
| 체크포인트 **쓰기** | 범위 밖. `UntypedStorage.__getstate__` 가 이름을 대며 거부 |

### 모르는 것

- **실제 사전훈련 체크포인트로는 검증하지 못했습니다.** `import transformers` 가 아직
  `torch.distributed` 에서 막히고(IMPORT_WALLS), 이 조사는 `torch.nn` 조립 + 수동 적재로
  판정했습니다. 여기서 쓴 것은 상류가 저장한 진짜 `.pt`/`.safetensors` 파일이지, 진짜 *모델* 은
  아닙니다. HF 체크포인트 특유의 것(공유 텐서 메타데이터, 샤딩된 `.index.json`, `_metadata`)은
  **미측정**입니다.
- **회귀 스위트에 박혀 있지 않습니다.** 위 숫자는 전부 `/Volumes/macMini/caches/ckpt-probe/`
  의 스크립트로 잰 것이고, 커밋 대상이 아닙니다(이 작업의 파일 범위가 `rust/torch_c/src/`,
  `tools/golden/cases.py`, 이 문서였습니다). **`pytests/test_shim.py` 에 넣는 것이 다음
  작업이고, 넣기 전까지 §1 의 어떤 성질도 회귀로부터 보호되지 않습니다.** docs/E2E.md 가 같은
  이유로 만들어졌던 자리입니다.
- `serialization_id()` 는 레코드가 없으면 빈 문자열을 답합니다. 상류가 그 값을 어떻게 쓰는지는
  텔레메트리 콜백 하나밖에 확인하지 못했습니다.

---

## 7. 재현 방법

```bash
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-ckpt
cd /path/to/repo
bash vendor/vendor_torch.sh
sh vendor/install_shim.sh

# 상류가 체크포인트를 만든다 (벤더 트리를 PYTHONPATH 에 넣지 않는다)
/Volumes/macMini/caches/spike-venv/bin/python make_ckpt.py

# shim 이 그것을 읽는다
TORCH_USE_RTLD_GLOBAL=1 PYTHONPATH=$PWD/torchnative/src/main \
    /Volumes/macMini/caches/spike-venv/bin/python verify.py
```

`TORCH_USE_RTLD_GLOBAL=1` 은 회피가 아니라 상류가 그 목적으로 둔 스위치입니다 —
`libtorch_global_deps` 가 없는 빌드 환경을 위한 것이고, 우리가 정확히 그 환경입니다
(VENDOR.md:181).

**`compare.py` / `verify_schemas.py` 는 벤더 트리를 `PYTHONPATH` 에 넣지 말고 돌립니다.**
그 둘은 상류 torch 와 대조하는 쪽이라 진짜 torch 가 보여야 합니다.
