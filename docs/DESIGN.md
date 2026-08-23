# BrainWave 설계 방향

온디바이스에서 PyTorch 생태계를 그대로 돌리기 위한 설계와, 그 판단의 근거를 정리한 문서입니다.

---

## 0. 이 문서가 답하는 것

BrainWave 는 연합학습(FL) 프레임워크이고, 모델을 기기로 배포해 그곳에서 학습·적응시키는 것이
목적입니다. 그러려면 **기기에서 PyTorch 모델이 실제로 돌아야 합니다.** 이 문서는 그것을 어떻게
가능하게 할지, 그리고 아직 결정되지 않은 것이 무엇인지를 정리합니다.

관련 저장소:

| 저장소 | 역할 |
|---|---|
| `thisisthepy/PythonMultiplatform` | CPython 3.13 을 Kotlin Multiplatform 에 임베딩. **기기에서 파이썬이 도는 기반** |
| `robustaim/test-time-adapters` | 비전 TTA. `AdaptationEngine.online()` 이 실제 역전파를 요구 |
| `retentionlabs/theRiverLethe` | TTT-Linear / Titans. 손유도 닫힌 형식이라 autograd 불필요 |

---

## 1. 전제 — 파사드를 만들지 않는다

`PythonMultiplatform` 이 존재하는 이유는 **진짜 CPython 이 기기에서 돌면 진짜 pip 패키지가 기기에서
돌기 때문**입니다. transformers 모양의 API 를 흉내내는 파사드를 만드는 순간 그 기반이 무의미해집니다.

그러므로 목표는 "transformers 처럼 보이는 것" 이 아니라 **`import transformers` 가 기기에서
성공하는 것** 이고, 유일한 장애물은 `torch` 입니다.

이 판단을 뒷받침하는 관찰이 하나 더 있습니다. `test-time-adapters` 는 RT-DETR, YOLO11,
Grounding DINO, RCNN, ResNet 을 COCO / Cityscapes / SHIFT 위에서 돌리는 **비전 라이브러리**입니다.
llama.cpp 류의 LLM 추론 엔진이 커버하는 것이 하나도 없습니다. 범용 텐서 계층 외에 답이 없습니다.

---

## 2. 계층 구조

```
Kotlin Multiplatform (Android / iOS / Desktop / GraalVM native image)
└─ 임베디드 CPython 3.13                          ← PythonMultiplatform 이 이미 제공
   ├─ transformers (진짜, 또는 thelethe 의 포크)
   ├─ tokenizers · safetensors · huggingface_hub    ← 이미 torch-free, 일부는 이미 Rust
   ├─ ttadapters · thelethe · torchbrain            ← 무수정으로 동작해야 함
   └─ torch                                         ← 만들 것
      ├─ torch/ 파이썬 트리    ← 상류에서 벤더링 (BSD)
      ├─ torch/_decomp/        ← 벤더링. Core ATen 밖 롱테일이 자동 분해됨
      └─ torch/_C              ← 여기만 새로 만듦
```

### 왜 `torch._C` 만인가

**PyTorch 는 대부분 파이썬입니다.** `torch/nn/modules/*.py`, `torch/nn/functional.py`,
`torch/_tensor.py`, `torch/optim/`, `torch/utils/_pytree.py` 가 전부 파이썬 소스입니다.
네이티브인 것은 `torch._C` — ATen 텐서 + 디스패처 + autograd — 하나뿐입니다.

따라서 "PyTorch 를 다시 만든다" 가 아니라 **"파이썬 계층은 벤더링하고 `_C` 만 교체한다"** 입니다.
BSD 라이선스라 벤더링에 법적 문제가 없고, `nn.Module` · `Parameter` · `GenerationMixin` ·
캐시 클래스 · pytree 등록이 전부 따라옵니다. transformers 가 매달 바꾸는 것들은 거의 전부 aten
위쪽(파이썬)이므로 re-vendoring 으로 추적됩니다. 추격 대상이 "torch API 전체" 가 아니라
"aten op 집합" 으로 줄어들고, aten 은 훨씬 안정적입니다.

### `torchbrain` 의 네임스페이스 주입과의 관계

현재 `torch/nn/federated.py` 가 `torchbrain.nn.federated` 를 `torch` 네임스페이스에 얹는
add-hook 형태입니다. **우리가 `torch` 파이썬 트리를 소유하게 되면 이 주입이 해킹이 아니라 정식
경로가 됩니다.** 다만 벤더링한 트리와 add-hook 이 같은 경로를 놓고 충돌하지 않도록, 주입 지점을
벤더링 트리 안의 확장 지점으로 일원화해야 합니다.

---

## 3. 범위를 정하는 축 — 미분 요구가 3단계다

이것이 설계 전체의 조직 원리입니다.

| 단계 | 무엇 | 필요한 것 | 어디서 |
|---|---|---|---|
| 0 | TTT-Linear / Titans 추론 | **forward 만** | 기기 |
| 1 | `ttadapters.online()` TTA, FL 로컬 스텝 | forward + **좁은** backward | 기기 |
| 2 | TTT 모델 사전학습 | scan 전체를 통과하는 full autograd | **데스크톱 전용, 영구히** |

**단계 2 를 기기에서 명시적으로 배제하는 것이 중요합니다.** 가장 어려운 요구 — scan 을 통과하는
역전파와, 2048 토큰 기준 1.5GB 를 넘는 활성값 보존 — 를 기기 타깃에서 통째로 들어냅니다.

단계 0 이 성립하는 근거는 `theRiverLethe` 의 TTT-Linear `backward()` 가 **손으로 유도한 닫힌
형식**이라는 점입니다 (`modular_ttt_linear.py:471-531`). autograd 테이프가 없습니다. 즉
**shim 에 backward 를 구현하지 않아도 TTT 추론이 성립합니다.**

단계 1 은 backward 가 필요하지만 `AdaptationEngine.online_parameters()` 가 갱신 대상을 명시적으로
좁히므로 (`base.py:196-222`), **backward 가 필요한 op 집합은 forward 집합보다 훨씬 작습니다.**

정리하면 `torch._C` 의 사양은 **"전체 forward op + 부분 backward op"** 이고, 두 집합 모두
측정 가능합니다 (§5).

---

## 4. 만들 것과 빌려올 것

| | |
|---|---|
| **빌려옴** | torch 파이썬 트리, `torch/_decomp` 분해표, candle (텐서 엔진), tokenizers · safetensors, Core ATen 목록 (사양서) |
| **이미 있음** | CPython 임베딩, KMP FFI, Gradle 플러그인 + 아티팩트 워커 |
| **새로 만듦** | `torch._C` PyO3 어댑터, 양자화 변환 파이프라인 |

### 텐서 엔진은 candle

[huggingface/candle](https://github.com/huggingface/candle) 을 `candle-core` 만 씁니다. 근거 셋:

1. **동적 랭크.** torch API 는 런타임 동적 랭크입니다 (`x.view(-1, d)` 의 결과 랭크가 실행 시점에
   정해짐). candle 의 텐서는 shape 가 타입이 아니라 값이라 그대로 맞습니다. 대안인
   [burn](https://github.com/tracel-ai/burn) 은 `Tensor<B, D>` 정적 랭크(const generics)라
   torch API 를 씌울 수 없습니다.
2. **양자화가 이미 있음.** llama.cpp 의 GGML/GGUF k-quant 를 읽습니다. §7 의 양자화 문제가 크게
   줄어듭니다.
3. **계보.** 저자 Laurent Mazare 는 candle 이전에 `tch-rs`(libtorch 의 Rust 바인딩, 5.5k stars)
   와 `ocaml-torch` 를 만들었습니다. **PyTorch 를 다른 언어로 옮기는 일에 가장 경험이 많은 사람이,
   배포 크기와 파이썬 제거가 중요해지자 libtorch 바인딩을 버리고 처음부터 다시 쓴 것이 candle
   입니다.** 우리 조건과 같은 조건에서 내려진 판단입니다.

`candle-transformers` 는 **쓰지 않습니다.** 그것은 모델 아키텍처를 Rust 로 재구현한 것이라,
§1 에서 거부한 접근 그 자체입니다.

burn 은 나중에 FL 학습 경로를 별도 트랙으로 둘 때 검토합니다. 그쪽은 torch 흉내가 필요 없어
정적 랭크가 오히려 장점이고, `Autodiff<B>` 백엔드 데코레이터와 `Learner` · 옵티마이저 ·
체크포인팅이 이미 갖춰져 있습니다.

### 주의: candle 과 torch 의 의미론 차이

candle 은 복사 지향이라 **view / stride aliasing 과 in-place 연산**(`add_`, `copy_`)에서
임피던스가 생깁니다. transformers 의 KV 캐시 갱신이 정확히 그 경로를 밟으므로, 스파이크 초기에
여기부터 확인해야 합니다.

---

## 5. 아직 닫히지 않은 결정 — A 대 B

`torch._C` 를 무엇으로 만들 것인가.

- **A. candle 위 PyO3 어댑터**
- **B. selective libtorch 빌드**

### B 는 처음 생각보다 살아 있다

"PyTorch 는 모바일 빌드가 불가능하다" 는 통념을 분해하면 이렇습니다.

| | 실상 |
|---|---|
| libtorch(C++) 를 Android / iOS 로 빌드 | **된다.** `scripts/build_mobile.sh` 가 지금도 main 에 있음 |
| TorchScript / `.pte` 아티팩트를 기기에서 실행 | **된다** |
| 임베디드 CPython 에서 `import torch` | **아무도 안 했다** ← 우리가 필요한 것 |

**"JIT 때문에 안 된다" 는 틀린 설명입니다.** TorchScript 는 JIT 컴파일러가 아니라 직렬화된
그래프에 대한 바이트코드 인터프리터입니다. 기계어를 생성하지 않으므로 iOS 의 W^X 를 위반하지 않고,
PyTorch Mobile 이 실제로 iOS 에서 수년간 돌았습니다.

**진짜 불가능한 것은 `torch.compile` / TorchInductor 뿐입니다.** 런타임에 C++ 컴파일러를 호출해
커널을 생성하므로 iOS 에서 원리적으로 불가능합니다. 그리고 우리는 eager 실행이 목표이므로
이 손실은 받아들일 수 있습니다.

크기도 in-tree 해법이 있습니다. `SELECTED_OP_LIST=<yaml> BUILD_PYTORCH_MOBILE=1` 로 필요한 op 만
포함하면 **MobileNetV2 op 집합 기준 arm-v7 압축 ~4.5MB** 입니다. op 목록은 `TRACING_BASED=1` 로
계측 빌드를 만들면 `model_tracer` 가 뽑아줍니다 — **§6 의 측정이 공식 도구로 이미 존재합니다.**

### 그럼 무엇이 실제 차이인가

**두 경로 모두 op 집합을 미리 확정해야 한다는 점은 같습니다.** selective build 도 닫힌 세계이고,
Rust 구현도 닫힌 세계입니다. 커버리지는 차이가 아닙니다.

| | A (candle + shim) | B (selective libtorch) |
|---|---|---|
| 소유하는 코드 | 작음 (`_C` 어댑터) | 300만 줄 C++ 의 크로스 컴파일 |
| 수치 의미론 | 다시 만들어야 함 (dtype promotion, broadcasting, stride/view) | **공짜** |
| 상류 지원 | 해당 없음 | **0** — 모바일 노력이 전부 ExecuTorch 로 이동 |
| 성격 | 의미론 문제 (torch 와 맞을 때까지 안 끝남) | 빌드 문제 (유한하고 기계적) |
| 주요 리스크 | 수치 불일치가 조용히 번짐 | `native_functions.yaml` 코드젠이 host==target 을 가정 |

**B 는 "이 논지가 성립하는가" 를 가장 빨리 답하고, A 는 "이걸 출시할 수 있는가" 의 답일 가능성이
높습니다.**

### 선행 사례

[`ljk53/upytorch`](https://github.com/ljk53/upytorch) — PyTorch 모바일 엔지니어가 만든
"MicroPython + PyTorch ATen 커널" 바인딩입니다.

| | 크기 |
|---|---|
| upytorch (비압축, x86-64) | **1MB 미만** (런타임 + 선택된 ATen 커널) |
| upytorch (압축) | **~430KB** |
| CPython 3.8.3 | ~2.6MB |
| PyTorch Mobile (선택 빌드) | 4.5 ~ 20MB |
| libtorch-cpu.so (전체) | **50MB 초과** |

**op 오버로드 약 40개로 AlexNet 이 돕니다.** 규모는 개념 증명 수준(53 stars, 85 commits,
추론 전용)이지만, 증명하는 것이 정확히 우리에게 필요한 두 가지입니다 — 파이썬 인터프리터에
ATen 커널을 torch 시그니처로 직접 붙이는 것이 되고, 결과가 작다.

### 사양서는 이미 공개되어 있다

op 집합을 추측할 필요가 없습니다. PyTorch 에 **[Core ATen Operator Set]**
(https://docs.pytorch.org/executorch/stable/ir-ops-set-definition.html) 이라는 공식 답이 있습니다.
공개 저장소와 유명 오픈소스 모델을 조사해 실제로 많이 쓰이는 aten op 을 추린 것이고, 목적이 명시적으로
"백엔드와 컴파일러가 처리해야 할 연산자 수를 줄이는 것" 입니다.

같이 딸려오는 **Core ATen Decomposition Table** 이 core 가 아닌 op 을 core op 으로 분해하는
규칙표이고, **이 규칙들은 파이썬 소스**(`torch/_decomp/decompositions.py`)입니다. 즉 §2 의
벤더링에 포함됩니다.

그러면 shim 이 3층으로 정리됩니다.

1. **Core ATen 을 Rust/candle 로 구현** — 하드 플로어
2. **분해 테이블을 벤더링** — 롱테일이 자동으로 core op 으로 분해됨
3. **핫한 op 만 손으로 최적화** — matmul, attention, 양자화 경로

부수 효과로, ExecuTorch 도 Core ATen 을 타깃하므로 나중에 무거운 모델만 `.pte` 로 빼는 하이브리드가
같은 계약 위에서 열립니다.

---

## 6. 이 결정을 닫는 숫자 넷

지금은 논쟁이지만, 아래 넷이 나오면 계산입니다.

1. **forward op 집합** — `TorchDispatchMode` 로 측정. TTT-Linear, Titans, RT-DETR, YOLO11
2. **backward op 집합** — 같은 방법을 `online()` 상태의 한 스텝에 적용
3. **selective libtorch 바이너리 크기** — 1+2 의 합집합으로 빌드
4. **그 크로스 컴파일이 완료되기는 하는지** — **타임박스를 걸 것**

```python
from torch.utils._python_dispatch import TorchDispatchMode

ops = set()

class Count(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        ops.add(str(func))
        return func(*args, **(kwargs or {}))

with Count():
    model.generate(**inputs, max_new_tokens=32)   # 또는 engine.online(); 한 스텝

print(len(ops), sorted(ops))
```

**판정:** 1+2 가 100개 남짓이고 3 이 크게 나오면 A. 반대면 B 가 진지해집니다.

**주의.** tracing-based selective build 는 *traced 모델* 을 추적하므로, eager 파이썬 transformers
가 밟는 경로(generation 루프, 캐시 클래스, dtype 변환, config 분기)를 놓칠 수 있습니다.
`TorchDispatchMode` 측정과 **합집합**을 내야 합니다.

---

## 7. 런타임 특성

### 디코드는 살아남는다

배치 1 디코드는 산술 강도가 바이트당 1~2 FLOP 수준이라 메모리 바운드입니다. op 하나가 밀리초
단위로 도는 동안 파이썬 디스패치 오버헤드(마이크로초)는 그 밑에 숨습니다. ggml 의 인터프리터가
경쟁력 있는 것과 같은 이유이고, **이 논거가 우리 편입니다.**

### prefill 이 약점이다

거기서는 compute 바운드라 오버헤드가 숨지 않고, op 융합도 못 합니다.

TTT-Linear 기준 구체적으로: `adapt_step` 하나가 텐서 연산 25~30 개이고, `chunk_size` 16 에
2048 토큰이면 미니배치 128회 × 24레이어 = **forward 당 8만 회 안팎의 파이썬→네이티브 호출**입니다.

레버는 `chunk_size` 입니다 (16 → 128회, 64 → 32회). 다만 이것은 TTT 미니배치 크기라 **모델링
하이퍼파라미터**이고 `max_position_embeddings` 가 여기 묶여 있어 (`modular_ttt_linear.py:137`)
자유 변수가 아닙니다. 값을 바꾸려면 그 값으로 학습된 체크포인트가 필요합니다.

**그러므로 `adapt_step` 전체를 네이티브 커널 하나로 내리는 탈출구를 처음부터 설계에 넣어둘 것.**
당장 만들지는 않되, 나중에 끼워넣을 수 있는 형태로.

### 양자화는 torch 코어 밖에 있다

4비트 체크포인트는 bitsandbytes / torchao / GPTQ / AWQ / compressed-tensors 중 하나를 타는데
**전부 별개의 C 확장**이고, transformers 가 체크포인트 config 를 보고 거기로 디스패치합니다.
즉 `torch` 를 제공해도 4비트 모델은 안 돌아갑니다.

**해법:** 경로를 **하나만** 고르고 (candle 의 GGUF k-quant 가 유력), 그 op 을 Rust 계층에 넣고,
**Gradle 플러그인에서 빌드 타임에 체크포인트를 변환**합니다. `PythonMultiplatform` 에 아티팩트
워커가 이미 있으므로 자리가 맞습니다.

**대가:** 지원 매트릭스가 "HF 전체" 가 아니라 "빌드 타임에 우리 경로로 변환한 체크포인트" 가 됩니다.
괜찮은 트레이드지만 **지금 의식적으로 정할 일이지 8개월 차에 발견할 일이 아닙니다.**

---

## 8. 기존 코드에서 나온 선결 과제

경로(A/B) 와 무관하게 옳은 것들입니다.

### `theRiverLethe`

| # | 위치 | 문제 |
|---|---|---|
| 1 | `thelethe/ops/normal_scan.py:214` | `compiled_scan = torch.compile(scan, mode="max-autotune")` 가 **모듈 스코프**. `ops/__init__.py` 가 export 하므로 import 시 실행됨. `torch.compile` 심볼이 없으면 import 단계에서 죽고, 호출되면 Inductor 가 런타임 C++ 컴파일러를 부름 → **iOS 불가**. `max-autotune` 은 커널 변형을 런타임 벤치마크하므로 그중 최악. **→ 지연 생성 + 플랫폼 가드** |
| 2 | `modular_ttt_linear.py:374-376` | `torch.isnan(bias).any()` 를 파이썬 `if` 로 분기. `rasterize` → `step()` → scan 반복마다 호출되어 **forward 당 3000회 넘는 호스트 동기화**. 게다가 데이터 의존 분기라 **export 를 영구히 막음**. bias 유무는 정적 정보이므로 **→ NaN 센티넬 대신 `use_bias` 불리언을 상태에 동반** |
| 3 | `normal_scan.py:11` | `from torch._higher_order_ops import scan` 이 아래 `def scan` 에 가려진 **죽은 import**. 그래도 모듈 경로가 존재해야 통과. **→ 삭제** |
| 4 | `modular_ttt_linear.py:319` | `torch.einsum` — 문자열 파싱 + 일반화 축약이라 shim 구현 비용이 큼. 여기서는 `"bhkc,hcd->bhkd"` 한 패턴뿐. **→ matmul 로 재작성** |

### `test-time-adapters`

| # | 위치 | 문제 |
|---|---|---|
| 5 | `methods/base.py:117` | `base_state` 가 전체 가중치의 CPU 사본을 엔진 수명 내내 보유. **폰에서 모델 메모리 2배**이고, 모바일은 CPU/GPU 가 같은 물리 RAM 이라 오프로드가 아니라 순수 중복. `reset()` 은 StandardTTA/GradualTTA **벤치마크 요구**이지 제품 요구가 아님. **→ 기기에서는 디스크 재로드로** |
| 6 | `methods/base.py:152-156` | `self._dtype = torch.dtype(*args, **kwargs)` — `torch.dtype` 은 생성자로 호출 불가라 항상 TypeError 이고 `except TypeError: pass` 가 삼킴. **`_dtype` 이 init 값에 고정**되고 `reset()` 이 `:254` 에서 그 값으로 되돌림. 데스크톱 fp32 고정에서는 안 보이지만 기기에서 fp16/fp32 를 오가면 즉시 물림 |
| 7 | `methods/base.py:37-55` | `torch.cuda.manual_seed()`, `torch.backends.cudnn.*` 가 가드 없이 호출됨 |
| 8 | `methods/base.py:180-191` | Muon / MuonWithAuxAdam 이 스텝마다 Newton-Schulz 직교화 반복(matmul 다회). `muon` 이 추가 의존성. **→ 기기 기본값은 SGD, Muon 은 데스크톱 게이팅** |

### carry in-place 에 관한 정정

`TTTLinearAdaptationState.step()` 이 매번 새 상태를 만드는 것은 **바깥쪽 학습(단계 2)의 역전파를
위해 필요하며 맞게 작성된 것입니다.** in-place 로 바꾸면 `W_{t-1}` 을 복원할 수 없어 PyTorch 가
버전 카운터로 잡아냅니다.

다만 판정 기준이 `self.training` 이 아닙니다. **`eval()` 은 grad 추적을 끄지 않습니다** —
파라미터의 `requires_grad` 가 True 면 eval 에서도 테이프가 쌓입니다. 올바른 술어는
`torch.is_grad_enabled()` 입니다.

```python
next_carry = carry.step(deltas) if torch.is_grad_enabled() else carry.step_(deltas)
```

`GenerationMixin.generate` 가 이미 `@torch.no_grad()` 로 감싸져 있으므로 기본 추론 경로는 분기만
넣으면 혜택을 받습니다. `inference_mode` 는 `no_grad` 보다 강해 그 안에서 만든 텐서를 이후
autograd 에 쓸 수 없으므로, **online/offline 을 오가는 `ttadapters` 에는 `no_grad` 가 안전합니다.**

**얻는 것의 크기는 크지 않습니다.** `memory_depth` 가 1 이라 스텝당 파이썬 객체는 2~3 개이고,
`no_grad` 하에서는 이전 carry 의 참조가 즉시 끊겨 allocator 가 블록을 재사용합니다. 실질적인 이득은
할당 제거보다 **항목 2 의 호스트 동기화가 같이 사라지는 쪽**입니다. prefill 비용의 본체는 디스패치
횟수이고 이것으로는 안 풀립니다 (§7).

### 이 저장소

- `torchbrain/api/__init__.py:4` — `def __init__(self, *args, *kwargs)` 는 **SyntaxError**
  입니다 (`**kwargs` 여야 함). 현재 이 모듈은 import 되지 않습니다.
- `torchbrain/nn/federated/__init__.py:1` — `from . import DistributedDataFederated` 인데 해당
  모듈이 디렉터리에 없어 ImportError 입니다.

둘 다 blank template 단계라 의도된 미완성일 수 있으나, 첫 실행 전에 걸립니다.

---

## 9. 순서

**측정 → 부트스트랩 → 이식.** 3 단계까지는 KMP 도 기기도 건드리지 않습니다.

| # | 할 일 | 산출물 |
|---|---|---|
| 1 | §6 의 숫자 넷 측정 (데스크톱) | A/B 결정 |
| 2 | 벤더링한 torch 파이썬 트리 + 빈 `_C` 스텁으로 `import transformers` 시도 | import-time 요구사항 전체 목록 |
| 3 | 데스크톱에서 `torch._C` 로 가장 작은 모델 하나 통과, 진짜 torch 와 골든 대조 | 수치 의미론 리스크의 실제 크기 |
| 4 | 기기 (Android 먼저 — iOS 보다 제약이 적음) | |
| 5 | GraalVM 네이티브 이미지 경로 | `PythonMultiplatform` 의 요구사항 |

**2 단계에서 나오는 벽의 개수가 이 계획의 실현 가능성을 거의 다 말해줍니다.**
`accelerate` 가 무조건 `import torch` 를 하는 것 같은 결합은 우리에게 유리합니다 — 그 이슈의
사람들은 torch 를 *피하려* 했고 우리는 *만족시키려는* 것이므로 방향이 반대입니다.

§8 의 1~4, 6~7 은 이 순서와 무관하게 지금 고쳐도 되는 것들이고, 5 와 8 이 설계 판단입니다.

---

## 부록: 기각된 대안과 근거

| 대안 | 기각 사유 |
|---|---|
| **llama.cpp 바인딩 + transformers 파사드** | §1 의 전제를 위반. 그리고 `ttadapters` 의 비전 모델을 하나도 커버하지 못함 |
| **micrograd / hypergrad 계열에서 출발** | 참조한 hypergrad 는 커밋 5개 · 스타 11개의 **스칼라 값** autograd 로, 텐서도 BLAS 도 없음. 거기서 "폰에서 int4 로 Llama" 까지는 증분이 아니라 ggml/candle 코드베이스 전체 |
| **torch API 를 처음부터 완전 복제** | transformers 는 `isinstance(x, torch.Tensor)`, dtype promotion, view/stride aliasing, `__torch_function__` 에 의존 — 동작 호환이 아니라 **버그 호환**이 요구됨. 게다가 transformers v5 가 TF/Flax 를 유예 없이 삭제했고 명시 사유가 "유지 비용" — **풀타임 팀의 HuggingFace 본인이 다중 백엔드 패리티를 포기함**. 그리고 그 결정 방향이 "추상화를 걷어내는 것" 이라 모델링 코드는 앞으로 torch 에 **더** 밀착함 |
| **tch-rs 사용** | (a) libtorch 가 타깃에 있어야 함 (b) **화살표 방향이 반대** — tch-rs 는 torch 를 *Rust 에* 노출하는데, 우리는 Rust 엔진을 *파이썬에* torch 얼굴로 노출해야 함. 만들 것은 크레이트가 아니라 `torch._C` 자리의 PyO3 확장 모듈 |
| **burn 을 shim 백엔드로** | `Tensor<B, D>` 정적 랭크가 torch 의 동적 랭크와 근본적으로 불일치. FL 학습 트랙에서는 여전히 후보 |
| **torch-xla / torch-mlir** | torch-xla 는 libtorch 를 요구하는 **런타임 확장**, torch-mlir 은 AOT **컴파일러 프론트엔드**로 결과가 컴파일된 아티팩트 — 둘 다 "기기에서 진짜 파이썬 transformers" 가 아님. 다만 각각 훔칠 것이 있음: 전자는 "한 계층에서 가로채면 위가 전부 따라온다" 의 대규모 증명, 후자는 "aten 수천 개가 작은 핵심 집합으로 분해된다" 의 독립 검증 (TOSA 목록이 특히 하드웨어 지향) |
| **`optimum-executorch` 로 `.pte` 배포** | 이미 공식으로 존재하고 잘 동작하지만, 데스크톱에서 export 한 아티팩트를 배포하는 것이라 §1 의 전제와 다름. §5 의 Core ATen 계약을 공유하므로 **나중에 무거운 모델만 빼는 하이브리드로는 열려 있음** |

---

## 참고

- [Core ATen Operator Set](https://docs.pytorch.org/executorch/stable/ir-ops-set-definition.html)
- [PyTorch's Tracing Based Selective Build](https://pytorch.org/blog/pytorchs-tracing-based-selective-build/)
- [`pytorch/scripts/build_mobile.sh`](https://github.com/pytorch/pytorch/blob/main/scripts/build_mobile.sh)
- [`ljk53/upytorch`](https://github.com/ljk53/upytorch)
- [huggingface/candle](https://github.com/huggingface/candle) · [tracel-ai/burn](https://github.com/tracel-ai/burn) · [LaurentMazare/tch-rs](https://github.com/LaurentMazare/tch-rs)
- [torch-mlir architecture](https://github.com/llvm/torch-mlir/blob/main/docs/architecture.md)
- [transformers v5 Migration Guide](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md)
