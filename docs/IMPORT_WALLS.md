# `import transformers` 가 torch 에 요구하는 것

DESIGN.md §11 의 1 단계 — "벤더링한 torch 파이썬 트리 + 빈 `_C` 스텁으로 `import transformers` 를
시도하고 나오는 벽을 기록한다" — 의 결과입니다.

환경: transformers **5.15.1**, CPython 3.13.0, torch 미설치.

---

## 가장 큰 발견 — `import transformers` 는 torch 없이도 성공한다

벽이 import 에 있을 것으로 보고 계획을 세웠는데, **없습니다.**

```
$ python -c "import transformers; print(transformers.__version__)"
[transformers] PyTorch was not found. Models won't be available and only
tokenizers, configuration and file/data utilities can be used.
5.15.1
```

`from transformers import AutoModelForCausalLM` 도 통과합니다 — 클래스 객체가 지연 생성이라
존재만 합니다. **실제로 쓸 때** 비로소 막힙니다.

```
AutoModelForCausalLM requires the PyTorch library but it was not found
```

덧붙여 **transformers v5 에서도 torch 는 하드 의존성이 아닙니다.** `uv pip install transformers`
가 torch 없이 설치를 끝냅니다(설치된 것은 tokenizers · huggingface-hub · safetensors 등).

**그러므로 §11 의 1 단계는 "벽의 개수를 센다" 가 아니라 "관문 하나를 통과시킨 뒤 그 너머를 센다"
입니다.** 계획의 전제를 하나 고쳐야 합니다.

## 관문은 `is_torch_available()` 하나다

`transformers/utils/import_utils.py:179`:

```python
is_available, torch_version = _is_package_available("torch", return_version=True)
return is_available and version.parse(torch_version) >= version.parse("2.5.0")
```

`_is_package_available` 은 `importlib.util.find_spec("torch")` 로 존재를, `importlib.metadata.
version(...)` 으로 버전을 봅니다(메타데이터가 없으면 직접 import 해 `__version__` 을 보는 폴백이
있습니다).

**즉 임포트 가능한 것만으로는 부족하고 배포 메타데이터가 필요하며, 버전이 2.5.0 이상이어야
합니다.** 벤더링한 트리는 어느 상류 버전에서 가져왔는지를 그대로 선언하면 됩니다.

확인: `torch/__init__.py` 에 `__version__` 을 두고 `torch-2.9.0.dist-info/METADATA` 를 놓는 것만으로
`is_torch_available() == True` 가 됩니다.

## 관문 너머 — 요구되는 것은 목록이 아니라 다섯 범주다

기록하는 스텁 `torch` 로 `AutoModelForCausalLM.from_config` 를 반복 실행하며 수집했습니다.
**중요한 것은 이름의 목록이 아니라 각 이름이 어떤 *종류* 여야 하는가입니다** — 그것이 벤더링에서
무엇을 진짜로 가져와야 하는지를 정합니다.

| # | 범주 | 근거 | 스텁으로 되는가 |
|---|---|---|---|
| 1 | **배포 메타데이터** | `importlib.metadata.version("torch") >= 2.5.0` | 예 |
| 2 | **어노테이션에 쓰이는 진짜 타입** | `generation/logits_process.py:142` 의 `eos_token_id: int \| list[int] \| torch.Tensor` — 어노테이션이 클래스 정의 시점에 평가되므로 `\|` 가 성립해야 함 | 예 (진짜 클래스면) |
| 3 | **실제 서브모듈 파일** | `generation/utils.py:26` 의 `import torch.distributed as dist` — **모듈 수준 `__getattr__` 로는 만족되지 않음** | 예 (파일이 있으면) |
| 4 | **상속 가능한 클래스** | `class _AllReduceBackward(torch.autograd.Function)` | 예 (진짜 클래스면) |
| 5 | **데코레이터로 호출되는 것** | `continuous_batching/input_outputs.py:244` 의 `@torch.no_grad()` — 호출 결과가 다시 호출 가능해야 함 | 미확인 (여기서 멈춤) |

범주 2 와 4 는 **속성마다 동적으로 진짜 클래스를 돌려주면** 한꺼번에 풀립니다. 범주 3 은 클래스로
풀리지 않습니다 — 파이썬의 `import a.b` 는 실제 모듈을 찾으므로 파일이 있어야 합니다.

## 지금까지 확인된 서브모듈 (11 개)

발견 순서입니다. 하나를 만들면 다음이 드러나는 식으로 수집했습니다.

```
torch.distributed
torch.distributed._composable
torch.distributed._composable.fsdp
torch.distributed.fsdp
torch.distributed.device_mesh
torch.nn
torch.nn.functional
torch.nn.attention
torch.nn.attention.flex_attention
torch._dynamo
torch._dynamo._trace_wrapped_higher_order_op
```

**두 가지가 눈에 띕니다.**

- **사설 모듈이 섞여 있습니다** — `torch._dynamo`, `torch.distributed._composable`. 벤더링 대상이
  공개 API 로 끝나지 않습니다.
- **분산 학습 모듈이 다수입니다** — FSDP · device_mesh · `_composable`. 기기에서 쓸 일이 없는데도
  **import 경로에 있다는 이유만으로** 존재해야 합니다. 비어 있는 스텁으로 충분한지, 아니면 실제
  구현이 필요한지는 아직 모릅니다.

`torch._dynamo` 가 요구되는 것은 §5 의 판단과도 맞물립니다 — `torch.compile` 은 기기에서 쓸 수
없지만 **그 모듈이 존재하기는 해야** 합니다.

## 아직 답하지 않은 것

- 범주 5(`@torch.no_grad()`) 너머는 미탐색입니다. 여기서 멈췄습니다.
- 위 서브모듈들이 **빈 스텁으로 충분한지** — 즉 transformers 가 그 안의 무엇을 실제로 호출하는지는
  다음 회차의 일입니다.
- 이 실험은 **모델 생성(`from_config`)** 까지만 봅니다. 순전파 · `generate` · `online()` 은 그
  너머입니다.

## 방법에 대한 메모

이 수집은 §6 의 "발견은 shim 이 스스로 한다" 를 파이썬 계층에 적용한 것입니다. 목록을 미리 만들지
않고, 스텁이 실패하면 그 실패가 다음에 필요한 것을 지목하게 했습니다. **요구 순서가 곧 우선순위
순서**라는 성질이 여기서도 성립합니다 — `torch.distributed` 가 먼저 나온 것은 그것이 중요해서가
아니라 import 경로에서 먼저 만나기 때문이고, 그 사실 자체가 "빈 스텁으로 족할 가능성이 높다" 는
신호입니다.
