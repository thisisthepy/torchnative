# 알려진 오탐

§9 선결 과제 패치를 이 스캐너로 교차 검증하다 드러난 것들입니다. **패치가 옳고 스캐너가 틀렸습니다.**

두 오탐 모두 `scan_torch_risks.py` 에서 해소했습니다(아래 각 항목 참고).

## 1. [해소] 가드·지연화된 `torch.compile` 을 여전히 CRITICAL 로 보던 문제

모듈 스코프의 `torch.compile` 은 import 만으로 실행되므로 치명적이지만, **함수 안으로 옮기고
가용성 가드를 붙이면 위험이 사라집니다.** 스캐너는 `import_time` 을 올바르게 `false` 로 보고하면서도
심각도는 CRITICAL 로 유지했습니다.

사례: `thelethe/ops/normal_scan.py:237` — `compiled_scan()` 안에서 `_torch_compile_available()`
뒤에 있음. 고쳐진 코드인데 고치기 전과 같은 심각도로 보고됨.

**적용한 수정:** `import_time == false` 이고, 같은 함수 스코프 안의 `if <가드>:` 또는
`<x> if <가드> else <y>` 삼항식의 참 분기에 놓여 있으면 finding 자체를 내지 않습니다
(`is_guarded_by_availability_check`, `Scanner.scan_attribute_namespaces`). 모듈 스코프의
`torch.compile` (여전히 `import_time == true`)은 가드 여부와 무관하게 그대로 CRITICAL 로 잡힙니다 —
import 시점 실행이라는 위험 자체는 가드로 없앨 수 없기 때문입니다.

검증: `scan-target-lethe` (미수정 원본)의 모듈 스코프 `torch.compile` 은 여전히 CRITICAL. `fix-lethe`
의 `compiled_scan()` 내부 가드된 `torch.compile` 은 더 이상 잡히지 않음.

## 2. [해소] 헬퍼 함수를 거친 가드를 따라가지 못하던 문제

`if torch.cuda.is_available():` 같은 리터럴만 가드로 인정했습니다. 아래는 못 알아봤습니다.

```python
def _cudnn_available() -> bool:
    cudnn = getattr(torch.backends, "cudnn", None)
    return cudnn is not None and cudnn.is_available()

use_cudnn = _cudnn_available()
if use_cudnn:
    torch.backends.cudnn.deterministic = True   # 가드 안인데 CUDA_UNGUARDED 로 보고됨
```

사례: `ttadapters/methods/base.py:51-53, 64`.

**적용한 수정:** 텐서 값 분기 탐지(`scan_tensor_value_branching`)에 있던 변수 오염 추적 기제를
그대로 재사용하되 방향을 반대로 적용했습니다. `collect_guard_taint` 가 `..._available(...)` 형태
(또는 `hasattr`/`cuda`/`available` 을 포함하는 호출)로 초기화되는 변수를 스코프·라인과 함께 기록하고,
`is_guarded_by_availability_check` 이 `if` 테스트의 리터럴 텍스트뿐 아니라 이 오염 테이블도 함께
확인합니다. `use_cudnn = _cudnn_available()` 뒤의 `if use_cudnn:` 이 이제 가드로 인식됩니다.

검증: `scan-target-tta` (미수정 원본)의 가드 없는 `torch.backends.cudnn.*` / `torch.cuda.*` 는
여전히 WARN. `fix-tta` 의 `base.py:51-53,64` (`if use_cudnn:` 안)는 더 이상 잡히지 않음. 같은 파일의
무관한 CUDA_UNGUARDED 항목들(`base.py` 밖, 예: `models/rcnn/*.py`, `utils/validator.py` 등)은
가드가 없으므로 그대로 유지됨 — 수정이 과도하게 억제하지 않음을 확인.

## 왜 중요했는가

**오탐이 많으면 CI 게이트는 무시당합니다.** 위 둘은 "고쳐도 경보가 안 꺼지는" 형태라 특히 나빴습니다 —
고친 사람이 스캐너를 신뢰하지 않게 됩니다.

## 남은 것

이 문서를 작성한 시점에 §9 교차 검증에서 나온 알려진 오탐은 이 둘뿐이었습니다. 추가로 발견되는
오탐은 이 문서에 새 항목으로 추가하고, 해소 시 위와 같은 형식으로 갱신합니다.
