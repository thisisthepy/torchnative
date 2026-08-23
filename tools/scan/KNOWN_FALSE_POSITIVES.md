# 알려진 오탐

§9 선결 과제 패치를 이 스캐너로 교차 검증하다 드러난 것들입니다. **패치가 옳고 스캐너가 틀렸습니다.**

## 1. 가드·지연화된 `torch.compile` 을 여전히 CRITICAL 로 본다

모듈 스코프의 `torch.compile` 은 import 만으로 실행되므로 치명적이지만, **함수 안으로 옮기고
가용성 가드를 붙이면 위험이 사라집니다.** 스캐너는 `import_time` 을 올바르게 `false` 로 보고하면서도
심각도는 CRITICAL 로 유지합니다.

사례: `thelethe/ops/normal_scan.py:237` — `compiled_scan()` 안에서 `_torch_compile_available()`
뒤에 있음. 고쳐진 코드인데 고치기 전과 같은 심각도로 보고됨.

**고칠 방향:** `import_time == false` 이고 같은 함수 안에 가용성 가드가 있으면 WARN 또는 INFO 로
내린다.

## 2. 헬퍼 함수를 거친 가드를 따라가지 못한다

`if torch.cuda.is_available():` 리터럴만 가드로 인정합니다. 아래는 못 알아봅니다.

```python
def _cudnn_available() -> bool:
    cudnn = getattr(torch.backends, "cudnn", None)
    return cudnn is not None and cudnn.is_available()

use_cudnn = _cudnn_available()
if use_cudnn:
    torch.backends.cudnn.deterministic = True   # 가드 안인데 CUDA_UNGUARDED 로 보고됨
```

사례: `ttadapters/methods/base.py:51-53, 64`.

**모순 하나:** 스캐너는 텐서 값 분기에서는 **변수 오염을 추적합니다** — `x = t.any()` 뒤의
`if x:` 를 잡아냅니다(간접 탐지). 가드에서는 같은 추적을 하지 않습니다. **같은 기제인데 방향만
반대입니다.** 오염 추적 코드를 가드 쪽에도 적용하면 됩니다.

## 왜 중요한가

**오탐이 많으면 CI 게이트는 무시당합니다.** 위 둘은 "고쳐도 경보가 안 꺼지는" 형태라 특히 나쁩니다 —
고친 사람이 스캐너를 신뢰하지 않게 됩니다.
