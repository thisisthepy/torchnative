# `torch/` — 벤더링 대상 자리

여기에 상류 torch 의 파이썬 트리(BSD)를 벤더링하고 `_C` 만 교체합니다 (DESIGN.md §2).
**아직 비어 있습니다.**

`__init__.py` 를 두지 않는 것은 의도입니다. DESIGN.md §11 의 1~3 단계가 전부 **데스크톱의 상류
torch 위에서** 이뤄지므로, 지금 이 디렉터리가 `torch` 를 가리면 그 검증 경로가 막힙니다.

## 미해결 — add-hook 을 어떻게 합치나

`nn/federated.py` 는 `torchnative.nn.federated` 를 `torch` 네임스페이스에 얹는 add-hook 입니다.
상류 torch 를 소유하기 전까지 이것을 실제 `torch.nn` 아래로 합치는 방법이 정해지지 않았습니다.
§2 의 규칙은 **add-hook 은 편의이지 의존이 아니어야 한다** 는 것이므로, 이것 없이도 `torchnative`
이 온전히 동작해야 합니다.
