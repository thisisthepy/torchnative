# PROJECT.md — torchnative 프로젝트 주요 사항

현황 · 구조 · 빌드와 테스트 · 결정과 그 근거 · 열린 질문. 에이전트 규정은 `AGENTS.md`,
의도는 `docs/INTENT.md`, 동작 계약은 `docs/SPEC.md`, 설계 근거는 `docs/design/DESIGN.md` 에 있습니다.
이 파일은 `develop` 에만 있고 `main` 에는 실리지 않습니다.

---

## 1. 한 줄 요약

**PyTorch/`transformers` 생태계는 그대로 두고 `torch._C` 만 Rust 네이티브 확장으로 교체해,
진짜 `torch` 와 `transformers` 가 기기(Android · iOS · 데스크톱)에서 돌게 합니다.** 재구현이
아니며, 정확성의 기준은 upstream PyTorch 와의 원소 단위 일치입니다.

## 2. 현황 (2026-10 기준, 상세는 `README.md` · `docs/platform/STATUS.md`)

| 영역 | 상태 |
|---|---|
| ATen 연산자 | 302 개, 골든 케이스 11,420 / 11,420 일치 |
| 아키텍처 | 297 / 297 forward(도달함), 판정 가능한 285 중 284 가 upstream 과 수치 일치 |
| 학습 | `loss.backward()` · 옵티마이저 스텝이 upstream 과 일치. double-backward 등은 이름으로 거부 |
| 적응 · 연합 | `Tent`(단계 1), 정규화 통계 재추정(단계 0), 다중 프로세스 FedAvg 동작 |
| 가속기 | Metal · Vulkan 은 실제 GPU 에서 계산. CoreML 은 Neural Engine 실행 확인. Intel NPU 는 가짜 프로브로만 검증, CUDA 는 배선만 |
| 배포 | 9 개 플랫폼 휠(`cp313-abi3`) 이 PyPI 에 있음(README 표 기준 `0.1.0b0`, `pyproject.toml` 은 `0.1.0b4`). Windows arm64 · iOS 기기는 실행된 적 없음 |
| 미구현 | `torch.compile`(영구 거부), `torchnative.kernels` 번들 리졸버, `TorchNativeAPI` |

`docs/SPEC.md` 가 항목별 상태(implemented / partial / planned)와 근거 테스트를 기록합니다.

## 3. 구조

```
crates/torch_c/          torch._C 확장 (Rust · pyo3 abi3-py313 · candle-core)
  src/aten.rs            연산자 커널 — 모든 op 은 _aten_dispatch 한 문으로 들어온다
  src/bootstrap.py       include_str! 로 확장에 구워지는 파이썬 부트스트랩
crates/vulkan_probe/     Vulkan 프로브 크레이트
crates/wasm_probe/       wasm 프로브 크레이트
python/
  torchnative/           파이썬 패키지: delta · adapt · nn/federated · device · transformers
                         · export · quant · kernels · api · distributed
  torch/                 upstream 벤더링 트리 (생성물, gitignore, 손대지 않음)
tests/                   게이트 스위트 (run.sh) · golden/ (upstream 값 대조) · docwatch/ (문서 검사기)
benches/                 측정 스크립트
scripts/                 vendor/ (vendor_torch.sh · install_shim.sh · vendor_candle.sh) · wheel/ ·
                         devices/ · scan/ · colab/
vendor/                  candle-core · candle-metal-kernels 포크와 그 패치
.github/                 workflows/ · scripts/ (CI 스크립트) · scripts/release/ (release-sync)
docs/<folder>/           회차별 측정 기록. 색인은 docs/README.md
docs/guide/              GitHub Pages 가이드 (영/한)
```

루트에 둘 수 있는 항목은 AGENTS.md §2 의 목록이 정합니다(#43).

## 4. 빌드와 테스트

```sh
# 벤더 트리와 확장 (worktree 마다 먼저 — AGENTS.md §14.1)
PYTHON=/Volumes/macMini/thisisthepy/torchnative/.caches/spike-venv/bin/python bash scripts/vendor/vendor_torch.sh
PYTHON=/Volumes/macMini/thisisthepy/torchnative/.caches/spike-venv/bin/python bash scripts/vendor/install_shim.sh

# 게이트 (단독 실행, 파이프 금지 — AGENTS.md §13)
PATH="$HOME/.cargo/bin:$PATH" PYTHON="$PWD/.caches/spike-venv/bin/python" \
    bash tests/run.sh > .scratch/gate.log 2>&1; echo "EXIT=$?"

# 휠
python scripts/wheel/build.py && python scripts/wheel/verify.py dist/torchnative-*.whl

# 가이드 사이트 검사
python3 docs/guide/check_guide.py

# 릴리스 브랜치 도구 검사
bash .github/scripts/release/test-sync-release.sh
```

## 5. 브랜치와 릴리스

- 작업은 `feat/<topic>` → PR → `develop` (AGENTS.md §4). `main` 은 `release` 에서 온 PR 로만 바뀝니다.
- `main` 으로 가는 길은 하나, release-sync 입니다. `release-sync.yml` 이 `develop` 푸시마다
  `.github/scripts/release/sync-release.sh` 로 `release` 를 재생성하고 `release → main` PR 을 엽니다.
  `main` 의 보호는 메인테이너의 몫이고, release PR 은 메인테이너가 병합합니다.
  `pages.yml` 이 `main` 푸시 때 `docs/guide/` 를 Pages 로 배포합니다 — `docs/guide/` 가 main 레이아웃에서
  살아남는다는 것은 `tests/test_publish.py` 가 실제 트리로 확인합니다.
- PyPI 배포는 `v*` 태그로 `publish-pypi.yml` 이 수행합니다(Trusted Publishing, 토큰 없음).
  **사용자 승인 없이 업로드하지 않습니다** (AGENTS.md §17.7).

옛 수동 경로 `publish_main.sh` 는 #43 에서 지웠습니다. `docs/` 를 통째로 빼서, 그 경로로 만든
`main` 은 Pages 사이트를 비웠을 것이고, 어떤 워크플로도 그것을 부르지 않았습니다.

## 6. 패키징 결정 (`pyproject.toml` · `setup.py`)

`pyproject.toml` 의 주석이 "PROJECT.md" 를 가리키는 곳이 여기입니다. **아래 필드를 바꾸기 전에
해당 항목을 읽으십시오** — 여러 번 "당연한" 값으로 바뀌었다가 깨졌습니다.

### 6.1 `setup.py` 는 휠 태그를 정한다 — 지우면 태그가 틀린다

선언적인 것은 전부 `pyproject.toml` 에 있고, `setup.py` 에는 `[tool.setuptools]` 로 쓸 수 없는
두 사실만 남았습니다.

1. **`has_ext_modules()` 가 `True` 여야 합니다.** `torch/_C.abi3.so` 는 `scripts/vendor/install_shim.sh` 가
   미리 빌드해 패키지 데이터로 들어오므로 setuptools 는 확장을 못 봅니다. 그대로 두면
   `py3-none-any` 휠이 나옵니다 — PyPI 의 **`0.0.1a0` 이 바로 그 휠**이고, `0.0.2a0` 부터는 올바릅니다.
2. **`py_limited_api = "cp313"`** (`bdist_wheel` 명령 옵션). 태그를 `cp313-abi3-<plat>` 로 만듭니다.

`setup.py` 를 설정 파일로 쓰는 것은 폐기되지 않았습니다(PEP 517, `setuptools.build_meta`).
Hatchling 은 둘 다 빌드 훅으로만 가능하고 `py_limited_api` 가 없어 기각했고, maturin 은 Rust 확장을
스스로 빌드한다는 전제가 맞지 않습니다.

### 6.2 `license` 는 upstream 의 표현식 그대로다

`Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT`
— torch 2.13.0 의 `License-Expression` 그대로입니다. 플랫폼 휠은 upstream 파이썬 트리를 싣기
때문입니다. **이 필드로는 어느 항이 우리 것인지 표현할 수 없습니다**(우리 라이선스 Apache-2.0 은
이미 포함). upstream 라이선스 원문은 `scripts/wheel/build.py` 가 torch 의 `dist-info` 와 함께 넣습니다.

### 6.3 `dependencies` — `torch` 는 없고, upstream 의 순수 파이썬 의존성은 있다

이 배포판은 `torch` 를 **제공**하므로 `torch` 를 요구하면 자기 자신에 대한 의존이 됩니다. 과거의
두 시도는 반대 방향으로 틀렸습니다 — 비워 두면 import 시 실패, `torch` 를 요구하면 upstream 휠이 없는
Android · iOS 에서 해결 불가. upstream torch 는 런타임 의존성이 아니라 **비교 기준**입니다. 따라서
설치된 PyTorch 와 공존할 수 없는 것은 결함이 아니라 귀결입니다.

나열된 것은 벤더링한 `torch-2.13.0.dist-info` 의 `Requires-Dist` 에서 가져온 순수 파이썬 의존성입니다
(`torch/__init__.py` 가 `typing_extensions` 를 먼저 import). **CUDA · triton 요구는 복사하지 않습니다** —
모두 `platform_system == "Linux"` 조건이고 Linux 설치마다 ~2 GB 를 끌어옵니다.

### 6.4 `classifiers`

- **`Development Status :: 2 - Pre-Alpha`** 는 의도적입니다.
- **`Python :: 3.13` 은 abi3 의 하한이지 상한이 아닙니다.** 3.14 · 3.15 는 측정으로 광고합니다 —
  `0.0.12a0` macOS arm64 휠이 3.14.7 과 3.15.0rc1 에서 같은 결과(`a @ b` 합 134.0)를 냈고,
  `test_release.py::test_the_abi3_wheel_loads_on_later_cpythons` 가 이를 지킵니다.
- **Linux · Windows** 분류자는 휠이 존재하고 pip 가 건네주기 때문에 선언합니다. 근거는 공개된 휠에
  대한 CI 실행이며, 정확한 기록은 README 의 플랫폼 표입니다.

### 6.5 `[project.optional-dependencies]` — 세 백엔드 extra

```toml
cpu = []
gpu = []
npu = ["openvino; (sys_platform == 'win32' and platform_machine == 'AMD64') or (sys_platform == 'linux' and platform_machine == 'x86_64')"]
```

휠 파일명에는 백엔드 축이 없어(pip 는 python · abi · platform 태그로만 고름) CPU/GPU/NPU 빌드를 한
플랫폼 태그 아래 나란히 둘 수 없습니다(`docs/platform/WHEEL.md` §13). 그래서 바이너리 하나가 셋을 다
싣고 런타임에 고릅니다 — 가속기 경로는 링크가 아니라 **dlopen** 이라 가능합니다
(`docs/devices/VULKAN.md`: `libvulkan` 이 `NEEDED` 에 없음). extra 는 휠 내용을 바꿀 수 없고 의존성만
더하므로, `cpu`/`gpu` 는 비어 있고 이름만 선언해 둡니다.

**`npu` 는 비어 있지 않습니다.** Intel NPU 경로(`torchnative/export/intelnpu.py`)는 OpenVINO C API 를
`ctypes` 로 부르고, `pip install openvino` 가 런타임 전체를 파이썬 패키지 안에 싣습니다.
`load_openvino_c` 가 자동으로 찾되 명시 경로와 `TORCHNATIVE_OPENVINO_C` 가 우선합니다. 마커는 Intel NPU 가
없는 플랫폼(macOS, 비 x86-64)을 제외합니다.

- **`federated = []`** — 적응만 쓸 때 연합 스택을 끌어오지 않게 합니다(`docs/design/DESIGN.md` §10).
- **`test = ["torch>=2.13,<2.14"]`** — 골든 하네스의 비교 기준. 한 릴리스의 `_C` 표면을 구현하므로
  그 릴리스에 고정합니다. 플랫폼 휠과 충돌하는 것은 설계상 당연합니다.

### 6.6 `[tool.setuptools]`

- **`package-dir = { "" = "python" }`** — 없으면 저장소 루트가 패키지 루트로 잡힙니다. 옛
  `torchnative/src/main` 레이아웃에서는 `import main.torchnative` 가 되는 휠이 나왔습니다.
- **`packages.find` 는 `torch` 를 포함합니다.** `python/torch` 는 남의 torch 에 붙이는 것이 아니라
  우리가 조립한 트리입니다. 제외했던 것이 PyPI 의 `py3-none-any` 배포판을 만들었습니다. 트리는 git 에
  없으므로 두 벤더 스크립트를 돌리기 전에는 찾을 것이 없고, `scripts/wheel/build.py` 는 그 상태에서 빌드를
  거부합니다.
- **upstream 패키지는 셋입니다** — `functorch`, `torch`, `torchgen` (`top_level.txt`). `import torch` 가
  `torch/utils/_python_dispatch.py:13` 에서 `torchgen` 에 닿습니다.
- **`package-data` 는 재귀 catch-all** — 확장자별 목록은 파일을 조용히 빠뜨렸습니다(특히 `torchgen/packaged/`).
- **`exclude-package-data`** — 바이트코드 캐시(빌드한 인터프리터의 매직 넘버에 묶임)와 **`.DS_Store`**
  (실제로 휠에 들어갔고, iOS 검사가 두 휠을 멤버 단위로 비교하다 잡음).
- **다른 배포판의 디렉터리에 쓰지 않습니다.** 위 `torch` 포함이 "남의 torch 에 이식"이 아니라는 구분이
  `torchnative/_cachedir.py` · `test_ovcache.py` 가 인용하는 원칙입니다.

### 6.7 `[tool.ppp]`

소스셋은 `python/` 입니다(#43 이전에는 pypackpack 의 `src/main` 레이아웃이었습니다). 그 최상위
패키지를 스캔하므로 한 패키지가 `torch` 와 `torchnative` 를 함께 제공할 수 있습니다
(`docs/design/DESIGN.md` §10).

## 7. 주요 결정 (요약)

| 결정 | 근거 |
|---|---|
| `torch._C` 만 교체, 나머지는 벤더링 | `docs/design/DESIGN.md` §2, §5 (A 안 채택) |
| 텐서 엔진은 candle | `docs/design/DESIGN.md` §4 |
| abi3 고정, `torch.compile` 영구 거부 | `docs/design/ABI3.md`, `docs/graph/COMPILE.md` |
| 가속기는 모듈 교체(앞단 고정 · 뒷단 교체) | `docs/graph/QUANT2.md` §3, AGENTS.md §20 |
| 핵심 추상은 수명이 타입에 박힌 가중치 델타 | `docs/design/DESIGN.md` §3 |
| `kernels` 는 계약 채용, 배포 역전(빌드 타임) | `docs/design/DESIGN.md` §8 |

## 8. 열린 질문

1. `docs/SPEC.md` 의 "Outside intent" 4 건 — WASM · Linux/Windows 휠, Vulkan 자작 백엔드, CUDA,
   `torchnative.transformers` 미러가 의도 안에 있는지.
2. 단계 0/1 을 **타입으로** 가르는 강제(DESIGN.md §2 의 첫째 강제 사항)는 아직 테스트가 없습니다.
3. (닫힘, #43) 임시 파일 위치는 `.scratch/` 하나입니다 — python-multiplatform 의 `.tmp/` 자리.
