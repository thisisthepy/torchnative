# 실제로 설치되는 휠 — 호스트 플랫폼 휠을 처음 만든 기록

PyPI 의 `torchnative 0.0.1a0` 은 **껍데기**입니다. 열어 보면 `torchnative/` 파이썬 스켈레톤뿐이고
`_C` 확장도, 벤더링된 `torch/` 트리도 없습니다. 태그는 `py3-none-any` 라서 어느 기계에나 설치되고,
설치된 다음 `import torch` 가 `ModuleNotFoundError` 로 죽습니다.

이 문서는 그것을 대체할 **진짜 플랫폼 휠**을 만들고, 깨끗한 가상환경에 설치해 동작을 확인한
기록입니다.

**판정은 하나입니다 — 저장소를 한 번도 본 적 없는 인터프리터가 `pip install` 한 뒤 `import torch`
하고 연산이 돌아야 하고, 그때 `torch.__file__` 이 그 가상환경 안을 가리켜야 합니다.** 개발 트리를
가리키면 휠이 아니라 소스를 쓴 것이고 아무것도 증명하지 못합니다.

---

## 0. 한눈에

| | 이전 (`0.0.1a0` on PyPI) | 지금 |
|---|---|---|
| 태그 | `py3-none-any` | **`cp313-abi3-macosx_11_0_arm64`** |
| `_C` 확장 | 없음 | **`torch/_C.abi3.so`** 3,185,936 B |
| 벤더링 트리 | 없음 | **`torch` · `torchgen` · `functorch`**, `.py` 2372 개 |
| 파일 수 | 8 | **2,683** |
| 압축 크기 | 8 KB | **13.2 MB** (13,175,324 B) |
| 설치 후 크기 | — | **56.6 MB** (`.pyc` 생성 후 114 MB) |
| `pip install` → `import torch` | **ImportError** | **동작** (§2) |
| `torch.ops.aten.mm.default` | 도달 불가 | **동작**, 값 일치 |
| `nn.Linear` 순전파 | 도달 불가 | **동작** |
| `importlib.metadata.version("torch")` | 없음 | **`2.13.0`** |
| CPython 3.14.7 | — | **같은 휠로 동작** (§2.3) |
| `twine check` | — | **PASSED** |

기존 검증은 그대로입니다 — shim 테스트 **113/113**, 골든 하네스 **2268/2268, ops=97**.
둘 다 이 작업 전후로 같은 값이고, 종료 코드로 판정했습니다.

**아직 안 되는 것**: `print(tensor)`. 휠 문제가 아니라 shim 표면의 구멍이고 개발 트리에서 똑같이
재현됩니다. §5 에 정확한 목록이 있습니다.

---

## 1. 만드는 법

```sh
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-wheel   # 선택

bash vendor/vendor_torch.sh      # 상류 파이썬 트리 3 개 패키지를 가져온다
bash vendor/install_shim.sh      # _C 를 빌드해 트리의 구멍에 넣는다
python tools/wheel/build.py      # 휠을 만든다  -> dist/*.whl
python tools/wheel/verify.py dist/torchnative-*.whl   # 진짜 되는지 본다
```

빌드 인터프리터에는 `pip` · `setuptools` · `wheel` 이 필요합니다 (`build` 는 필요 없습니다 —
`tools/wheel/build.py` 가 `pip wheel --no-build-isolation` 으로 몰아넣습니다). 이 기록은
`/Volumes/macMini/caches/wheel-build-venv` (CPython 3.13.0, setuptools 84.0.0, wheel 0.48.0)
에서 만들었습니다.

**빈 체크아웃에서 위 네 줄이 그대로 돕니다.** 벤더링 트리를 통째로 지우고 (`git checkout` 으로
추적 파일 2 개만 복구) 처음부터 다시 돌려 같은 휠을 얻는 것으로 확인했습니다.

### `tools/wheel/build.py` 가 `pip wheel .` 보다 더 하는 것

| | 왜 |
|---|---|
| **preflight** | 벤더링 트리와 `_C` 가 없으면 **빌드를 거부**한다. 둘 다 `.gitignore` 라서 새 클론에는 없고, 그 상태로 setuptools 를 돌리면 **PyPI 에 있는 그 껍데기가 다시 나온다**. 조용히 실패하면 안 되는 지점이다 |
| **global-deps 스텁** | `torch/lib/libtorch_global_deps.dylib` 를 빈 라이브러리로 만들어 넣는다 (§3.2) |
| **retag** | `universal2` → `arm64`. setuptools 는 확장의 실제 아키텍처를 보지 않는다 (§3.3) |
| **install name** | cargo 가 박아 넣은 빌드 머신 절대경로를 `@rpath/_C.abi3.so` 로 바꾼다 (§3.4) |
| **상류 dist-info 주입** | `importlib.metadata.version("torch")` 가 답하게 한다 (§3.5) |
| **verify** | 아카이브를 소스 트리와 **파일 단위로 대조**하고 빠진 것이 있으면 실패한다. 작기만 한 휠은 설치는 되고 나중에 아무 임포트에서나 죽는다 |

---

## 2. 실제로 설치해서 동작하는 것

### 2.1 판정 출력

`tools/wheel/verify.py` 는 새 venv 를 만들고, 휠을 **의존성까지 함께** 설치하고, 저장소 **밖**
디렉터리에서 `-I`(PYTHONPATH·user site 무시)로 프로브를 돌립니다.

```
+ /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m venv /tmp/wheeltest
+ pip install torchnative-0.0.1a0-cp313-abi3-macosx_11_0_arm64.whl
  pulled in: filelock 3.32.4, fsspec 2026.7.0, Jinja2 3.1.6, MarkupSafe 3.0.3, mpmath 1.3.0,
             networkx 3.6.1, setuptools 84.0.0, sympy 1.14.0, torch 2.13.0,
             typing_extensions 4.16.0
+ import torch

  py                       3.13.0
  prefix                   /private/tmp/wheeltest
  torch_version            2.13.0
  torch_file               /private/tmp/wheeltest/lib/python3.13/site-packages/torch/__init__.py
  C_file                   /private/tmp/wheeltest/lib/python3.13/site-packages/torch/_C.abi3.so
  C_names                  1242
  aten_ops                 896
  metadata_version_torch   2.13.0
  torchnative              /private/tmp/wheeltest/lib/python3.13/site-packages/torchnative/__init__.py
  aten.mm.default          [[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]]  (torch.float32)
  x + x                    [2.0, 2.0, 2.0, 2.0]

PASS -- torchnative-0.0.1a0-cp313-abi3-macosx_11_0_arm64.whl installs into a clean venv and its torch computes
```

**`torch.__file__` 도 `torch._C.__file__` 도 `/private/tmp/wheeltest/` 안입니다.** 이것이 이
작업의 증명입니다. `torch_version` 이 `2.13.0` 인 것은 벤더링한 상류 트리의 버전이고,
`C_file` 이 `_C.abi3.so` 인 것은 그 트리가 상류 `_C` 가 아니라 우리 것을 쓰고 있다는 뜻입니다.

조금 더 얹은 것 (같은 venv):

```
$ ./bin/python -I -c "import torch, torch.nn as nn; m = nn.Linear(4,3); y = m(torch.ones(2,4)); print(tuple(y.shape), y.dtype)"
(2, 3) torch.float32
```

### 2.2 이 판정은 실패할 수 있다 — 확인함

"실패할 수 없는 검증은 검증이 아니다". `torch.__file__` 검사가 실제로 개발 트리를 잡아내는지
직접 확인했습니다. 같은 venv, 같은 명령, 차이는 `-I` 뿐입니다.

```sh
$ (cd /tmp/wheeltest && PYTHONPATH=<repo>/torchnative/src/main ./bin/python -I -c \
     "import torch; print(torch.__file__)")
/private/tmp/wheeltest/lib/python3.13/site-packages/torch/__init__.py     # 휠

$ (cd /tmp/wheeltest && PYTHONPATH=<repo>/torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 \
     ./bin/python -c "import torch; print(torch.__file__)")
<repo>/torchnative/src/main/torch/__init__.py                             # 개발 트리
```

아래쪽이 `verify.py` 가 `FAIL` 로 잡는 상태이고, 실제로 개발 중에 한 번 잡혔습니다 — macOS 의
`/tmp` → `/private/tmp` 심볼릭 링크 때문에 접두사 비교가 어긋나 **통과한 실행을 실패로** 보고했고,
그래서 양쪽을 `realpath` 로 정규화했습니다.

### 2.3 abi3 는 말뿐이 아니다 — 3.14 에서 같은 휠이 돈다

`cp313-abi3` 은 "3.13 **이상** 전부" 라는 주장입니다. 그 주장을 그대로 시험했습니다.

```
$ /Users/ibrew/.local/bin/python3.14 -m venv /tmp/wheeltest314
$ /tmp/wheeltest314/bin/pip install <같은 휠>      # 재빌드 없음, 태그 그대로
$ /tmp/wheeltest314/bin/python -I -c "..."
py 3.14.7
torch 2.13.0 /private/tmp/wheeltest314/lib/python3.14/site-packages/torch/__init__.py
_C /private/tmp/wheeltest314/lib/python3.14/site-packages/torch/_C.abi3.so
mm [[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]]
add [2.0, 2.0, 2.0, 2.0]
```

**3.13 으로 빌드한 바이너리 하나가 3.14.7 에서 그대로 로드되고 계산합니다.** docs/ABI3.md 가
비용 대칭성으로 고른 선택인데, 여기서 처음으로 그 이득이 실물로 확인됐습니다 — 버전 고정이었다면
3.14 용 휠을 따로 빌드해야 했습니다.

---

## 3. 무엇을 고쳐야 했는가

### 3.1 상류 torch 는 패키지 **하나가 아니다** — 이것이 제일 컸다

`vendor_torch.sh` 는 `torch/` 만 복사하고 있었습니다. 상류 배포본의 `top_level.txt` 는 세 개를
말합니다.

```
functorch
torch
torchgen
```

**PYTHONPATH 워크플로에서는 이 결손이 보이지 않습니다.** `PYTHONPATH=$PWD/torchnative/src/main`
은 `torch` 에 대해서만 site-packages 를 가리고, `torchgen` 과 `functorch` 는 **그 밑의 상류
설치본으로 조용히 해소**됩니다. 즉 지금까지의 "`import torch` 완주" 는 배포본의 3 분의 2 를
참조 설치가 대주는 상태에서 측정된 것입니다.

상류 torch 가 없는 기계에 휠을 얹는 순간 드러납니다:

```
torch/__init__.py:2254  from torch import _VF, functional
  -> torch/nn/modules/module.py:17  from torch.utils._python_dispatch import ...
     -> torch/utils/_python_dispatch.py:13  import torchgen
ModuleNotFoundError: No module named 'torchgen'
```

`torch/__init__.py` 3087 행 중 2254 행에서 죽습니다. 고친 방법은 `vendor_torch.sh` 가
`top_level.txt` 를 **읽어서** 형제 패키지를 함께 벤더링하게 한 것입니다 — 하드코딩하지 않은 이유는
상류가 형제를 추가·삭제했을 때 같은 방식으로 다시 놓치지 않기 위해서입니다.

부수 효과로 `.stamp` 의 `native_left` 계산을 고쳤습니다. rsync 의 `--exclude '*.so'` 는
`--delete` 로부터도 그 파일을 **보호**하므로, `install_shim.sh` 를 돌린 트리 위에 다시 벤더링하면
우리 `_C.abi3.so` 가 세어져 `native_left=1` 이 나옵니다. 그 숫자는 VENDOR.md §2 에서
"상류 네이티브가 하나도 안 남았다" 는 뜻으로 읽히는 값이므로, 우리 것을 세지 않도록 했습니다.

### 3.2 `libtorch_global_deps` — 벽 1 은 휠에서는 우회로 못 넘는다

VENDOR.md 벽 1: `torch/__init__.py:_load_global_deps()` 가 무조건
`ctypes.CDLL(torch/lib/libtorch_global_deps.dylib, RTLD_GLOBAL)` 을 하는데,
`vendor_torch.sh` 는 `torch/lib/` 를 통째로 버립니다. 이 저장소의 모든 실행이
`TORCH_USE_RTLD_GLOBAL=1` 로 그 분기를 건너뜁니다.

**소스 트리에서는 그래도 됩니다. 휠에서는 안 됩니다.** `pip install torchnative` 한 사람에게
"이 환경변수를 설정하세요" 라고 할 수는 없습니다. 게다가 그 우회 분기는 공짜가 아닙니다 —
`from torch._C import *` 전체를 `sys.setdlopenflags(RTLD_GLOBAL | RTLD_LAZY)` 아래에서
수행하고, 그 기기 영향은 VENDOR.md §7 항목 4 가 **미확인**으로 남긴 것입니다.

그래서 **빈 공유 라이브러리를 하나 넣습니다.** 상류의 그 파일은 MKL/OpenMP 를 libtorch 보다 먼저
전역 네임스페이스에 올리려고 존재하는데, 이 빌드는 둘 다 링크하지 않습니다 (`_C.abi3.so` 는
Accelerate 에 대해 자기완결적입니다). 즉 **비어 있는 것이 스텁이 아니라 정확한 내용**입니다.
다만 `install_shim.sh` 가 `torch/bin/torch_shm_manager` 에 쓰는 0 바이트 마커와는 달리 이쪽은
실제로 `dlopen` 되므로 **로드 가능한 이미지**여야 합니다.

`install_shim.sh` 가 아니라 휠 빌드에서 만드는 이유: 소스 트리는 기존 문서가 기술하는 대로
그대로 두기 위해서입니다. 벽 1 은 거기서 여전히 서 있고, 환경변수를 설정하는 스위트들은
자기가 잰다고 말하는 것을 계속 잽니다.

### 3.3 태그의 아키텍처가 거짓말이었다

setuptools 는 플랫폼 태그를 `sysconfig.get_platform()` 에서 가져온 뒤
`calculate_macosx_platform_tag` 로 아카이브를 훑는데, **그 훑기는 배포 대상 버전만 올리고
아키텍처는 건드리지 않습니다.** python.org 의 universal2 인터프리터로 빌드하면 안에 무엇이 있든
`macosx_11_0_universal2` 가 나옵니다. 우리 안에는 arm64 Mach-O 하나뿐입니다.

이것은 미관 문제가 아닙니다. `universal2` 는 인텔 맥의 pip 에게 "이건 네 것" 이라고 말하고,
설치는 성공하고, `import torch` 가 `dlopen` 의 아키텍처 불일치로 죽습니다 — 그러면 사용자의 증거는
이 태그가 아니라 자기 기계를 가리킵니다. 좁히면 그 사용자는 "no matching distribution" 을 받고,
그게 사실입니다.

`tools/wheel/build.py` 가 Mach-O 헤더(fat/thin, cputype)를 직접 읽어 `arm64` / `x86_64` /
`universal2` 를 정하고, 파일 이름과 `.dist-info/WHEEL` 의 `Tag:` 를 함께 고칩니다.

```
retag: macosx_11_0_universal2 -> macosx_11_0_arm64 (extension is arm64)
```

### 3.4 `_C.abi3.so` 가 빌드 머신 경로를 광고하고 있었다

```
$ otool -L torchnative/src/main/torch/_C.abi3.so
	/Volumes/macMini/caches/cargo-target-wheel/release/deps/lib_C.dylib   # LC_ID_DYLIB
```

cargo 가 cdylib 을 `-install_name <CARGO_TARGET_DIR>/...` 로 링크합니다. **동작에는 영향이
없습니다** — 경로로 `dlopen` 되는 모듈은 `LC_ID_DYLIB` 을 보지 않고, 그래서 이걸 고치기 전에도
임포트는 됐습니다. 다만 배포되는 산출물이 빌드한 기계의 디렉터리를 적어 두는 것은 결함입니다.

휠 안의 사본에만 `install_name_tool -id @rpath/_C.abi3.so` 를 걸고 **애드혹 재서명**합니다
(arm64 Mach-O 는 서명이 *없는* 것은 괜찮아도 *깨진* 것은 로드가 거부됩니다). 소스 트리의 파일은
cargo 가 낸 그대로 두므로, 스위트가 초록인 것은 이 재작성이 아니라 그 산출물에 대한 증거입니다.

```
$ otool -L /tmp/wheeltest/lib/python3.13/site-packages/torch/_C.abi3.so
	@rpath/_C.abi3.so
```

### 3.5 상류 `torch-2.13.0.dist-info` 를 어디에 넣는가

`importlib.metadata.version("torch")` 는 transformers 의 `is_torch_available()` 이 걸려 있는
호출이므로 (VENDOR.md) 장식이 아닙니다. 그런데 아카이브 **루트**에 둘 수는 없습니다 — pip 는
최상위에서 `.dist-info` 로 끝나는 이름을 훑어 자기 메타데이터 디렉터리를 찾고, 둘 이상이면
`UnsupportedWheel: multiple .dist-info directories found` 로 거부합니다.

`torchnative-0.0.1a0.data/purelib/torch-2.13.0.dist-info/` 로 넣습니다. 설치 시점에
site-packages 로 옮겨지고, 루트에는 여전히 하나만 있습니다. 상류의 `RECORD` 는 복사하지
않습니다 (`vendor_torch.sh` 도 안 합니다) — 우리 `RECORD` 가 이 파일들을 들고 있어서
`pip uninstall torchnative` 가 정리하고, 자기 `RECORD` 를 주면 `pip uninstall torch` 가
트리를 통째로 지우도록 초대하는 셈이 됩니다.

실측한 결과:

```
$ pip list | grep -i torch
torch             2.13.0
torchnative       0.0.1a0

$ pip install --dry-run --no-deps torch
Requirement already satisfied: torch in ./lib/python3.13/site-packages (2.13.0)

$ pip uninstall -y torch
× Cannot uninstall torch 2.13.0
╰─> The package's contents are unknown: no RECORD file was found for torch.

$ pip uninstall -y torchnative
Successfully uninstalled torchnative-0.0.1a0
$ ls site-packages | grep -i torch     # 남는 것 없음
```

앞의 셋은 의도한 것입니다 — 상류 torch 로 덮어쓰이지 않고, `is_torch_available()` 이 참이 되고,
제거는 우리 배포본을 통해서만 일어납니다. `pip uninstall torch` 의 거절 메시지가 "uv 가 설치한 것"
이라고 잘못 추측하는 것은 다듬을 여지입니다 (§6).

### 3.6 벤더링 트리는 자기 의존성을 데려온다

`dependencies = []` 가 옳았던 이유는 "**torch** 를 요구하면 자기 자신을 요구하는 것" 이라서인데,
그 문장이 상류의 **다른** 의존성까지 덮는 것으로 읽혀 있었습니다. 첫 플랫폼 휠은 그래서 설치는
되고 `torch/__init__.py:35` 의 `from typing_extensions import ...` 에서 죽었습니다.

상류 `Requires-Dist` 중 순수 파이썬 7 개를 그대로 가져왔습니다: `filelock`,
`typing-extensions>=4.10.0`, `setuptools>=77.0.3`, `sympy>=1.13.3`, `networkx>=2.5.1`,
`jinja2`, `fsspec>=0.8.5`.

**CUDA · triton 항목은 가져오지 않았습니다.** 전부 `platform_system == "Linux"` 가드가 붙어
있고 이 빌드에는 CUDA 경로가 아예 없습니다. 그대로 두면 모든 리눅스 설치가 ~2 GB 의 nvidia 휠을
끌어옵니다 — 이 프로젝트의 대상인 aarch64 보드까지 포함해서.

---

## 4. 휠 안에 무엇이 있는가

```
torch/                     상류 2.13.0 파이썬 트리 + 우리 _C
  _C.abi3.so               3,185,936 B   arm64, abi3, install name @rpath
  _C/*.pyi                 상류 스텁 26 개. `_C` 확장 옆의 *디렉터리*이고
                           importlib 는 확장을 먼저 고른다 (상류도 같은 모양)
  lib/libtorch_global_deps.dylib   16,840 B   빈 것 (§3.2)
  bin/torch_shm_manager    0 B   존재만 검사되는 마커 (VENDOR.md 벽 4)
  nn/federated.py          우리 add-hook
torchgen/                  native_functions.yaml 등. import torch 가 요구한다
functorch/
torchnative/               API 스켈레톤
torchnative-0.0.1a0.data/purelib/torch-2.13.0.dist-info/    §3.5
torchnative-0.0.1a0.dist-info/
```

크기:

| | |
|---|---|
| 휠 (압축) | **13,175,324 B** = 12.6 MiB |
| 아카이브 엔트리 | 2,683 |
| 압축 해제 | 56.6 MB |
| 설치 후 (pip 가 `.pyc` 생성) | torch 110 MB + torchgen 3.5 MB + functorch 420 KB + torchnative 52 KB |

**PyPI 의 기본 파일 크기 한도는 100 MB 이고 여유가 큽니다.** 벤더링 트리가 수백 MB 가 될 것이라는
예상은 빗나갔는데, 이유는 `vendor_torch.sh` 가 버리는 것이 바로 무거운 것들이기 때문입니다 —
`torch/lib/` 353 MB, `torch/include/` 61 MB, `torch/bin/` 7 MB. 남는 것은 파이썬 소스이고
잘 압축됩니다.

---

## 5. 이 휠로 아직 안 되는 것 — `print(tensor)`

작업 지시의 판정 스니펫은 이렇게 되어 있었습니다.

```python
print(torch.ops.aten.mm.default(x, torch.ones(4, 2)))
```

**연산은 됩니다. `print` 가 안 됩니다.**

```
File "torch/_tensor_str.py", line 409, in _str_intern
    if torch._C._functorch.is_functorch_wrapped_tensor(inp):
NotImplementedError: not implemented in torch._C shim:
    torch._C._functorch.is_functorch_wrapped_tensor
```

**이것은 휠의 결함이 아닙니다.** 개발 트리에서 `TORCH_USE_RTLD_GLOBAL=1 PYTHONPATH=...` 로
돌려도 같은 자리에서 같은 예외가 납니다. `_C` 표면의 구멍이고 포장과 무관합니다.

막고 있는 이름을 하나씩 스텁하며 `repr` 경로를 따라가 본 결과 **최소 8 개**입니다.

```
torch._C._functorch.is_functorch_wrapped_tensor
TensorBase.is_nested
torch.get_default_dtype                 (torch._C._get_default_dtype 자체가 없음)
TensorBase.is_sparse
TensorBase.layout
TensorBase.is_quantized
torch._is_functional_tensor
TensorBase._is_zerotensor
```

"최소" 인 이유: 이 열거는 막힌 이름을 `False` 로 갈아끼우고 다시 돌리는 거친 방법이라, 8 번째
뒤에서 스텁 자체의 타입 오류로 멈췄습니다. **8 은 하한이지 총계가 아닙니다.**

값을 꺼내는 것은 문제없습니다 — `.tolist()`, 산술, `nn.Linear` 순전파 전부 동작합니다. 그래서
`verify.py` 는 `repr` 대신 `.tolist()` 로 판정합니다. 막힌 것은 **텐서 서식화**입니다.

이 여덟 개는 거의 전부 "이 빌드에는 그런 것이 없다" 로 답이 정해져 있어 보이지만
(functorch 래핑 없음, 희소 텐서 없음, 양자화 없음), **그 판단은 `_C` 표면 작업의 몫이지 휠
작업의 몫이 아닙니다.** 여기서 고치지 않고 남겼습니다.

---

## 6. 아직 판단이 필요한 것

| | |
|---|---|
| **라이선스 메타데이터** | `pyproject.toml` 은 `license = "MIT"` 인데 휠의 대부분은 **BSD-3 인 상류 PyTorch 소스**다. 상류 LICENSE 와 third-party 라이선스들은 주입한 dist-info 안에 함께 실려 있지만, 배포본 메타데이터가 그것을 진술하지 않는다. **업로드 전에 정리해야 한다** |
| **`torch` 이름을 점유한다** | 설치되면 `pip list` 에 `torch 2.13.0` 이 뜨고 `pip install torch` 가 무시된다. 의도한 것이지만(pyproject 주석), 사용자가 상류 torch 를 원할 때 되돌리는 절차가 문서화돼 있지 않다. 지금은 `pip uninstall torchnative` 뒤 재설치가 유일한 경로다 |
| **`pip uninstall torch` 의 오진단** | 거절 메시지가 "uv 가 설치한 것 같다" 고 추측한다. RECORD 가 없기 때문인데, 우리가 일부러 안 넣은 것이므로 메시지가 사용자를 엉뚱한 곳으로 보낸다 |
| **sdist** | 만들지 않았다. 벤더링 트리가 git 에 없으므로 sdist 는 지금의 `py3-none-any` 와 같은 껍데기가 된다. `python -m build` 를 기본으로 쓰지 않는 이유이기도 하다 |

---

## 7. 호스트 밖 — Android · iOS

**지금 되는 것은 호스트 휠 하나뿐입니다.** 나머지는 아래가 정확한 현황입니다.

### 7.1 확장은 오늘 크로스 빌드된다 — 실측

Android 확장을 이 작업 중에 직접 빌드했습니다.

```sh
export ANDROID_NDK_HOME=$HOME/Library/Android/sdk/ndk/27.1.12297006
export PYO3_CROSS=1
export PYO3_CROSS_LIB_DIR=/Volumes/macMini/caches/target-python/aarch64-linux-android/prefix/lib
cargo ndk -t arm64-v8a --platform 21 build --release
```

```
EXIT=0
lib_C.so  4,471,688 B
ELF 64-bit LSB shared object, ARM aarch64, version 1 (SYSV), dynamically linked
```

iOS 배선은 docs/RUST_CROSSBUILD.md §0.5 가 기술하고 있고 (`PYO3_CONFIG_FILE` +
`TORCHNATIVE_PYTHON_FRAMEWORK_DIR`), 이번에 다시 돌리지는 않았습니다.

즉 **막고 있는 것은 컴파일이 아닙니다.**

### 7.2 크로스 휠에 필요한 것 — 항목별

| | Android | iOS |
|---|---|---|
| `_C` 크로스 빌드 | **된다** (위 실측) | 배선 문서화됨, 이번 미실행 |
| 플랫폼 태그 | PEP 738 `android_<api>_<abi>`, 예 `android_21_arm64_v8a` | PEP 730 `ios_<ver>_<arch>_iphoneos` / `_iphonesimulator` |
| 태그를 다는 방법 | `tools/wheel/build.py` 는 지금 **호스트 전용**이다. 산출물 경로와 태그를 인자로 받게 열어야 한다 | 같음 |
| global-deps 스텁 | **NDK `cc` 로 함께 크로스 컴파일해야 한다.** 지금은 호스트 `cc` 로 만든다 — 그대로 넣으면 Mach-O 를 안드로이드에 싣게 된다 | 같음 (`xcrun --sdk iphoneos cc`) |
| 의존성 해소 | `filelock` 등 7 개는 전부 순수 파이썬이라 `py3-none-any` 로 받아진다 | 같음 |
| 설치 주체 | 기기의 pip 가 아니라 앱 패키징 도구가 언팩한다 | 같음 |
| **검증** | **기기/에뮬레이터 필요.** 이번 작업 범위 밖 | 시뮬레이터 필요 |

**마지막 줄이 핵심입니다.** 크로스 휠을 만드는 것 자체는 위 네 항목이면 되지만, 만든 것이
동작하는지는 기기에서만 알 수 있습니다. 검증하지 않은 크로스 휠을 내는 것은 지금 PyPI 에 있는
껍데기와 같은 종류의 약속입니다 — 설치는 되고 동작은 확인된 바 없는 것. **그래서 만들지
않았습니다.**

(Android 기기에서 `import torch` 와 97 개 연산이 도는 것 자체는 별도로 측정돼 있습니다 —
README Status 와 docs/DEVICE.md. 다만 그것은 `PYTHONPATH` 로 트리를 얹은 것이지 휠 설치가
아닙니다.)

### 7.3 `docs/CARGO_KT.md` 와의 관계

설계 문서는 이 배선을 `pypackpack` 저장소의 Cargo 백엔드가 맡는 것으로 상정합니다. **이 작업은
그 저장소를 건드리지 않았습니다.** `tools/wheel/build.py` 는 그 백엔드가 생기기 전까지의 경로이고,
그때 옮겨질 것이라는 전제로 짧게 유지했습니다.

---

## 8. PyPI 로 갈 때

**아직 올리지 않았습니다.** 확인한 것과 남은 것:

| | |
|---|---|
| `twine check` | **PASSED** |
| 파일 크기 | 12.6 MiB — 기본 한도 100 MB 대비 여유 |
| 프로젝트 총량 | 플랫폼당 ~13 MB. macOS(arm64/x86_64) · Linux · Android · iOS 를 다 내도 기본 총량 한도에 한참 못 미친다 |
| 태그 | `cp313-abi3-macosx_11_0_arm64`. abi3 이므로 **파이썬 버전마다 낼 필요가 없다** — 3.13 · 3.14 를 하나로 덮는 것을 §2.3 에서 확인했다 |
| 여러 플랫폼 | 태그가 다르므로 같은 버전에 여러 파일을 올린다. 크로스 휠은 §7.2 가 끝나야 한다 |
| **라이선스** | §6 의 첫 줄. **이것이 업로드 전 진짜 블로커다** |
| `0.0.1a0` 재사용 불가 | PyPI 는 같은 버전에 같은 파일명을 다시 받지 않는다. 지금 `0.0.1a0` 에는 `py3-none-any` 가 이미 올라가 있다. 플랫폼 휠은 파일명이 다르므로 **같은 버전에 추가**할 수는 있는데, 그러면 그 버전이 "설치는 되는데 안 되는 것" 과 "되는 것" 을 동시에 갖게 된다. 새 버전을 쓰는 편이 낫다 — 판단 필요 |

`py3-none-any` 는 다른 모든 태그보다 **덜** 구체적이라, 같은 버전에 둘 다 있으면 pip 는 플랫폼
휠을 먼저 고릅니다. 그래도 태그가 맞지 않는 플랫폼(예: 리눅스)에서는 여전히 껍데기가 설치됩니다.

---

## 9. 이 작업이 만든 것 / 고친 것

**기능 추가**

- `tools/wheel/build.py` — 플랫폼 휠 빌드. preflight · global-deps 스텁 · retag · install name ·
  상류 dist-info 주입 · 아카이브 대조
- `tools/wheel/verify.py` — 깨끗한 venv 설치 + `torch.__file__` 판정
- `setup.py` — `has_ext_modules()` 와 `py_limited_api="cp313"`. 이 둘이 태그를
  `py3-none-any` 에서 `cp313-abi3-<plat>` 로 바꾼다

**결함 수정**

- `vendor/vendor_torch.sh` — `torchgen` · `functorch` 미벤더링 (§3.1). PYTHONPATH 워크플로가
  가리고 있던 결손
- `vendor/vendor_torch.sh` — 재벤더링 시 `native_left` 가 우리 `_C` 를 세던 것
- `pyproject.toml` — `dependencies = []` 가 상류의 순수 파이썬 의존성까지 비우고 있던 것 (§3.6)
- `pyproject.toml` — `packages.find` 가 `torch` 를 제외해 껍데기를 만들던 것

**문서**

- 이 문서
- `README.md` Install 절 — "`0.0.1a0` 은 이름 예약" 서술을 현재 상태로 갱신

**변경하지 않은 것**

- `rust/torch_c/` — 한 줄도. §5 의 구멍은 그대로 남겼다
- `vendor/install_shim.sh` — 소스 트리 동작을 기존 문서대로 유지
- 벤더링 트리 자체 — 전부 `vendor_torch.sh` 가 생성한 것
- `pypackpack` 저장소

**기존 검증** (이 작업 전후 동일, 종료 코드로 판정)

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh   ->  EXIT=0,  ok 113
$PY tools/golden/compare.py                 ->  EXIT=0,  2268/2268, ops=97
```
