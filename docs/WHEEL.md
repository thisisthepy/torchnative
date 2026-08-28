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
| 태그 | `py3-none-any` 하나 | **플랫폼 휠 4 개** (아래) |
| `_C` 확장 | 없음 | **`torch/_C.abi3.so`**, 타깃별로 진짜 그 플랫폼의 바이너리 |
| 벤더링 트리 | 없음 | **`torch` · `torchgen` · `functorch`**, `.py` 2372 개 |
| 파일 수 | 8 | **2,686** (네 휠 모두 같음) |
| `pip install` → `import torch` | **ImportError** | **동작** (§2) |
| `torch.ops.aten.mm.default` | 도달 불가 | **동작**, 값 일치 |
| `nn.Linear` 순전파 | 도달 불가 | **동작** |
| `importlib.metadata.version("torch")` | 없음 | **`2.13.0`** |
| CPython 3.14.7 | — | **같은 휠로 동작** (§2.3) |
| `twine check` | — | **4/4 PASSED** |

| 휠 | 압축 | 설치 후 | `_C` | 어디까지 확인됐나 |
|---|---|---|---|---|
| `cp313-abi3-macosx_11_0_arm64` | 13,287,413 B | 56.9 MB | 3,509,008 B | **계산됨** — 깨끗한 venv (§2) |
| `cp313-abi3-android_21_arm64_v8a` | 13,623,651 B | 58.2 MB | 4,821,448 B | **계산됨** — 에뮬레이터 (§7.3) |
| `cp313-abi3-ios_12_0_arm64_iphoneos` | 13,313,090 B | 57.1 MB | 3,618,000 B | **아티팩트까지** (§7.4) |
| `cp313-abi3-ios_14_0_arm64_iphonesimulator` | 13,270,683 B | 56.9 MB | 3,476,720 B | **아티팩트까지** (§7.4) |

기존 검증은 그대로입니다 — shim 테스트 **168/168**, 골든 하네스 **2702/2702, ops=118**.
둘 다 이 작업 전후로 같은 값이고, 종료 코드로 판정했습니다.

**아직 안 되는 것**: iOS 휠이 로드·임포트·계산되는지는 **측정하지 않았습니다** (§7.0).
안드로이드는 `_multiprocessing` 스텁이 여전히 필요한데, 휠이 아니라 안드로이드 CPython
배포본의 성질입니다 (§7.3.2).

---

## 1. 만드는 법

```sh
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-wheel   # 선택

bash vendor/vendor_torch.sh      # 상류 파이썬 트리 3 개 패키지를 가져온다
bash vendor/install_shim.sh      # _C 를 빌드해 트리의 구멍에 넣는다
python tools/wheel/build.py      # 휠을 만든다  -> dist/*.whl
python tools/wheel/verify.py dist/torchnative-*macosx*.whl   # 진짜 되는지 본다
```

안드로이드 · iOS 휠은 `--target` 을 줍니다. 배선과 판정은 §7 에 있습니다.

빌드 인터프리터에는 `pip` · `setuptools` · `wheel` 이 필요합니다 (`build` 는 필요 없습니다 —
`tools/wheel/build.py` 가 `pip wheel --no-build-isolation` 으로 몰아넣습니다). 이 기록은
`/Volumes/macMini/caches/wheel-build-venv` (CPython 3.13.0, setuptools 84.0.0, wheel 0.48.0)
에서 만들었습니다.

**빈 체크아웃에서 위 네 줄이 그대로 돕니다.** 벤더링 트리를 통째로 지우고 (`git checkout` 으로
추적 파일 2 개만 복구) 처음부터 다시 돌려 같은 휠을 얻는 것으로 확인했습니다.

### `tools/wheel/build.py` 가 `pip wheel .` 보다 더 하는 것

| | 왜 |
|---|---|
| **preflight** | 벤더링 트리와 `_C` 가 없으면 **빌드를 거부**한다. 둘 다 `.gitignore` 라서 새 클론에는 없고, 그 상태로 setuptools 를 돌리면 **PyPI 에 있는 그 껍데기가 다시 나온다**. 조용히 실패하면 안 되는 지점이다. `--target` 을 줘도 이 검사는 **그대로 돕니다** — 크로스 산출물은 그 위에 얹는 *추가* 요구사항이지 대체가 아니다 |
| **global-deps 스텁** | `torch/lib/libtorch_global_deps.{dylib,so}` 를 빈 라이브러리로 만들어 넣는다. **타깃의 컴파일러로** 만들고, 만든 것이 정말 그 플랫폼인지 확인한다 (§3.2 · §7.4) |
| **retag** | 호스트: `universal2` → `arm64`, setuptools 는 확장의 실제 아키텍처를 보지 않는다 (§3.3). 크로스: PEP 738/730 태그를 타깃 CPython 에서 유도한다 (§7.2) |
| **install name** | cargo 가 박아 넣은 빌드 머신 절대경로를 `@rpath/_C.abi3.so` 로 바꾼다 (§3.4). 이미지 형식을 보고 판단하므로 ELF(안드로이드)에는 걸지 않는다 |
| **상류 dist-info 주입** | `importlib.metadata.version("torch")` 가 답하게 한다 (§3.5) |
| **verify** | 아카이브를 소스 트리와 **파일 단위로 대조**하고 빠진 것이 있으면 실패한다. 작기만 한 휠은 설치는 되고 나중에 아무 임포트에서나 죽는다. 크로스면 완성된 아카이브 안의 바이너리를 다시 열어 플랫폼을 확인한다 |

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

**이 파일이 실제로 로드되고 있다는 것은 이후에 기기에서 확인됐습니다** — 에뮬레이터에서 이
파일만 치우면 `import torch` 가 `dlopen failed: ... libtorch_global_deps.so not found` 로 죽습니다
(§7.3.1). 처음 만들 때는 "있으면 우회 분기를 안 탄다" 는 논증뿐이었습니다.

#### 3.2.1 파일 이름과 배포 대상 — 나중에 드러난 두 가지

**이름은 `.dylib` 가 아닐 수 있습니다.** `_load_global_deps()` 는
`".dylib" if platform.system() == "Darwin" else ".so"` 로 이름을 만드는데, `platform.system()` 은
**안드로이드에서 `"Android"`, iOS 에서 `"iOS"`** 입니다. 즉 **iOS 도 `.so` 를 찾습니다** — 내용은
Mach-O dylib 인 채로. 이것을 틀리면 조용합니다: `CDLL` 이 던진 `OSError` 를
`_load_global_deps` 가 `_preload_cuda_deps` 경로로 삼켜서, 임포트는 전혀 관계없는 곳에서 죽습니다.

**호스트용도 배포 대상이 틀려 있었습니다** — 이번에 고쳤습니다. 호스트 `cc` 는 자기가 가진 SDK
로 스탬프를 찍으므로 `macosx_11_0_arm64` 로 태그된 휠 안에 `Mach-O arm64 macos 26.0+` 짜리
파일이 들어가 있었습니다. dyld 는 그 필드를 강제하므로 **태그가 약속한 macOS 대부분에서 로드할
수 없는 파일**입니다. §3.3 과 같은 종류의 불일치이고, 같은 방향입니다 — 남의 기계에서만 드러나는
쪽. 지금은 빌드 인터프리터의 `MACOSX_DEPLOYMENT_TARGET` 을 `-mmacosx-version-min` 으로 넘겨
`macos 11.0+` 로 맞춥니다.

(이 결함이 실제로 사용자를 깨뜨린 것은 확인하지 않았습니다 — macOS 11~25 인 기계가 없습니다.
고친 근거는 재현이 아니라 **아카이브 내용과 태그가 어긋나 있다**는 것 자체입니다.)

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

> **닫혔습니다 (2026-08-28). `docs/E2E_REAL.md` §2 를 보십시오.** 아래 측정은 그대로 두되,
> 이 절이 "최소 8개" 라고 적은 목록은 **거절한 이름의 목록이지 필요한 셈의 목록이 아니었습니다.**
> 상류의 `repr` 이 실제로 무슨 op 을 디스패치하는지 재 보니 없던 것은 커널 6개였고,
> 여기 3번으로 적힌 `torch.get_default_dtype` 은 그 사이에 이미 구현되었으며
> (`docs/DISTRIBUTED.md` §3.4), 반대로 이 목록에 없던 `abs`·`ceil`·`gt.Scalar`·`min`·
> `unbind.int`·`masked_select` 가 필요했습니다. 그 차이를 만든 것이 계측기의 위치입니다 —
> `_str` 은 `_disable_current_modes()` 로 시작하므로 `TorchDispatchMode` 를 **밖에** 걸면
> 아무것도 기록되지 않습니다.

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

**이제 네 개의 휠이 있습니다.** 다만 넷이 같은 정도로 증명된 것은 아니고, 그 차이가 이 절의
전부입니다.

### 7.0 어디까지 확인됐는가

| | 빌드됨 | 설치됨 | 임포트됨 | 계산됨 |
|---|---|---|---|---|
| `macosx_11_0_arm64` | ✅ | ✅ 깨끗한 venv 에 `pip install` (§2) | ✅ | ✅ `aten.mm`, `nn.Linear` |
| `android_21_arm64_v8a` | ✅ | ✅ **에뮬레이터의 site-packages 에 언팩** (§7.3) | ✅ | ✅ `aten.mm`, `nn.Linear` |
| `ios_12_0_arm64_iphoneos` | ✅ | ❌ 실기 없음 | ❌ | ❌ |
| `ios_14_0_arm64_iphonesimulator` | ✅ | ❌ 시뮬레이터 하네스 없음 | ❌ | ❌ |

**iOS 두 개에 대해 이 문서가 주장하는 것은 "아티팩트가 맞다" 까지입니다** — 태그가 설치기가
매칭할 형태이고, 안의 바이너리가 진짜 그 플랫폼용이고, 기기용 쪽이 `Python.framework` 를
링크한다는 것. **로드된다·임포트된다·계산한다는 어느 것도 측정하지 않았습니다.** iOS 확장을
실행하려면 앱 번들과 서명이 필요하고 이 기계에 그 하네스가 없습니다.

빈 칸을 채우는 방법은 §7.6 에 적어 두었습니다.

### 7.1 만드는 법

```sh
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-wheel2
export ANDROID_NDK_HOME=$HOME/Library/Android/sdk/ndk/27.1.12297006
TP=/Volumes/macMini/caches/target-python

# 1) _C 를 타깃별로 크로스 빌드한다
scripts/device_android.sh build                       # aarch64-linux-android

cat > /tmp/pyo3-ios.cfg <<'EOF'
implementation=CPython
version=3.13
shared=true
lib_name=Python
pointer_width=64
suppress_build_script_link_lines=true
EOF

( cd rust/torch_c && PYO3_CONFIG_FILE=/tmp/pyo3-ios.cfg \
  PYO3_CROSS=1 PYO3_CROSS_PYTHON_VERSION=3.13 \
  PYO3_CROSS_LIB_DIR=$TP/arm64-iphoneos/lib \
  TORCHNATIVE_PYTHON_FRAMEWORK_DIR=$TP/arm64-iphoneos \
  cargo build --release --target aarch64-apple-ios )

( cd rust/torch_c && PYO3_CONFIG_FILE=/tmp/pyo3-ios.cfg \
  PYO3_CROSS=1 PYO3_CROSS_PYTHON_VERSION=3.13 \
  PYO3_CROSS_LIB_DIR=$TP/arm64-iphonesimulator/lib \
  cargo build --release --target aarch64-apple-ios-sim )

# 2) 휠을 만든다
python tools/wheel/build.py --target android-arm64-v8a
python tools/wheel/build.py --target ios-arm64
python tools/wheel/build.py --target ios-arm64-sim

# 3) 판정한다
python tools/wheel/verify_cross.py   dist/torchnative-*android*.whl
python tools/wheel/verify_cross.py   --self-test dist/torchnative-*android*.whl
ANDROID_SERIAL=emulator-5554 \
  python tools/wheel/verify_android.py dist/torchnative-*android*.whl
```

**시뮬레이터도 `PYO3_CONFIG_FILE` 이 필요합니다.** RUST_CROSSBUILD.md §0.5 는 그것을 실기
전용으로 적어 두었는데(`.cargo/config.toml` 주석과 `build.rs` 의 `is_ios_device` 도 같습니다),
실제로 돌려 보니 시뮬레이터 배포본에도 `libpython3.13.{a,dylib}` 이 없어 PyO3 의 `-lpython3.13`
이 그대로 실패합니다.

```
ld: warning: -undefined dynamic_lookup is deprecated on iOS-simulator
ld: library 'python3.13' not found
```

`suppress_build_script_link_lines=true` 만 추가하면 통과합니다 — 실기와 달리 `-framework Python`
은 필요 없고, `.cargo/config.toml` 이 이미 주는 `-undefined dynamic_lookup` 으로 해소됩니다.
그래서 시뮬레이터 산출물에는 `Python.framework` 의존이 **없고**, 그것이 정상입니다.
(`build.rs` 와 `.cargo/config.toml` 은 이 작업의 소유 범위 밖이라 손대지 않았습니다. 고칠 곳은
그 두 파일의 주석이지 배선이 아닙니다.)

### 7.2 태그 — 추측하지 않고 확인한 것

| | Android | iOS |
|---|---|---|
| PEP | 738 | 730 |
| 형태 | `android_<api>_<abi>` | `ios_<major>_<minor>_<multiarch>` |
| 실제 값 | `android_21_arm64_v8a` | `ios_12_0_arm64_iphoneos` · `ios_14_0_arm64_iphonesimulator` |
| 그 숫자의 출처 | 타깃 CPython 의 `ANDROID_API_LEVEL` | 타깃 CPython 의 `IPHONEOS_DEPLOYMENT_TARGET`, **단 산출물 쪽이 더 높으면 그쪽** |
| 숫자의 의미 | **하한**. `packaging` 은 기기의 레벨에서 16 까지 내려가며 태그를 만든다 | **하한**. 12.0 까지 내려간다 |

두 가지를 실제로 확인했습니다.

**하나. 형태를 명세가 아니라 `packaging` 에 물었습니다.** pip 가 "이 휠이 내 것인가" 를 판정할 때
쓰는 코드가 `packaging.tags.{android,ios}_platforms` 이므로, 거기서 나오는 목록에 우리 태그가
들어 있는지를 빌드가 검사합니다. 들어 있지 않으면 빌드가 실패합니다.

```
  tag android_21_arm64_v8a accepted by packaging.tags.android_platforms
  tag ios_12_0_arm64_iphoneos accepted by packaging.tags.ios_platforms
```

이 기계의 `packaging` 은 26.3 이고 두 생성기를 모두 가지고 있습니다. 없는 버전이면 검사를
**건너뛰었다고 출력**합니다 — 조용히 통과시키지 않습니다.

**둘. 시뮬레이터의 하한은 CPython 이 아니라 산출물이 정했습니다.** Rust 의
`aarch64-apple-ios-sim` 기본 배포 대상이 14.0 이라, CPython 의 12.0 을 그대로 쓰면 **자기 확장을
로드할 수 없는 OS 를 광고하는 휠**이 됩니다. `LC_BUILD_VERSION` 의 `minos` 를 읽어 더 높은 쪽을
택합니다.

```
  tag floor from the artefact (14.0), not from CPython (12.0)
```

실기 쪽은 반대로 산출물이 10.0(`LC_VERSION_MIN_IPHONEOS`)이라 CPython 의 12.0 이 이겼습니다.

### 7.3 Android — 기기에서 실제로 임포트하고 계산한다

`tools/wheel/verify_android.py` 가 하는 것은 §2 의 판정을 기기로 옮긴 것입니다. 판정 문장이
같습니다 — **`torch.__file__` 이 설치 위치 안을 가리켜야 한다.**

`PYTHONPATH` 를 쓰지 않는 것이 이전 측정과의 차이입니다. docs/DEVICE.md 의 모든 안드로이드
측정은 `PYTHONPATH=$ROOT/site` 로 트리를 얹은 것이고, **`PYTHONPATH` 엔트리는 망가진 설치를
가려 줄 수 있는 바로 그것**입니다. 여기서는 휠을 기기 CPython 의 **site-packages** 에 풀고,
`-s -P` 로(유저 site 없음, 작업 디렉터리 없음) 돌립니다.

```
$ ANDROID_SERIAL=emulator-5554 python tools/wheel/verify_android.py \
      dist/torchnative-0.0.1a0-cp313-abi3-android_21_arm64_v8a.whl

  sys.path  ['/data/local/tmp/bw_wheel/lib/python313.zip',
             '/data/local/tmp/bw_wheel/lib/python3.13',
             '/data/local/tmp/bw_wheel/lib/python3.13/lib-dynload',
             '/data/local/tmp/bw_wheel/lib/python3.13/site-packages']

  torch_file   .../site-packages/torch/__init__.py
  C_file       .../site-packages/torch/_C.abi3.so
  C_names      1249   aten_ops 896
  aten.mm      [[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]]
  x + x        [2.0, 2.0, 2.0, 2.0]
  nn.Linear    [2, 3] torch.float32
  metadata     torch 2.13.0

PASS -- torchnative-0.0.1a0-cp313-abi3-android_21_arm64_v8a.whl unpacks into an
        Android CPython's site-packages and its torch computes on the device
```

`sys.path` 에 저장소로 이어지는 항목이 하나도 없습니다. 기기에는 개발 트리 자체가 없으므로
`torch.__file__` 검사는 §2 만큼 날카롭지는 않지만, **`PYTHONPATH` 를 쓰지 않는다는 것**이 그
자리를 대신합니다.

`pip install` 은 아닙니다 — 기기 CPython 용 pip 가 이 기계에 없고 기기에서 받게 하면 네트워크가
필요합니다. 대신 이 휠이 실제로 쓰는 PEP 427 설치 동작만 흉내 냅니다: 아카이브를 site-packages
에 풀고, `.data/purelib/` 를 그 위로 옮기고(그래서 `importlib.metadata.version("torch")` 가
`2.13.0` 을 답합니다), 휠의 `Requires-Dist` 가 이름 대는 순수 파이썬 배포본들을 옆에 놓습니다.
그 목록은 여기 적어 둔 것이 아니라 **휠의 METADATA 를 읽어서** 만듭니다.

#### 7.3.1 `TORCH_USE_RTLD_GLOBAL` 이 더 이상 필요 없다

docs/DEVICE.md §4 는 이 변수를 안드로이드에서 **필수**로 기록했습니다. 그 측정은
`torch/lib/libtorch_global_deps.so` 가 **없는** 트리에 대한 것이었고, 휠은 그것을 싣습니다
(§3.2). 그래서 같은 기기에서 셋을 다 돌렸습니다.

| 실행 | 결과 |
|---|---|
| 스텁만, `TORCH_USE_RTLD_GLOBAL` 없이 | **성공** — 위 출력 |
| 스텁 + `TORCH_USE_RTLD_GLOBAL=1` | 성공, 같은 값 |
| 스텁, 변수 없이, **기기의 global-deps 파일을 치우고** | **실패** |

세 번째가 음성 대조입니다. 파일 이름만 바꾸고 같은 명령을 돌리면:

```
OSError: dlopen failed: library ".../site-packages/torch/lib/libtorch_global_deps.so" not found
  ctypes.CDLL(global_deps_lib_path, mode=ctypes.RTLD_GLOBAL)
```

**즉 그 파일은 장식이 아니라 실제로 로드되고 있고, 그것이 있어서 우회 변수가 필요 없어진
것입니다.** 이것으로 VENDOR.md 벽 1 이 기기에서도 닫혔습니다.

#### 7.3.2 아직 필요한 것 — `_multiprocessing` 스텁

`_multiprocessing` 스텁 없이 돌리면 여전히 죽습니다.

```
torch/multiprocessing/__init__.py:110  from multiprocessing.resource_tracker import ...
  -> lib/python3.13/multiprocessing/resource_tracker.py:41  import _multiprocessing
ModuleNotFoundError: No module named '_multiprocessing'
```

**이것은 휠의 결함이 아니라 안드로이드 CPython 배포본의 성질입니다** — 그 배포본은
`_multiprocessing` 도 `_posixshmem` 도 빌드하지 않습니다(안드로이드에 SysV IPC 가 없습니다).
`scripts/device_parity.py` 가 같은 이유로 같은 스텁을 답니다. `verify_android.py` 는 스텁 없이도
한 번 돌려서 그 실패를 **출력에 남깁니다** — 필요한 것을 배경에 숨기지 않기 위해서입니다.

이것을 진짜로 닫으려면 안드로이드 CPython 을 다시 빌드하거나 `torch/multiprocessing` 을
지연 임포트로 바꿔야 하고, 둘 다 휠 작업의 몫이 아닙니다.

### 7.4 iOS — 아티팩트까지가 답할 수 있는 전부다

실기도, 시뮬레이터 앱 하네스도 없으므로 **로드·임포트·계산은 측정하지 않았습니다.** 확인한
것은 파일 안에 무엇이 들어 있는가입니다.

```
torchnative-0.0.1a0-cp313-abi3-ios_12_0_arm64_iphoneos.whl
  binaries            2
    torch/_C.abi3.so                   Mach-O arm64 ios 10.0+
    torch/lib/libtorch_global_deps.so  Mach-O arm64 ios 12.0+
  ext suffix          .abi3.so present in Python
  file list           identical to torchnative-...-macosx_11_0_arm64.whl (2,684 entries)
```

각 줄이 무엇을 말하는지:

| | |
|---|---|
| `Mach-O arm64 ios` | `LC_BUILD_VERSION`/`LC_VERSION_MIN_*` 의 플랫폼 필드를 직접 읽은 것. **`ios`(2) 와 `iossimulator`(7) 은 이것 말고 구별되지 않습니다** — 크기도 아키텍처도 심볼도 같습니다 |
| `Python.framework` | 실기 산출물의 `LC_LOAD_DYLIB` 에 `@rpath/Python.framework/Python` 이 있습니다. 실기에는 폴백할 libpython 이 없으므로 이것이 없으면 로드 자체가 불가능합니다. 시뮬레이터 쪽에는 없고, 그것이 맞습니다(§7.1) |
| `.so` 인데 Mach-O | 파일 **이름**이 `.so` 여야 합니다. `_load_global_deps()` 가 `".dylib" if platform.system() == "Darwin" else ".so"` 로 이름을 만드는데 **iOS 에서 `platform.system()` 은 `"iOS"`** 입니다. 내용은 Mach-O dylib 이고 `dlopen` 은 확장자를 보지 않습니다 |
| `ext suffix` | `.abi3.so` 라는 문자열이 그 배포본의 `Python.framework/Python` 안에 실제로 있습니다. CPython 의 `_PyImport_DynLoadFiletab` 이 `{SOABI 접미사, ".abi3.so", SHLIB_SUFFIX}` 이고 그 상수들이 인터프리터에 컴파일되어 들어가므로, **기기 없이 이 질문에 답할 수 있는 가장 강한 형태**입니다. 안드로이드 쪽은 기기에서 직접 목록을 찍은 측정이 따로 있습니다(`scripts/device_android.sh` 주석) |
| `file list identical` | 트리가 하나도 빠지지 않았다는 것. 호스트 휠과 엔트리 집합이 정확히 같습니다 |

**여기까지가 이 문서가 iOS 에 대해 주장하는 전부입니다.** "빌드된다" 는 판정이 아니라는 것을
호스트 휠에서 배웠고, 그 교훈은 여기에도 그대로 적용됩니다 — 다만 지금은 판정할 수단이 없다는
것이 답입니다.

또 하나 미결로 남는 것: iOS 앱은 App Store 규칙상 바이너리 모듈을 `.framework` 로 담아야 하고,
PEP 730 휠 안의 `.so` 를 프레임워크로 바꾸는 것은 **앱 패키징 도구의 일**입니다(briefcase 가
그렇게 합니다). 이 휠은 규격대로 `.so` 를 싣고, 그 변환이 이 저장소 안에서 일어나는지는
아직 아무도 정하지 않았습니다.

### 7.5 검증이 실제로 실패하는지 확인했다

"실패할 수 없는 검증은 검증이 아니다". `verify_cross.py --self-test` 는 좋은 휠을 아홉 가지로
망가뜨린 뒤, 각각이 **올바른 이유로** 거절되는지 봅니다. 종료 코드만 보지 않고 보고 문구를
맞춰 보는 이유는, 파일 목록 비교가 나머지 여덟 개를 전부 흡수해 버리기 때문입니다.

```
SELF-TEST against torchnative-0.0.1a0-cp313-abi3-android_21_arm64_v8a.whl
  caught      extension built for the wrong platform
  caught      global-deps library missing
  caught      global-deps library under the host's name
  caught      wall-4 marker missing
  caught      a member edited without updating RECORD
  caught      part of the vendored tree dropped
  caught      WHEEL Tag: out of step with the filename
  caught      platform tag no installer would generate
  caught      abi tag downgraded from abi3

SELF-TEST: PASS -- 9/9 fault modes rejected
```

세 크로스 휠 전부에서 9/9 입니다. 손상은 헤더 필드를 직접 고쳐 만들므로 — ELF 의 `e_machine`,
Mach-O 의 `LC_BUILD_VERSION.platform` — **디스크에 다른 아티팩트가 있든 없든 아홉 개가 매번 다
돕니다.** 건너뛴 항목이 통과처럼 보이는 일이 없습니다.

기기 쪽 음성 대조는 §7.3.1 의 세 번째 줄입니다.

### 7.6 남은 칸을 채우려면

| 빈 칸 | 필요한 것 |
|---|---|
| iOS 시뮬레이터 임포트 | 최소 앱 번들 + `xcrun simctl` 로 실행하는 하네스. `Python.framework` 를 번들에 넣고 휠을 그 앱의 `app_packages` 에 풀면 `verify_android.py` 와 같은 프로브를 돌릴 수 있습니다 |
| iOS 실기 | 기기 · 프로비저닝 프로파일. 이 기계에 없습니다 |
| Android `_multiprocessing` | 배포본 재빌드, 또는 상류 `torch/multiprocessing` 지연 임포트 |
| `pip install` 로서의 안드로이드 설치 | 기기용 pip. 지금은 언팩으로 대신하고 있고, 그 차이를 §7.3 이 적어 두었습니다 |

### 7.7 `docs/CARGO_KT.md` 와의 관계

설계 문서는 이 배선을 `pypackpack` 저장소의 Cargo 백엔드가 맡는 것으로 상정합니다. **이 작업은
그 저장소를 건드리지 않았습니다.** `tools/wheel/build.py` 의 `--target` 은 그 백엔드가 생기기
전까지의 경로이고, `TARGETS` 표가 그때 옮겨질 것 — 타깃 하나는 "산출물 경로 · 컴파일러 ·
태그" 세 답이고, 그 세 개가 `Cargo.kt` 가 인코딩해야 할 것과 같습니다.

---

## 8. PyPI 로 갈 때

**아직 올리지 않았습니다.** 확인한 것과 남은 것:

| | |
|---|---|
| `twine check` | **4/4 PASSED** (macOS · Android · iOS · iOS 시뮬레이터) |
| 파일 크기 | 12.7~13.0 MiB — 기본 한도 100 MB 대비 여유 |
| 프로젝트 총량 | 플랫폼당 ~13 MB. 지금 있는 넷을 다 내도 기본 총량 한도에 한참 못 미친다 |
| 태그 | 전부 `cp313-abi3-*`. abi3 이므로 **파이썬 버전마다 낼 필요가 없다** — 3.13 · 3.14 를 하나로 덮는 것을 §2.3 에서 확인했다 |
| 여러 플랫폼 | 태그가 다르므로 같은 버전에 여러 파일을 올린다 |
| **iOS 두 개** | 만들어졌고 아티팩트는 맞지만 **한 번도 실행된 적이 없다** (§7.0). 검증되지 않은 휠을 올리는 것은 지금 PyPI 에 있는 껍데기와 같은 종류의 약속이다 — 설치는 되고 동작은 확인된 바 없는 것. **업로드 판단이 필요한 지점이고, 이 작업은 아무것도 올리지 않았다** |
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

---

## 10. 크로스 휠 회차 (2026-08-28) — 만든 것 / 고친 것

앞 절들은 호스트 휠을 처음 만든 기록이고, 이 절은 안드로이드 · iOS 를 붙인 회차입니다.
칸을 나눈 이유는 CLAUDE.md §5.3 입니다 — 테스트 수는 진척이 아니고, 무엇이 어느 칸인지
밝히지 않으면 삭제와 구현이 같은 줄에 섞입니다.

**기능 추가**

- `tools/wheel/build.py --target {android-arm64-v8a,ios-arm64,ios-arm64-sim}` —
  크로스 산출물 주입 · 타깃 컴파일러로 global-deps · PEP 738/730 태그 유도. 호스트 경로는
  동작이 바뀌지 않았고, 태그도 그대로 `macosx_11_0_arm64` 입니다
- `tools/wheel/binfmt.py` — Mach-O / ELF 를 직접 읽는다. `LC_BUILD_VERSION` ·
  `LC_VERSION_MIN_*` · `LC_LOAD_DYLIB` · ELF `e_machine`. 아카이브 안의 **바이트**에 대해
  물어야 하므로 `file(1)` 로는 안 됩니다
- `tools/wheel/verify_cross.py` — 크로스 휠 정적 판정 + `--self-test` (9 개 오류 모드)
- `tools/wheel/verify_android.py` — 기기 site-packages 설치 + `import torch` + 연산 판정

**결함 수정**

- `tools/wheel/build.py` — 호스트 global-deps 가 빌드 SDK 의 배포 대상(`macos 26.0+`)으로
  스탬프되어 태그(`macosx_11_0`)와 어긋나 있던 것 (§3.2.1)

**측정 (새 사실)**

- 안드로이드 휠에서 **`TORCH_USE_RTLD_GLOBAL=1` 이 더 이상 필요 없다.** docs/DEVICE.md §4 는
  이것을 필수로 기록하고 있고, 그 표는 이 결과로 정정되어야 합니다 (§7.3.1). 파일을 치우면
  다시 실패하는 것까지 확인했습니다
- iOS **시뮬레이터도 `PYO3_CONFIG_FILE` 이 필요하다.** RUST_CROSSBUILD.md §0.5 와
  `rust/torch_c/build.rs` 주석은 실기 전용으로 적고 있습니다 (§7.1)
- 세 타깃 산출물의 배포 대상: 실기 `ios 10.0+`, 시뮬레이터 `iossimulator 14.0+`,
  안드로이드 API 21. 시뮬레이터 쪽이 CPython 의 12.0 보다 높아 **태그 하한을 산출물이 정합니다**

**문서**

- 이 문서 §0 · §1 · §3.2 · §7 · §8. §0 의 호스트 숫자와 스위트 수(113/2268 → 168/2702)가
  낡아 있어 실측으로 갱신했습니다

**손대지 않은 것**

- `rust/torch_c/` · `tools/golden/` · `scripts/` · `vendor/` — 한 줄도
- `rust/torch_c/build.rs` 와 `.cargo/config.toml` — §7.1 이 지적하는 것은 그 두 파일의
  **주석**이지 배선이 아니고, 소유 범위 밖이라 남겼습니다
- `docs/DEVICE.md` — §7.3.1 이 그 문서의 표를 정정해야 하지만 소유 범위 밖입니다
- 업로드. 아무것도 PyPI 에 올리지 않았습니다

**기존 검증** (이 작업 전후 동일, 종료 코드로 판정)

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh   ->  EXIT=0,  ok 168
$PY tools/golden/compare.py                 ->  EXIT=0,  2702/2702, ops=118
$BPY tools/wheel/build.py                   ->  EXIT=0
$BPY tools/wheel/verify.py dist/*macosx*    ->  EXIT=0,  PASS
$BPY -m twine check dist/*.whl              ->  EXIT=0,  4/4 PASSED
```
