# iOS 시뮬레이터 — 우리 `torch` 가 임포트되고 계산한다

**결론: 시뮬레이터 휠은 로드·임포트·계산까지 전부 확인됐다.** `docs/WHEEL.md` §7 이
"아티팩트만 확인" 으로 남겨 두었던 칸이 시뮬레이터 쪽에 한해 채워졌다.

**기기 휠은 여전히 확인되지 않았다.** 이유는 §7 에 있다 — 편의상의 유보가 아니라 별개의
아티팩트이기 때문이다.

## 0. 판정표

| | 빌드됨 | 설치됨 | 임포트됨 | 계산됨 |
|---|---|---|---|---|
| `ios_14_0_arm64_iphonesimulator` | ✅ | ✅ 시뮬레이터 CPython 의 site-packages 에 언팩 | ✅ | ✅ `aten.mm`, `x+x`, `nn.Linear` |
| `ios_12_0_arm64_iphoneos` (실기) | ✅ | ❌ 기기 없음 | ❌ | ❌ |

판정 하네스: `tools/wheel/verify_ios_sim.py`. `verify_android.py` 와 **판정 문장이 같다** —
*`torch.__file__` 이 설치 위치 안을 가리켜야 한다.*

```
PASS -- torchnative-0.0.1a0-cp313-abi3-ios_14_0_arm64_iphonesimulator.whl unpacks into
        an iOS CPython's site-packages and its torch computes in the simulator
```

## 1. 시뮬레이터가 답할 수 없는 것 — 먼저 적는다

**시뮬레이터는 이 M1 맥 위에서 도는 일반 프로세스다.** `simctl spawn` 이 해 주는 것은
`DYLD_ROOT_PATH` 로 시뮬레이터 런타임 루트를 씌우는 것뿐이고, **CPU 는 호스트 M1 그대로다.**

| 시뮬레이터가 답하는 것 | 시뮬레이터가 **답하지 않는** 것 |
|---|---|
| 확장이 로드되는가 (`dlopen`, 심볼 해결) | 아이폰 처리량 · 지연 |
| `import torch` 가 끝까지 가는가 | 실기 AMX / NEON 경로의 실제 성능 |
| 계산 결과가 호스트와 일치하는가 | 발열 · 스로틀링 · 전력 |
| 임베드 경로 (`PYTHONHOME`, 프레임워크 링크) | 실기 코드 서명 · 프로비저닝 |

**그래서 이 문서에는 성능 수치가 하나도 없다.** 재면 다음 사람이 그것을 아이폰 숫자로 읽는다.
이 저장소는 이미 같은 함정에 빠진 적이 있다 — 안드로이드 에뮬레이터가 광고한 `shaderFloat16`
비트가 사실 **호스트 M1 의 것이 그대로 전달된 것**이었다 (`docs/DEVICE.md` §10 정정).

### 그 노출이 여기서도 그대로 보인다

시뮬레이터 안에서 찍은 `platform.uname()` 이다:

```
system  = 'iOS'                     ← 시뮬레이터의 것
release = '18.0'                    ← 시뮬레이터의 것
machine = 'arm64'
version = 'Darwin Kernel Version 25.5.0: ... xnu-12377.121.6~2/RELEASE_ARM64_T8103'
                                    ↑ 호스트 맥의 커널. T8103 은 M1 이다
```

`system`/`release` 는 iOS 인데 **커널은 호스트의 것이 그대로 보인다.** 시뮬레이터는 iOS 커널을
흉내 내지 않고 호스트 커널 위에 iOS 사용자 공간만 씌운다. 성능이나 CPU 특성을 여기서 읽으면
안 되는 이유가 이 한 줄에 다 들어 있다. 하네스는 이 값을 매번 출력하고 그 옆에 경고를 찍는다.

## 2. 사다리 — 작은 것부터, 각 단계가 실제로 관측된 것

| 단계 | 결과 |
|---|---|
| 런처가 `print("hello")` | ✅ |
| `import sys` | ✅ `sys.platform == 'ios'`, CPython `3.13.0+` |
| stdlib (`json` `zipfile` `ctypes` `sysconfig`) | ✅ |
| `import _multiprocessing` | ❌ 배포본에 없음 — 스텁 필요 (§5) |
| 휠을 site-packages 에 언팩 | ✅ 호스트 네이티브 침입 0건 |
| `import torch` | ✅ 단, 런처가 UIKit 를 링크해야 한다 (§4) |
| `aten.mm` · `x+x` · `nn.Linear` | ✅ 호스트와 동일 (§3) |

## 3. 계산 결과 — 호스트와 대조

| | 시뮬레이터 (iOS 18.0) | 호스트 (macOS arm64) |
|---|---|---|
| `torch.__version__` | 2.13.0 | 2.13.0 |
| `len(dir(torch._C))` | **1251** | **1251** |
| `len(dir(torch.ops.aten))` | **896** | **896** |
| `aten.mm.default(ones(3,4), ones(4,2))` | `[[4,4],[4,4],[4,4]]` | `[[4,4],[4,4],[4,4]]` |
| `x + x` | `[2,2,2,2]` | `[2,2,2,2]` |
| `nn.Linear(4,3)(ones(2,4))` | `[2,3] float32` | `[2,3] float32` |
| `importlib.metadata.version("torch")` | 2.13.0 | 2.13.0 |

**전부 일치한다.** 개수 두 개(1251·896)까지 같다는 것은 크로스 빌드가 op 등록을 하나도 흘리지
않았다는 뜻이다.

`sys.path` 에 저장소로 이어지는 항목이 **하나도 없다** — 하네스가 이것을 단언으로 검사한다:

```
<prefix>/lib/python313.zip
<prefix>/lib/python3.13
<prefix>/lib/python3.13/lib-dynload
<prefix>/lib/python3.13/site-packages     ← torch.__file__ 이 여기 안
```

`PYTHONPATH` 를 쓰지 않는다. 안드로이드에서 얻은 교훈 그대로 — `PYTHONPATH` 엔트리는 **망가진
설치를 가려 주는 바로 그것**이다 (`WHEEL.md` §7.3).

## 4. 무엇이 막고 있었나 — `platform.system()` 이 UIKit 를 부른다

`import torch` 는 처음에 **`torch/__init__.py:370` 에서** 죽었다.

```
torch/__init__.py:444  _load_global_deps()
torch/__init__.py:370    if platform.system() == "Windows":
platform.py:1072         return uname().system
platform.py:1056         system, release, _, _ = ios_ver()
platform.py:517          result = _ios_support.get_platform_ios()
_ios_support.py:67       system = objc.objc_msgSend(device_systemName, SEL_UTF8String).decode()
AttributeError: 'NoneType' object has no attribute 'decode'
```

**휠의 결함이 아니다.** iOS 의 CPython 은 `platform.system()` 을 Objective-C 런타임에 물어본다
(`_ios_support.py:40` — `objc_getClass(b"UIDevice")`). `simctl spawn` 으로 띄운 **맨 프로세스에는
UIKit 가 로드되어 있지 않아** 그 클래스 조회가 nil 을 돌려주고, 이어지는 `objc_msgSend` 가 전부
nil 을 타고 내려가 마지막 `.decode()` 에서 터진다.

**진짜 앱에는 UIKit 가 항상 로드되어 있다.** 즉 이것은 앱 경로에는 없고 **spawn 하네스에만 있는
결손**이다. 그래서 프로브에 우회를 넣지 않고 **런처를 `-framework UIKit` 로 링크했다** — 하네스를
앱에 가깝게 만든 것이지 측정을 피해 간 것이 아니다.

```
UIKit 링크 후:  platform.system() -> 'iOS'
```

이 결론은 추측이 아니라 음성 대조로 확인했다 (§6).

## 5. `DEVICE.md` §4 의 환경 변수를 하나씩 빼서 쟀다

안드로이드에 필수라고 기록된 것들이 iOS 에서도 필요한지 **하나씩 제거하며** 측정했다.
안드로이드에서 그중 하나(`TORCH_USE_RTLD_GLOBAL`)가 실은 불필요했던 것이 최근 밝혀졌기 때문에
(`WHEEL.md` §7.3.1), 물려받지 않고 다시 쟀다.

| 뺀 것 | 결과 | 판정 |
|---|---|---|
| `PYTHONHOME` | `ModuleNotFoundError: No module named 'encodings'` (인터프리터 초기화 단계에서 죽음) | **필수** |
| `_multiprocessing`/`_posixshmem` 스텁 | `ModuleNotFoundError: No module named '_multiprocessing'` (`torch/multiprocessing/__init__.py:110`) | **필수** |
| `TORCH_USE_RTLD_GLOBAL` | 없어도 성공. 붙여도 같은 값 | **불필요** |
| UIKit 링크 | `AttributeError: 'NoneType' object has no attribute 'decode'` | **필수** (하네스에만) |

**`TORCH_USE_RTLD_GLOBAL` 은 안드로이드와 같은 이유로 불필요하다** — 휠이
`torch/lib/libtorch_global_deps.so` 를 싣기 때문이다. 그리고 그것이 장식이 아니라 실제로
로드되고 있다는 것을 음성 대조로 확인했다 (§6).

**`_multiprocessing` 은 휠의 결함이 아니라 iOS CPython 배포본의 성질이다** — 안드로이드 배포본과
똑같이 그것도 `_posixshmem` 도 빌드하지 않는다. 하네스는 스텁 없이 한 번 돌려서 그 실패를
**출력에 남긴다** — 필요한 것을 배경에 숨기지 않기 위해서다.

## 6. 실패할 수 있는 검증인지 확인했다

"실패할 수 없는 검증은 검증이 아니다." 통과를 만들어 낸 것 세 개를 각각 없애 보고, **올바른
이유로** 깨지는지 확인했다.

| 망가뜨린 것 | 관측된 실패 | 무엇을 증명하는가 |
|---|---|---|
| `libtorch_global_deps.so` 를 옆으로 치움 | `OSError: dlopen(.../libtorch_global_deps.so ...): (no such file)` | 그 파일이 **실제로 로드되고 있다.** `TORCH_USE_RTLD_GLOBAL` 이 필요 없어진 진짜 이유 |
| 런처를 UIKit 없이 재링크 | `AttributeError: 'NoneType' ... decode` 재현 | UIKit 링크가 §4 의 실제 원인 수정임 |
| `PYTHONHOME` 제거 | `No module named 'encodings'` | 스테이징한 prefix 를 실제로 쓰고 있다 |

하네스 자체도 잘못된 아티팩트를 거절한다:

```
$ verify_ios_sim.py dist/...-macosx_11_0_arm64.whl
torchnative-...-macosx_11_0_arm64.whl is not an iOS-simulator wheel.
The device wheel is a different artefact and this harness cannot run it.
```

## 7. 시뮬레이터를 검증해도 기기 휠은 검증되지 않는다

**둘은 별개의 아티팩트다.** Mach-O `LC_BUILD_VERSION` 의 platform 필드가 다르다:

| | platform | 값 |
|---|---|---|
| `ios_14_0_arm64_iphonesimulator` | `IOSSIMULATOR` | **7** |
| `ios_12_0_arm64_iphoneos` | `IOS` | **2** |

**크기도 아키텍처도 심볼도 같아서 그 필드 말고는 구별되지 않는다** (`WHEEL.md` §7.4).
게다가 링크 방식 자체가 다르다:

- **기기** 산출물은 `@rpath/Python.framework/Python` 을 `LC_LOAD_DYLIB` 에 박는다. 실기에는
  폴백할 libpython 이 없으므로 이것이 없으면 로드 자체가 불가능하다.
- **시뮬레이터** 산출물에는 그 의존이 **없다.** `-undefined dynamic_lookup` 으로 해소되고,
  그것이 정상이다 (`WHEEL.md` §7.1).

즉 시뮬레이터에서 심볼이 해결됐다는 것은 **기기에서 프레임워크 링크가 맞다는 근거가 되지
않는다.** 서로 다른 메커니즘이다. 기기 쪽을 닫으려면 실기 + 프로비저닝 프로파일이 필요하고
이 기계에 둘 다 없다.

## 8. 재현

```sh
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-ios
BPY=/Volumes/macMini/caches/wheel-build-venv/bin/python
PY=/Volumes/macMini/caches/spike-venv/bin/python

bash vendor/vendor_torch.sh
bash vendor/install_shim.sh

# 1) 시뮬레이터용 _C 를 크로스 빌드한다.
#    시뮬레이터도 PYO3_CONFIG_FILE 이 필요하다 (WHEEL.md §7.1). 기기와 달리
#    TORCHNATIVE_PYTHON_FRAMEWORK_DIR 은 필요 없다 — 프레임워크를 링크하지 않는다.
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
  PYO3_CROSS_LIB_DIR=/Volumes/macMini/caches/target-python/arm64-iphonesimulator/lib \
  cargo build --release --target aarch64-apple-ios-sim )

# 2) 휠을 만든다
$BPY tools/wheel/build.py --target ios-arm64-sim

# 3) 판정한다 — 시뮬레이터를 띄우고, 임포트시키고, 계산시킨다
$PY tools/wheel/verify_ios_sim.py dist/torchnative-*iphonesimulator*.whl
```

하네스는 시뮬레이터를 **자기가 부팅했을 때만** 종료시킨다. 이미 떠 있던 것은 건드리지 않는다.
`--udid` 또는 `IOS_SIMULATOR_UDID` 로 기기를 고정할 수 있고, 없으면 사용 가능한 첫 iPhone 을
고르고 **어느 것을 골랐는지 출력한다.**

측정에 쓴 것: iPhone 16 Pro / iOS 18.0 (`8F9F769D-...`), Xcode 26.6, macOS 26.5.1.

## 9. 하네스가 안드로이드와 다른 지점

| | Android (`verify_android.py`) | iOS 시뮬레이터 (`verify_ios_sim.py`) |
|---|---|---|
| 배포본에 인터프리터가 있는가 | 있음 (`bin/python3.13`) | **없음** — `Python.framework/Python` 은 `MH_DYLIB` 뿐 |
| 실행 파일 | 배포본의 것을 push | **직접 컴파일한다** — `Py_BytesMain` 3줄 (아래) |
| 기기로 옮기기 | `adb push` | **불필요** — 시뮬레이터는 호스트 파일시스템을 그대로 본다 |
| 실행 진입점 | `adb shell` | `xcrun simctl spawn <UDID>` (`DYLD_ROOT_PATH` 를 대신 채워 줌) |
| 환경 변수 전달 | 그대로 | `SIMCTL_CHILD_` 접두사 필요 |
| 추가로 필요한 것 | — | **UIKit 링크** (§4) |
| 스텁 | `_multiprocessing`, `_posixshmem` | **같음** |

### 인터프리터를 직접 만든 것에 대해

iOS 배포본에는 실행 파일이 없다. 그래서 하네스가 이것을 컴파일한다:

```c
#include <Python.h>
int main(int argc, char **argv) { return Py_BytesMain(argc, argv); }
```

`Py_BytesMain` 은 CPython 의 **CLI 진입점 전체**이고 프레임워크가 그것을 export 한다
(`nm -gU` 로 확인). 그래서 결과물은 시뮬레이터용 진짜 `python3.13` 이고, 프로브를
`verify_android.py` 와 **똑같은 `-s -P <script>` 인자로** 돌릴 수 있다.

이것이 `PyRun_SimpleString` 보다 나은 이유가 있다. `DEVICE_LOAD_IOS.md` 는 임베딩 API 경로에서
`sys.path` 에 cwd 가 안 붙어 `PYTHONPATH` 를 손으로 채워야 했다고 적었는데, **그 차이가 바로 두
플랫폼의 측정을 비교 불가능하게 만드는 지점**이다. `Py_BytesMain` 은 `-s -P` 를 진짜로 해석하므로
그 차이가 사라진다.

빌드된 런처가 시뮬레이터 플랫폼인지 하네스가 매번 확인한다:

```
$ xcrun vtool -show-build-version <launcher>
 platform IOSSIMULATOR
    minos 14.0
```

### 공유 캐시를 오염시키지 않는다

`/Volumes/macMini/caches/target-python/arm64-iphonesimulator` 는 다른 작업과 공유된다.
하네스는 그것을 `/Volumes/macMini/caches/ios-wheel-check/prefix` 로 **복사한 뒤** 그 사본의
site-packages 에 휠을 푼다 (205 MB). 공유본의 site-packages 는 그대로 남는다.

## 10. 아직 남은 것

| 빈 칸 | 필요한 것 |
|---|---|
| iOS **실기** 로드·임포트·계산 | 기기 + 프로비저닝 프로파일. 이 기계에 없다. §7 때문에 시뮬레이터 결과로 대신할 수 없다 |
| **실제 앱 번들** 경로 | `Python.framework` 를 `Frameworks/` 에 `Embed & Sign` 하고 앱 프로세스가 스스로 `Py_Initialize` 를 부르는 형태. rpath 가 `@rpath` 에서 `@executable_path/Frameworks` 로 바뀐다 |
| 아이폰 성능 | 실기에서만. 시뮬레이터에서 재면 M1 숫자다 (§1) |
| `_multiprocessing` | iOS CPython 배포본 재빌드, 또는 상류 `torch/multiprocessing` 지연 임포트. 안드로이드와 같은 항목이고 휠 작업의 몫이 아니다 |
| PEP 730 휠의 `.so` → `.framework` 변환 | App Store 규칙상 바이너리 모듈은 `.framework` 여야 한다. 그것은 앱 패키징 도구의 일이고(briefcase 가 그렇게 한다), 이 저장소 안에서 일어나는지는 아직 아무도 정하지 않았다 (`WHEEL.md` §7.4) |
