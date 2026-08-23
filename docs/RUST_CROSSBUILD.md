# PyO3 CPython 3.13 크로스 빌드 조사 보고서

> 초안은 `agy` 가 작성했고, 경로와 디스크 결론은 조율 세션에서 아카이브를 직접 열어 검증·수정했습니다.
> 수정된 항목: §1 의 Android · iOS 경로(둘 다 틀렸음), §5 의 결론. §1 의 abi3 항목은 **미결**입니다.

## 1. PyO3 CPython 3.13 확장 크로스 빌드 필수 요소

*   **abi3 사용 여부 — 결정되지 않았습니다.** 초안은 "CPython 3.13 전용이므로 `abi3` 는 불필요"
    라고 적었으나, 그 판단은 이 생태계의 전제를 보지 않은 것입니다. `PythonMultiplatform` 의
    FFI 계층이 **CPython Stable ABI 를 대상으로** 설계되어 있고(`EmbedAPI.kt` 의 `expect` 선언
    약 313개), 플랫폼마다 파이썬을 따로 싣는 구조에서 Limited API 를 쓰면 버전 고정이 풀립니다.
    반대로 abi3 는 쓸 수 있는 C API 가 좁아 `torch._C` 가 필요로 하는 것을 다 못 쓸 수 있습니다.
    **양쪽 다 실질적인 근거가 있으므로 결정 항목으로 남깁니다.**
*   **크로스 빌드 환경 변수 (PyO3 공식 문서 기준)**:
    *   `PYO3_CROSS=1`: PyO3에 크로스 컴파일 중임을 명시합니다.
    *   `PYO3_CROSS_PYTHON_VERSION=3.13`: 대상 파이썬 버전을 명시합니다.
    *   `PYO3_CROSS_LIB_DIR`: 타깃에 맞는 파이썬 라이브러리(`libpython`)가 위치한 경로입니다.
*   **타깃 파이썬 헤더/라이브러리 경로** — 아카이브를 직접 열어 확인했습니다. **두 플랫폼의 링크 방식이 다릅니다.**

    **Android (`aarch64-linux-android.tar.xz`)** — 통상적인 prefix 레이아웃.

    | | 경로 |
    |---|---|
    | 헤더 | `aarch64-linux-android/prefix/include/python3.13/` |
    | `PYO3_CROSS_LIB_DIR` | `aarch64-linux-android/prefix/lib/` — `libpython3.13.so`, `libpython3.so` |
    | 정적 대안 | `aarch64-linux-android/build/libpython3.13.a` |

    `prefix/lib/` 에 `libssl.a` · `libffi` · `libbz2.a` · `liblzma.a` · `libsqlite3` 가 함께 들어 있어
    정적 링크 시 이것들도 걸립니다. **`build/` 가 아니라 `prefix/` 를 씁니다** — 전자는 빌드 트리,
    후자가 설치 트리이고 헤더와 짝이 맞는 쪽이 후자입니다.

    **iOS (`arm64-iphoneos.zip`)** — **프레임워크입니다.**

    | | 경로 |
    |---|---|
    | 헤더 | `arm64-iphoneos/Python.framework/Headers/` (최상위 `include/python3.13` 에도 있음) |
    | 링크 대상 | `arm64-iphoneos/Python.framework/Python` — 5,335,216 B |
    | `lib/` | **비어 있음** (`.DS_Store` 뿐) |

    **`libpython*.a` 도 `.so` 도 없습니다.** 링크 가능한 산출물이 프레임워크 바이너리 하나뿐이므로
    `-lpython3.13` 이 아니라 프레임워크 링크가 되고, `PYO3_CROSS_LIB_DIR` 도 프레임워크를 가리켜야
    합니다. **`Cargo.kt` 는 이 비대칭을 다뤄야 합니다** — Android 는 lib, iOS 는 framework.

## 2. Android: cargo-ndk 필요성 및 설정

*   **cargo-ndk 필요 여부**: 필요합니다. Rust의 `cargo`만으로는 Android NDK 경로와 링커/컴파일러 플래그 설정이 매우 까다로우므로, 이를 대신 처리해주는 `cargo-ndk` 래퍼 도구를 사용하는 것이 표준이자 권장 방식입니다.
*   **NDK 버전**: 특정한 버전이 강제되지는 않으나, `pypackpack/docs/SPEC.md`의 `build/crossenv/android_21_arm64` 예시에 나타나듯 최소 API Level 21 이상을 기준으로 빌드해야 합니다.
*   **링커 설정**: 안드로이드에는 시스템 파이썬이 존재하지 않습니다. 따라서 `libpython.so` 등 크로스 컴파일된 파이썬 런타임을 링커가 찾을 수 있도록 링커 경로를 설정해야 하며, 배포 시 앱에 파이썬 런타임을 임베딩해야 합니다.
*   **`.so` 이름 규약**: 파이썬에서 `import torch._C`로 로드되려면 최종 산출물 파일명이 `libtorch_C.so`가 아닌 `_C.so`여야 합니다. Rust 빌드는 기본적으로 `lib` 접두사를 붙이므로, `Cargo.toml` 구성이나 빌드 후 이름 변경 처리가 필요합니다.

## 3. iOS: XCFramework 및 cargo-lipo 동향

*   **cargo-lipo 상태**: 현재 **Deprecated(유지보수 중단)** 상태입니다. 과거에는 Universal Binary 생성을 위해 쓰였으나 최신 워크플로우에서는 권장되지 않습니다.
*   **현재 권장 방식 (xcframework)**: `cargo build --target aarch64-apple-ios` (실기기용)와 `--target aarch64-apple-ios-sim` (시뮬레이터용) 등 개별 아키텍처로 타깃을 각각 빌드합니다. 이후 생성된 정적 라이브러리(`.a`)들을 Apple의 `xcodebuild -create-xcframework` 명령어를 통해 하나의 `XCFramework` 묶음으로 패키징하는 방식이 최신 표준입니다.
*   **iOS 프레임워크 패키징과 Rust cdylib 제약**: iOS는 동적 라이브러리 임의 로드에 대한 코드 서명(Code Signing) 및 보안 제약이 매우 엄격합니다. 파이썬 확장을 동적 라이브러리(`cdylib`, `.so`) 형태로 임의 로드하는 것은 제약이 따르므로, 통상적으로 Rust 코드를 정적 라이브러리(`staticlib`)로 빌드한 후 앱 메인 바이너리나 프레임워크와 정적으로 묶어서(Static Linking) 배포하는 것이 일반적입니다.

## 4. pypackpack 의 Cargo.kt 구현 방향

`pypackpack/docs/SPEC.md` 와 `BackendInterface.kt`, `Meson.kt` 소스코드를 분석한 결과, `Cargo.kt`가 구현되어야 할 인터페이스는 다음과 같습니다.

*   **인터페이스 준수**: `BackendInterface`를 상속받아 `initialize()`와 `suspend fun compile(packageName: String, extraArgs: Map<String, String>?)` 메서드를 구현해야 합니다.
*   **어댑터 패턴**: `Meson.kt`가 내부적으로 `meson` 프로세스를 호출하듯, `Cargo.kt`는 `cargo`를 래핑해야 합니다.
*   **NDK / XCode 어댑터 연계**:
    *   **Android 타깃**: `NDK.kt` 어댑터와 연계하여 내부적으로 `cargo` 대신 `cargo-ndk` 커맨드로 치환하거나, NDK 관련 링커 환경 변수를 주입하도록 분기해야 합니다.
    *   **iOS 타깃**: `XCode.kt` 어댑터와 연계하여 `cargo build --target aarch64-apple-ios` 등을 호출하고, 산출물을 `xcodebuild -create-xcframework`로 패키징하는 래퍼 역할을 수행해야 합니다.

## 5. 디스크 비용 추정 (rustup + 두 타깃 + cargo-ndk)

*   **rustup (Rust Toolchain)**: 약 450MB (Minimal Profile 사용 시) ~ 1.2GB (Default Profile 사용 시)
*   **두 타깃 (Android, iOS)**: `aarch64-linux-android`와 `aarch64-apple-ios` 툴체인은 각각 약 100~150MB를 차지하여 도합 약 300MB 소모
*   **cargo-ndk**: Cargo 플러그인용 단일 바이너리로 약 10~20MB 미만 소모
*   **툴체인 소계**: 대략 **750MB ~ 1.5GB**.

### 그러나 이 숫자로 블로커가 풀렸다고 볼 수 없다

위 추정은 **툴체인만** 셉니다. 실제로 `torch._C` 를 빌드할 때 자리를 차지하는 것은 빠져 있습니다.

| 빠진 항목 | 규모 |
|---|---|
| Android NDK | **수 GB** (위 보고서도 제외를 명시함) |
| `~/.cargo/registry` — 크레이트 소스와 캐시 | candle + PyO3 의존 트리 기준 **1~2GB** 추정 |
| `target/` 빌드 산출물 | Rust 디버그 빌드가 크고, **타깃마다 별도**로 쌓임. 수 GB |
| Xcode / iOS SDK | 수십 GB (이미 설치돼 있을 가능성이 높음 — 확인 필요) |

즉 "여유 4.6GB 로 도구 설치에는 무리가 없다" 는 **맞지만 답하는 질문이 다릅니다.** rustup 을
설치할 수 있느냐와 `torch._C` 를 빌드할 수 있느냐는 다른 질문이고, **후자는 여전히 열려 있습니다.**

`/Volumes/macMini` 는 349Gi 중 4.6Gi 여유(99% 사용)이고 내부 디스크는 48Gi 여유입니다. 저장소의
CLAUDE.md 는 외장 335GB 여유를 기록하고 있으나 **그 기록은 낡았습니다.** 공간을 확보하거나 캐시
위치를 정하는 것은 판단이 필요한 사안이라 여기서는 사실만 남깁니다.
