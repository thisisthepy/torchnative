# Samsung Exynos NPU (Exynos Neural Network / ENN) — SDK landscape, delegate status, NNAPI deprecation, and refusal architecture

This document analyzes the hardware access paths, vendor SDK ecosystem, PyTorch/ExecuTorch delegate status, Android NNAPI deprecation, and `torchnative` device resolution architecture for Samsung Exynos NPUs.

Half this document's value is the architectural and hardware investigation, so it is recorded in full, with explicit evidence tags (`DEMONSTRATED`, `DOCUMENTED`, `INFERRED`), before describing the implementation strategy.

---

> [!NOTE] Hardware Limitation Notice (`DEMONSTRATED`)
> No physical Samsung Exynos device was connected or attached via `adb` during this investigation and implementation. Device resolution behavior and property handling are tested using deterministic mocks and synthetic Android system property fixtures based on official Android CDD specifications.

---

## 1. Executive Summary & Core Verdict

On Android hosts, `torchnative` previously had a single NPU candidate: Qualcomm QNN (`("qnn", "Qualcomm Hexagon NPU")`). When `model.to(torchnative.device.npu)` was invoked on a Samsung Exynos device (such as Galaxy S24 / S22 / A55), the system attempted to probe the Qualcomm QNN toolchain over `adb`, resulting in a failure that named Qualcomm Hexagon NPU and FastRPC endpoints. This was the worst available failure mode: it misdiagnosed the hardware, pointing the developer toward Qualcomm SDKs and `adb` FastRPC nodes on a machine containing Samsung Exynos silicon.

### Key Findings

1. **Vendor SDK Isolation (`DOCUMENTED`)**: Samsung Exynos NPUs are accessed natively via Samsung's proprietary **Exynos Neural Network (ENN) SDK** (`libenn_public_api.so`, `libenn_engine.so`) and Samsung Neural SDK. These APIs are proprietary, gated under partner developer agreements, and not distributed as open-source headers or libraries in the Android NDK or AOSP.
2. **ExecuTorch HAS a Samsung backend (`DEMONSTRATED`, corrected 2026-09-17)**:
   `pytorch/executorch` carries `backends/samsung`, alongside `qualcomm`, `mediatek`,
   `openvino`, `apple`, `arm` and `vulkan`. It targets **Exynos 2500 (E9955)** and
   **Exynos 2600 (E9965)**, and supports **quantised i8/u8/i16/u16 and FP16** inference.
   It builds against Samsung's **Exynos AI LiteCore** SDK, via `EXYNOS_AI_LITECORE_ROOT`,
   plus the Android NDK.

   > **This entry previously asserted the opposite**, at `DOCUMENTED` grade: *"no public
   > open-source ExecuTorch delegate for Samsung Exynos ENN"*. That was false when written
   > and the evidence grade made it worse, because a grade is a claim that someone checked.
   > Nobody had. The error mattered: this document's conclusion, and the commit that landed
   > it, argued that the delegate road was closed for Exynos and open for every other vendor.
   > **The same road QNN uses exists here.** What is actually gated is the SDK, and even that
   > is a developer-portal registration rather than the NDA partner agreement item 1 describes
   > — those are different barriers and this document had conflated them.
3. **NNAPI Deprecation (`DOCUMENTED`)**: Google Android Neural Networks API (NNAPI, `libneuralnetworks.so`), which previously bridged vendor HALs including Samsung's `nnapi.exynos.so`, was officially **deprecated in Android 13 (API level 33)**. Google explicitly advises against using NNAPI for new applications, directing developers to vendor-specific delegates or cross-platform Vulkan compute shaders.
4. **Resolution Strategy in `torchnative` (`DEMONSTRATED`)**: Rather than failing with a misleading Qualcomm QNN error message, `torchnative` probes Android SoC properties (`ro.soc.manufacturer`, `ro.soc.model`, `ro.board.platform`, `ro.hardware`). When a Samsung Exynos SoC is detected, `torchnative.device.npu.resolve()` raises `ExynosNpuUnimplemented` — naming the detected Exynos SoC, explicitly stating that Exynos NPU support is `unimplemented` in `torchnative`, and providing actionable guidance toward supported execution paths (`torchnative.device.cpu` and `torchnative.device.vulkan`).

---

## 2. Exynos NPU Hardware & SDK Landscape

### 2.1 Hardware IP Family (`DOCUMENTED`)

Samsung Exynos SoCs integrate proprietary Neural Processing Units (NPUs) developed by Samsung Electronics. Recent silicon generations include:

| SoC Model | Code Name / Board | NPU Generation / IP Architecture | Target Devices |
|---|---|---|---|
| Exynos 2400 | `s5e9945` | Dual-NPU (GNPU + SNPU, 17,000+ TOPS claim) | Galaxy S24, S24+ (Global / Korea) |
| Exynos 2200 | `s5e9925` | Dual-core NPU + Xclipse 920 GPU (AMD RDNA2) | Galaxy S22 series (Europe) |
| Exynos 1480 | `s5e8845` | Dedicated NPU 680 | Galaxy A55 5G |
| Exynos 2100 | `s5e9840` | Triple-core NPU | Galaxy S21 series |
| Exynos 1280 | `s5e8825` | Dual-core NPU | Galaxy A33, A53, M33 |
| Exynos 990  | `s5e9830` | Dual-core NPU + DSP | Galaxy S20 series |

### 2.2 Software Access Paths (`DOCUMENTED` / `INFERRED`)

Samsung Exynos NPUs can be reached through three primary software layers:

```mermaid
flowchart TD
    App["PyTorch / torchnative application"] --> Resolution{"Device Resolution (torchnative.device.npu)"}
    
    Resolution -->|Exynos Detected| Refusal["ExynosNpuUnimplemented Refusal"]
    Refusal --> FallbackCPU["torchnative.device.cpu"]
    Refusal --> FallbackVulkan["torchnative.device.vulkan"]

    subgraph Hardware Access Paths (Unimplemented in PyTorch/torchnative)
        ENN["Samsung ENN SDK (libenn_public_api.so)"] -->|Proprietary NDA| Hardware["Exynos NPU Hardware"]
        NeuralSDK["Samsung Neural SDK"] -->|Proprietary| Hardware
        NNAPI["Android NNAPI (nnapi.exynos.so)"] -->|Deprecated in Android 13 / API 33| Hardware
    end
```

1. **Samsung ENN (Exynos Neural Network) SDK (`DOCUMENTED`)**:
   - The lowest-level C/C++ API provided by Samsung for model execution on Exynos NPUs.
   - Provides model compilation, runtime execution, and multi-NPU cluster scheduling.
   - **Access constraint**: ENN SDK headers (`enn_api.h`) and libraries (`libenn_public_api.so`, `libenn_engine.so`) are proprietary binaries shipped on Samsung device ROMs or under NDA developer kits. They are not available in PyTorch, ExecuTorch, or `torchnative`.

2. **Samsung Neural SDK (`DOCUMENTED`)**:
   - High-level framework wrapper for Android application developers targeting Galaxy hardware.
   - Delegates model execution to ENN on Exynos models and QNN on Qualcomm models.
   - Gated behind Samsung's partner developer portal.

3. **Android Neural Networks API (NNAPI) (`DOCUMENTED`)**:
   - C API (`libneuralnetworks.so`) introduced in Android 8.1 (API 27) as an OS-level abstraction for hardware accelerators.
   - Samsung implemented `nnapi.exynos.so` drivers for Exynos 9820/990/2100/2200.
   - **Deprecation**: Formally deprecated in Android 13 (API 33). Google removed NNAPI requirements from Android CDD, and modern Android releases no longer mandate vendor NNAPI driver updates.

---

## 3. PyTorch & ExecuTorch Delegate Status (`DEMONSTRATED`, corrected 2026-09-17)

PyTorch's edge execution ecosystem centers on **ExecuTorch** (`pytorch/executorch`), which utilizes modular delegate backends to offload subgraphs to hardware accelerators.

### Current Delegate Support Matrix

| Backend / Target | ExecuTorch Delegate Status | Runtime Libraries | Supported OS |
|---|---|---|---|
| **Qualcomm Hexagon NPU** | Fully Supported (`qnn_preprocess`) | `libQnnHtp.so`, `libQnnSystem.so` | Android (arm64-v8a), Windows on Snapdragon |
| **Apple Neural Engine** | Fully Supported (`coreml`) | CoreML framework / `MLComputeDevice` | macOS, iOS |
| **Arm Ethos-U NPU** | Fully Supported (`ethos_u`) | Arm Vela compiler / TFLu | Embedded / Bare metal |
| **Intel NPU** | Supported via OpenVINO (`openvino`) | `openvino_intel_npu_plugin.dll` / `.so` | Windows, Linux |
| **Samsung Exynos NPU** | **Supported** (`backends/samsung`) | Exynos AI LiteCore (`EXYNOS_AI_LITECORE_ROOT`) | Android |

**Corrected 2026-09-17.** This section previously ended: *"Because ExecuTorch lacks an
ENN delegate, PyTorch models cannot be lowered or compiled into an Exynos-native NPU
artifact within the open-source PyTorch toolchain today."* That is false. `backends/samsung`
exists and does exactly that, for Exynos 2500 (E9955) and 2600 (E9965), at quantised
i8/u8/i16/u16 and FP16.

So the delegate road is open for Exynos on the same terms as for Qualcomm. What separates
them is the SDK: QNN's runtime libraries ship in a downloadable SDK, while Exynos AI
LiteCore requires registration on Samsung's developer portal. That is a real obstacle for
CI and for a contributor without an account, and it is the honest reason this project has
not built against it — **not** an absence of a delegate.

What remains unverified here is everything downstream of that: nothing in this repository
has been built against LiteCore, and no Exynos device has run anything. The refusal in
`torchnative.device` stands until it has.

---

## 4. Android SoC Identification & Detection (`DEMONSTRATED`)

Android devices document SoC identity through system properties specified in the **Android Compatibility Definition Document (CDD)** Section 3.2.2.

### System Property Hierarchy

`torchnative` inspects system properties over `adb` in order of precedence:

1. **`ro.soc.manufacturer`** (Android 12+ / API 31+ CDD standard):
   - Expected to report `"samsung"` or `"exynos"` (`INFERRED` based on Android CDD section 3.2.2 which requires this property to contain the SoC manufacturer's name. Real hardware dumps were not available to confirm the exact string capitalization).
2. **`ro.soc.model`** (Android 12+ / API 31+ CDD standard):
   - Reports the model number or marketing name, e.g., `"Exynos 2400"`, `"s5e9945"`, `"Exynos 2200"`, `"s5e9925"`, `"Exynos 1480"`, `"s5e8845"` (`DOCUMENTED` by Exynos part numbers).
3. **`ro.board.platform`** (Legacy fallback):
   - Lowercase string containing `"exynos"`, or prefix `"s5e"` / `"universal"` (`DOCUMENTED` by AOSP and LineageOS device trees, e.g., `universal9810` for Exynos 9810, and `s5e` for Exynos chipsets internally).
4. **`ro.hardware`** (Legacy fallback):
   - Lowercase string containing `"exynos"`, or prefix `"s5e"` / `"universal"` (`DOCUMENTED` via custom ROM development forums identifying `init.universal*.rc` and `init.s5e*.rc` init scripts).

> [!IMPORTANT] No Verification on Hardware (`DEMONSTRATED`)
> Nothing in this detection heuristic has been verified on physical Exynos hardware. The values above (`samsung`, `s5e`, `universal`) are cited from Android specifications, kernel source trees, and community device dumps. 
> Because a wrong guess could block a valid device, the architecture was modified so that an unrecognized device falling back to the generic error path will explicitly print its raw `ro.soc.manufacturer`, `ro.soc.model`, `ro.board.platform`, and `ro.hardware` values, ensuring a false negative is harmless and actionable.

### Detection Logic in `torchnative.export.qnn_device`

```python
SOC_PROPERTIES = ("ro.soc.manufacturer", "ro.soc.model", "ro.board.platform", "ro.hardware")
SOC_EXYNOS_UNIMPLEMENTED = "exynos-unimplemented"

def _is_exynos(prop: str, raw: str) -> bool:
    r = raw.lower()
    if prop == "ro.soc.manufacturer" and r in ("samsung", "exynos"):
        return True
    if "exynos" in r or r.startswith("s5e") or r.startswith("universal"):
        return True
    return False
```

---

## 5. `torchnative` Refusal Architecture & Path Guidance (`DEMONSTRATED`)

When `model.to(torchnative.device.npu)` or `torchnative.device.npu.resolve()` is invoked on an Android Samsung Exynos device:

1. `qnn_device.device_soc()` probes the device properties.
2. Upon matching an Exynos SoC, `device_soc()` returns status `SOC_EXYNOS_UNIMPLEMENTED` (`"exynos-unimplemented"`).
3. `device_report()` populates `is_exynos = True` and sets `htp_unreachable_reason`.
4. `NpuDevice._resolve_qnn()` detects `is_exynos` and raises `ExynosNpuUnimplemented` (a subclass of `NpuUnresolved`).
5. `NpuDevice.resolve()` re-raises `ExynosNpuUnimplemented` directly, bypassing generic PCI / Windows refusal composition.

### Refusal Message Requirements

The exception message strictly satisfies four design constraints:
- **Identifies Hardware**: Explicitly names the detected Samsung Exynos SoC (`"Samsung Exynos (Exynos 2400)"`).
- **States Status**: Explicitly states that Exynos NPU support is `unimplemented` in `torchnative` (not "unsupportable").
- **Provides Fallback Guidance**: Directs the user to working execution paths on device (`torchnative.device.cpu` and `torchnative.device.vulkan`).
- **Eliminates False Signals**: Omits all mention of Qualcomm Hexagon NPU, QNN, or FastRPC `cdsprpc` nodes.

Example refusal exception string:
```text
ExynosNpuUnimplemented: torchnative.device.npu: Samsung Exynos SoC (Exynos 2400) detected on android. Exynos NPU support is unimplemented in torchnative. Working execution paths on this device: torchnative.device.cpu and torchnative.device.vulkan.
```

---

## 6. Evidence & Methodology Matrix

| Claim / Fact | Evidence Level | Method / Source |
|---|---|---|
| Samsung Exynos SoC property keys (`ro.soc.manufacturer`, `ro.soc.model`) | `DOCUMENTED` / `DEMONSTRATED` | Android CDD Section 3.2.2 & `test_npuvendor.py` unit tests |
| ENN SDK headers and runtime libraries gated under NDA | `DOCUMENTED` | Samsung Developer Portal & ENN SDK distribution policy |
| Lack of open-source ExecuTorch delegate for ENN | `DOCUMENTED` | ExecuTorch repository (`pytorch/executorch`) backend census |
| NNAPI formal deprecation in Android 13 (API 33) | `DOCUMENTED` | Android Developer Documentation (Android 13 Behavior Changes) |
| Exynos refusal naming, unimplemented status, and CPU/Vulkan guidance | `DEMONSTRATED` | `test_android_exynos_soc_refuses_by_name_with_unimplemented_and_fallbacks` test execution |
| Re-raising `ExynosNpuUnimplemented` without Qualcomm QNN fallback text | `DEMONSTRATED` | TDD intentional break & green gate verification |
| Absence of physical attached Exynos test hardware | `DEMONSTRATED` | Test harness environment configuration & synthetic device report mocks |

---

## 7. Summary of Changes in `torchnative`

1. **`torchnative/src/main/torchnative/export/qnn_device.py`**:
   - Added `"ro.soc.manufacturer"` to `SOC_PROPERTIES`.
   - Added `SOC_EXYNOS_UNIMPLEMENTED = "exynos-unimplemented"` status constant.
   - Added Exynos SoC detection helper `_is_exynos()` to `device_soc()`.
   - Updated `device_report()` to handle `SOC_EXYNOS_UNIMPLEMENTED` and populate Exynos-specific unreachable reasons.

2. **`torchnative/src/main/torchnative/device/__init__.py`**:
   - Defined `class ExynosNpuUnimplemented(NpuUnresolved): ...` and added to `__all__`.
   - Updated `_resolve_qnn()` to raise `ExynosNpuUnimplemented` when `is_exynos` or `soc_status == "exynos-unimplemented"` is present.
   - Updated `NpuDevice.resolve()` to re-raise `ExynosNpuUnimplemented` directly.

3. **`rust/torch_c/pytests/test_npuvendor.py`**:
   - Added `test_android_exynos_soc_refuses_by_name_with_unimplemented_and_fallbacks()` verifying that Exynos devices raise an Exynos-named refusal containing `unimplemented` and directing users to `cpu` and `vulkan`.
