//! The `vulkan` device: a tensor representation that lives outside candle.
//!
//! `docs/devices/VULKAN2.md` §5 sized this and deliberately wrote no code. Its finding
//! is the shape of this file: `candle_core::Device` is a closed enum with
//! nowhere to put a Vulkan handle, so a Vulkan tensor **cannot be a
//! `candle::Tensor` wearing a label**. It has to be a fourth arm of
//! `tensor::Repr`, exactly as `Repr::Quantized` is a third arm for the same
//! structural reason (candle's `QTensor` is a separate type system).
//!
//! **That inheritance is the single most important property here and it must
//! not be weakened.** `PyTensorBase::tensor()` refuses on every non-`Dense`
//! arm, and there are 396 call sites of it in this crate. So a kernel that has
//! not been taught the Vulkan arm *cannot* read CPU storage off a Vulkan
//! tensor by forgetting to check -- it gets a `PyResult` it has to handle, and
//! the only thing it can do with it is raise. **A silent CPU fallback is
//! structurally impossible**, which is what `docs/devices/VULKAN.md` §5 names as the
//! worst available outcome. Ops opt in one at a time, by name, in `dispatch`
//! below; everything else refuses *naming the op*.
//!
//! What is deliberately not here:
//!
//! * **No performance claim of any kind.** The only Vulkan driver reachable on
//!   this machine is `kosmickrisp`, a Vulkan-on-Metal translation layer, so a
//!   number measured here would describe the translator. `docs/devices/VULKAN2.md` §4.4
//!   says so and nothing in this file or its tests times anything.
//! * **No fallback.** If the loader is absent, `ones(..., device="vulkan")`
//!   raises with the loader's own error text. It never quietly returns a CPU
//!   tensor.
//! * **f32 compute, contiguous storage.** Arithmetic is float32 only; int64
//!   indices are *stored* (as int32) and moved, never computed on. Every
//!   tensor is a contiguous buffer, so a view materialises
//!   (docs/devices/VULKAN7.md §2). Every other case refuses by name rather
//!   than being approximated.
//!
//! ## Where the loader comes from
//!
//! The loader is *dlopened* rather than linked, so this module compiles and the
//! artefact loads on a machine with no Vulkan at all -- absence is a value that
//! gets reported, not a link error. `loader_candidates_for` lists where it is
//! looked for; on macOS that includes the Homebrew prefixes, where
//! `vulkan-loader` + `molten-vk` put it (docs/devices/VULKAN5.md §1).
//! On this host the loader and ICDs also exist inside the Android emulator's
//! bundle (`docs/devices/VULKAN2.md` §4), and pointing `DYLD_LIBRARY_PATH` and
//! `VK_DRIVER_FILES` at `libkosmickrisp_icd.json` gives the real `Apple M1`
//! GPU. **Those files are the Android SDK's private implementation detail and
//! this is a development/test path, never a shipping one.**

use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use ash::vk;
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;
use pyo3::types::{PyDict, PyModule, PyTuple};

use crate::device::PyDevice;
use crate::dtype::TorchDType;
use crate::err::not_implemented;
use crate::tensor::PyTensorBase;

/// The one kernel this round ships, as SPIR-V words.
///
/// Checked in beside its GLSL rather than compiled by a `build.rs`: this crate
/// builds for Android, iOS and the host, and making all three depend on a
/// shader compiler being on PATH for a file that changes about once a round is
/// a bad trade. `shaders/compile.sh` regenerates it and
/// `test_the_checked_in_spirv_is_not_stale` is the guard that an edit to the
/// `.comp` was actually compiled.
const ADD_F32_SPV: &[u8] = include_bytes!("../shaders/add_f32.spv");

macro_rules! spv {
    ($name:ident, $file:literal) => {
        const $name: &[u8] = include_bytes!(concat!("../shaders/", $file, ".spv"));
    };
}
spv!(SUB_F32_SPV, "sub_f32");
spv!(MUL_F32_SPV, "mul_f32");
spv!(DIV_F32_SPV, "div_f32");
spv!(RELU_F32_SPV, "relu_f32");
spv!(NEG_F32_SPV, "neg_f32");
spv!(GELU_F32_SPV, "gelu_f32");
spv!(SOFTMAX_LASTDIM_F32_SPV, "softmax_lastdim_f32");
    spv!(ALL_BOOL_I32_SPV, "all_bool_i32");
    spv!(SAFE_SOFTMAX_LASTDIM_F32_SPV, "safe_softmax_lastdim_f32");
spv!(NATIVE_LAYER_NORM_F32_SPV, "native_layer_norm_f32");
spv!(BMM_F32_SPV, "bmm_f32");
spv!(COPY_F32_SPV, "copy_f32");
spv!(TRANSPOSE2D_F32_SPV, "transpose2d_f32");
spv!(MATMUL_F32_SPV, "matmul_f32");
spv!(BIAS_ADD_F32_SPV, "bias_add_f32");
spv!(EMBEDDING_I32_F32_SPV, "embedding_i32_f32");
spv!(TRANSPOSE_BATCHED2D_F32_SPV, "transpose_batched2d_f32");
spv!(SCALAR_F32_SPV, "scalar_f32");
spv!(STRIDED_GATHER_U32_SPV, "strided_gather_u32");
spv!(GATHER_U32_I32_SPV, "gather_u32_i32");
spv!(TANH_F32_SPV, "tanh_f32");
spv!(BROADCAST_BINARY_F32_SPV, "broadcast_binary_f32");

// ---------------------------------------------------------------------------
// The instrument: how "it ran on the GPU" stops being an inference
// ---------------------------------------------------------------------------

/// Three process-wide counters, incremented at the only three places where
/// this module can cross the host/device boundary or make the GPU do work.
///
/// **Why these exist rather than a source-scanning test.** `docs/devices/MPSATTN.md`
/// §3.1 records, against its own round, a way to take an op off a refusal list
/// while keeping its host readback that *both* of that device's derivation
/// tests would still pass: the per-op scan looks for six helper names in a
/// kernel body and the classification test looks for three markers, so moving
/// the readback one call deeper -- into a helper named something else -- is
/// invisible to both. Every check of that shape reads the source and can be
/// defeated by moving the thing it greps for.
///
/// These counters cannot be defeated that way, because they are not a
/// description of the code: `SHADER_DISPATCHES` is incremented inside
/// `dispatch_kernel`, after `vkWaitForFences` has returned success, and
/// `HOST_DOWNLOADS` is incremented inside `download`, which is the only
/// `vkMapMemory`-for-reading in the module. An op that computed on the host
/// would have to read its operands, and reading them goes through `download`
/// however many helpers deep it is buried. So the test asserts, at runtime and
/// per op, **`shader_dispatches` went up by the expected number and
/// `host_downloads` did not move at all** -- which is a statement about what
/// the process did, not about what the source looks like.
static SHADER_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static HOST_UPLOADS: AtomicU64 = AtomicU64::new(0);
static HOST_DOWNLOADS: AtomicU64 = AtomicU64::new(0);

fn push_bytes(push: &[u32]) -> Vec<u8> {
    push.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// Instance, device, queue and the two caches, created once per process.
///
/// **Lifetime is the piece `vk_probe` does not have and `docs/devices/VULKAN2.md` §5.2
/// item 2 names.** The probe builds an instance, a pipeline and a command pool
/// per dispatch and tears them all down again, which is correct for a probe and
/// indefensible for a runtime: creating a `VkDevice` per tensor op would put a
/// driver-side allocation on the hot path and lose the pipeline cache entirely.
/// Here the context is a process-lifetime singleton and pipelines are built
/// once per kernel.
///
/// It is never destroyed, and that is a decision rather than an omission.
/// `vkDestroyDevice` at process exit would have to run *after* every
/// `Repr::Vulkan` tensor Python still holds has been dropped, and Python
/// guarantees no such ordering for module teardown. Leaking the device at exit
/// is what the driver already survives (the OS reclaims it); freeing it while a
/// live `VkBuffer` still names it is a use-after-free. So `Drop` frees buffers
/// and nothing frees the device.
pub struct VkContext {
    // Kept alive because the `Instance`/`Device` function pointers were loaded
    // out of it; dropping it would unload the library underneath them.
    _entry: ash::Entry,
    // Kept for the same reason `_entry` is: the `Device`'s function pointers
    // were loaded out of it. Never called after `init`, hence the underscore.
    _instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    qfi: u32,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    /// What the driver says about integers, **measured rather than assumed**.
    ///
    /// Asked because `docs/devices/VULKAN6.md` §1 had to decide how an index
    /// buffer is stored, and **the measurement contradicted the guess**: this
    /// MoltenVK/Apple M1 reports `shaderInt64 = true`. Native int64 indices are
    /// therefore possible *here* -- and are still not what this device does,
    /// because taking them would make the supported index range a property of
    /// the GPU rather than of the library (§1.2). Reported to Python so that
    /// argument can be checked against the device instead of trusted.
    pub shader_int64: bool,
    pub shader_int16: bool,
    pub shader_float64: bool,
    pub khr_8bit_storage: bool,
    pub khr_16bit_storage: bool,
    /// For the report `_vulkan_probe()` gives Python, so a skipped test can say
    /// which driver it skipped and a passing one can say what it ran on.
    pub device_name: String,
    pub device_type: String,
    /// Which of `loader_candidates()` was opened -- so a passing run can say
    /// which loader (and so which ICD search) it measured.
    pub loader: String,
    /// Vulkan requires *external* synchronisation on a queue and on a command
    /// pool -- the driver does no locking of its own. One mutex covers both,
    /// which is right while there is one queue: the critical section is the
    /// whole record-submit-wait, so a finer lock would buy nothing.
    submit: Mutex<vk::CommandPool>,
    /// The pipeline cache of `docs/devices/VULKAN2.md` §5.2 item 3. Keyed by kernel
    /// name; a `&'static str` because the set of kernels is closed and authored
    /// here, not discovered.
    pipelines: Mutex<HashMap<&'static str, Kernel>>,
}

/// Everything creating a compute pipeline produces, kept so it is created once.
#[derive(Clone, Copy)]
struct Kernel {
    dsl: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    dpool: vk::DescriptorPool,
}

// The handles above are opaque `u64`s and `ash` marks them `Send`/`Sync`; the
// mutability that Vulkan requires to be externally synchronised is behind the
// two mutexes.
unsafe impl Send for VkContext {}
unsafe impl Sync for VkContext {}

static CONTEXT: OnceLock<Result<VkContext, String>> = OnceLock::new();

/// The process's Vulkan context, or why there isn't one.
///
/// The error is a `String` and it is the *driver's* message, not ours: on a
/// machine with no loader the useful thing to print is `dlopen`'s own text.
/// `OnceLock` means a machine with no Vulkan pays one failed `dlopen` for the
/// life of the process rather than one per call.
pub fn context() -> Result<&'static VkContext, &'static str> {
    match CONTEXT.get_or_init(|| unsafe { init() }) {
        Ok(ctx) => Ok(ctx),
        Err(message) => Err(message.as_str()),
    }
}

/// The context, or a `NotImplementedError` carrying the driver's reason.
fn require(op: &str) -> PyResult<&'static VkContext> {
    context().map_err(|reason| {
        not_implemented(format!(
            "{op}: the vulkan device is not available in this process -- {reason}. \
             Nothing falls back to the CPU here (docs/devices/VULKAN.md §5): a vulkan \
             tensor is a VkBuffer, so there is nothing to fall back with."
        ))
    })
}

/// Where the Vulkan loader is looked for, in order (docs/devices/VULKAN5.md §1).
///
/// `ash::Entry::load()` tries one bare name, and on macOS that name is found
/// only on dyld's search path. Homebrew installs `vulkan-loader` under its own
/// prefix -- `/opt/homebrew/lib` on Apple Silicon, `/usr/local/lib` on Intel --
/// and neither is on that path, so a Mac with a loader installed reported "no
/// such file" and every Vulkan test skipped. Measured: `dlopen("libvulkan.dylib")`
/// failed and `dlopen("/opt/homebrew/lib/libvulkan.dylib")` loaded.
///
/// The bare name stays **first**, so `DYLD_LIBRARY_PATH` still chooses (that
/// is how the Android emulator's bundled loader is selected, docs/devices/VULKAN2.md
/// §4). After it: `$VULKAN_SDK/lib` (the LunarG SDK's convention), then the two
/// Homebrew prefixes. Each is a *fallback*, not an assumption -- a path that
/// does not exist costs one failed `dlopen` and is named in the error. Linux
/// keeps the soname lookup the dynamic linker already does well; no absolute
/// path is guessed there.
fn loader_candidates_for(vulkan_sdk: Option<&str>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    out.push("libvulkan.dylib".into());
    #[cfg(target_os = "macos")]
    {
        out.push("libvulkan.1.dylib".into());
        if let Some(sdk) = vulkan_sdk.filter(|s| !s.is_empty()) {
            out.push(format!("{sdk}/lib/libvulkan.1.dylib"));
        }
        out.push("/opt/homebrew/lib/libvulkan.1.dylib".into());
        out.push("/usr/local/lib/libvulkan.1.dylib".into());
    }
    #[cfg(windows)]
    out.push("vulkan-1.dll".into());
    #[cfg(any(target_os = "android", target_os = "fuchsia"))]
    out.push("libvulkan.so".into());
    #[cfg(all(
        unix,
        not(any(target_os = "macos", target_os = "ios", target_os = "android", target_os = "fuchsia"))
    ))]
    {
        out.push("libvulkan.so.1".into());
        out.push("libvulkan.so".into());
    }
    let _ = vulkan_sdk;
    out
}

pub fn loader_candidates() -> Vec<String> {
    loader_candidates_for(std::env::var("VULKAN_SDK").ok().as_deref())
}

/// The first candidate that loads, and its name -- or every attempt's error.
unsafe fn load_loader() -> Result<(ash::Entry, String), String> {
    let mut tried = Vec::new();
    for candidate in loader_candidates() {
        match ash::Entry::load_from(&candidate) {
            Ok(entry) => return Ok((entry, candidate)),
            Err(e) => tried.push(format!("{candidate}: {e}")),
        }
    }
    Err(format!("failed to load the Vulkan loader -- tried {}", tried.join(" | ")))
}

fn offers(props: &[vk::ExtensionProperties], name: &CStr) -> bool {
    props.iter().any(|p| p.extension_name_as_c_str() == Ok(name))
}

unsafe fn init() -> Result<VkContext, String> {
    let (entry, loader) = load_loader()?;
    let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 1, 0));
    // MoltenVK is a *portability* driver, and loaders from 1.3.216 on hide
    // those unless the instance opts in -- the refusal docs/devices/VULKAN2.md §4.2
    // recorded verbatim. Opted into only when the loader offers the extension,
    // so a loader that predates it (Android's) sees the same request as before.
    let offered = entry.enumerate_instance_extension_properties(None).unwrap_or_default();
    let mut instance_exts = Vec::new();
    let mut ci = vk::InstanceCreateInfo::default().application_info(&app);
    if offers(&offered, ash::khr::portability_enumeration::NAME) {
        instance_exts.push(ash::khr::portability_enumeration::NAME.as_ptr());
        ci = ci.flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
    }
    // MoltenVK compiles every shader with Metal's fast-math on by default, and
    // fast-math licenses rewriting `a / b`: `div` came back 3 of 20 elements
    // off upstream's bits on MoltenVK and bit-identical with
    // `MVK_CONFIG_FAST_MATH_ENABLED=0` (docs/devices/VULKAN5.md §3.1). The
    // exactly-rounded kernels here promise IEEE results, so this instance asks
    // for them, through the extension MoltenVK documents for configuration
    // (layer name "MoltenVK"; settings made here override its environment
    // variables). A driver that does not offer the extension is not asked.
    //
    // The name is the environment variable's, prefix included: measured,
    // `FAST_MATH_ENABLED` and `fastMathEnabled` were ignored exactly as a
    // nonsense name was (1199 of 4149 `div` elements off upstream's bits for
    // all three, 0 for this one).
    let fast_math_off = 0i32.to_ne_bytes();
    let mut settings = [vk::LayerSettingEXT::default()
        .layer_name(c"MoltenVK")
        .setting_name(c"MVK_CONFIG_FAST_MATH_ENABLED")
        .ty(vk::LayerSettingTypeEXT::INT32)
        .values(&fast_math_off)];
    settings[0].value_count = 1;
    let mut layer_settings = vk::LayerSettingsCreateInfoEXT::default().settings(&settings);
    if offers(&offered, ash::ext::layer_settings::NAME) {
        instance_exts.push(ash::ext::layer_settings::NAME.as_ptr());
        ci = ci.push_next(&mut layer_settings);
    }
    ci = ci.enabled_extension_names(&instance_exts);
    let instance = entry
        .create_instance(&ci, None)
        .map_err(|e| format!("vkCreateInstance: {e}"))?;

    let pds = instance
        .enumerate_physical_devices()
        .map_err(|e| format!("vkEnumeratePhysicalDevices: {e}"))?;
    // Prefer a real GPU over a software rasteriser when both ICDs are loaded.
    // Both are useful -- lavapipe exercises the whole stack on a machine with
    // no GPU -- but if an `Apple M1` is enumerated it is the one to run on, and
    // `_vulkan_probe()` reports which was chosen so a test can say so.
    let mut best: Option<(vk::PhysicalDevice, u32, String, String, u32)> = None;
    for &pd in &pds {
        let p = instance.get_physical_device_properties(pd);
        let qfs = instance.get_physical_device_queue_family_properties(pd);
        let Some(qfi) = qfs
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
        else {
            continue;
        };
        let name = CStr::from_ptr(p.device_name.as_ptr())
            .to_string_lossy()
            .into_owned();
        let rank = match p.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 3,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
            _ => 0,
        };
        let better = match &best {
            None => true,
            Some(b) => rank > b.4,
        };
        if better {
            best = Some((pd, qfi as u32, name, format!("{:?}", p.device_type), rank));
        }
    }
    let (pd, qfi, device_name, device_type, _) =
        best.ok_or_else(|| "no physical device has a COMPUTE queue family".to_string())?;

    let prio = [1.0f32];
    let qci = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(qfi)
        .queue_priorities(&prio)];
    // A device that advertises the portability subset must have it enabled
    // (the extension's own spec text); MoltenVK's does.
    let device_offered = instance.enumerate_device_extension_properties(pd).unwrap_or_default();
    let device_exts = [ash::khr::portability_subset::NAME.as_ptr()];
    let mut dci = vk::DeviceCreateInfo::default().queue_create_infos(&qci);
    if offers(&device_offered, ash::khr::portability_subset::NAME) {
        dci = dci.enabled_extension_names(&device_exts);
    }
    let device = instance
        .create_device(pd, &dci, None)
        .map_err(|e| format!("vkCreateDevice: {e}"))?;
    let queue = device.get_device_queue(qfi, 0);
    let mem_props = instance.get_physical_device_memory_properties(pd);
    // Asked once, here, and reported: `docs/devices/VULKAN6.md` §1 chose int32
    // index storage *because* of what these say on this machine, and a machine
    // where they say something else can see that in `_vulkan_probe()`.
    let features = instance.get_physical_device_features(pd);
    let khr_8bit_storage = offers(&device_offered, ash::khr::_8bit_storage::NAME);
    let khr_16bit_storage = offers(&device_offered, ash::khr::_16bit_storage::NAME);

    let cp_ci = vk::CommandPoolCreateInfo::default()
        .queue_family_index(qfi)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let pool = device
        .create_command_pool(&cp_ci, None)
        .map_err(|e| format!("vkCreateCommandPool: {e}"))?;

    Ok(VkContext {
        _entry: entry,
        _instance: instance,
        device,
        queue,
        qfi,
        mem_props,
        shader_int64: features.shader_int64 != 0,
        shader_int16: features.shader_int16 != 0,
        shader_float64: features.shader_float64 != 0,
        khr_8bit_storage,
        khr_16bit_storage,
        device_name,
        device_type,
        loader,
        submit: Mutex::new(pool),
        pipelines: Mutex::new(HashMap::new()),
    })
}

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------

/// One device allocation, freed when the last tensor naming it is dropped.
///
/// **One allocation per tensor, and that is stated as a limitation rather than
/// presented as an allocator.** `docs/devices/VULKAN2.md` §5.2 item 3 asks for a
/// suballocating allocator, and this is not one: a real one matters because
/// `maxMemoryAllocationCount` is a few thousand on desktop drivers and can be
/// 4096 on mobile, so a model's worth of weights would exhaust it. It is the
/// right shape for a two-by-two and the wrong shape for a model, and the next
/// round is where that changes.
pub struct VkBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    bytes: usize,
    /// `(min, max)` of the int64 values this buffer was filled with, recorded
    /// at upload -- the only moment an integer tensor can come into existence
    /// on this device (no kernel here produces one).
    ///
    /// **This is what lets `embedding` check its indices against the table
    /// without reading the device back.** The alternative -- download the
    /// index buffer and look -- would be a host round trip per call, which in
    /// a decode loop is one per token; `docs/devices/VULKAN6.md` §2 states
    /// that cost and why this side of the trade was taken. It lives on the
    /// `VkBuffer` rather than on the `VkTensor` so that `view`, `reshape`,
    /// `detach` and `alias` -- which share the `Arc<VkBuffer>` and rebuild the
    /// `VkTensor` -- carry it without a line of code each.
    ///
    /// A view of an index tensor (`slice`, `select`, `expand`, `gather`) holds
    /// a subset of its source's values, so it inherits the source's range as
    /// a **bound that may be loose** -- the exact one would need a read-back
    /// (docs/devices/VULKAN7.md §3). `IndexRange::exact` says which it is, so
    /// a refusal on a loose bound can say that rather than claim a value.
    index_range: OnceLock<IndexRange>,
}

/// The recorded `(lo, hi)` of an index buffer, and whether it is the buffer's
/// own or one inherited from the tensor it was viewed out of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexRange {
    pub lo: i64,
    pub hi: i64,
    pub exact: bool,
}

unsafe impl Send for VkBuffer {}
unsafe impl Sync for VkBuffer {}

impl Drop for VkBuffer {
    fn drop(&mut self) {
        // The context outlives every buffer by construction (it is a
        // `OnceLock` that is never cleared), so this cannot be `None` for a
        // buffer that exists -- a buffer can only have been made through it.
        if let Ok(ctx) = context() {
            unsafe {
                ctx.device.destroy_buffer(self.buffer, None);
                ctx.device.free_memory(self.memory, None);
            }
        }
    }
}

/// A tensor whose bytes are on the GPU. The fourth arm of `Repr`.
///
/// `Arc` because `Repr` is `Clone` and cloning a tensor here must not copy a
/// GPU allocation. Two clones therefore *share* storage, which matches candle's
/// `Dense` arm (a candle clone is an `Arc` clone) -- and it is safe in the same
/// way, because nothing on this device is in-place: every taught op allocates
/// its output.
#[derive(Clone)]
pub struct VkTensor {
    pub buffer: Arc<VkBuffer>,
    pub shape: Vec<usize>,
}

impl VkTensor {
    pub fn elem_count(&self) -> usize {
        self.shape.iter().product()
    }
}

impl VkContext {
    /// Allocate `bytes` of host-visible device memory.
    ///
    /// **Host-visible, and on this machine that is not a compromise.** The
    /// `Apple M1` reports as `INTEGRATED_GPU` with unified memory, so the
    /// memory type that is `DEVICE_LOCAL` is also `HOST_VISIBLE` and the map is
    /// a pointer into the same bytes the shader reads -- the upload is real and
    /// the copy is the only one there is. The `DEVICE_LOCAL` bit is *preferred*
    /// and falls back to plain host-visible, so a discrete GPU still works and
    /// is simply slower than it should be; that is where a staging buffer plus
    /// `vkCmdCopyBuffer` belongs, and it is not here.
    unsafe fn alloc(&self, bytes: usize) -> Result<VkBuffer, String> {
        let size = bytes.max(4) as vk::DeviceSize;
        let ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = self
            .device
            .create_buffer(&ci, None)
            .map_err(|e| format!("vkCreateBuffer: {e}"))?;
        let req = self.device.get_buffer_memory_requirements(buffer);

        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let mut chosen = None;
        for want in [host | vk::MemoryPropertyFlags::DEVICE_LOCAL, host] {
            chosen = (0..self.mem_props.memory_type_count).find(|&i| {
                req.memory_type_bits & (1 << i) != 0
                    && self.mem_props.memory_types[i as usize]
                        .property_flags
                        .contains(want)
            });
            if chosen.is_some() {
                break;
            }
        }
        let Some(idx) = chosen else {
            self.device.destroy_buffer(buffer, None);
            return Err("no host-visible coherent memory type".to_string());
        };

        let ai = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(idx);
        let memory = match self.device.allocate_memory(&ai, None) {
            Ok(m) => m,
            Err(e) => {
                self.device.destroy_buffer(buffer, None);
                return Err(format!("vkAllocateMemory: {e}"));
            }
        };
        if let Err(e) = self.device.bind_buffer_memory(buffer, memory, 0) {
            self.device.destroy_buffer(buffer, None);
            self.device.free_memory(memory, None);
            return Err(format!("vkBindBufferMemory: {e}"));
        }
        Ok(VkBuffer {
            buffer,
            memory,
            bytes: req.size as usize,
            index_range: OnceLock::new(),
        })
    }

    /// Host to device. The upload half of `docs/devices/VULKAN2.md` §5.2 item 5.
    unsafe fn upload(&self, buf: &VkBuffer, data: &[f32]) -> Result<(), String> {
        let p = self
            .device
            .map_memory(
                buf.memory,
                0,
                buf.bytes as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|e| format!("vkMapMemory: {e}"))? as *mut f32;
        std::ptr::copy_nonoverlapping(data.as_ptr(), p, data.len());
        self.device.unmap_memory(buf.memory);
        HOST_UPLOADS.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Host to device, int32 words. The index half of `docs/devices/VULKAN6.md` §1.
    ///
    /// Separate from `upload` rather than generic over the word type, because
    /// the two are not interchangeable at the call site: this one is reached
    /// only after every value has been checked to fit in an int32, and that
    /// check is what makes the narrowing a stated bound instead of a truncation.
    unsafe fn upload_i32(&self, buf: &VkBuffer, data: &[i32]) -> Result<(), String> {
        let p = self
            .device
            .map_memory(
                buf.memory,
                0,
                buf.bytes as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|e| format!("vkMapMemory: {e}"))? as *mut i32;
        std::ptr::copy_nonoverlapping(data.as_ptr(), p, data.len());
        self.device.unmap_memory(buf.memory);
        HOST_UPLOADS.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Device to host, int32 words, widened back to the int64 torch asked for.
    unsafe fn download_i32(&self, buf: &VkBuffer, n: usize) -> Result<Vec<i64>, String> {
        let p = self
            .device
            .map_memory(
                buf.memory,
                0,
                buf.bytes as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|e| format!("vkMapMemory: {e}"))? as *const i32;
        let mut words = vec![0i32; n];
        std::ptr::copy_nonoverlapping(p, words.as_mut_ptr(), n);
        self.device.unmap_memory(buf.memory);
        HOST_DOWNLOADS.fetch_add(1, Ordering::Relaxed);
        Ok(words.into_iter().map(i64::from).collect())
    }

    /// Device to host. The download half -- what `.cpu()` runs.
    unsafe fn download(&self, buf: &VkBuffer, n: usize) -> Result<Vec<f32>, String> {
        let p = self
            .device
            .map_memory(
                buf.memory,
                0,
                buf.bytes as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|e| format!("vkMapMemory: {e}"))? as *const f32;
        let mut out = vec![0.0f32; n];
        std::ptr::copy_nonoverlapping(p, out.as_mut_ptr(), n);
        self.device.unmap_memory(buf.memory);
        HOST_DOWNLOADS.fetch_add(1, Ordering::Relaxed);
        Ok(out)
    }

    /// Build (or fetch) the compute pipeline for a kernel.
    ///
    /// Three storage-buffer bindings and a `uint` push constant is the whole
    /// interface every kernel here uses, so the layout is shared rather than
    /// described per kernel.
    unsafe fn kernel(
        &self,
        name: &'static str,
        spv: &[u8],
        num_bindings: u32,
        push_words: u32,
    ) -> Result<Kernel, String> {
        let mut cache = self.pipelines.lock().map_err(|_| "pipeline cache poisoned")?;
        if let Some(k) = cache.get(name) {
            return Ok(*k);
        }
        // `include_bytes!` has no alignment guarantee and
        // `VkShaderModuleCreateInfo` wants `u32`; copying is the honest fix and
        // it happens once per kernel per process.
        if spv.len() % 4 != 0 {
            return Err(format!("{name}: SPIR-V length is not a multiple of 4"));
        }
        let words: Vec<u32> = spv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let module = self
            .device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            .map_err(|e| format!("vkCreateShaderModule({name}): {e}"))?;

        let bindings: Vec<_> = (0..num_bindings)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let dsl = self
            .device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
            .map_err(|e| format!("vkCreateDescriptorSetLayout({name}): {e}"))?;

        // Sixteen bytes for every kernel that declares
        // `Push { uint n; uint p1; uint p2; uint p3; }` -- most use only `n`;
        // a shader that reads fewer bytes than the range declares is legal,
        // which is why widening this did not require recompiling
        // `add_f32.spv` -- and `shaders/compile.sh` reproduced that file
        // byte-for-byte, which is the control that the toolchain here is the
        // one that produced the committed kernel (docs/devices/VULKAN4.md §3).
        //
        // Eighty for the two view kernels, which carry a shape and a stride of
        // rank up to `VIEW_RANK_MAX` (docs/devices/VULKAN7.md §2) -- inside the
        // 128 bytes every Vulkan device must accept, so this is not a
        // capability that varies by GPU. The size is fixed per kernel name,
        // and the cache is keyed by name, so one pipeline never sees two sizes.
        let pc = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(push_words.max(4) * 4)];
        let set_layouts = [dsl];
        let layout = self
            .device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&pc),
                None,
            )
            .map_err(|e| format!("vkCreatePipelineLayout({name}): {e}"))?;

        let entry = CStr::from_bytes_with_nul(b"main\0").expect("literal");
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(entry);
        let cp = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout)];
        let pipeline = self
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &cp, None)
            .map_err(|(_, e)| format!("vkCreateComputePipelines({name}): {e}"))?[0];
        // The module is only needed while the pipeline is being created.
        self.device.destroy_shader_module(module, None);

        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            // Per binding: `native_layer_norm` binds six, and a pool sized for
            // three would run out at half its stated set count.
            .descriptor_count(num_bindings * 64)];
        let dpool = self
            .device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(64)
                    .pool_sizes(&sizes),
                None,
            )
            .map_err(|e| format!("vkCreateDescriptorPool({name}): {e}"))?;

        let k = Kernel {
            dsl,
            layout,
            pipeline,
            dpool,
        };
        cache.insert(name, k);
        Ok(k)
    }

    /// Bind three buffers, push `n`, dispatch, wait.
    ///
    /// Synchronous on purpose: an asynchronous device needs a stream and an
    /// event per tensor, and a half-built one is a class of bug (reading a
    /// buffer whose dispatch has not retired) that produces *plausible wrong
    /// numbers*. The fence here is the same one the probe used and it means
    /// every tensor this device produces is complete when it is returned.
    unsafe fn dispatch_kernel(
        &self,
        name: &'static str,
        spv: &[u8],
        bufs: &[&VkBuffer],
        push: [u32; 4],
    ) -> Result<(), String> {
        self.dispatch_kernel_words(name, spv, bufs, &push)
    }

    /// `dispatch_kernel` with a push-constant block of any length -- the view
    /// kernels' is twenty words. `push[0]` is the output element count, as
    /// for every kernel here.
    unsafe fn dispatch_kernel_words(
        &self,
        name: &'static str,
        spv: &[u8],
        bufs: &[&VkBuffer],
        push: &[u32],
    ) -> Result<(), String> {
        let k = self.kernel(name, spv, bufs.len() as u32, push.len() as u32)?;
        let pool = self.submit.lock().map_err(|_| "submit lock poisoned")?;

        let set_layouts = [k.dsl];
        let dset = self
            .device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(k.dpool)
                    .set_layouts(&set_layouts),
            )
            .map_err(|e| format!("vkAllocateDescriptorSets: {e}"))?[0];
        let infos: Vec<_> = bufs
            .iter()
            .map(|b| {
                [vk::DescriptorBufferInfo::default()
                    .buffer(b.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)]
            })
            .collect();
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(dset)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        self.device.update_descriptor_sets(&writes, &[]);

        let cb = self
            .device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(*pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .map_err(|e| format!("vkAllocateCommandBuffers: {e}"))?[0];
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        self.device
            .begin_command_buffer(cb, &begin)
            .map_err(|e| format!("vkBeginCommandBuffer: {e}"))?;
        self.device
            .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, k.pipeline);
        self.device.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::COMPUTE,
            k.layout,
            0,
            &[dset],
            &[],
        );
        self.device.cmd_push_constants(
            cb,
            k.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &push_bytes(push),
        );
        // One invocation per output element is the convention every `.comp` in
        // `shaders/` follows, so the grid is sized from `push[0]` alone.
        self.device.cmd_dispatch(cb, push[0].div_ceil(64), 1, 1);
        self.device
            .end_command_buffer(cb)
            .map_err(|e| format!("vkEndCommandBuffer: {e}"))?;

        let fence = self
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(|e| format!("vkCreateFence: {e}"))?;
        let cbs = [cb];
        let submit = [vk::SubmitInfo::default().command_buffers(&cbs)];
        let result = self
            .device
            .queue_submit(self.queue, &submit, fence)
            .map_err(|e| format!("vkQueueSubmit: {e}"))
            .and_then(|()| {
                // Ten seconds, for the same reason the probe has a timeout: a
                // translation layer that deadlocks would otherwise hang the
                // interpreter with no output.
                self.device
                    .wait_for_fences(&[fence], true, 10_000_000_000)
                    .map_err(|e| format!("vkWaitForFences: {e}"))
            });
        self.device.destroy_fence(fence, None);
        self.device.free_command_buffers(*pool, &cbs);
        let _ = self.device.free_descriptor_sets(k.dpool, &[dset]);
        drop(pool);
        // After the fence, and only on success: the counter means "a compute
        // shader ran to completion on the device", not "one was submitted".
        if result.is_ok() {
            SHADER_DISPATCHES.fetch_add(1, Ordering::Relaxed);
        }
        result
    }
}

// ---------------------------------------------------------------------------
// The pieces the rest of the crate calls
// ---------------------------------------------------------------------------

/// A tensor of `size` filled with `value`, on the GPU.
///
/// This is `torch.ones(2, 2, device="vulkan")`, and there is no kernel in it:
/// the fill is built on the host and *uploaded*, which is the honest minimum --
/// the bytes really do end up in a `VkBuffer` and really do come back out of
/// one through `.cpu()`. A `fill` shader would move the loop to the GPU and
/// prove nothing more about the wiring; `docs/devices/VULKAN2.md` §5.3 says the same.

pub fn arange_factory(
    py: Python<'_>,
    op: &str,
    start: f64,
    step: f64,
    n: usize,
    tag: TorchDType,
) -> PyResult<Py<PyAny>> {
    let ctx = require(op)?;
    check_dtype(op, tag)?;
    let buffer = unsafe {
        let buf = ctx.alloc(n * 4).map_err(|e| vk_error(op, e))?;
        if tag == TorchDType::Int64 || tag == TorchDType::Int32 {
            let mut data = Vec::with_capacity(n);
            let s = start as i64;
            let d = step as i64;
            let mut lo = s;
            let mut hi = s;
            for i in 0..n {
                let v = s + (i as i64) * d;
                if i == 0 || v < lo { lo = v; }
                if i == 0 || v > hi { hi = v; }
                data.push(v as i32);
            }
            ctx.upload_i32(&buf, &data).map_err(|e| vk_error(op, e))?;
            let _ = buf.index_range.set(IndexRange { lo, hi, exact: true });
        } else {
            let mut data = Vec::with_capacity(n);
            let s = start as f32;
            let d = step as f32;
            for i in 0..n {
                data.push(s + (i as f32) * d);
            }
            ctx.upload(&buf, &data).map_err(|e| vk_error(op, e))?;
        }
        buf
    };
    let vk_tensor = VkTensor {
        buffer: Arc::new(buffer),
        shape: vec![n],
    };
    wrap_vk(py, vk_tensor, tag)
}

pub fn factory(
    py: Python<'_>,
    op: &str,
    size: Vec<usize>,
    tag: TorchDType,
    value: f32,
) -> PyResult<Py<PyAny>> {
    let ctx = require(op)?;
    check_dtype(op, tag)?;
    let n: usize = size.iter().product();
    let data = vec![value; n];
    let buffer = unsafe {
        let buf = ctx.alloc(n * 4).map_err(|e| vk_error(op, e))?;
        ctx.upload(&buf, &data).map_err(|e| vk_error(op, e))?;
        buf
    };
    let vk_tensor = VkTensor {
        buffer: Arc::new(buffer),
        shape: size,
    };
    let wrapped = PyTensorBase::vulkan(vk_tensor, tag);
    crate::tensor::promote(py, wrapped.into_pyobject(py)?.into_any().unbind())
}

/// The largest index this device can hold, and the reason it is not 2^63.
///
/// **This is a bound, not a truncation, and not a capability report.**
/// `docs/devices/VULKAN6.md` §1: 64-bit integers in a compute shader need the
/// device's `shaderInt64`, which is optional in Vulkan and absent on a great
/// many mobile GPUs. This machine was measured rather than assumed and it
/// *has* it (`_vulkan_probe()["shader_int64"] == True` on MoltenVK 1.4.2 /
/// Apple M1) -- and the index buffer is int32 anyway, because an index range
/// that depended on the GPU would mean the same model refusing on one device
/// and not another, which is a worse failure than a stated ceiling. So the
/// ceiling is stated once, here, for every device, and any value above it
/// **refuses by name** on the way in rather than wrapping into a plausible
/// wrong row. A vocabulary is nowhere near it: 2^31-1 is about 43000 times
/// GPT-2's 50257.
pub const VULKAN_INDEX_MAX: i64 = i32::MAX as i64;

/// How a dtype is stored on this device, or a refusal saying it is not.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VkStore {
    /// One float32 word per element -- every kernel in `shaders/` but one.
    F32,
    /// One int32 word per element, widened back to int64 on the way out.
    ///
    /// Reached only by `aten._to_copy.default` from the host and read only by
    /// the `embedding` kernel's index binding: **there is no arithmetic on
    /// integers on this device**, so this storage class cannot be used to
    /// smuggle an unimplemented integer op through a float shader.
    I64AsI32,
    BoolAsI32,
}

/// Which storage class a dtype gets, or a refusal.
///
/// The split from `check_dtype` is the design change of this round and it is
/// deliberately narrow: *storing* an int64 index tensor is now possible,
/// *computing* on one is exactly as impossible as before, because every op but
/// `embedding` still calls `check_dtype` and `check_dtype` still admits float32
/// only. `docs/devices/VULKAN6.md` §1.
pub fn check_storage_dtype(op: &str, tag: TorchDType) -> PyResult<VkStore> {
    match tag {
        TorchDType::Float32 => Ok(VkStore::F32),
        TorchDType::Int64 => Ok(VkStore::I64AsI32),
        TorchDType::Bool => Ok(VkStore::BoolAsI32),
        _ => Err(not_implemented(format!(
            "{op}: the vulkan device in this build stores float32, and int64 \
             indices as int32 (docs/devices/VULKAN6.md §1) -- not {}. The \
             shader is f32 and there is no conversion path that would not be a \
             silent one (docs/devices/VULKAN3.md).",
            tag.name()
        ))),
    }
}

/// f32 is the only dtype with a kernel, and every other one refuses here rather
/// than being silently widened or narrowed.
fn check_dtype(op: &str, tag: TorchDType) -> PyResult<()> {
    if tag == TorchDType::Float32 {
        return Ok(());
    }
    Err(not_implemented(format!(
        "{op}: the vulkan device in this build stores float32 only, not {}. \
         The shader is f32 and there is no conversion path that would not be a \
         silent one (docs/devices/VULKAN3.md).",
        tag.name()
    )))
}

fn vk_error(op: &str, message: String) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("{op}: vulkan: {message}"))
}

/// Bring a Vulkan tensor's bytes back to the CPU as a candle tensor.
///
/// `tag` decides how the words are read, and it has to: an index buffer holds
/// int32 words that mean int64 values, and reading them as f32 would return
/// denormal nonsense that looks like an answer.
pub fn to_cpu(op: &str, vk_tensor: &VkTensor, tag: TorchDType) -> PyResult<candle_core::Tensor> {
    let ctx = require(op)?;
    let n = vk_tensor.elem_count();
    match check_storage_dtype(op, tag)? {
        VkStore::F32 => {
            let host = unsafe { ctx.download(&vk_tensor.buffer, n) }.map_err(|e| vk_error(op, e))?;
            candle_core::Tensor::from_vec(host, vk_tensor.shape.clone(), &candle_core::Device::Cpu)
                .map_err(|e| crate::err::candle_err(op, e))
        }
        VkStore::I64AsI32 => {
            let host =
                unsafe { ctx.download_i32(&vk_tensor.buffer, n) }.map_err(|e| vk_error(op, e))?;
            candle_core::Tensor::from_vec(host, vk_tensor.shape.clone(), &candle_core::Device::Cpu)
                .map_err(|e| crate::err::candle_err(op, e))
        }
        VkStore::BoolAsI32 => {
            let host =
                unsafe { ctx.download_i32(&vk_tensor.buffer, n) }.map_err(|e| vk_error(op, e))?;
            let host_u8: Vec<u8> = host.into_iter().map(|v| if v != 0 { 1 } else { 0 }).collect();
            candle_core::Tensor::from_vec(host_u8, vk_tensor.shape.clone(), &candle_core::Device::Cpu)
                .map_err(|e| crate::err::candle_err(op, e))
        }
    }
}

/// The `vulkan` half of the dispatcher, and the *only* way an op computes on a
/// Vulkan tensor.
///
/// Structured exactly like `aten.rs::meta_dispatch` and for the same reason:
/// the answer to "does this op work on the vulkan device?" is a list here, not
/// a reading of ninety-odd kernels. The `other` arm is what makes the whole
/// design honest -- **an op that has not been taught this device refuses and
/// names itself.**
pub fn dispatch(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    match op {
        "aten._to_copy.default" => to_copy(py, op, args, kwargs),

        // Elementwise, equal shapes, f32. One exactly-rounded IEEE operation
        // per element, so each of these is compared bit-for-bit against the
        // CPU kernel rather than within a tolerance (docs/devices/VULKAN4.md §4).
        "aten.add.Tensor" => add_tensor(py, op, args, kwargs),
        "aten.all.default" => all_vulkan(py, op, args, kwargs),
        "aten._local_scalar_dense.default" => local_scalar_dense_vulkan(py, op, args, kwargs),
        "aten.sub.Tensor" => binary(py, op, args, kwargs, "sub_f32", SUB_F32_SPV, true),
        "aten.mul.Tensor" => binary(py, op, args, kwargs, "mul_f32", MUL_F32_SPV, false),
        "aten.div.Tensor" => binary(py, op, args, kwargs, "div_f32", DIV_F32_SPV, false),
        // Tensor-by-number. On the measured trace this is attention's
        // `scores / sqrt(d)` (docs/devices/VULKAN6.md §3); the scalar rides in
        // the push constants rather than in a buffer.
        "aten.mul.Scalar" => scalar_op(py, op, args, kwargs, 0),
        "aten.div.Scalar" => scalar_op(py, op, args, kwargs, 1),

        // Unary elementwise.
        "aten.relu.default" => unary(py, op, args, kwargs, "relu_f32", RELU_F32_SPV),
        "aten.gelu.default" => gelu_vulkan(py, op, args, kwargs),
        "aten.native_layer_norm.default" => native_layer_norm_vulkan(py, op, args, kwargs),
        "aten.bmm.default" => bmm_vulkan(py, op, args, kwargs),
        // The first op of every transformer forward, and the only one here
        // that reads an integer buffer (docs/devices/VULKAN6.md).
        "aten.embedding.default" => embedding_vulkan(py, op, args, kwargs),
        "aten._scaled_dot_product_flash_attention_for_cpu.default" => sdpa_vulkan(py, op, args, kwargs),
        "aten._softmax.default" => softmax_vulkan(py, op, args, kwargs),
        "aten._safe_softmax.default" => safe_softmax_vulkan(py, op, args, kwargs),
        "aten.neg.default" => unary(py, op, args, kwargs, "neg_f32", NEG_F32_SPV),
        // BERT's pooler (docs/devices/VULKAN7.md).
        "aten.tanh.default" => unary(py, op, args, kwargs, "tanh_f32", TANH_F32_SPV),
        // The views of docs/devices/VULKAN7.md, materialised by one strided
        // copy each -- float32 and index words alike -- and the gather BERT's
        // embeddings run on their `token_type_ids` buffer.
        "aten.slice.Tensor" => slice_vulkan(py, op, args, kwargs),
        "aten.select.int" => select_vulkan(py, op, args, kwargs),
        "aten.expand.default" => expand_vulkan(py, op, args, kwargs),
        "aten.gather.default" => gather_vulkan(py, op, args, kwargs),

        // `clone` allocates and copies on the device. `detach`/`alias` share
        // the buffer, which is what the dense arm does too (a candle clone is
        // an `Arc` clone) and is safe here for a stronger reason: no op on
        // this device writes in place.
        "aten.clone.default" => unary(py, op, args, kwargs, "copy_f32", COPY_F32_SPV),
        "aten.detach.default" | "aten.alias.default" | "aten.contiguous.default" => {
            let input = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
            let vk_tensor = input.vk_tensor(op)?.clone();
            let out = PyTensorBase::vulkan(vk_tensor, input.tag());
            crate::tensor::promote(py, out.into_pyobject(py)?.into_any().unbind())
        }

        // Shape-only. Every tensor on this device is contiguous by
        // construction -- there are no strides to be non-contiguous with --
        // so a reshape is a new `shape` over the same buffer and moves no
        // bytes. `_vulkan_counters()` shows zero shader dispatches for these,
        // which is the honest answer and not a claim that the GPU did work.
        "aten.view.default" | "aten._unsafe_view.default" | "aten.reshape.default" => {
            reshape(py, op, args, kwargs)
        }

        // Transpose *materialises* here, because a `VkTensor` has a shape and
        // no strides. Any pair, up to rank `VIEW_RANK_MAX` (docs/devices/VULKAN7.md).
        "aten.t.default" => t_default(py, op, args, kwargs),
        "aten.transpose.int" => transpose_int(py, op, args, kwargs),

        // The matmuls -- the ops the measured trace says a forward pass
        // actually spends itself on (docs/devices/VULKAN4.md §2).
        "aten.mm.default" => mm(py, op, args, kwargs),
        // Kept whole by the shim (docs/devices/VULKAN4.md §2.1), so it is its
        // own arm rather than upstream's expand + bmm + view.
        "aten.matmul.default" => matmul_vulkan(py, op, args, kwargs),
        "aten.addmm.default" => addmm(py, op, args, kwargs),

        other => Err(not_implemented(format!(
            "{other}: not implemented for the vulkan device. This build teaches \
             the vulkan device {} ops by name -- {} -- and every other op \
             refuses here rather than falling back to the CPU \
             (docs/devices/VULKAN4.md). Move the tensor with .cpu() to compute {other}.",
            vulkan_ops().len(),
            vulkan_ops().join(", ")
        ))),
    }
}

// ---------------------------------------------------------------------------
// The taught ops
// ---------------------------------------------------------------------------

/// Every op here is f32-only, and this is where that is enforced for a whole
/// tensor list rather than once per call site.
fn check_all_f32(op: &str, tensors: &[&PyTensorBase]) -> PyResult<()> {
    for t in tensors {
        check_dtype(op, t.tag())?;
    }
    Ok(())
}

/// `self` for an op whose only tensor argument is the receiver.
fn self_vk(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<(PyTensorBase, VkTensor)> {
    let input = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
    check_dtype(op, input.tag())?;
    let vk = input.vk_tensor(op)?.clone();
    Ok((input, vk))
}

/// Wrap a freshly produced buffer as a tensor of `shape` with `tag`.
fn wrap(
    py: Python<'_>,
    buffer: VkBuffer,
    shape: Vec<usize>,
    tag: TorchDType,
) -> PyResult<Py<PyAny>> {
    let out = PyTensorBase::vulkan(
        VkTensor {
            buffer: Arc::new(buffer),
            shape,
        },
        tag,
    );
    crate::tensor::promote(py, out.into_pyobject(py)?.into_any().unbind())
}

/// One input, one output, same shape.
fn unary(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    name: &'static str,
    spv: &'static [u8],
) -> PyResult<Py<PyAny>> {
    let (input, a) = self_vk(op, args, kwargs)?;
    let ctx = require(op)?;
    let n = a.elem_count();
    let out = unsafe {
        let out = ctx.alloc(n * 4).map_err(|e| vk_error(op, e))?;
        // Binding 1 is the input again: the descriptor set layout is three
        // storage buffers for every kernel here and the shader declares the
        // slot it does not read (`shaders/*_f32.comp`).
        ctx.dispatch_kernel(name, spv, &[&a.buffer, &a.buffer, &out], [n as u32, 0, 0, 0])
            .map_err(|e| vk_error(op, e))?;
        out
    };
    wrap(py, out, a.shape.clone(), input.tag())
}

/// Two inputs, one output. Equal shapes take the op's own elementwise kernel;
/// unequal ones broadcast through `broadcast_binary`. `alpha` is accepted only
/// when it is 1, for the ops that have one.
fn binary(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    name: &'static str,
    spv: &'static [u8],
    has_alpha: bool,
) -> PyResult<Py<PyAny>> {
    let lhs = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
    let rhs = crate::aten::tensor_arg(op, args, kwargs, 1, "other")?;
    if has_alpha {
        reject_alpha(op, args, kwargs)?;
    }
    check_all_f32(op, &[&lhs, &rhs])?;
    let a = lhs.vk_tensor(op)?.clone();
    let b = rhs.vk_tensor(op)?.clone();
    if a.shape != b.shape {
        return broadcast_binary(py, op, &lhs, &a, &b);
    }
    let ctx = require(op)?;
    let n = a.elem_count();
    let out = unsafe {
        let out = ctx.alloc(n * 4).map_err(|e| vk_error(op, e))?;
        ctx.dispatch_kernel(name, spv, &[&a.buffer, &b.buffer, &out], [n as u32, 0, 0, 0])
            .map_err(|e| vk_error(op, e))?;
        out
    };
    wrap(py, out, a.shape.clone(), lhs.tag())
}

fn reject_alpha(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    if let Some(alpha) = crate::aten::optional(args, kwargs, 2, "alpha")? {
        if !alpha.is_none() && alpha.extract::<f64>().unwrap_or(1.0) != 1.0 {
            return Err(not_implemented(format!(
                "{op}: the vulkan kernel has no alpha (docs/devices/VULKAN4.md §6)."
            )));
        }
    }
    Ok(())
}

/// `view` / `_unsafe_view` / `reshape`: a new shape over the same buffer.
fn reshape(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    // `check_storage_dtype`, not `check_dtype`: a reshape moves no bytes and
    // runs no shader, so it is the one op here that is honest about an int64
    // index tensor -- and it has to be, because a transformer reshapes its
    // `input_ids` before the embedding sees them.
    let input = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
    check_storage_dtype(op, input.tag())?;
    let a = input.vk_tensor(op)?.clone();
    // `aten.view` and `aten._unsafe_view` spell the argument `size`;
    // `aten.reshape` spells it `shape`. Both are accepted rather than one
    // being assumed -- the first draft assumed `size` and `t.reshape(-1)`
    // raised "missing argument 'size'" on a tensor that had one.
    let requested = match crate::aten::optional(args, kwargs, 1, "size")? {
        Some(v) if !v.is_none() => Some(v),
        _ => crate::aten::optional(args, kwargs, 1, "shape")?,
    };
    let requested = requested
        .filter(|v| !v.is_none())
        .ok_or_else(|| not_implemented(format!("{op}: vulkan: missing argument 'size'")))?;
    let requested: Vec<i64> = match requested.extract::<Vec<i64>>() {
        Ok(v) => v,
        Err(_) => vec![requested.extract::<i64>()?],
    };
    let n = a.elem_count();
    let shape = resolve_shape(op, &requested, n)?;
    let out = PyTensorBase::vulkan(
        VkTensor {
            buffer: a.buffer.clone(),
            shape,
        },
        input.tag(),
    );
    crate::tensor::promote(py, out.into_pyobject(py)?.into_any().unbind())
}

/// The `-1` sentinel and the element-count check, in upstream's words.
fn resolve_shape(op: &str, requested: &[i64], n: usize) -> PyResult<Vec<usize>> {
    let mut wild: Option<usize> = None;
    let mut known: usize = 1;
    for (i, &d) in requested.iter().enumerate() {
        if d == -1 {
            if wild.is_some() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "only one dimension can be inferred",
                ));
            }
            wild = Some(i);
        } else if d < 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{op}: invalid shape dimension {d}"
            )));
        } else {
            known = known.saturating_mul(d as usize);
        }
    }
    let mut shape: Vec<usize> = requested
        .iter()
        .map(|&d| if d == -1 { 0 } else { d as usize })
        .collect();
    match wild {
        Some(i) => {
            if known == 0 || n % known != 0 {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "shape {requested:?} is invalid for input of size {n}"
                )));
            }
            shape[i] = n / known;
        }
        None => {
            if known != n {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "shape {requested:?} is invalid for input of size {n}"
                )));
            }
        }
    }
    Ok(shape)
}

/// `aten.t.default`. Rank 0 and 1 are the identity upstream, and are here too;
/// rank 2 materialises through the transpose kernel; anything else is upstream's
/// own error rather than a vulkan one.
fn t_default(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let (input, a) = self_vk(op, args, kwargs)?;
    match a.shape.len() {
        0 | 1 => {
            let out = PyTensorBase::vulkan(a, input.tag());
            crate::tensor::promote(py, out.into_pyobject(py)?.into_any().unbind())
        }
        2 => transpose2d(py, op, &input, &a),
        rank => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "t() expects a tensor with <= 2 dimensions, but self is {rank}D"
        ))),
    }
}

/// `aten.transpose.int`. Any pair, up to rank `VIEW_RANK_MAX`; 2-D and the
/// last two axes of a float 3-D tensor keep their own kernels.
fn transpose_int(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    // `check_storage_dtype`, not `self_vk`'s `check_dtype`: a transpose moves
    // words and computes on none, so an int64 index tensor is as welcome as a
    // float one (docs/devices/VULKAN7.md). The two older kernels below are
    // float-only shaders, so an index tensor always takes the strided one.
    let input = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
    let store = check_storage_dtype(op, input.tag())?;
    let a = input.vk_tensor(op)?.clone();
    let rank = a.shape.len();
    let dim = |i: usize, name: &str| -> PyResult<isize> {
        let v = crate::aten::optional(args, kwargs, i, name)?
            .ok_or_else(|| not_implemented(format!("{op}: vulkan: missing argument '{name}'")))?;
        v.extract::<isize>()
    };
    let extent = rank.max(1) as isize;
    let norm = |d: isize| -> PyResult<usize> {
        let i = if d < 0 { d + extent } else { d };
        if i < 0 || i >= extent {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Dimension out of range (expected to be in range of [{}, {}], but got {d})",
                -extent,
                extent - 1
            )));
        }
        Ok(i as usize)
    };
    let d0 = norm(dim(1, "dim0")?)?;
    let d1 = norm(dim(2, "dim1")?)?;
    if d0 == d1 || rank < 2 {
        let out = PyTensorBase::vulkan(a, input.tag());
        return crate::tensor::promote(py, out.into_pyobject(py)?.into_any().unbind());
    }
    // The last two dimensions of a 3-D tensor: `k.transpose(1, 2)` between an
    // attention block's two `bmm`s. **Measured, not guessed** -- with
    // `embedding` taught and this not, a transformer forward stopped here, one
    // op past the wall this round came to remove (docs/devices/VULKAN6.md §3).
    // Every other pair and every higher rank still refuses by name: a general
    // permutation needs strides a `VkTensor` does not have.
    if store == VkStore::F32 && rank == 3 && d0.min(d1) == 1 && d0.max(d1) == 2 {
        return transpose_batched2d(py, op, &input, &a);
    }
    if store == VkStore::F32 && rank == 2 {
        return transpose2d(py, op, &input, &a);
    }
    // Every other pair and rank, up to the stated bound: BERT's attention
    // transposes (batch, head, seq, dim) tensors, which is the 4-D wall
    // docs/devices/VULKAN6.md §3.1 named. Above the bound this refuses by name.
    transpose_strided(py, op, &input, &a, d0, d1)
}

/// The materialising 2-D transpose. Pure data movement, so the result is
/// bit-identical to the CPU answer by construction.

/// The last two dimensions of a 3-D tensor, materialised per batch.
///
/// Shares `transpose2d`'s reasoning and its proof: no arithmetic happens, so
/// the result is bit-identical to the CPU answer rather than close to it.
/// `aten.mul.Scalar` / `aten.div.Scalar` -- one tensor, one number.
///
/// `which` is 0 for multiply and 1 for divide, and the divide stays a divide
/// (see `shaders/scalar_f32.comp`). The scalar is required to be a real number
/// that survives the trip to float32 exactly as upstream's own `Scalar` does;
/// a non-finite or complex one refuses rather than being coerced.
fn scalar_op(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    which: u32,
) -> PyResult<Py<PyAny>> {
    let (input, a) = self_vk(op, args, kwargs)?;
    let other = crate::aten::optional(args, kwargs, 1, "other")?
        .ok_or_else(|| not_implemented(format!("{op}: vulkan: missing argument 'other'")))?;
    let value: f64 = other.extract().map_err(|_| {
        not_implemented(format!(
            "{op}: the vulkan device's scalar kernel takes a number, not {}.              A 0-d tensor goes through the .Tensor overload.",
            other.get_type().name().map(|n| n.to_string()).unwrap_or_default()
        ))
    })?;
    let ctx = require(op)?;
    let n = a.elem_count();
    let bits = (value as f32).to_bits();
    let out = unsafe {
        let out = ctx.alloc(n.max(1) * 4).map_err(|e| vk_error(op, e))?;
        if n > 0 {
            ctx.dispatch_kernel(
                "scalar_f32",
                SCALAR_F32_SPV,
                &[&a.buffer, &a.buffer, &out],
                [n as u32, bits, which, 0],
            )
            .map_err(|e| vk_error(op, e))?;
        }
        out
    };
    wrap(py, out, a.shape.clone(), input.tag())
}

fn transpose_batched2d(
    py: Python<'_>,
    op: &str,
    input: &PyTensorBase,
    a: &VkTensor,
) -> PyResult<Py<PyAny>> {
    let (batch, rows, cols) = (a.shape[0], a.shape[1], a.shape[2]);
    let ctx = require(op)?;
    let n = batch * rows * cols;
    let out = unsafe {
        let out = ctx.alloc(n.max(1) * 4).map_err(|e| vk_error(op, e))?;
        if n > 0 {
            ctx.dispatch_kernel(
                "transpose_batched2d_f32",
                TRANSPOSE_BATCHED2D_F32_SPV,
                &[&a.buffer, &a.buffer, &out],
                [n as u32, rows as u32, cols as u32, 0],
            )
            .map_err(|e| vk_error(op, e))?;
        }
        out
    };
    wrap(py, out, vec![batch, cols, rows], input.tag())
}

fn transpose2d(
    py: Python<'_>,
    op: &str,
    input: &PyTensorBase,
    a: &VkTensor,
) -> PyResult<Py<PyAny>> {
    let (rows, cols) = (a.shape[0], a.shape[1]);
    let ctx = require(op)?;
    let n = rows * cols;
    let out = unsafe {
        let out = ctx.alloc(n * 4).map_err(|e| vk_error(op, e))?;
        ctx.dispatch_kernel(
            "transpose2d_f32",
            TRANSPOSE2D_F32_SPV, &[&a.buffer, &a.buffer, &out],
            [n as u32, rows as u32, cols as u32, 0],
        )
        .map_err(|e| vk_error(op, e))?;
        out
    };
    wrap(py, out, vec![cols, rows], input.tag())
}

/// The product itself, shared by `mm` and `addmm`.
fn matmul_into(op: &str, a: &VkTensor, b: &VkTensor) -> PyResult<(VkBuffer, usize, usize)> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(not_implemented(format!(
            "{op}: the vulkan matmul kernel is 2-D and was given {:?} and {:?}. \
             Batched and broadcast products go through `aten.bmm.default` / \
             `aten.matmul.default` (docs/devices/VULKAN7.md).",
            a.shape, b.shape
        )));
    }
    let (m, k) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
    if k != k2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "mat1 and mat2 shapes cannot be multiplied ({m}x{k} and {k2}x{n})"
        )));
    }
    let ctx = require(op)?;
    let count = m * n;
    let out = unsafe {
        let out = ctx.alloc(count * 4).map_err(|e| vk_error(op, e))?;
        ctx.dispatch_kernel(
            "matmul_f32",
            MATMUL_F32_SPV, &[&a.buffer, &b.buffer, &out],
            [count as u32, m as u32, k as u32, n as u32],
        )
        .map_err(|e| vk_error(op, e))?;
        out
    };
    Ok((out, m, n))
}

fn mm(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let lhs = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
    let rhs = crate::aten::tensor_arg(op, args, kwargs, 1, "mat2")?;
    check_all_f32(op, &[&lhs, &rhs])?;
    let a = lhs.vk_tensor(op)?.clone();
    let b = rhs.vk_tensor(op)?.clone();
    let (out, m, n) = matmul_into(op, &a, &b)?;
    wrap(py, out, vec![m, n], lhs.tag())
}

/// `aten.addmm.default(bias, mat1, mat2, *, beta=1, alpha=1)`.
///
/// Two dispatches, not one: the product, then the bias. `beta` and `alpha` are
/// refused unless they are 1 -- `nn.Linear` never sets them, and scaling here
/// would be a third arithmetic operation with no kernel behind it.
fn addmm(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let bias = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
    let lhs = crate::aten::tensor_arg(op, args, kwargs, 1, "mat1")?;
    let rhs = crate::aten::tensor_arg(op, args, kwargs, 2, "mat2")?;
    for (i, name) in [(3usize, "beta"), (4usize, "alpha")] {
        if let Some(v) = crate::aten::optional(args, kwargs, i, name)? {
            if !v.is_none() && v.extract::<f64>().unwrap_or(1.0) != 1.0 {
                return Err(not_implemented(format!(
                    "{op}: the vulkan device implements addmm with beta = alpha = 1 \
                     and was given {name} != 1 (docs/devices/VULKAN4.md §6)."
                )));
            }
        }
    }
    check_all_f32(op, &[&bias, &lhs, &rhs])?;
    let c = bias.vk_tensor(op)?.clone();
    let a = lhs.vk_tensor(op)?.clone();
    let b = rhs.vk_tensor(op)?.clone();
    let (product, m, n) = matmul_into(op, &a, &b)?;

    // A 1-D bias of width `n` is what `nn.Linear` passes. A 2-D `[m, n]` bias
    // is elementwise and goes through the same kernel with `p1 = n`, because
    // `col` then indexes the row too. Everything else refuses.
    let bias_ok = match c.shape.as_slice() {
        [w] => *w == n,
        [r, w] => *r == m && *w == n,
        _ => false,
    };
    if !bias_ok {
        return Err(not_implemented(format!(
            "{op}: the vulkan device adds a bias of shape [{n}] or [{m}, {n}] and \
             was given {:?}. Wider broadcasting refuses rather than being \
             emulated (docs/devices/VULKAN4.md §6).",
            c.shape
        )));
    }
    let ctx = require(op)?;
    let count = m * n;
    let out = unsafe {
        let out = ctx.alloc(count * 4).map_err(|e| vk_error(op, e))?;
        ctx.dispatch_kernel(
            "bias_add_f32",
            BIAS_ADD_F32_SPV, &[&product, &c.buffer, &out],
            [count as u32, n as u32, 0, 0],
        )
        .map_err(|e| vk_error(op, e))?;
        out
    };
    wrap(py, out, vec![m, n], lhs.tag())
}

/// `x.to("vulkan")` for a tensor that is **not** on Vulkan yet -- the upload
/// half of `docs/devices/VULKAN2.md` §5.2 item 5, which did not exist until this round.
///
/// **Why this had to be added before anything could be measured.** Before it,
/// the only way onto this device was `torch.ones`/`zeros`/`empty`, so every
/// tensor that had ever reached a Vulkan kernel was a constant. A matmul of
/// all-ones agrees with any implementation that sums the right number of ones;
/// it cannot distinguish a correct kernel from one that transposed an index or
/// accumulated in the wrong order. Comparing element-wise against upstream on
/// real data (`docs/devices/VULKAN4.md` §4) needs real data, and this is how it gets
/// there.
///
/// Returns `None` when the call is not a copy *to* vulkan, so the caller falls
/// through to its ordinary path and nothing else changes.
pub fn maybe_upload(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<Py<PyAny>>> {
    if op != "aten._to_copy.default" {
        return Ok(None);
    }
    let Ok(input) = crate::aten::tensor_arg(op, args, kwargs, 0, "self") else {
        return Ok(None);
    };
    let current = input.device_label();
    let label = crate::aten::device_arg_or_label(args, kwargs, 3, "device", &current)?;
    if label.kind != "vulkan" || current.kind == "vulkan" {
        return Ok(None);
    }
    // From here on the call *is* a copy to vulkan, so every remaining problem
    // refuses by name rather than returning `None` and letting `resolve()`
    // answer with the generic "device not available", which would say nothing
    // about which narrowing was hit.
    let tag = crate::aten::dtype_arg(args, kwargs, 1, "dtype")?.unwrap_or(input.tag());
    if tag != input.tag() {
        let allow = (tag == TorchDType::Bool && input.tag() == TorchDType::Int64) ||
                    (tag == TorchDType::Int64 && input.tag() == TorchDType::Bool);
        if !allow {
            return Err(not_implemented(format!(
                "{op}: the vulkan device cannot change dtype on the way in ({} to \
                 {}) -- there is no conversion shader. Cast on the cpu first \
                 (docs/devices/VULKAN4.md §6).",
                input.tag().name(),
                tag.name()
            )));
        }
    }
    let store = check_storage_dtype(op, tag)?;
    if current.kind != "cpu" {
        return Err(not_implemented(format!(
            "{op}: the vulkan device accepts a copy from the cpu, not from {} \
             (docs/devices/VULKAN4.md §6).",
            current.kind
        )));
    }
    let host = input.tensor()?;
    let shape: Vec<usize> = host.dims().to_vec();
    let flat = host
        .flatten_all()
        .and_then(|t| t.contiguous())
        .map_err(|e| crate::err::candle_err(op, e))?;
    let ctx = require(op)?;
    let buffer = match store {
        VkStore::F32 => {
            let words = flat.to_vec1::<f32>().map_err(|e| crate::err::candle_err(op, e))?;
            unsafe {
                let buf = ctx.alloc(words.len() * 4).map_err(|e| vk_error(op, e))?;
                ctx.upload(&buf, &words).map_err(|e| vk_error(op, e))?;
                buf
            }
        }
        VkStore::BoolAsI32 => {
            let values = flat.to_dtype(candle_core::DType::U8).map_err(|e| crate::err::candle_err(op, e))?.to_vec1::<u8>().map_err(|e| crate::err::candle_err(op, e))?;
            let words: Vec<i32> = values.into_iter().map(|v| if v != 0 { 1 } else { 0 }).collect();
            unsafe {
                let buf = ctx.alloc(words.len() * 4).map_err(|e| vk_error(op, e))?;
                ctx.upload_i32(&buf, &words).map_err(|e| vk_error(op, e))?;
                buf
            }
        }
        VkStore::I64AsI32 => {
            let values = flat.to_vec1::<i64>().map_err(|e| crate::err::candle_err(op, e))?;
            // **The narrowing, refused by name rather than performed.** One
            // pass over values that are being copied anyway: the offending
            // element is reported with its position and the bound, so the
            // message says which value and against what -- and the same pass
            // records the range `embedding` will check against, so the range
            // check later costs no host round trip.
            let mut lo = 0i64;
            let mut hi = 0i64;
            let mut words = Vec::with_capacity(values.len());
            for (i, &v) in values.iter().enumerate() {
                if !(-VULKAN_INDEX_MAX - 1..=VULKAN_INDEX_MAX).contains(&v) {
                    return Err(not_implemented(format!(
                        "{op}: int64 element {i} is {v}, which does not fit the \
                         int32 index storage every vulkan device in this build \
                         uses (bound {}..={}). 64-bit integers in a shader need \
                         the optional shaderInt64 feature, so the ceiling is the \
                         same everywhere rather than a property of this GPU -- \
                         and a value above it is refused here rather than \
                         truncated into a different, plausible index \
                         (docs/devices/VULKAN6.md §1).",
                        -VULKAN_INDEX_MAX - 1,
                        VULKAN_INDEX_MAX
                    )));
                }
                if i == 0 || v < lo {
                    lo = v;
                }
                if i == 0 || v > hi {
                    hi = v;
                }
                words.push(v as i32);
            }
            let buf = unsafe {
                let buf = ctx.alloc(words.len().max(1) * 4).map_err(|e| vk_error(op, e))?;
                ctx.upload_i32(&buf, &words).map_err(|e| vk_error(op, e))?;
                buf
            };
            let _ = buf.index_range.set(IndexRange { lo, hi, exact: true });
            buf
        }
    };
    Ok(Some(wrap(py, buffer, shape, tag)?))
}

/// `.cpu()`, `.to("cpu")` and `.to("vulkan")` for a tensor already on Vulkan.
fn to_copy(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let input = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
    let tag = crate::aten::dtype_arg(args, kwargs, 1, "dtype")?.unwrap_or(input.tag());
    let label = crate::aten::device_arg_or_label(args, kwargs, 3, "device", &input.device_label())?;
    if tag != input.tag() {
        let allow = (tag == TorchDType::Bool && input.tag() == TorchDType::Int64) ||
                    (tag == TorchDType::Int64 && input.tag() == TorchDType::Bool);
        if !allow {
            return Err(not_implemented(format!(
                "{op}: the vulkan device cannot change dtype ({} to {}) -- there is \
                 no conversion shader. Bring the tensor to the cpu first \
                 (docs/devices/VULKAN3.md).",
                input.tag().name(),
                tag.name()
            )));
        }
    }
    let vk_tensor = input.vk_tensor(op)?.clone();
    match label.kind.as_str() {
        "cpu" => {
            let host = to_cpu(op, &vk_tensor, input.tag())?;
            let out = PyTensorBase::new(host)?;
            crate::tensor::promote(py, out.into_pyobject(py)?.into_any().unbind())
        }
        "vulkan" => {
            let out = PyTensorBase::vulkan(vk_tensor, tag);
            crate::tensor::promote(py, out.into_pyobject(py)?.into_any().unbind())
        }
        other => Err(not_implemented(format!(
            "{op}: the vulkan device can copy to cpu and to vulkan, not to \
             {other} (docs/devices/VULKAN3.md)."
        ))),
    }
}

/// `aten.add.Tensor` -- f32, alpha == 1.
///
/// Unequal shapes broadcast on the device, in one dispatch
/// (`broadcast_binary`, docs/devices/VULKAN7.md §3.2). Every other narrowing
/// refuses by name instead of being emulated: a scalar `other` and an `alpha`
/// both have perfectly good CPU implementations half a metre away and reaching
/// for one of them here is exactly the silent fallback this device exists to
/// make impossible.
fn add_tensor(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let lhs = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
    let rhs = crate::aten::tensor_arg(op, args, kwargs, 1, "other")?;
    if let Some(alpha) = crate::aten::optional(args, kwargs, 2, "alpha")? {
        if !alpha.is_none() && alpha.extract::<f64>().unwrap_or(1.0) != 1.0 {
            return Err(not_implemented(format!(
                "{op}: the vulkan kernel is a+b and has no alpha (docs/devices/VULKAN3.md)."
            )));
        }
    }
    check_dtype(op, lhs.tag())?;
    check_dtype(op, rhs.tag())?;
    let a = lhs.vk_tensor(op)?.clone();
    let b = rhs.vk_tensor(op)?.clone();
    if a.shape != b.shape {
        return broadcast_binary(py, op, &lhs, &a, &b);
    }
    let ctx = require(op)?;
    let n = a.elem_count();
    let out = unsafe {
        let out = ctx.alloc(n * 4).map_err(|e| vk_error(op, e))?;
        ctx.dispatch_kernel(
            "add_f32",
            ADD_F32_SPV, &[&a.buffer, &b.buffer, &out],
            [n as u32, 0, 0, 0],
        )
        .map_err(|e| vk_error(op, e))?;
        out
    };
    let result = VkTensor {
        buffer: Arc::new(out),
        shape: a.shape.clone(),
    };
    let wrapped = PyTensorBase::vulkan(result, lhs.tag());
    crate::tensor::promote(py, wrapped.into_pyobject(py)?.into_any().unbind())
}

/// The label a Vulkan tensor reports. There is one device, so there is no index
/// to reconstruct and none is invented -- the mirror of `PyDevice::meta()`.
pub fn label() -> PyDevice {
    PyDevice::checked("vulkan", None).expect("\"vulkan\" is in DEVICE_TYPES")
}

// ---------------------------------------------------------------------------
// What Python can ask
// ---------------------------------------------------------------------------

/// `_C._vulkan_probe()` -- **the reason tests can skip by name.**
///
/// A test that fails where there is no Vulkan is a broken gate, and a test that
/// silently passes there is worse. This returns a dict, so a skip can say
/// *which* driver was missing and a pass can say which GPU it ran on, and the
/// report distinguishes the two.
#[pyfunction]
#[pyo3(name = "_vulkan_probe")]
fn vulkan_probe(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    match context() {
        Ok(ctx) => {
            d.set_item("available", true)?;
            d.set_item("device", ctx.device_name.as_str())?;
            d.set_item("type", ctx.device_type.as_str())?;
            d.set_item("queue_family", ctx.qfi)?;
            d.set_item("loader", ctx.loader.as_str())?;
            d.set_item("shader_int64", ctx.shader_int64)?;
            d.set_item("shader_int16", ctx.shader_int16)?;
            d.set_item("shader_float64", ctx.shader_float64)?;
            d.set_item("khr_8bit_storage", ctx.khr_8bit_storage)?;
            d.set_item("khr_16bit_storage", ctx.khr_16bit_storage)?;
            d.set_item("index_max", VULKAN_INDEX_MAX)?;
            d.set_item("error", py.None())?;
        }
        Err(reason) => {
            d.set_item("available", false)?;
            d.set_item("device", py.None())?;
            d.set_item("type", py.None())?;
            d.set_item("queue_family", py.None())?;
            d.set_item("loader", py.None())?;
            d.set_item("shader_int64", py.None())?;
            d.set_item("shader_int16", py.None())?;
            d.set_item("shader_float64", py.None())?;
            d.set_item("khr_8bit_storage", py.None())?;
            d.set_item("khr_16bit_storage", py.None())?;
            d.set_item("index_max", VULKAN_INDEX_MAX)?;
            d.set_item("error", reason)?;
        }
    }
    Ok(d.into_any().unbind())
}

/// `_C._vulkan_ops()` -- the closed list of ops taught this device, so a test
/// can pick something that is *not* on it and check the refusal names it.
#[pyfunction]
#[pyo3(name = "_vulkan_ops")]
fn vulkan_ops() -> Vec<&'static str> {
    vec![
        "aten._local_scalar_dense.default",
        "aten._safe_softmax.default",
        "aten._scaled_dot_product_flash_attention_for_cpu.default",
        "aten._softmax.default",
        "aten._to_copy.default",
        "aten._unsafe_view.default",
        "aten.add.Tensor",
        "aten.addmm.default",
        "aten.alias.default",
        "aten.all.default",
        "aten.bmm.default",
        "aten.clone.default",
        "aten.contiguous.default",
        "aten.detach.default",
        "aten.div.Scalar",
        "aten.div.Tensor",
        "aten.embedding.default",
        "aten.expand.default",
        "aten.gather.default",
        "aten.gelu.default",
        "aten.matmul.default",
        "aten.mm.default",
        "aten.mul.Scalar",
        "aten.mul.Tensor",
        "aten.native_layer_norm.default",
        "aten.neg.default",
        "aten.relu.default",
        "aten.reshape.default",
        "aten.select.int",
        "aten.slice.Tensor",
        "aten.sub.Tensor",
        "aten.t.default",
        "aten.tanh.default",
        "aten.transpose.int",
        "aten.view.default",
    ]
}

/// `_C._vulkan_counters()` -- **the runtime answer to "did the GPU do it?"**
///
/// Returns the three counters described beside their definitions above:
///
/// * `shader_dispatches` -- compute shaders that ran to completion on the
///   device, counted after `vkWaitForFences` returned success.
/// * `host_uploads` / `host_downloads` -- crossings of the host boundary,
///   counted inside the only two `vkMapMemory` sites in the module.
///
/// A test asserts the delta across one op. That is a statement about what the
/// process did rather than about what the source looks like, which is the
/// distinction `docs/devices/MPSATTN.md` §3.1 says its own evidence failed to make.
#[pyfunction]
#[pyo3(name = "_vulkan_counters")]
fn vulkan_counters(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item("shader_dispatches", SHADER_DISPATCHES.load(Ordering::Relaxed))?;
    d.set_item("host_uploads", HOST_UPLOADS.load(Ordering::Relaxed))?;
    d.set_item("host_downloads", HOST_DOWNLOADS.load(Ordering::Relaxed))?;
    Ok(d.into_any().unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(vulkan_probe, m)?)?;
    m.add_function(wrap_pyfunction!(vulkan_ops, m)?)?;
    m.add_function(wrap_pyfunction!(vulkan_counters, m)?)?;
    m.add_function(wrap_pyfunction!(vulkan_loader_candidates, m)?)?;
    Ok(())
}

/// `_C._vulkan_loader_candidates()` -- where `init` looks, in the order it
/// looks, so a test can check that a loader present at one of them is found
/// without restating the list.
#[pyfunction]
#[pyo3(name = "_vulkan_loader_candidates")]
fn vulkan_loader_candidates() -> Vec<String> {
    loader_candidates()
}

// ---------------------------------------------------------------------------
// The transformer kernels: gelu, _softmax, native_layer_norm, bmm
// ---------------------------------------------------------------------------
//
// docs/devices/VULKAN5.md §3. Each refuses what it does not implement *before*
// allocating or dispatching, with upstream's message where upstream raises.

fn wrap_vk(py: Python<'_>, t: VkTensor, tag: TorchDType) -> PyResult<Py<PyAny>> {
    crate::tensor::promote(py, PyTensorBase::vulkan(t, tag).into_pyobject(py)?.into_any().unbind())
}

fn gelu_vulkan(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let input = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
    if args.len() > 1 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "aten::gelu() takes 1 positional argument(s) but {} were given. \
             Declaration: aten::gelu(Tensor self, *, str approximate=\"none\") -> Tensor",
            args.len()
        )));
    }
    let approximate: String = match kwargs.map(|kw| kw.get_item("approximate")).transpose()?.flatten() {
        None => "none".to_string(),
        Some(v) => v.extract().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "aten::gelu() expected a value of type 'str' for argument 'approximate'",
            )
        })?,
    };
    // Anything else used to compute the exact gelu silently.
    let tanh = match approximate.as_str() {
        "none" => 0,
        "tanh" => 1,
        _ => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "approximate argument must be either none or tanh.",
            ))
        }
    };
    check_dtype(op, input.tag())?;
    let x = input.vk_tensor(op)?.clone();
    let ctx = require(op)?;
    let n = x.elem_count();
    let out = unsafe {
        let out = ctx.alloc(n * 4).map_err(|e| vk_error(op, e))?;
        ctx.dispatch_kernel("gelu_f32", GELU_F32_SPV, &[&x.buffer, &x.buffer, &out], [n as u32, tanh, 0, 0])
            .map_err(|e| vk_error(op, e))?;
        out
    };
    wrap_vk(py, VkTensor { buffer: Arc::new(out), shape: x.shape }, input.tag())
}

fn softmax_vulkan(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let input = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
    let dim_raw = crate::aten::dim_arg(args, kwargs, 1, "dim")?.ok_or_else(|| crate::aten::missing(op, "dim"))?;
    let half_to_float = crate::aten::bool_arg(args, kwargs, 2, "half_to_float")?
        .ok_or_else(|| crate::aten::missing(op, "half_to_float"))?;
    check_dtype(op, input.tag())?;
    let x = input.vk_tensor(op)?.clone();
    let rank = x.shape.len() as isize;
    // A 0-d tensor accepts dim 0 and -1, as upstream's does.
    let span = rank.max(1);
    let dim = if dim_raw < 0 { dim_raw + span } else { dim_raw };
    if dim < 0 || dim >= span {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "Dimension out of range (expected to be in range of [{}, {}], but got {dim_raw})",
            -span,
            span - 1
        )));
    }
    if rank > 0 && dim != rank - 1 {
        return Err(not_implemented(format!(
            "{op}: the vulkan device implements softmax over the last dimension only, \
             got dim={dim_raw} of a {rank}-D tensor. Move it with .cpu() or permute first."
        )));
    }
    if half_to_float {
        return Err(not_implemented(format!(
            "{op}: half_to_float=True is not implemented on the vulkan device, which \
             stores float32 only"
        )));
    }
    let ctx = require(op)?;
    let n = x.elem_count();
    let row_len = x.shape.last().copied().unwrap_or(1);
    let rows = if row_len > 0 { n / row_len } else { 0 };
    let out = unsafe {
        let out = ctx.alloc(n * 4).map_err(|e| vk_error(op, e))?;
        ctx.dispatch_kernel(
            "softmax_lastdim_f32",
            SOFTMAX_LASTDIM_F32_SPV,
            &[&x.buffer, &x.buffer, &out],
            [rows as u32, row_len as u32, 0, 0],
        )
        .map_err(|e| vk_error(op, e))?;
        out
    };
    wrap_vk(py, VkTensor { buffer: Arc::new(out), shape: x.shape }, input.tag())
}

/// `native_layer_norm(input, normalized_shape, weight?, bias?, eps)`.
///
/// Returns `(out, mean, invstd)` with `mean`/`invstd` shaped
/// `input.shape[:axis] + [1] * len(normalized_shape)`, which is upstream's
/// shape -- the inherited kernel returned `input.shape[:axis]`.
fn native_layer_norm_vulkan(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let input = crate::aten::tensor_arg(op, args, kwargs, 0, "input")?;
    let requested = crate::aten::shape_arg(op, args, kwargs, 1, "normalized_shape")?;
    let weight = crate::aten::optional_tensor_arg(op, args, kwargs, 2, "weight")?;
    let bias = crate::aten::optional_tensor_arg(op, args, kwargs, 3, "bias")?;
    let eps = crate::aten::scalar_arg(op, args, kwargs, 4, "eps")?
        .map(|s| s.as_f64())
        .ok_or_else(|| crate::aten::missing(op, "eps"))?;

    check_dtype(op, input.tag())?;
    let x = input.vk_tensor(op)?.clone();
    let dims = x.shape.clone();
    let fmt = |v: &[usize]| format!("{v:?}").replace(' ', "");

    if requested.is_empty() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Expected normalized_shape to be at least 1-dimensional, i.e., containing at least one element, but got normalized_shape = []",
        ));
    }
    // Negative entries cannot match a real size; map them to a value that
    // fails the comparison below rather than wrapping.
    let norm: Vec<usize> = requested.iter().map(|&d| usize::try_from(d).unwrap_or(usize::MAX)).collect();
    // Previously unchecked: a mismatch normalised the wrong span, and a
    // `normalized_shape` longer than the input underflowed a `usize`.
    if norm.len() > dims.len() || dims[dims.len() - norm.len()..] != norm[..] {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Given normalized_shape={}, expected input with shape [*, {}], but got input of size{}",
            fmt(&norm),
            fmt(&norm).trim_start_matches('[').trim_end_matches(']'),
            fmt(&dims)
        )));
    }
    let affine = |name: &str, t: &Option<PyTensorBase>| -> PyResult<Option<VkTensor>> {
        let Some(t) = t else { return Ok(None) };
        check_dtype(op, t.tag())?;
        let v = t.vk_tensor(op)?.clone();
        if v.shape != norm {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Expected {name} to be of same shape as normalized_shape, but got {name} of shape {} and normalized_shape = {}",
                fmt(&v.shape),
                fmt(&norm)
            )));
        }
        Ok(Some(v))
    };
    let w = affine("weight", &weight)?;
    let b = affine("bias", &bias)?;

    let axis = dims.len() - norm.len();
    let n = x.elem_count();
    let row_len: usize = norm.iter().product();
    let rows = if row_len > 0 { n / row_len } else { 0 };
    let mut stat_shape = dims[..axis].to_vec();
    stat_shape.extend(std::iter::repeat(1).take(norm.len()));

    let ctx = require(op)?;
    let (out, mean, invstd) = unsafe {
        // Bound in the slot of an absent weight/bias; the shader does not read
        // a slot whose bit is clear.
        let dummy = ctx.alloc(4).map_err(|e| vk_error(op, e))?;
        let out = ctx.alloc(n * 4).map_err(|e| vk_error(op, e))?;
        let mean = ctx.alloc(rows * 4).map_err(|e| vk_error(op, e))?;
        let invstd = ctx.alloc(rows * 4).map_err(|e| vk_error(op, e))?;
        let wbuf: &VkBuffer = w.as_ref().map(|t| &*t.buffer).unwrap_or(&dummy);
        let bbuf: &VkBuffer = b.as_ref().map(|t| &*t.buffer).unwrap_or(&dummy);
        let flags = u32::from(w.is_some()) | (u32::from(b.is_some()) << 1);
        ctx.dispatch_kernel(
            "native_layer_norm_f32",
            NATIVE_LAYER_NORM_F32_SPV,
            &[&x.buffer, wbuf, bbuf, &out, &mean, &invstd],
            [rows as u32, row_len as u32, flags, (eps as f32).to_bits()],
        )
        .map_err(|e| vk_error(op, e))?;
        (out, mean, invstd)
    };
    let tag = input.tag();
    let items = [
        wrap_vk(py, VkTensor { buffer: Arc::new(out), shape: dims }, tag)?,
        wrap_vk(py, VkTensor { buffer: Arc::new(mean), shape: stat_shape.clone() }, tag)?,
        wrap_vk(py, VkTensor { buffer: Arc::new(invstd), shape: stat_shape }, tag)?,
    ];
    Ok(PyTuple::new(py, items)?.into_any().unbind())
}

fn bmm_vulkan(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let lhs = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
    let rhs = crate::aten::tensor_arg(op, args, kwargs, 1, "mat2")?;
    check_all_f32(op, &[&lhs, &rhs])?;
    let a = lhs.vk_tensor(op)?.clone();
    let b = rhs.vk_tensor(op)?.clone();
    let runtime = |m: String| pyo3::exceptions::PyRuntimeError::new_err(m);
    if a.shape.len() != 3 {
        return Err(runtime("batch1 must be a 3D tensor".into()));
    }
    if b.shape.len() != 3 {
        return Err(runtime("batch2 must be a 3D tensor".into()));
    }
    let (batch, m, k) = (a.shape[0], a.shape[1], a.shape[2]);
    if b.shape[0] != batch || b.shape[1] != k {
        return Err(runtime(format!(
            "Expected size for first two dimensions of batch2 tensor to be: [{batch}, {k}] but got: [{}, {}].",
            b.shape[0], b.shape[1]
        )));
    }
    let n_dim = b.shape[2];
    let ctx = require(op)?;
    let n = batch * m * n_dim;
    let out = unsafe {
        let out = ctx.alloc(n * 4).map_err(|e| vk_error(op, e))?;
        ctx.dispatch_kernel(
            "bmm_f32",
            BMM_F32_SPV,
            &[&a.buffer, &b.buffer, &out],
            [n as u32, m as u32, k as u32, n_dim as u32],
        )
        .map_err(|e| vk_error(op, e))?;
        out
    };
    wrap_vk(py, VkTensor { buffer: Arc::new(out), shape: vec![batch, m, n_dim] }, lhs.tag())
}

#[cfg(test)]
mod tests {
    use super::loader_candidates_for;

    /// docs/devices/VULKAN5.md §1. The bare name stays first so that
    /// `DYLD_LIBRARY_PATH` (and run.sh's `TORCHNATIVE_VULKAN_DYLD`) still
    /// choose the loader; the absolute paths are fallbacks after it.
    #[test]
    #[cfg(target_os = "macos")]
    fn the_bare_name_wins_and_the_sdk_precedes_homebrew() {
        let c = loader_candidates_for(Some("/sdk"));
        assert_eq!(c[0], "libvulkan.dylib");
        let at = |p: &str| c.iter().position(|x| x == p).unwrap_or_else(|| panic!("{p} not in {c:?}"));
        assert!(at("/sdk/lib/libvulkan.1.dylib") < at("/opt/homebrew/lib/libvulkan.1.dylib"));
        assert!(at("/opt/homebrew/lib/libvulkan.1.dylib") < at("/usr/local/lib/libvulkan.1.dylib"));
        // An empty VULKAN_SDK is not the filesystem root.
        assert!(!loader_candidates_for(Some("")).iter().any(|p| p.starts_with("/lib")));
        assert!(!loader_candidates_for(None).iter().any(|p| p.contains("/sdk")));
    }

    #[test]
    #[cfg(all(target_os = "linux", not(target_env = "ohos")))]
    fn linux_looks_for_the_soname_first_and_has_no_macos_paths() {
        let c = loader_candidates_for(Some("/sdk"));
        assert_eq!(c[0], "libvulkan.so.1");
        assert!(!c.iter().any(|p| p.contains("homebrew") || p.ends_with(".dylib")));
    }
}


/// `aten.embedding.default` -- the gather that starts every transformer.
///
/// `embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False,
/// sparse=False)`. The two trailing flags are **autograd hints**: upstream's
/// forward reads neither (they change how the backward accumulates), and there
/// is no backward on this device, so they are accepted and the forward is
/// asserted equal to upstream's with them set. `padding_idx` is likewise a
/// backward-only argument in `aten::embedding`; the row is still gathered.
/// That is a claim about upstream's kernel, and
/// `test_embedding_ignores_the_autograd_only_arguments_exactly_as_upstream_does`
/// checks it against upstream rather than against this comment.
///
/// **The index range is checked before anything is dispatched**, against the
/// `(min, max)` recorded when the indices were uploaded -- so an out-of-range
/// index is upstream's `IndexError` with zero shader dispatches, and the check
/// costs no read-back. `docs/devices/VULKAN6.md` §2.
fn embedding_vulkan(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let weight = crate::aten::tensor_arg(op, args, kwargs, 0, "weight")?;
    let indices = crate::aten::tensor_arg(op, args, kwargs, 1, "indices")?;
    // The table is float32 like every other tensor with a kernel here...
    check_dtype(op, weight.tag())?;
    // ...and the indices are the one integer tensor this device holds.
    if indices.tag() != TorchDType::Int64 {
        return Err(not_implemented(format!(
            "{op}: the vulkan embedding kernel indexes with int64 (stored as \
             int32, docs/devices/VULKAN6.md §1), not {}. Upstream's \
             aten::embedding requires Long indices too.",
            indices.tag().name()
        )));
    }
    let w = weight.vk_tensor(op)?.clone();
    let idx = indices.vk_tensor(op)?.clone();
    if w.shape.len() != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "'weight' must be 2-D, got {}-D",
            w.shape.len()
        )));
    }
    let (rows, dim) = (w.shape[0], w.shape[1]);

    // The range the upload recorded. `None` can only mean a buffer that was
    // not filled by the int64 upload path, which nothing in this module can
    // produce -- so it refuses rather than guessing a range.
    let count = idx.elem_count();
    // Upstream's wording for an index outside the table.
    check_index_range(op, &idx, rows, count, |bad| {
        pyo3::exceptions::PyIndexError::new_err(format!(
            "index out of range in self: {bad} is not in [0, {rows})"
        ))
    })?;

    let mut shape = idx.shape.clone();
    shape.push(dim);
    let n = count * dim;
    let ctx = require(op)?;
    let out = unsafe {
        let out = ctx.alloc(n.max(1) * 4).map_err(|e| vk_error(op, e))?;
        if n > 0 {
            ctx.dispatch_kernel(
                "embedding_i32_f32",
                EMBEDDING_I32_F32_SPV,
                &[&w.buffer, &idx.buffer, &out],
                [n as u32, dim as u32, 0, 0],
            )
            .map_err(|e| vk_error(op, e))?;
        }
        out
    };
    wrap(py, out, shape, weight.tag())
}

// ---------------------------------------------------------------------------
// The pretrained-BERT wall: slice, select, expand, any-rank transpose, gather,
// tanh (docs/devices/VULKAN7.md)
// ---------------------------------------------------------------------------
//
// `docs/devices/VULKAN6.md` §3.1 measured what stopped a pretrained BERT after
// a single-head block ran: `expand` 10, `slice` 1, `gather` 1, `select` 1,
// `tanh` 1, and `transpose.int` on 4-D tensors. Four of those six are *views*
// upstream, and a `VkTensor` has a shape and no strides -- so here they
// materialise, all through one kernel that takes the view's `(shape, stride,
// base)` and copies. That keeps the property every other op here has (each
// tensor is a contiguous buffer, so nothing downstream needs to learn strides)
// at the price of one dispatch per view, which is stated in the per-op
// dispatch table rather than hidden.

/// The highest rank a view kernel carries in its push constants.
///
/// A stated bound, like `VULKAN_INDEX_MAX`, and for the same reason: the push
/// block is eighty bytes (inside the 128 every device must accept), so the
/// rank a view can have is a property of this library, not of the GPU. BERT
/// needs 4; a rank above 8 refuses by name before any GPU work.
pub const VIEW_RANK_MAX: usize = 8;

/// A count or offset a shader indexes with, or a refusal naming it.
///
/// The shaders index with `uint`, so anything above `u32::MAX` would wrap into
/// a different, plausible element. Checked *before* allocating, so the
/// refusal costs nothing -- `expand` in particular can ask for an output far
/// larger than its input, which upstream answers as a free view and this
/// device would have to materialise.
fn shader_u32(op: &str, what: &str, value: usize) -> PyResult<u32> {
    u32::try_from(value).map_err(|_| {
        not_implemented(format!(
            "{op}: the {what} is {value}, which does not fit the 32-bit index \
             every vulkan shader in this build uses (bound {}). A view is \
             materialised on this device, so its size is a real allocation \
             and a real index, not a free stride (docs/devices/VULKAN7.md §2).",
            u32::MAX
        ))
    })
}

fn check_view_rank(op: &str, rank: usize) -> PyResult<()> {
    if rank > VIEW_RANK_MAX {
        return Err(not_implemented(format!(
            "{op}: a rank-{rank} view is above the rank {VIEW_RANK_MAX} the vulkan \
             view kernels carry in their push constants. The bound is this \
             build's, the same on every device, and a higher rank refuses \
             rather than being reshaped by some other route \
             (docs/devices/VULKAN7.md §2)."
        )));
    }
    Ok(())
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// The source's element count, the storage class, and the dtype check that
/// every view kernel shares: float32 and the int64-as-int32 index words are
/// both accepted, because the kernel copies words and computes on neither.
fn view_source(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<(PyTensorBase, VkTensor, VkStore)> {
    let input = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
    let store = check_storage_dtype(op, input.tag())?;
    let a = input.vk_tensor(op)?.clone();
    Ok((input, a, store))
}

/// What the output of a view says about its index values: its source's range,
/// marked inherited. A float tensor has none.
fn inherit_range(store: VkStore, from: &VkBuffer, to: &VkBuffer) {
    if store == VkStore::I64AsI32 {
        if let Some(r) = from.index_range.get() {
            let _ = to.index_range.set(IndexRange { exact: false, ..*r });
        }
    }
}

/// Materialise `out[i] = src[base + sum coord_d * stride_d]` over `shape`.
///
/// Returns the input's own buffer, with no dispatch, when the view is the
/// identity -- the output would be a byte-for-byte copy of a buffer nothing on
/// this device ever writes to again, and upstream's answer there is a view of
/// the same storage too.
fn materialise_view(
    py: Python<'_>,
    op: &str,
    input: &PyTensorBase,
    a: &VkTensor,
    store: VkStore,
    shape: Vec<usize>,
    stride: Vec<usize>,
    base: usize,
) -> PyResult<Py<PyAny>> {
    check_view_rank(op, shape.len())?;
    let n: usize = shape.iter().product();
    let identity = base == 0 && shape == a.shape && stride == contiguous_strides(&a.shape);
    if identity {
        let out = PyTensorBase::vulkan(
            VkTensor { buffer: a.buffer.clone(), shape },
            input.tag(),
        );
        return crate::tensor::promote(py, out.into_pyobject(py)?.into_any().unbind());
    }
    let n32 = shader_u32(op, "output element count", n)?;
    let mut push = vec![n32, shape.len() as u32, 0, 0];
    if n > 0 {
        // The largest source offset this view can form, checked against the
        // source rather than trusted: a stride derivation that was wrong by
        // one axis would otherwise read past the buffer in a shader that
        // cannot raise.
        let last = base
            + shape
                .iter()
                .zip(&stride)
                .map(|(&e, &s)| (e - 1) * s)
                .sum::<usize>();
        if last >= a.elem_count() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{op}: vulkan: internal view error -- offset {last} is outside a \
                 source of {} elements (shape {:?}, stride {:?}, base {base})",
                a.elem_count(),
                shape,
                stride
            )));
        }
        push[2] = shader_u32(op, "source offset", base)?;
        shader_u32(op, "source offset", last)?;
    }
    let mut dims = [0u32; VIEW_RANK_MAX];
    let mut strides = [0u32; VIEW_RANK_MAX];
    for d in 0..shape.len() {
        dims[d] = shape[d] as u32;
        strides[d] = if n > 0 { shader_u32(op, "stride", stride[d])? } else { 0 };
    }
    push.extend_from_slice(&dims);
    push.extend_from_slice(&strides);
    let ctx = require(op)?;
    let out = unsafe {
        let out = ctx.alloc(n.max(1) * 4).map_err(|e| vk_error(op, e))?;
        if n > 0 {
            ctx.dispatch_kernel_words(
                "strided_gather_u32",
                STRIDED_GATHER_U32_SPV,
                &[&a.buffer, &a.buffer, &out],
                &push,
            )
            .map_err(|e| vk_error(op, e))?;
        }
        out
    };
    inherit_range(store, &a.buffer, &out);
    wrap(py, out, shape, input.tag())
}

/// `aten::slice.Tensor(self, dim=0, start=None, end=None, step=1)`.
///
/// Upstream clamps rather than raising; the arithmetic is the dense kernel's
/// (`aten.rs::slice_tensor`) and the meta kernel's, transcribed.
fn slice_vulkan(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let (input, a, store) = view_source(op, args, kwargs)?;
    let rank = a.shape.len();
    let dim = crate::aten::normalise_dim(
        op,
        crate::aten::dim_arg(args, kwargs, 1, "dim")?.unwrap_or(0),
        rank,
    )?;
    if rank == 0 {
        return Err(pyo3::exceptions::PyIndexError::new_err(
            "slice() cannot be applied to a 0-dim tensor.",
        ));
    }
    let extent = a.shape[dim] as i64;
    let step = crate::aten::int_arg(args, kwargs, 4, "step")?.unwrap_or(1);
    if step <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "step must be greater than zero, got {step}"
        )));
    }
    let clamp = |value: i64| -> i64 {
        let shifted = if value < 0 { value + extent } else { value };
        shifted.clamp(0, extent)
    };
    let start = clamp(crate::aten::int_arg(args, kwargs, 2, "start")?.unwrap_or(0));
    let end = match crate::aten::int_arg(args, kwargs, 3, "end")? {
        Some(value) if value >= extent => extent,
        Some(value) => clamp(value),
        None => extent,
    };
    let length = ((end - start).max(0) as usize).div_ceil(step as usize);
    let mut shape = a.shape.clone();
    let mut stride = contiguous_strides(&a.shape);
    let base = start as usize * stride[dim];
    shape[dim] = length;
    stride[dim] *= step as usize;
    materialise_view(py, op, &input, &a, store, shape, stride, base)
}

/// `aten::select.int(self, dim, index)` -- BERT's pooler takes `[:, 0]`.
fn select_vulkan(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let (input, a, store) = view_source(op, args, kwargs)?;
    let rank = a.shape.len();
    if rank == 0 {
        return Err(pyo3::exceptions::PyIndexError::new_err(
            "invalid index of a 0-dim tensor",
        ));
    }
    let dim = crate::aten::normalise_dim(
        op,
        crate::aten::dim_arg(args, kwargs, 1, "dim")?.unwrap_or(0),
        rank,
    )?;
    let index = crate::aten::normalise_index(
        op,
        crate::aten::int_arg(args, kwargs, 2, "index")?
            .ok_or_else(|| crate::aten::missing(op, "index"))? as isize,
        a.shape[dim],
    )?;
    let mut shape = a.shape.clone();
    let mut stride = contiguous_strides(&a.shape);
    let base = index * stride[dim];
    shape.remove(dim);
    stride.remove(dim);
    materialise_view(py, op, &input, &a, store, shape, stride, base)
}

/// `aten::expand(self, size, *, implicit=False)` -- a grown axis reads the
/// same source element along its whole length (stride 0).
fn expand_vulkan(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let (input, a, store) = view_source(op, args, kwargs)?;
    let requested = crate::aten::shape_arg(op, args, kwargs, 1, "size")?;
    let target = crate::aten::expand_target(op, &a.shape, &requested)?;
    crate::aten::check_expand_extents(&a.shape, &target, &requested)?;
    check_view_rank(op, target.len())?;
    let lead = target.len() - a.shape.len();
    let source = contiguous_strides(&a.shape);
    let stride: Vec<usize> = (0..target.len())
        .map(|d| {
            if d < lead || a.shape[d - lead] != target[d] {
                0
            } else {
                source[d - lead]
            }
        })
        .collect();
    materialise_view(py, op, &input, &a, store, target, stride, 0)
}

/// `aten.transpose.int` for any pair of a tensor of rank up to
/// `VIEW_RANK_MAX`: the two strides swap.
fn transpose_strided(
    py: Python<'_>,
    op: &str,
    input: &PyTensorBase,
    a: &VkTensor,
    d0: usize,
    d1: usize,
) -> PyResult<Py<PyAny>> {
    let store = check_storage_dtype(op, input.tag())?;
    let mut shape = a.shape.clone();
    let mut stride = contiguous_strides(&a.shape);
    shape.swap(d0, d1);
    stride.swap(d0, d1);
    materialise_view(py, op, input, a, store, shape, stride, 0)
}

/// Refuse an index whose recorded range falls outside `[0, extent)`.
///
/// `exact_refusal` is upstream's error for the op, given the offending value;
/// when the range was **inherited** (the index tensor is a view of another)
/// the refusal says so and gives the bound, rather than claiming a value it
/// cannot have read.
fn check_index_range(
    op: &str,
    idx: &VkTensor,
    extent: usize,
    count: usize,
    exact_refusal: impl Fn(i64) -> PyErr,
) -> PyResult<()> {
    // `None` can only mean a buffer that no int64 upload or view filled,
    // which nothing in this module produces -- so it refuses rather than
    // guessing a range.
    let Some(range) = idx.buffer.index_range.get() else {
        return Err(not_implemented(format!(
            "{op}: this index tensor carries no recorded range, so its values \
             cannot be checked against the table without reading the device \
             back. Copy the indices to the vulkan device with .to(\"vulkan\") \
             (docs/devices/VULKAN6.md §2)."
        )));
    };
    if count == 0 || (range.lo >= 0 && range.hi < extent as i64) {
        return Ok(());
    }
    let bad = if range.lo < 0 { range.lo } else { range.hi };
    if range.exact {
        return Err(exact_refusal(bad));
    }
    Err(pyo3::exceptions::PyIndexError::new_err(format!(
        "{op}: this index tensor is a view of another, and the only bound on its \
         values this device holds without reading it back is the one inherited \
         from that source: [{}, {}]. {bad} is not in [0, {extent}), so the \
         indices cannot be shown to be in range and are not gathered. Its own \
         values may all be in range; upload them directly with .to(\"vulkan\") \
         to record their exact range (docs/devices/VULKAN7.md §3).",
        range.lo, range.hi
    )))
}

/// `aten::gather(self, dim, index, *, sparse_grad=False)`.
///
/// `sparse_grad` is an autograd hint, like `embedding`'s trailing flags, and
/// the test asks upstream for the same answer with it set. The shape checks
/// are the dense kernel's (`aten.rs::gather_default`), in upstream's words.
fn gather_vulkan(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let input = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
    let dim_raw = crate::aten::dim_arg(args, kwargs, 1, "dim")?
        .ok_or_else(|| crate::aten::missing(op, "dim"))?;
    let index = crate::aten::tensor_arg(op, args, kwargs, 2, "index")?;
    let _sparse_grad = crate::aten::bool_arg(args, kwargs, 3, "sparse_grad")?;
    let store = check_storage_dtype(op, input.tag())?;
    if index.tag() != TorchDType::Int64 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "gather(): Expected dtype int32/int64 for index -- and the vulkan \
             device holds int64 indices only (stored as int32, \
             docs/devices/VULKAN6.md §1), not {}",
            index.tag().name()
        )));
    }
    let a = input.vk_tensor(op)?.clone();
    let idx = index.vk_tensor(op)?.clone();
    // `ensure_nonempty_dim`: a 0-d tensor counts as one axis of extent 1.
    let self_dims = if a.shape.is_empty() { vec![1] } else { a.shape.clone() };
    let idx_dims = if idx.shape.is_empty() { vec![1] } else { idx.shape.clone() };
    if idx_dims.len() != self_dims.len() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Index tensor must have the same number of dimensions as input tensor",
        ));
    }
    let dim = crate::aten::normalise_dim(op, dim_raw, a.shape.len())?;
    for d in 0..self_dims.len() {
        if d != dim && idx_dims[d] > self_dims[d] {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Size does not match at dimension {d} expected index {idx_dims:?} \
                 to be no larger than self {self_dims:?} apart from dimension {dim}"
            )));
        }
    }
    check_view_rank(op, self_dims.len())?;
    let n = idx.elem_count();
    let extent = self_dims[dim];
    check_index_range(op, &idx, extent, n, |bad| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "index {bad} is out of bounds for dimension {dim} with size {extent}"
        ))
    })?;
    let n32 = shader_u32(op, "output element count", n)?;
    shader_u32(op, "source element count", a.elem_count())?;
    let mut push = vec![n32, self_dims.len() as u32, dim as u32, 0];
    let mut dims = [0u32; VIEW_RANK_MAX];
    let mut strides = [0u32; VIEW_RANK_MAX];
    for (d, (&e, s)) in idx_dims.iter().zip(contiguous_strides(&self_dims)).enumerate() {
        dims[d] = e as u32;
        strides[d] = s as u32;
    }
    push.extend_from_slice(&dims);
    push.extend_from_slice(&strides);
    let ctx = require(op)?;
    let out = unsafe {
        let out = ctx.alloc(n.max(1) * 4).map_err(|e| vk_error(op, e))?;
        if n > 0 {
            ctx.dispatch_kernel_words(
                "gather_u32_i32",
                GATHER_U32_I32_SPV,
                &[&a.buffer, &idx.buffer, &out],
                &push,
            )
            .map_err(|e| vk_error(op, e))?;
        }
        out
    };
    inherit_range(store, &a.buffer, &out);
    wrap(py, out, idx.shape.clone(), input.tag())
}

/// The strides of `shape` right-aligned into `out`, 0 where it broadcasts.
fn broadcast_strides(shape: &[usize], out: &[usize]) -> Vec<usize> {
    let lead = out.len() - shape.len();
    let own = contiguous_strides(shape);
    (0..out.len())
        .map(|d| {
            if d < lead || (shape[d - lead] == 1 && out[d] != 1) {
                0
            } else {
                own[d - lead]
            }
        })
        .collect()
}

/// `add`/`sub`/`mul`/`div` `.Tensor` over unequal, broadcastable shapes, in
/// one dispatch (docs/devices/VULKAN7.md §3.2).
///
/// What upstream refuses -- a pair of extents that are neither equal nor 1 --
/// refuses here in upstream's words (`aten.rs::broadcast_shape`), before any
/// GPU work.
fn broadcast_binary(
    py: Python<'_>,
    op: &str,
    lhs: &PyTensorBase,
    a: &VkTensor,
    b: &VkTensor,
) -> PyResult<Py<PyAny>> {
    let which = match op {
        "aten.add.Tensor" => 0u32,
        "aten.sub.Tensor" => 1,
        "aten.mul.Tensor" => 2,
        "aten.div.Tensor" => 3,
        other => {
            return Err(not_implemented(format!(
                "{other}: the vulkan broadcast kernel implements add, sub, mul and div"
            )))
        }
    };
    let shape = crate::aten::broadcast_shape(op, &a.shape, &b.shape)?;
    check_view_rank(op, shape.len())?;
    let n = shape.iter().product::<usize>();
    let n32 = shader_u32(op, "output element count", n)?;
    shader_u32(op, "operand element count", a.elem_count().max(b.elem_count()))?;
    let sa = broadcast_strides(&a.shape, &shape);
    let sb = broadcast_strides(&b.shape, &shape);
    let mut push = vec![n32, shape.len() as u32, which, 0];
    let mut dims = [0u32; VIEW_RANK_MAX];
    let mut stride_a = [0u32; VIEW_RANK_MAX];
    let mut stride_b = [0u32; VIEW_RANK_MAX];
    for d in 0..shape.len() {
        dims[d] = shape[d] as u32;
        stride_a[d] = sa[d] as u32;
        stride_b[d] = sb[d] as u32;
    }
    push.extend_from_slice(&dims);
    push.extend_from_slice(&stride_a);
    push.extend_from_slice(&stride_b);
    let ctx = require(op)?;
    let out = unsafe {
        let out = ctx.alloc(n.max(1) * 4).map_err(|e| vk_error(op, e))?;
        if n > 0 {
            ctx.dispatch_kernel_words(
                "broadcast_binary_f32",
                BROADCAST_BINARY_F32_SPV,
                &[&a.buffer, &b.buffer, &out],
                &push,
            )
            .map_err(|e| vk_error(op, e))?;
        }
        out
    };
    wrap(py, out, shape, lhs.tag())
}

/// A batch of matrices, broadcast to `batch` leading dimensions and made
/// contiguous: the operand itself when nothing needs to move, otherwise one
/// strided copy.
fn broadcast_batch(
    op: &str,
    t: &VkTensor,
    batch: &[usize],
) -> PyResult<(Arc<VkBuffer>, bool)> {
    let r = t.shape.len();
    let mut want = batch.to_vec();
    want.extend_from_slice(&t.shape[r - 2..]);
    if want == t.shape {
        return Ok((t.buffer.clone(), false));
    }
    check_view_rank(op, want.len())?;
    let stride = broadcast_strides(&t.shape, &want);
    let n = want.iter().product::<usize>();
    let n32 = shader_u32(op, "broadcast operand element count", n)?;
    let mut push = vec![n32, want.len() as u32, 0, 0];
    let mut dims = [0u32; VIEW_RANK_MAX];
    let mut strides = [0u32; VIEW_RANK_MAX];
    for d in 0..want.len() {
        dims[d] = want[d] as u32;
        strides[d] = stride[d] as u32;
    }
    push.extend_from_slice(&dims);
    push.extend_from_slice(&strides);
    let ctx = require(op)?;
    let out = unsafe {
        let out = ctx.alloc(n.max(1) * 4).map_err(|e| vk_error(op, e))?;
        if n > 0 {
            ctx.dispatch_kernel_words(
                "strided_gather_u32",
                STRIDED_GATHER_U32_SPV,
                &[&t.buffer, &t.buffer, &out],
                &push,
            )
            .map_err(|e| vk_error(op, e))?;
        }
        out
    };
    Ok((Arc::new(out), true))
}

/// `aten::matmul(self, other)` with upstream's rank rules: a 1-D operand is a
/// row (left) or column (right) that is removed from the result, and batch
/// dimensions broadcast. The product itself is the `bmm` kernel over the
/// flattened batch -- every tensor here is contiguous, so the flattening moves
/// no bytes -- and a broadcast batch is materialised first, one strided copy
/// per operand that needs it.
fn matmul_vulkan(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let lhs = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
    let rhs = crate::aten::tensor_arg(op, args, kwargs, 1, "other")?;
    check_all_f32(op, &[&lhs, &rhs])?;
    let a0 = lhs.vk_tensor(op)?.clone();
    let b0 = rhs.vk_tensor(op)?.clone();
    let runtime = |m: String| pyo3::exceptions::PyRuntimeError::new_err(m);
    let (ra, rb) = (a0.shape.len(), b0.shape.len());
    if ra == 0 || rb == 0 {
        return Err(runtime(format!(
            "both arguments to matmul need to be at least 1D, but they are {ra}D and {rb}D"
        )));
    }
    // Promote a vector to a matrix, remembering to drop the axis afterwards.
    let a = VkTensor {
        buffer: a0.buffer.clone(),
        shape: if ra == 1 { vec![1, a0.shape[0]] } else { a0.shape.clone() },
    };
    let b = VkTensor {
        buffer: b0.buffer.clone(),
        shape: if rb == 1 { vec![b0.shape[0], 1] } else { b0.shape.clone() },
    };
    let (m, k) = (a.shape[a.shape.len() - 2], a.shape[a.shape.len() - 1]);
    let (k2, n) = (b.shape[b.shape.len() - 2], b.shape[b.shape.len() - 1]);
    if k != k2 {
        return Err(runtime(if ra <= 2 && rb <= 2 {
            format!("mat1 and mat2 shapes cannot be multiplied ({m}x{k} and {k2}x{n})")
        } else {
            format!(
                "Expected size for first two dimensions of batch2 tensor to be: \
                 [*, {k}] but got: [*, {k2}]."
            )
        }));
    }
    let batch_a = &a.shape[..a.shape.len() - 2];
    let batch_b = &b.shape[..b.shape.len() - 2];
    let batch = crate::aten::broadcast_shape(op, batch_a, batch_b)
        .map_err(|e| runtime(e.to_string().trim_start_matches(&format!("{op}: ")).to_string()))?;
    check_view_rank(op, batch.len() + 2)?;
    let flat: usize = batch.iter().product();
    let count = flat * m * n;
    let count32 = shader_u32(op, "output element count", count)?;
    shader_u32(op, "operand element count", (flat * m * k).max(flat * k * n))?;
    let (abuf, _) = broadcast_batch(op, &a, &batch)?;
    let (bbuf, _) = broadcast_batch(op, &b, &batch)?;
    let ctx = require(op)?;
    let out = unsafe {
        let out = ctx.alloc(count.max(1) * 4).map_err(|e| vk_error(op, e))?;
        if count > 0 {
            ctx.dispatch_kernel(
                "bmm_f32",
                BMM_F32_SPV,
                &[&abuf, &bbuf, &out],
                [count32, m as u32, k as u32, n as u32],
            )
            .map_err(|e| vk_error(op, e))?;
        }
        out
    };
    let mut shape = batch;
    if ra > 1 {
        shape.push(m);
    }
    if rb > 1 {
        shape.push(n);
    }
    wrap(py, out, shape, lhs.tag())
}

fn safe_softmax_vulkan(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let input = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
    let dim_raw = crate::aten::dim_arg(args, kwargs, 1, "dim")?.ok_or_else(|| crate::aten::missing(op, "dim"))?;
    let dtype = crate::aten::dtype_arg(args, kwargs, 2, "dtype")?;
    let tag = dtype.unwrap_or(input.tag());
    if tag != input.tag() {
        return Err(not_implemented(format!(
            "{op}: the vulkan device cannot cast to dtype {} -- cast first.",
            tag.name()
        )));
    }
    check_dtype(op, input.tag())?;
    let x = input.vk_tensor(op)?.clone();
    let rank = x.shape.len() as isize;
    let span = rank.max(1);
    let dim = if dim_raw < 0 { dim_raw + span } else { dim_raw };
    if dim < 0 || dim >= span {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "Dimension out of range (expected to be in range of [{}, {}], but got {dim_raw})",
            -span,
            span - 1
        )));
    }
    if rank > 0 && dim != rank - 1 {
        return Err(not_implemented(format!(
            "{op}: the vulkan device implements safe_softmax over the last dimension only,              got dim={dim_raw} of a {rank}-D tensor. Move it with .cpu() or permute first."
        )));
    }
    let ctx = require(op)?;
    let n = x.elem_count();
    let row_len = x.shape.last().copied().unwrap_or(1);
    let rows = if row_len > 0 { n / row_len } else { 0 };
    let out = unsafe {
        let out = ctx.alloc(n * 4).map_err(|e| vk_error(op, e))?;
        ctx.dispatch_kernel(
            "safe_softmax_lastdim_f32",
            SAFE_SOFTMAX_LASTDIM_F32_SPV,
            &[&x.buffer, &x.buffer, &out],
            [rows as u32, row_len as u32, 0, 0],
        )
        .map_err(|e| vk_error(op, e))?;
        out
    };
    wrap(py, out, x.shape, input.tag())
}

fn all_vulkan(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let input = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
    if input.tag() != TorchDType::Bool {
        return Err(not_implemented(format!("{op} on vulkan is implemented only for bool tensors")));
    }
    let x = input.vk_tensor(op)?.clone();
    let n = x.elem_count();
    let ctx = require(op)?;
    let out = unsafe {
        let out = ctx.alloc(4).map_err(|e| vk_error(op, e))?;
        ctx.dispatch_kernel(
            "all_bool_i32",
            ALL_BOOL_I32_SPV,
            &[&x.buffer, &out],
            [n as u32, 0, 0, 0],
        )
        .map_err(|e| vk_error(op, e))?;
        out
    };
    wrap_vk(py, VkTensor { buffer: Arc::new(out), shape: vec![] }, TorchDType::Bool)
}

fn local_scalar_dense_vulkan(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let input = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
    let x = input.vk_tensor(op)?.clone();
    if x.elem_count() != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{op}: expected a tensor with exactly one element, got {} elements",
            x.elem_count()
        )));
    }
    // We must return a Python scalar!
    let ctx = require(op)?;
    match input.tag() {
        TorchDType::Bool => {
            let host = unsafe { ctx.download_i32(&x.buffer, 1) }.map_err(|e| vk_error(op, e))?;
            Ok((host[0] != 0).into_bound_py_any(py)?.unbind())
        }
        TorchDType::Float32 => {
            let host = unsafe { ctx.download(&x.buffer, 1) }.map_err(|e| vk_error(op, e))?;
            Ok((host[0] as f64).into_bound_py_any(py)?.unbind())
        }
        TorchDType::Int64 => {
            let host = unsafe { ctx.download_i32(&x.buffer, 1) }.map_err(|e| vk_error(op, e))?;
            Ok((host[0] as i64).into_bound_py_any(py)?.unbind())
        }
        _ => Err(not_implemented(format!("{op} for dtype {} is not implemented", input.tag().name()))),
    }
}


/// One step of the math path, dispatched as an aten op.
///
/// **Not a Python-API call.** `crate::aten::aten_dispatch` is the same
/// function `_aten_dispatch` reaches, so every operand stays a
/// `torch._C.TensorBase` and nothing here depends on the caller having
/// arrived through `torch.Tensor`.
fn sdpa_step(
    py: Python<'_>,
    op: &str,
    args: Vec<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let tuple = PyTuple::new(py, args)?;
    crate::aten::aten_dispatch(py, op, &tuple, None)
}

fn sdpa_vulkan(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    // The math path of `scaled_dot_product_attention`, composed from ops that
    // are already taught on this device:
    //
    //   transpose.int -> matmul (q @ kT) -> mul.Scalar (scale)
    //                 -> add.Tensor (mask, if any) -> _softmax -> matmul (@ v)
    //
    // This is the fallback for a device with no fused flash kernel, which is
    // every device this module has.
    //
    // **Each step is dispatched, not called through the Python API.** This
    // handler sits on the device dispatch table, so it is reached with
    // whatever the dispatcher is holding -- bare `torch._C.TensorBase`
    // objects. It used to be inline Python ending in `torch.softmax(...)`,
    // and `torch.softmax` is a Python-level entry point that accepts only the
    // `torch.Tensor` subclass:
    //
    //     TypeError: softmax(): argument 'input' (position 1) must be Tensor,
    //                not torch._C.TensorBase
    //
    // A model forward never saw that, because a model's operands came in
    // through `torch.Tensor`; calling the op directly, which is what the
    // dispatcher does, failed every time. A dispatch handler that re-enters
    // the API above it only works for callers who came in that way, so the
    // handler dispatches `aten._softmax.default` instead and the asymmetry is
    // gone. `test_sdpa_is_reachable_from_the_dispatcher_and_agrees` in
    // test_vulkan4.py is the test that fails if it comes back.
    //
    // The argument indices are the schema's, and they are *not* the ones the
    // inline Python used: `(query, key, value, dropout_p=0.0, is_causal=false,
    // attn_mask=None, scale=None)`. The old body read `args[3]` as the mask,
    // which is `dropout_p` -- so a positional call meant one thing here and
    // another in `sdpa_flash_cpu`, which parses these same indices.
    let query = crate::aten::tensor_arg(op, args, kwargs, 0, "query")?;
    let _key = crate::aten::tensor_arg(op, args, kwargs, 1, "key")?;
    let _value = crate::aten::tensor_arg(op, args, kwargs, 2, "value")?;
    let dropout_p = match crate::aten::optional(args, kwargs, 3, "dropout_p")? {
        Some(value) if !value.is_none() => value.extract::<f64>()?,
        _ => 0.0,
    };
    let is_causal = crate::aten::bool_arg(args, kwargs, 4, "is_causal")?.unwrap_or(false);
    let attn_mask = crate::aten::optional_tensor_arg(op, args, kwargs, 5, "attn_mask")?;
    let scale = match crate::aten::optional(args, kwargs, 6, "scale")? {
        Some(value) if !value.is_none() => Some(value.extract::<f64>()?),
        _ => None,
    };

    // Upstream's CPU kernel refuses both of these by name rather than
    // returning an answer computed as if they had not been given, and a
    // device that quietly ignored them would be wrong in the one direction
    // nobody checks. The wording is upstream's, kept as `sdpa_flash_cpu`
    // keeps it.
    if dropout_p > 0.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "scaled_dot_product_attention_flash_attention: Currently do not support dropout > 0",
        ));
    }
    if is_causal {
        return Err(not_implemented(format!(
            "{op} on vulkan is implemented only for is_causal=False"
        )));
    }

    // `1/sqrt(E)` over the last dimension, read off the device tensor's own
    // shape -- there is no host readback here, a shape is metadata.
    let scale = match scale {
        Some(value) => value,
        None => {
            let shape = &query.vk_tensor(op)?.shape;
            let last = *shape.last().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "{op}: query must have at least one dimension"
                ))
            })?;
            1.0 / (last as f64).sqrt()
        }
    };

    // The operands are passed on as the objects they arrived as, so this
    // composes ops without ever constructing a Python-level tensor.
    let q = crate::aten::required(op, args, kwargs, 0, "query")?.unbind();
    let k = crate::aten::required(op, args, kwargs, 1, "key")?.unbind();
    let v = crate::aten::required(op, args, kwargs, 2, "value")?.unbind();

    let kt = sdpa_step(py, "aten.transpose.int", vec![
        k,
        (-2i64).into_bound_py_any(py)?.unbind(),
        (-1i64).into_bound_py_any(py)?.unbind(),
    ])?;
    let scores = sdpa_step(py, "aten.matmul.default", vec![q, kt])?;
    let mut scores = sdpa_step(py, "aten.mul.Scalar", vec![
        scores,
        scale.into_bound_py_any(py)?.unbind(),
    ])?;
    if attn_mask.is_some() {
        let mask = crate::aten::required(op, args, kwargs, 5, "attn_mask")?.unbind();
        scores = sdpa_step(py, "aten.add.Tensor", vec![scores, mask])?;
    }
    let attn = sdpa_step(py, "aten._softmax.default", vec![
        scores,
        (-1i64).into_bound_py_any(py)?.unbind(),
        false.into_bound_py_any(py)?.unbind(),
    ])?;
    let out = sdpa_step(py, "aten.matmul.default", vec![attn, v])?;

    let none = py.None();
    let tup = pyo3::types::PyTuple::new(py, [out.into_bound(py), none.into_bound(py)])?;
    Ok(tup.into_any().unbind())
}

