use super::{Buffer, Device};
use objc2::{
    rc::Retained,
    runtime::{AnyClass, NSObjectProtocol as _, ProtocolObject},
    sel,
};
use objc2_metal::{MTLAllocation, MTLDevice as _, MTLResidencySet, MTLResidencySetDescriptor};
use std::sync::atomic::{AtomicU64, Ordering};

// torchnative (issue #65): `MTLResidencySet` is a macOS 15 / iOS 18 API, and
// `MTLResidencySetDescriptor::new()` panics with "class MTLResidencySetDescriptor
// could not be found" on anything older, so creating any Metal device panicked
// there while the wheels are tagged macosx_11_0. The set was already optional
// (`raw: Option<..>`, every use is `if let Some`), so the fix is to decide at
// runtime whether to build it, and take the `None` path when the OS lacks it.
// Without a residency set, buffers are made resident by Metal's default
// per-command-buffer tracking, which is what Metal did before macOS 15.

/// Set to any non-empty value to build no residency set even where the OS has
/// one. It can only *disable* the set, never force one on an OS without it, so
/// it cannot fail open; tests use it to drive the pre-macOS-15 path on a newer
/// machine.
pub const DISABLE_ENV: &str = "CANDLE_METAL_NO_RESIDENCY_SET";

/// Residency sets built, and devices that went without one. Process-wide.
static CREATED: AtomicU64 = AtomicU64::new(0);
static SKIPPED: AtomicU64 = AtomicU64::new(0);

/// `[created, skipped]`: how many devices got a residency set and how many
/// took the pre-macOS-15 path.
pub fn residency_set_counters() -> [u64; 2] {
    [CREATED.load(Ordering::Relaxed), SKIPPED.load(Ordering::Relaxed)]
}

/// Whether this process can build a residency set at all: the descriptor class
/// exists in the loaded Metal framework. The device-side selector is checked
/// separately in `ResidencySet::new`.
pub fn residency_sets_supported() -> bool {
    AnyClass::get(c"MTLResidencySetDescriptor").is_some()
}

/// The decision, pure so it can be tested with every input faked. A set is
/// built only when the class exists AND the device answers the selector AND
/// nobody disabled it. Any `false` takes the path without a set.
fn should_build(class_present: bool, device_responds: bool, disabled: bool) -> bool {
    class_present && device_responds && !disabled
}

fn disabled_by_env() -> bool {
    std::env::var_os(DISABLE_ENV).is_some_and(|v| !v.is_empty())
}

/// Keeps Metal buffers resident in GPU memory, removing the need for `useResource` calls.
///
/// Add the set to the command queue once via `MTLCommandQueue::addResidencySet`. Then
/// register every buffer at allocation time and unregister at free time.
pub struct ResidencySet {
    raw: Option<Retained<ProtocolObject<dyn MTLResidencySet>>>,
}

unsafe impl Send for ResidencySet {}
unsafe impl Sync for ResidencySet {}

impl ResidencySet {
    pub fn new(device: &Device) -> Self {
        let class_present = residency_sets_supported();
        // Only ask the device once the class is known to exist; on an OS
        // without it the selector is absent as well.
        let device_responds = class_present
            && device
                .as_ref()
                .respondsToSelector(sel!(newResidencySetWithDescriptor:error:));
        if !should_build(class_present, device_responds, disabled_by_env()) {
            SKIPPED.fetch_add(1, Ordering::Relaxed);
            return ResidencySet { raw: None };
        }
        let descriptor = MTLResidencySetDescriptor::new();
        let raw = device
            .as_ref()
            .newResidencySetWithDescriptor_error(&descriptor)
            .ok()
            .inspect(|set| set.requestResidency());
        if raw.is_some() {
            CREATED.fetch_add(1, Ordering::Relaxed);
        } else {
            SKIPPED.fetch_add(1, Ordering::Relaxed);
        }
        ResidencySet { raw }
    }

    pub fn raw(&self) -> Option<&ProtocolObject<dyn MTLResidencySet>> {
        self.raw.as_deref()
    }

    pub fn insert(&self, buf: &Buffer) {
        if let Some(set) = &self.raw {
            set.addAllocation(as_allocation(buf));
            set.commit();
        }
    }

    pub fn remove(&self, buf: &Buffer) {
        if let Some(set) = &self.raw {
            set.removeAllocation(as_allocation(buf));
            set.commit();
        }
    }
}

/// Cast a `&Buffer` to `&ProtocolObject<dyn MTLAllocation>`.
///
/// Safe because `MTLBuffer: MTLResource: MTLAllocation`. All `ProtocolObject<P>` are
/// `repr(C)` thin ObjC pointers — the cast only changes the static protocol marker,
/// not the pointer value or runtime dispatch.
fn as_allocation(buf: &Buffer) -> &ProtocolObject<dyn MTLAllocation> {
    unsafe {
        &*(buf.as_ref() as *const ProtocolObject<dyn objc2_metal::MTLBuffer>
            as *const ProtocolObject<dyn MTLAllocation>)
    }
}

#[cfg(test)]
mod gate_tests {
    use super::should_build;

    // torchnative (issue #65): the gate with every input faked, so the
    // pre-macOS-15 answer is checked on a machine that has macOS 15.
    #[test]
    fn a_residency_set_needs_the_class_the_selector_and_no_override() {
        assert!(should_build(true, true, false));
        assert!(!should_build(false, false, false), "macOS 14: no class");
        assert!(!should_build(false, true, false));
        assert!(!should_build(true, false, false));
        assert!(!should_build(true, true, true), "the override only disables");
    }
}
