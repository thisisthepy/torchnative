//! `torch._C.dtype`.
//!
//! In real PyTorch `torch.float32` is not a Python-level constant -- it is an
//! instance of a type defined in `_C`, and the vendored Python tree re-exports
//! it (`torch/__init__.py`). So the shim has to own the type, not just a name.
//!
//! It also has to own the *set*. This module used to wrap `candle_core::DType`
//! directly, which meant the shim could only name a dtype candle could store:
//! ten of them. The vendored tree needs thirty-three. `torch/_prims_common`,
//! `torch/_tensor_str.py` and `torch/_refs` all build tables over
//! `torch.bool`, `torch.int8`, `torch.complex64` and the quantised types while
//! `import torch` is still running, so a shim with ten dtypes cannot finish the
//! import at all.
//!
//! BOOL.md settled the shape of the answer for `torch.bool` and it generalises
//! to the rest: **`_C` owns the dtype tag and candle owns the storage.** The
//! two are related by `storage()`, which is `None` for every dtype candle
//! cannot hold. That is the same decision `device` already embodies (a label,
//! resolved on use) and it has the same payoff -- the failure lands where torch
//! puts it, by name, instead of on a near neighbour.
//!
//! Specifically *not* done: aliasing `bool` onto `uint8`. BOOL.md §3 measured
//! what that costs -- `bool + bool` is a logical or while `uint8 + uint8` is an
//! arithmetic sum, `~bool` is a logical not while `~uint8` is a bit flip, and
//! `masked_fill` stops refusing the wrong mask dtype. Six of torch's own
//! guardrails are keyed on the tag; sharing one tag between the two turns all
//! six off.
use candle_core::DType;
use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Every dtype the vendored tree names. Derived from the tree's own stub
/// (`torch/_C/__init__.pyi` declares each as a module-level `: dtype`), not
/// from an installed upstream binary.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TorchDType {
    Float32,
    Float64,
    Float16,
    BFloat16,
    Complex32,
    Complex64,
    Complex128,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Bool,
    QInt8,
    QUInt8,
    QInt32,
    QUInt4x2,
    QUInt2x4,
    Float8E4M3FN,
    Float8E4M3FNUZ,
    Float8E5M2,
    Float8E5M2FNUZ,
    Float8E8M0FNU,
    Float4E2M1FNX2,
    Bits1x8,
    Bits2x4,
    Bits4x2,
    Bits8,
    Bits16,
    // The sub-byte integer tags. These are *not* in `torch/_C/__init__.pyi`
    // (which is where the rest of this enum came from) -- upstream registers
    // them from C++ without a stub entry, so the generated surface never
    // named them and the shim did not either.
    //
    // They are here because `torch.load` demands them by name.
    // `torch/_weights_only_unpickler.py:194` builds its allowlist with
    //
    //     for t in [getattr(torch, f"uint{x}") for x in range(1, 8)]: ...
    //     for t in [getattr(torch, f"int{x}") for x in range(1, 8)]: ...
    //
    // unconditionally, before a single byte of the checkpoint is read. Missing
    // any one of the fourteen makes *every* `torch.load` raise
    // `AttributeError: module 'torch' has no attribute 'uint1'` -- measured,
    // and it was the second wall on that path (docs/CKPT.md §2).
    //
    // Nothing can be stored under them: `storage()` returns `None` for all
    // fourteen, so a tensor tagged with one refuses by name rather than
    // quietly becoming `uint8`. They are names on the allowlist, and that is
    // the whole of what they are.
    UInt1,
    UInt2,
    UInt3,
    UInt4,
    UInt5,
    UInt6,
    UInt7,
    Int1,
    Int2,
    Int3,
    Int4,
    Int5,
    Int6,
    Int7,
}

use TorchDType::*;

/// The canonical spellings, and the full set. `register` walks this.
pub const ALL: &[TorchDType] = &[
    Float32,
    Float64,
    Float16,
    BFloat16,
    Complex32,
    Complex64,
    Complex128,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Bool,
    QInt8,
    QUInt8,
    QInt32,
    QUInt4x2,
    QUInt2x4,
    Float8E4M3FN,
    Float8E4M3FNUZ,
    Float8E5M2,
    Float8E5M2FNUZ,
    Float8E8M0FNU,
    Float4E2M1FNX2,
    Bits1x8,
    Bits2x4,
    Bits4x2,
    Bits8,
    Bits16,
    UInt1,
    UInt2,
    UInt3,
    UInt4,
    UInt5,
    UInt6,
    UInt7,
    Int1,
    Int2,
    Int3,
    Int4,
    Int5,
    Int6,
    Int7,
];

/// torch's short spellings. `torch.float is torch.float32` is `True` upstream,
/// so these name the same dtype rather than a distinct one.
pub const ALIASES: &[(&str, TorchDType)] = &[
    ("float", Float32),
    ("double", Float64),
    ("half", Float16),
    ("short", Int16),
    ("int", Int32),
    ("long", Int64),
    ("chalf", Complex32),
    ("cfloat", Complex64),
    ("cdouble", Complex128),
];

impl TorchDType {
    pub fn name(self) -> &'static str {
        match self {
            Float32 => "float32",
            Float64 => "float64",
            Float16 => "float16",
            BFloat16 => "bfloat16",
            Complex32 => "complex32",
            Complex64 => "complex64",
            Complex128 => "complex128",
            UInt8 => "uint8",
            UInt16 => "uint16",
            UInt32 => "uint32",
            UInt64 => "uint64",
            Int8 => "int8",
            Int16 => "int16",
            Int32 => "int32",
            Int64 => "int64",
            Bool => "bool",
            QInt8 => "qint8",
            QUInt8 => "quint8",
            QInt32 => "qint32",
            QUInt4x2 => "quint4x2",
            QUInt2x4 => "quint2x4",
            Float8E4M3FN => "float8_e4m3fn",
            Float8E4M3FNUZ => "float8_e4m3fnuz",
            Float8E5M2 => "float8_e5m2",
            Float8E5M2FNUZ => "float8_e5m2fnuz",
            Float8E8M0FNU => "float8_e8m0fnu",
            Float4E2M1FNX2 => "float4_e2m1fn_x2",
            Bits1x8 => "bits1x8",
            Bits2x4 => "bits2x4",
            Bits4x2 => "bits4x2",
            Bits8 => "bits8",
            Bits16 => "bits16",
            UInt1 => "uint1",
            UInt2 => "uint2",
            UInt3 => "uint3",
            UInt4 => "uint4",
            UInt5 => "uint5",
            UInt6 => "uint6",
            UInt7 => "uint7",
            Int1 => "int1",
            Int2 => "int2",
            Int3 => "int3",
            Int4 => "int4",
            Int5 => "int5",
            Int6 => "int6",
            Int7 => "int7",
        }
    }

    /// How candle stores a tensor with this tag. `None` means the shim can
    /// name the dtype but cannot hold a tensor of it -- every op path that
    /// reaches such a dtype must refuse by name rather than substitute.
    ///
    /// `Bool` is the interesting entry: it stores as `U8` but is *not*
    /// `UInt8`. BOOL.md §6.3 states the invariant that buys -- the bytes under
    /// a `bool` tag are 0 or 1 -- and keeping it is the shim's job, which is
    /// why there is one tagging constructor rather than a free-for-all.
    pub fn storage(self) -> Option<DType> {
        Some(match self {
            Float32 => DType::F32,
            Float64 => DType::F64,
            Float16 => DType::F16,
            BFloat16 => DType::BF16,
            UInt8 => DType::U8,
            UInt32 => DType::U32,
            Int16 => DType::I16,
            Int32 => DType::I32,
            Int64 => DType::I64,
            Float8E4M3FN => DType::F8E4M3,
            Bool => DType::U8,
            _ => return None,
        })
    }

    /// The tag a tensor candle handed us carries. Deliberately *not* the
    /// inverse of `storage()`: `U8` maps back to `uint8`, never to `bool`,
    /// because a bool tensor's tag has to be carried alongside the storage
    /// rather than reconstructed from it (BOOL.md §5-B).
    pub fn from_storage(dtype: DType) -> Option<Self> {
        Some(match dtype {
            DType::F32 => Float32,
            DType::F64 => Float64,
            DType::F16 => Float16,
            DType::BF16 => BFloat16,
            DType::U8 => UInt8,
            DType::U32 => UInt32,
            DType::I16 => Int16,
            DType::I32 => Int32,
            DType::I64 => Int64,
            DType::F8E4M3 => Float8E4M3FN,
            // candle's MX formats (`F6E2M3`, `F6E3M2`, `F4`, `F8E8M0`) stay
            // unnamed. torch has `float8_e8m0fnu` and `float4_e2m1fn_x2`, but
            // the latter packs two values per byte, so the correspondence is
            // not established and lending the name would be the silent drift
            // this file exists to avoid.
            _ => return None,
        })
    }

    pub fn is_floating_point(self) -> bool {
        matches!(
            self,
            Float32
                | Float64
                | Float16
                | BFloat16
                | Float8E4M3FN
                | Float8E4M3FNUZ
                | Float8E5M2
                | Float8E5M2FNUZ
                | Float8E8M0FNU
                | Float4E2M1FNX2
        )
    }

    pub fn is_complex(self) -> bool {
        matches!(self, Complex32 | Complex64 | Complex128)
    }

    pub fn is_signed(self) -> bool {
        !matches!(
            self,
            UInt8 | UInt16
                | UInt32
                | UInt64
                | UInt1
                | UInt2
                | UInt3
                | UInt4
                | UInt5
                | UInt6
                | UInt7
                | Bool
                | QUInt8
                | QUInt4x2
                | QUInt2x4
                | Float8E8M0FNU
                | Bits1x8
                | Bits2x4
                | Bits4x2
                | Bits8
                | Bits16
        )
    }

    /// torch's short spelling, used as a key by `torch/utils/_dtype_abbrs.py`
    /// (`{dt: dt.abbr for dt in torch._C._get_all_dtypes()}`, at import). The
    /// mapping is torch's, read off torch 2.13.0 -- it is not derivable from
    /// the name (`bits8` is `b8x1`, and `bool` is `b8`).
    pub fn abbr(self) -> &'static str {
        match self {
            Float32 => "f32",
            Float64 => "f64",
            Float16 => "f16",
            BFloat16 => "bf16",
            Complex32 => "c32",
            Complex64 => "c64",
            Complex128 => "c128",
            UInt8 => "u8",
            UInt16 => "u16",
            UInt32 => "u32",
            UInt64 => "u64",
            Int8 => "i8",
            Int16 => "i16",
            Int32 => "i32",
            Int64 => "i64",
            Bool => "b8",
            QInt8 => "qi8",
            QUInt8 => "qu8",
            QInt32 => "qi32",
            QUInt4x2 => "qu4x2",
            QUInt2x4 => "qu2x4",
            Float8E4M3FN => "f8e4m3fn",
            Float8E4M3FNUZ => "f8e4m3fnuz",
            Float8E5M2 => "f8e5m2",
            Float8E5M2FNUZ => "f8e5m2fnuz",
            Float8E8M0FNU => "f8e8m0fnu",
            Float4E2M1FNX2 => "f4e2m1fnx2",
            Bits1x8 => "b1x8",
            Bits2x4 => "b2x4",
            Bits4x2 => "b4x2",
            Bits8 => "b8x1",
            Bits16 => "b16x1",
            // Upstream has no abbreviation for these: `dtype_abbrs` is built
            // from `_get_all_dtypes()`, which leaves all fourteen out, and
            // `torch.uint1.abbr` reads back as the full name (measured on
            // torch 2.13.0). Spelling them out here says the same thing.
            UInt1 => "uint1",
            UInt2 => "uint2",
            UInt3 => "uint3",
            UInt4 => "uint4",
            UInt5 => "uint5",
            UInt6 => "uint6",
            UInt7 => "uint7",
            Int1 => "int1",
            Int2 => "int2",
            Int3 => "int3",
            Int4 => "int4",
            Int5 => "int5",
            Int6 => "int6",
            Int7 => "int7",
        }
    }

    /// Whether `torch._C._get_all_dtypes()` lists this one. torch leaves the
    /// five quantised dtypes out of that list, and the fourteen sub-byte
    /// integer tags too -- upstream torch 2.13.0 returns exactly 27 entries,
    /// re-measured after `uint1..7`/`int1..7` were added here. That count is
    /// the reason those fourteen are excluded rather than a guess about them:
    /// `torch/utils/_dtype_abbrs.py` builds a dict keyed on exactly this list.
    pub fn in_all_dtypes(self) -> bool {
        !matches!(
            self,
            QInt8
                | QUInt8
                | QInt32
                | QUInt4x2
                | QUInt2x4
                | UInt1
                | UInt2
                | UInt3
                | UInt4
                | UInt5
                | UInt6
                | UInt7
                | Int1
                | Int2
                | Int3
                | Int4
                | Int5
                | Int6
                | Int7
        )
    }

    /// Bytes per element. torch reports 1 for `bool`, the same as `uint8`
    /// (BOOL.md §2.1) -- the two differ in semantics, not in storage width.
    pub fn itemsize(self) -> usize {
        match self {
            Complex128 => 16,
            Float64 | Complex64 | Int64 | UInt64 => 8,
            Float32 | Int32 | UInt32 | QInt32 => 4,
            Float16 | BFloat16 | Int16 | UInt16 | Complex32 | Bits16 => 2,
            _ => 1,
        }
    }
}

/// `torch.float32` and friends.
#[pyclass(name = "dtype", module = "torch._C", frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PyDtype {
    inner: TorchDType,
}

impl PyDtype {
    pub fn new(inner: TorchDType) -> Self {
        Self { inner }
    }

    pub fn tag(&self) -> TorchDType {
        self.inner
    }

    /// The candle dtype to store this in, or a named refusal. Callers holding
    /// a dtype and needing storage go through here rather than matching on
    /// `storage()` themselves, so the message stays in one place.
    pub fn storage(&self, op: &str) -> PyResult<DType> {
        self.inner.storage().ok_or_else(|| {
            crate::err::not_implemented(format!(
                "{op}: dtype not storable by the candle backend in torch._C shim: torch.{}",
                self.inner.name()
            ))
        })
    }
}

#[pymethods]
impl PyDtype {
    fn __repr__(&self) -> String {
        format!("torch.{}", self.inner.name())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.extract::<PyDtype>() {
            Ok(other) => self.inner == other.inner,
            Err(_) => false,
        }
    }

    fn __hash__(&self) -> u64 {
        self.inner as u64
    }

    /// torch pickles a dtype by its bare name; matching that keeps the
    /// vendored tree's serialisation paths from needing to know about us.
    fn __reduce__(&self) -> String {
        self.inner.name().to_string()
    }

    #[getter]
    fn is_floating_point(&self) -> bool {
        self.inner.is_floating_point()
    }

    #[getter]
    fn is_complex(&self) -> bool {
        self.inner.is_complex()
    }

    #[getter]
    fn is_signed(&self) -> bool {
        self.inner.is_signed()
    }

    #[getter]
    fn itemsize(&self) -> usize {
        self.inner.itemsize()
    }

    #[getter]
    fn abbr(&self) -> &'static str {
        self.inner.abbr()
    }

    /// Not a torch attribute. It answers the one question that is specific to
    /// this shim -- "can candle hold this?" -- so that the vendored tree, the
    /// tests and the golden harness can ask instead of keeping a second copy
    /// of `storage()`'s table.
    #[getter]
    fn _has_storage(&self) -> bool {
        self.inner.storage().is_some()
    }

    fn to_real(&self) -> PyDtype {
        PyDtype::new(match self.inner {
            Complex32 => Float16,
            Complex64 => Float32,
            Complex128 => Float64,
            other => other,
        })
    }

    fn to_complex(&self) -> PyDtype {
        PyDtype::new(match self.inner {
            Float16 => Complex32,
            Float32 => Complex64,
            Float64 => Complex128,
            other => other,
        })
    }
}

/// `torch._C._get_all_dtypes()`. `torch/utils/_dtype_abbrs.py:5` calls this
/// during `import torch` and indexes every result by `.abbr`, so both the list
/// and the attribute are import-blocking.
#[pyfunction]
#[pyo3(name = "_get_all_dtypes")]
pub fn get_all_dtypes() -> Vec<PyDtype> {
    ALL.iter()
        .filter(|d| d.in_all_dtypes())
        .map(|d| PyDtype::new(*d))
        .collect()
}

/// Look a dtype up by its torch spelling, aliases included.
pub fn by_name(name: &str) -> Option<TorchDType> {
    ALL.iter()
        .copied()
        .find(|d| d.name() == name)
        .or_else(|| ALIASES.iter().find(|(n, _)| *n == name).map(|(_, d)| *d))
}

/// Registers the type and the module-level dtype instances.
///
/// Upstream puts these on `torch`, not on `torch._C` -- the C side does
/// `PyModule_AddObject` against the `torch` module during `_initExtension`
/// (VENDOR.md wall 7). Registering them on `_C` and letting
/// `from torch._C import *` carry them across lands them in the same place by
/// a different route; `bootstrap.py`'s `_initExtension` fills in whatever the
/// star import did not cover.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDtype>()?;
    m.add_function(wrap_pyfunction!(get_all_dtypes, m)?)?;

    // One Python object per dtype, shared with its aliases. torch guarantees
    // `torch.float is torch.float32`, and the tree uses dtypes as dict keys
    // in enough places that two objects comparing equal but not identical is
    // a difference worth not having.
    let py = m.py();
    let mut made: Vec<(TorchDType, Py<PyAny>)> = Vec::with_capacity(ALL.len());
    for dtype in ALL {
        let object = PyDtype::new(*dtype).into_pyobject(py)?.into_any().unbind();
        m.add(dtype.name(), object.clone_ref(py))?;
        made.push((*dtype, object));
    }
    for (alias, dtype) in ALIASES {
        let object = made
            .iter()
            .find(|(d, _)| d == dtype)
            .map(|(_, o)| o.clone_ref(py))
            .expect("every alias names a dtype in ALL");
        m.add(*alias, object)?;
    }
    Ok(())
}
