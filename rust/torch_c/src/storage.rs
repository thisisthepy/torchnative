//! `torch._C.StorageBase` -- a flat, untyped byte buffer.
//!
//! The vendored tree's `torch/storage.py:467` opens with
//! `class UntypedStorage(torch._C.StorageBase, _StorageBase)`, so this is the
//! exact name the Python layer subclasses, the same relationship `TensorBase`
//! has with `torch.Tensor`. It is in Rust rather than in `bootstrap.py` for the
//! reason that file's docstring gives: it holds state and enforces an
//! invariant, and bootstrap.py holds neither.
//!
//! # What a storage is here, and what it is not
//!
//! Upstream a storage is the *owner* of a tensor's memory, and a tensor is a
//! view onto it -- `set_` makes the tensor alias the storage, so writing to the
//! storage afterwards changes the tensor. candle owns its own storage and has
//! no way to express that aliasing, so here a storage is a byte buffer that
//! `TensorBase.set_` **copies out of**.
//!
//! That difference is not cosmetic, and docs/CKPT.md §4 has the measurement:
//! `torch.load`'s two container formats fill the storage at different moments
//! relative to `set_`.
//!
//! ```text
//!   zip    (default)  fill storage from the record  ->  _rebuild_tensor -> set_
//!   legacy (v0)       _rebuild_tensor -> set_       ->  fill storage from file
//! ```
//!
//! Under aliasing both orders give the same answer, which is why upstream can
//! use one code path for both. Under copying the second order gives a tensor
//! full of whatever the storage held at `set_` time -- **zeros**. Measured: a
//! copying `set_` loads a legacy checkpoint to a complete state dict in which
//! every single weight is `0.0`, with no error raised anywhere.
//!
//! Silently wrong weights are the worst failure this shim can have, so the
//! ordering is not left to be respected by convention. `filled` records whether
//! any bytes were ever written in, and `set_` refuses on a storage that was
//! never filled. That makes the legacy order impossible to take by accident:
//! anyone who later implements `_set_from_file` gets a refusal naming this
//! invariant instead of a checkpoint of zeros.
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};

use crate::device::PyDevice;
use crate::err::not_implemented;

#[pyclass(name = "StorageBase", module = "torch._C", subclass)]
pub struct PyStorageBase {
    buf: Vec<u8>,
    /// Whether bytes were ever written in. See the module docstring -- this is
    /// the guard that makes the copy/alias difference loud instead of silent.
    /// Allocation does not set it; only `_shim_fill` does.
    filled: bool,
    /// Storages are CPU-only here. `torch.load` asks for `device="meta"` when
    /// it is loading under a fake mode, and that is refused at construction
    /// rather than answered with a CPU buffer.
    device: String,
}

impl PyStorageBase {
    pub fn bytes(&self) -> &[u8] {
        &self.buf
    }

    pub fn is_filled(&self) -> bool {
        self.filled
    }
}

#[pymethods]
impl PyStorageBase {
    /// `UntypedStorage(nbytes)`. The other upstream spellings -- from a
    /// sequence, wrapping another storage -- are not reachable from the load
    /// path this exists for, and are refused by `_StorageBase`'s own stubs
    /// rather than guessed at here.
    #[new]
    #[pyo3(signature = (size = 0, *, device = None))]
    fn new(size: usize, device: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let device = match device {
            None => "cpu".to_string(),
            Some(d) => {
                let s = d.str()?.to_string();
                if s != "cpu" {
                    return Err(not_implemented(format!(
                        "torch._C shim: UntypedStorage(device={s:?}) -- storages here \
                         are CPU byte buffers only"
                    )));
                }
                s
            }
        };
        Ok(Self {
            buf: vec![0u8; size],
            filled: false,
            device,
        })
    }

    /// Write the payload in. Not an upstream name: upstream fills a storage
    /// through `_set_from_file` / `_write_file` / the C++ reader, none of which
    /// are implemented here. This is the one door, so `filled` cannot be set by
    /// anything that did not actually deliver bytes.
    fn _shim_fill(&mut self, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let view = pyo3::buffer::PyBuffer::<u8>::get(data)?;
        let bytes = view.to_vec(data.py())?;
        if bytes.len() != self.buf.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "torch._C shim: UntypedStorage._shim_fill got {} bytes for a \
                 storage of {} bytes",
                bytes.len(),
                self.buf.len()
            )));
        }
        self.buf.copy_from_slice(&bytes);
        self.filled = true;
        Ok(())
    }

    /// Readable from Python so the invariant in the module docstring can be
    /// asserted by a test rather than argued about.
    #[getter]
    fn _shim_filled(&self) -> bool {
        self.filled
    }

    fn nbytes(&self) -> usize {
        self.buf.len()
    }

    fn size(&self) -> usize {
        self.buf.len()
    }

    fn __len__(&self) -> usize {
        self.buf.len()
    }

    /// Upstream's untyped storage reports 1: it is a byte buffer, and the
    /// element size belongs to the *typed* storage wrapped around it.
    fn element_size(&self) -> usize {
        1
    }

    /// The address of the buffer. Real, and stable for the life of the storage,
    /// because `torch/serialization.py` and `safetensors` both use it only to
    /// tell two storages apart -- never to read through it.
    fn data_ptr(&self) -> usize {
        self.buf.as_ptr() as usize
    }

    #[getter]
    fn device(&self) -> PyDevice {
        PyDevice::cpu()
    }

    #[getter]
    fn is_cuda(&self) -> bool {
        false
    }

    #[getter]
    fn is_hpu(&self) -> bool {
        false
    }

    #[getter]
    fn is_sparse(&self) -> bool {
        false
    }

    #[getter]
    fn is_sparse_csr(&self) -> bool {
        false
    }

    fn __getstate__(&self) -> PyResult<()> {
        Err(not_implemented(
            "torch._C shim: UntypedStorage does not pickle -- this shim reads \
             checkpoints, it does not write them",
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "<torch._C.StorageBase {} bytes on {}{}>",
            self.buf.len(),
            self.device,
            if self.filled { "" } else { ", unfilled" }
        )
    }

    /// `bytes(storage)`, for tests and for anything that wants the payload back
    /// without going through a tensor.
    fn _shim_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.buf)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStorageBase>()?;
    Ok(())
}
