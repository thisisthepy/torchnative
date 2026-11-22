//! exir's program serializer, re-implemented in Rust (issue #423, step R5-S1).
//!
//! Owner, 2026-11-21 (asked by the leader, option (c)): "Rust로 재구현": the
//! serializer step is re-implemented in Rust and only exir's passes are
//! vendored, so no executorch Python package goes into the wheel.
//!
//! What it reproduces, measured from the executorch 1.4.1 wheel:
//!
//! * `exir/_serialize/_program.py::serialize_pte_binary`: the constant
//!   segment, the mutable-data segment, delegate segments (optional), named
//!   data, the segment table, the extended header (`eh00`) and the padding
//!   between program and segments.
//! * `exir/_serialize/_flatbuffer_program.py::_program_to_flatbuffer`: the
//!   `Program` dataclasses packed through flatc's generated Python object API
//!   (`ProgramT.Pack`) by the `flatbuffers` 25.12.19 Python `Builder`, with
//!   its two replaced packers for `Buffer` and `BackendDelegateInlineData`
//!   (raw byte copy aligned to the patched `force_align`).
//!
//! The schema is the vendored `schema/program.fbs` (and `scalar_type.fbs`),
//! read once by a parser for the subset of the flatbuffers IDL they use, so the
//! field order, slots and defaults come from the same file the runtime's
//! generated headers come from. The `Builder` below is a line-by-line port of
//! the Python one (`Prep`, `WriteVtable` with its vtable de-duplication,
//! `StartVector`/`EndVector`, `CreateString`, `Finish` with a file
//! identifier); the generated `Pack` order is: every string, vector, table and
//! union value of the table first, in declaration order, then `Start`, then
//! every field in declaration order (scalars only when they differ from the
//! default, offsets only when present), then `End`. Upstream bytes are the
//! specification: `tests/export/test_et_serialize.py` compares both byte for
//! byte on CI.
//!
//! It is pure Rust and needs no runtime, so it is built on every target.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyFloat, PyList, PyString, PyTuple};
use std::collections::HashMap;
use std::sync::OnceLock;

const SCALAR_TYPE_FBS: &str = include_str!("../../vendor/executorch/schema/scalar_type.fbs");
const PROGRAM_FBS: &str = include_str!("../../vendor/executorch/schema/program.fbs");

/// The comment markers `_flatbuffer.py::_patch_schema_alignment` patches.
const TENSOR_MARKER: &str = "@executorch-tensor-alignment";
const DELEGATE_MARKER: &str = "@executorch-delegate-alignment";

// ------------------------------------------------------------------ schema

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Scalar {
    Bool,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
}

impl Scalar {
    fn from_name(s: &str) -> Option<Scalar> {
        Some(match s {
            "bool" => Scalar::Bool,
            "byte" | "int8" => Scalar::I8,
            "ubyte" | "uint8" => Scalar::U8,
            "short" | "int16" => Scalar::I16,
            "ushort" | "uint16" => Scalar::U16,
            "int" | "int32" => Scalar::I32,
            "uint" | "uint32" => Scalar::U32,
            "long" | "int64" => Scalar::I64,
            "ulong" | "uint64" => Scalar::U64,
            "float" | "float32" => Scalar::F32,
            "double" | "float64" => Scalar::F64,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            Scalar::Bool | Scalar::I8 | Scalar::U8 => 1,
            Scalar::I16 | Scalar::U16 => 2,
            Scalar::I32 | Scalar::U32 | Scalar::F32 => 4,
            Scalar::I64 | Scalar::U64 | Scalar::F64 => 8,
        }
    }

    fn is_float(self) -> bool {
        matches!(self, Scalar::F32 | Scalar::F64)
    }

    fn range(self) -> (i128, i128) {
        match self {
            Scalar::Bool => (0, 1),
            Scalar::I8 => (i8::MIN as i128, i8::MAX as i128),
            Scalar::U8 => (0, u8::MAX as i128),
            Scalar::I16 => (i16::MIN as i128, i16::MAX as i128),
            Scalar::U16 => (0, u16::MAX as i128),
            Scalar::I32 => (i32::MIN as i128, i32::MAX as i128),
            Scalar::U32 => (0, u32::MAX as i128),
            Scalar::I64 => (i64::MIN as i128, i64::MAX as i128),
            Scalar::U64 => (0, u64::MAX as i128),
            Scalar::F32 | Scalar::F64 => (0, 0),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Ty {
    Scalar(Scalar),
    /// index into `Schema::enums`
    Enum(usize),
    Str,
    /// index into `Schema::tables`
    Table(usize),
    /// index into `Schema::unions`
    Union(usize),
    Vec(Box<Ty>),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum AlignRole {
    Tensor,
    Delegate,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Num {
    I(i128),
    F(f64),
    B(bool),
}

#[derive(Debug)]
pub(crate) struct Field {
    pub name: String,
    pub ty: Ty,
    pub default: Num,
    pub force_align: Option<usize>,
    pub role: Option<AlignRole>,
    /// vtable slot; a union field takes `slot` (its type) and `slot + 1`
    pub slot: usize,
}

#[derive(Debug)]
pub(crate) struct Table {
    pub name: String,
    pub fields: Vec<Field>,
    pub slots: usize,
}

#[derive(Debug)]
pub(crate) struct EnumDef {
    pub name: String,
    pub base: Scalar,
    pub values: Vec<(String, i128)>,
}

#[derive(Debug)]
pub(crate) struct UnionDef {
    pub name: String,
    /// member `i` has tag `i + 1`; tag 0 is NONE
    pub members: Vec<String>,
}

#[derive(Debug)]
pub(crate) struct Schema {
    pub tables: Vec<Table>,
    pub enums: Vec<EnumDef>,
    pub unions: Vec<UnionDef>,
    pub root: usize,
    pub file_identifier: [u8; 4],
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Str(String),
    Num(String),
    P(char),
}

fn tokenize(text: &str) -> Result<(Vec<(Tok, usize)>, Vec<(usize, AlignRole)>), String> {
    let mut toks = Vec::new();
    let mut marks = Vec::new();
    for (lineno, raw) in text.lines().enumerate() {
        let (code, comment) = match raw.find("//") {
            Some(i) => (&raw[..i], &raw[i..]),
            None => (raw, ""),
        };
        if raw.contains("/*") {
            return Err(format!("line {}: block comments are not in the parsed subset", lineno + 1));
        }
        if comment.contains(TENSOR_MARKER) {
            marks.push((lineno, AlignRole::Tensor));
        } else if comment.contains(DELEGATE_MARKER) {
            marks.push((lineno, AlignRole::Delegate));
        }
        let c: Vec<char> = code.chars().collect();
        let mut i = 0;
        while i < c.len() {
            let ch = c[i];
            if ch.is_whitespace() {
                i += 1;
            } else if ch.is_ascii_alphabetic() || ch == '_' {
                let s = i;
                while i < c.len() && (c[i].is_ascii_alphanumeric() || c[i] == '_' || c[i] == '.') {
                    i += 1;
                }
                toks.push((Tok::Ident(c[s..i].iter().collect()), lineno));
            } else if ch.is_ascii_digit() || ((ch == '-' || ch == '+') && i + 1 < c.len() && c[i + 1].is_ascii_digit()) {
                let s = i;
                i += 1;
                while i < c.len() && (c[i].is_ascii_alphanumeric() || c[i] == '.') {
                    i += 1;
                }
                toks.push((Tok::Num(c[s..i].iter().collect()), lineno));
            } else if ch == '"' {
                let s = i + 1;
                i += 1;
                while i < c.len() && c[i] != '"' {
                    i += 1;
                }
                if i >= c.len() {
                    return Err(format!("line {}: unterminated string", lineno + 1));
                }
                toks.push((Tok::Str(c[s..i].iter().collect()), lineno));
                i += 1;
            } else if "{}:;,()[]=".contains(ch) {
                toks.push((Tok::P(ch), lineno));
                i += 1;
            } else {
                return Err(format!("line {}: unexpected character {ch:?}", lineno + 1));
            }
        }
    }
    Ok((toks, marks))
}

struct RawField {
    name: String,
    ty: String,
    is_vec: bool,
    default: Option<String>,
    force_align: Option<usize>,
    role: Option<AlignRole>,
}

/// Parses the flatbuffers IDL subset program.fbs and scalar_type.fbs use:
/// include, namespace, file_identifier, file_extension, root_type, attribute,
/// enum, union and table (no structs, no nested namespaces). Anything else is
/// an error naming it, so a schema change that leaves the subset is caught at
/// the first call rather than serialised wrongly.
pub(crate) fn parse_schema(texts: &[&str]) -> Result<Schema, String> {
    let mut raw_tables: Vec<(String, Vec<RawField>)> = Vec::new();
    let mut enums: Vec<EnumDef> = Vec::new();
    let mut unions: Vec<UnionDef> = Vec::new();
    let mut root: Option<String> = None;
    let mut ident: Option<String> = None;
    for text in texts {
        let (toks, marks) = tokenize(text)?;
        let mut p = 0usize;
        let next = |p: &mut usize| -> Result<(Tok, usize), String> {
            let t = toks.get(*p).cloned().ok_or_else(|| "unexpected end of schema".to_string())?;
            *p += 1;
            Ok(t)
        };
        let expect = |p: &mut usize, ch: char| -> Result<(), String> {
            match toks.get(*p) {
                Some((Tok::P(c), _)) if *c == ch => {
                    *p += 1;
                    Ok(())
                }
                other => Err(format!("expected {ch:?}, got {other:?}")),
            }
        };
        let ident_of = |t: (Tok, usize)| -> Result<String, String> {
            match t.0 {
                Tok::Ident(s) => Ok(s),
                other => Err(format!("line {}: expected a name, got {other:?}", t.1 + 1)),
            }
        };
        while p < toks.len() {
            let (kw, line) = next(&mut p)?;
            let kw = match kw {
                Tok::Ident(s) => s,
                other => return Err(format!("line {}: unexpected {other:?}", line + 1)),
            };
            match kw.as_str() {
                "include" | "file_extension" | "attribute" => {
                    next(&mut p)?;
                    expect(&mut p, ';')?;
                }
                "namespace" => {
                    next(&mut p)?;
                    expect(&mut p, ';')?;
                }
                "file_identifier" => {
                    match next(&mut p)?.0 {
                        Tok::Str(s) => ident = Some(s),
                        other => return Err(format!("file_identifier: {other:?}")),
                    }
                    expect(&mut p, ';')?;
                }
                "root_type" => {
                    root = Some(ident_of(next(&mut p)?)?);
                    expect(&mut p, ';')?;
                }
                "enum" => {
                    let name = ident_of(next(&mut p)?)?;
                    expect(&mut p, ':')?;
                    let base_name = ident_of(next(&mut p)?)?;
                    let base = Scalar::from_name(&base_name).ok_or_else(|| format!("enum {name}: base {base_name}"))?;
                    expect(&mut p, '{')?;
                    let mut values = Vec::new();
                    let mut v: i128 = 0;
                    loop {
                        match next(&mut p)? {
                            (Tok::P('}'), _) => break,
                            (Tok::P(','), _) => continue,
                            (Tok::Ident(vn), _) => {
                                if let Some((Tok::P('='), _)) = toks.get(p) {
                                    p += 1;
                                    match next(&mut p)?.0 {
                                        Tok::Num(n) => v = n.parse::<i128>().map_err(|e| format!("enum {name}.{vn}: {e}"))?,
                                        other => return Err(format!("enum {name}.{vn}: {other:?}")),
                                    }
                                }
                                values.push((vn, v));
                                v += 1;
                            }
                            other => return Err(format!("enum {name}: unexpected {other:?}")),
                        }
                    }
                    enums.push(EnumDef { name, base, values });
                }
                "union" => {
                    let name = ident_of(next(&mut p)?)?;
                    expect(&mut p, '{')?;
                    let mut members = Vec::new();
                    loop {
                        match next(&mut p)? {
                            (Tok::P('}'), _) => break,
                            (Tok::P(','), _) => continue,
                            (Tok::Ident(m), _) => {
                                if let Some((Tok::P('='), _)) = toks.get(p) {
                                    return Err(format!("union {name}: explicit values are not in the parsed subset"));
                                }
                                members.push(m)
                            }
                            other => return Err(format!("union {name}: unexpected {other:?}")),
                        }
                    }
                    unions.push(UnionDef { name, members });
                }
                "table" => {
                    let name = ident_of(next(&mut p)?)?;
                    expect(&mut p, '{')?;
                    let mut fields = Vec::new();
                    loop {
                        let (t, fline) = next(&mut p)?;
                        let fname = match t {
                            Tok::P('}') => break,
                            Tok::Ident(s) => s,
                            other => return Err(format!("table {name}: unexpected {other:?}")),
                        };
                        expect(&mut p, ':')?;
                        let (ty, is_vec) = match next(&mut p)? {
                            (Tok::P('['), _) => {
                                let t = ident_of(next(&mut p)?)?;
                                expect(&mut p, ']')?;
                                (t, true)
                            }
                            (Tok::Ident(t), _) => (t, false),
                            other => return Err(format!("table {name}.{fname}: type {other:?}")),
                        };
                        let mut default = None;
                        if let Some((Tok::P('='), _)) = toks.get(p) {
                            p += 1;
                            default = Some(match next(&mut p)?.0 {
                                Tok::Num(n) | Tok::Ident(n) => n,
                                other => return Err(format!("table {name}.{fname}: default {other:?}")),
                            });
                        }
                        let mut force_align = None;
                        if let Some((Tok::P('('), _)) = toks.get(p) {
                            p += 1;
                            loop {
                                match next(&mut p)? {
                                    (Tok::P(')'), _) => break,
                                    (Tok::P(','), _) => continue,
                                    (Tok::Ident(a), _) => {
                                        let mut val = None;
                                        if let Some((Tok::P(':'), _)) = toks.get(p) {
                                            p += 1;
                                            val = Some(next(&mut p)?.0);
                                        }
                                        match (a.as_str(), val) {
                                            ("force_align", Some(Tok::Num(n))) => {
                                                force_align = Some(n.parse::<usize>().map_err(|e| format!("{name}.{fname}: {e}"))?)
                                            }
                                            ("deprecated", None) => {
                                                return Err(format!("table {name}.{fname}: deprecated fields are not in the parsed subset"))
                                            }
                                            (other, _) => {
                                                return Err(format!("table {name}.{fname}: attribute {other} is not in the parsed subset"))
                                            }
                                        }
                                    }
                                    other => return Err(format!("table {name}.{fname}: attribute {other:?}")),
                                }
                            }
                        }
                        expect(&mut p, ';')?;
                        let role = marks.iter().find(|(l, _)| *l == fline).map(|(_, r)| *r);
                        fields.push(RawField { name: fname, ty, is_vec, default, force_align, role });
                    }
                    raw_tables.push((name, fields));
                }
                other => return Err(format!("line {}: `{other}` is not in the parsed subset of the IDL", line + 1)),
            }
        }
    }
    let table_index: HashMap<String, usize> = raw_tables.iter().enumerate().map(|(i, (n, _))| (n.clone(), i)).collect();
    let resolve = |t: &str| -> Result<Ty, String> {
        if t == "string" {
            return Ok(Ty::Str);
        }
        if let Some(s) = Scalar::from_name(t) {
            return Ok(Ty::Scalar(s));
        }
        if let Some(i) = enums.iter().position(|e| e.name == t) {
            return Ok(Ty::Enum(i));
        }
        if let Some(i) = unions.iter().position(|u| u.name == t) {
            return Ok(Ty::Union(i));
        }
        if let Some(i) = table_index.get(t) {
            return Ok(Ty::Table(*i));
        }
        Err(format!("unknown type {t}"))
    };
    let mut tables = Vec::new();
    for (name, raws) in raw_tables.iter() {
        let mut fields = Vec::new();
        let mut slot = 0usize;
        for r in raws {
            let base = resolve(&r.ty).map_err(|e| format!("{name}.{}: {e}", r.name))?;
            if r.is_vec && matches!(base, Ty::Union(_)) {
                return Err(format!("{name}.{}: vectors of unions are not in the parsed subset", r.name));
            }
            let scalar = match &base {
                Ty::Scalar(s) => Some(*s),
                Ty::Enum(i) => Some(enums[*i].base),
                _ => None,
            };
            let default = match (&r.default, &base, scalar) {
                (None, _, Some(s)) if s == Scalar::Bool => Num::B(false),
                (None, _, Some(s)) if s.is_float() => Num::F(0.0),
                (None, _, _) => Num::I(0),
                (Some(d), Ty::Enum(i), _) if !r.is_vec => match enums[*i].values.iter().find(|(n, _)| n == d) {
                    Some((_, v)) => Num::I(*v),
                    None => Num::I(d.parse::<i128>().map_err(|e| format!("{name}.{}: default {d}: {e}", r.name))?),
                },
                (Some(d), _, Some(Scalar::Bool)) => Num::B(d == "true"),
                (Some(d), _, Some(s)) if s.is_float() => Num::F(d.parse::<f64>().map_err(|e| format!("{name}.{}: {e}", r.name))?),
                (Some(d), _, _) => Num::I(d.parse::<i128>().map_err(|e| format!("{name}.{}: default {d}: {e}", r.name))?),
            };
            let ty = if r.is_vec { Ty::Vec(Box::new(base.clone())) } else { base.clone() };
            let width = if matches!(ty, Ty::Union(_)) { 2 } else { 1 };
            fields.push(Field { name: r.name.clone(), ty, default, force_align: r.force_align, role: r.role, slot });
            slot += width;
        }
        tables.push(Table { name: name.clone(), fields, slots: slot });
    }
    let root = root.ok_or("no root_type")?;
    let root = *table_index.get(&root).ok_or_else(|| format!("root_type {root} is not a table"))?;
    let id = ident.ok_or("no file_identifier")?;
    let id: [u8; 4] = id.as_bytes().try_into().map_err(|_| format!("file_identifier {id:?} is not 4 bytes"))?;
    Ok(Schema { tables, enums, unions, root, file_identifier: id })
}

pub(crate) fn program_schema() -> PyResult<&'static Schema> {
    static S: OnceLock<Result<Schema, String>> = OnceLock::new();
    S.get_or_init(|| parse_schema(&[SCALAR_TYPE_FBS, PROGRAM_FBS]))
        .as_ref()
        .map_err(|e| PyValueError::new_err(format!("torchnative's program.fbs parser (issue #423): {e}")))
}

impl Schema {
    fn table(&self, name: &str) -> usize {
        self.tables.iter().position(|t| t.name == name).unwrap_or_else(|| panic!("program.fbs has no table {name}"))
    }

    fn field(&self, t: usize, name: &str) -> usize {
        self.tables[t]
            .fields
            .iter()
            .position(|f| f.name == name)
            .unwrap_or_else(|| panic!("program.fbs table {} has no field {name}", self.tables[t].name))
    }
}

// ----------------------------------------------------------------- builder

/// A port of `flatbuffers.Builder` (Python, 25.12.19): the same buffer
/// growth from the back, the same `Prep` padding, the same vtable
/// de-duplication keyed on the trimmed vtable, so the same call sequence
/// gives the same bytes.
pub(crate) struct Builder {
    bytes: Vec<u8>,
    head: usize,
    minalign: usize,
    vtable: Option<Vec<usize>>,
    object_end: usize,
    vtables: HashMap<Vec<usize>, usize>,
    nested: bool,
    vector_num_elems: usize,
}

const MAX_BUFFER_SIZE: usize = 1 << 31;

/// The Builder's and the XNNPACK framing's error: plain Rust, so the crate's
/// unit tests (which run without libpython on Linux, run.sh issue #302) link
/// none of pyo3. It becomes the same `ValueError` at the Python boundary.
#[derive(Debug)]
pub(crate) struct FbError(String);

impl From<FbError> for PyErr {
    fn from(e: FbError) -> PyErr {
        PyValueError::new_err(e.0)
    }
}

type FbResult<T> = Result<T, FbError>;

fn fb_err(msg: impl Into<String>) -> FbError {
    FbError(msg.into())
}

fn err(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

impl Builder {
    pub fn new() -> Builder {
        Builder {
            bytes: vec![0u8; 1024],
            head: 1024,
            minalign: 1,
            vtable: None,
            object_end: 0,
            vtables: HashMap::new(),
            nested: false,
            vector_num_elems: 0,
        }
    }

    fn offset(&self) -> usize {
        self.bytes.len() - self.head
    }

    fn grow(&mut self) -> FbResult<()> {
        let old = self.bytes.len();
        if old == MAX_BUFFER_SIZE {
            return Err(fb_err("flatbuffers: cannot grow buffer beyond 2 gigabytes"));
        }
        let new = (old * 2).clamp(1, MAX_BUFFER_SIZE);
        let mut b = vec![0u8; new];
        b[new - old..].copy_from_slice(&self.bytes);
        self.bytes = b;
        self.head += new - old;
        Ok(())
    }

    fn pad(&mut self, n: usize) {
        for _ in 0..n {
            self.head -= 1;
            self.bytes[self.head] = 0;
        }
    }

    fn prep(&mut self, size: usize, additional: usize) -> FbResult<()> {
        if size > self.minalign {
            self.minalign = size;
        }
        let align_size = (self.bytes.len() - self.head + additional).wrapping_neg() & (size - 1);
        let needed = align_size + size + additional;
        while self.head < needed {
            self.grow()?;
        }
        self.pad(align_size);
        Ok(())
    }

    fn place(&mut self, le: &[u8]) {
        self.head -= le.len();
        let h = self.head;
        self.bytes[h..h + le.len()].copy_from_slice(le);
    }

    fn prepend(&mut self, le: &[u8]) -> FbResult<()> {
        self.prep(le.len(), 0)?;
        self.place(le);
        Ok(())
    }

    fn prepend_uoffset_rel(&mut self, off: usize) -> FbResult<()> {
        self.prep(4, 0)?;
        if off > self.offset() {
            return Err(fb_err("flatbuffers: Offset arithmetic error."));
        }
        let off2 = (self.offset() - off + 4) as u32;
        self.place(&off2.to_le_bytes());
        Ok(())
    }

    fn assert_not_nested(&self) -> FbResult<()> {
        if self.nested {
            return Err(fb_err("flatbuffers: object serialization must not be nested"));
        }
        Ok(())
    }

    fn start_vector(&mut self, elem: usize, n: usize, align: usize) -> FbResult<()> {
        self.assert_not_nested()?;
        self.nested = true;
        self.vector_num_elems = n;
        self.prep(4, elem * n)?;
        self.prep(align, elem * n)?;
        Ok(())
    }

    fn end_vector(&mut self) -> usize {
        self.nested = false;
        let n = self.vector_num_elems as u32;
        self.place(&n.to_le_bytes());
        self.vector_num_elems = 0;
        self.offset()
    }

    /// `StartVector(elem, n, align)`, the payload copied in one piece (the
    /// generated per-element `Prepend` writes the same bytes: after
    /// `StartVector` every element is already aligned), `EndVector`.
    fn raw_vector(&mut self, data: &[u8], elem: usize, n: usize, align: usize) -> FbResult<usize> {
        self.start_vector(elem, n, align)?;
        self.head -= data.len();
        let h = self.head;
        self.bytes[h..h + data.len()].copy_from_slice(data);
        Ok(self.end_vector())
    }

    fn create_string(&mut self, s: &[u8]) -> FbResult<usize> {
        self.assert_not_nested()?;
        self.nested = true;
        self.prep(4, s.len() + 1)?;
        self.place(&[0]);
        self.head -= s.len();
        let h = self.head;
        self.bytes[h..h + s.len()].copy_from_slice(s);
        self.vector_num_elems = s.len();
        Ok(self.end_vector())
    }

    fn offset_vector(&mut self, offs: &[usize]) -> FbResult<usize> {
        self.start_vector(4, offs.len(), 4)?;
        for o in offs.iter().rev() {
            self.prepend_uoffset_rel(*o)?;
        }
        Ok(self.end_vector())
    }

    fn start_object(&mut self, n: usize) -> FbResult<()> {
        self.assert_not_nested()?;
        self.vtable = Some(vec![0; n]);
        self.object_end = self.offset();
        self.nested = true;
        Ok(())
    }

    fn slot(&mut self, i: usize) {
        let off = self.offset();
        if let Some(vt) = self.vtable.as_mut() {
            vt[i] = off;
        }
    }

    fn prepend_uoffset_slot(&mut self, i: usize, off: usize) -> FbResult<()> {
        if off != 0 {
            self.prepend_uoffset_rel(off)?;
            self.slot(i);
        }
        Ok(())
    }

    fn end_object(&mut self) -> FbResult<usize> {
        if !self.nested {
            return Err(fb_err("flatbuffers: object serialization must be nested"));
        }
        self.nested = false;
        // PrependSOffsetTRelative(0)
        self.prep(4, 0)?;
        let off2 = (self.offset() + 4) as i32;
        self.place(&off2.to_le_bytes());
        let object_offset = self.offset();
        let vt = self.vtable.take().unwrap_or_default();
        let mut key = Vec::with_capacity(vt.len() + 1);
        let mut trim = true;
        for &elem in vt.iter().rev() {
            if elem == 0 {
                if trim {
                    continue;
                }
                key.push(0);
            } else {
                key.push(object_offset - elem);
                trim = false;
            }
        }
        let object_size = object_offset - self.object_end;
        key.push(object_size);
        if let Some(&vt2) = self.vtables.get(&key) {
            let object_start = self.bytes.len() - object_offset;
            self.head = object_start;
            let v = (vt2 as i64 - object_offset as i64) as i32;
            let h = self.head;
            self.bytes[h..h + 4].copy_from_slice(&v.to_le_bytes());
        } else {
            let mut trailing = 0usize;
            let mut trim = true;
            for &elem in vt.iter().rev() {
                let mut off = 0usize;
                if elem == 0 {
                    if trim {
                        trailing += 1;
                        continue;
                    }
                } else {
                    off = object_offset - elem;
                    trim = false;
                }
                self.prepend_voffset(off)?;
            }
            self.prepend_voffset(object_size)?;
            let vbytes = (vt.len() - trailing + 2) * 2;
            self.prepend_voffset(vbytes)?;
            let object_start = self.bytes.len() - object_offset;
            let v = (self.offset() as i64 - object_offset as i64) as i32;
            self.bytes[object_start..object_start + 4].copy_from_slice(&v.to_le_bytes());
            self.vtables.insert(key, self.offset());
        }
        Ok(object_offset)
    }

    fn prepend_voffset(&mut self, x: usize) -> FbResult<()> {
        let v = u16::try_from(x).map_err(|_| fb_err(format!("flatbuffers: voffset {x} does not fit 16 bits")))?;
        self.prepend(&v.to_le_bytes())
    }

    pub fn finish(mut self, root: usize, file_identifier: Option<[u8; 4]>) -> FbResult<Vec<u8>> {
        let mut prep_size = 4;
        if file_identifier.is_some() {
            prep_size += 4;
        }
        let minalign = self.minalign;
        self.prep(minalign, prep_size)?;
        if let Some(id) = file_identifier {
            self.prep(4, 4)?;
            for i in (0..4).rev() {
                self.place(&[id[i]]);
            }
        }
        self.prepend_uoffset_rel(root)?;
        Ok(self.bytes[self.head..].to_vec())
    }
}

// ------------------------------------------------------------ value tree

/// One field's value, after `_flatbuffer_program.convert_program`.
#[derive(Clone, Debug)]
pub(crate) enum Val {
    Absent,
    Num(Num),
    Str(Vec<u8>),
    /// a vector of scalars as little-endian bytes: `n` elements of `elem`
    /// bytes, aligned to `align`
    Raw { data: Vec<u8>, elem: usize, n: usize, align: usize },
    Strs(Vec<Vec<u8>>),
    Table(Box<Node>),
    Tables(Vec<Node>),
    Union(u8, Option<Box<Node>>),
}

#[derive(Clone, Debug)]
pub(crate) struct Node {
    pub t: usize,
    pub v: Vec<Val>,
}

/// The alignments `_prepare_schema` patches into `force_align`.
#[derive(Clone, Copy)]
pub(crate) struct Aligns {
    pub tensor: usize,
    pub delegate: usize,
}

fn field_align(f: &Field, a: Aligns) -> Option<usize> {
    match f.role {
        Some(AlignRole::Tensor) => Some(a.tensor),
        Some(AlignRole::Delegate) => Some(a.delegate),
        None => f.force_align,
    }
}

fn type_name(o: &Bound<'_, PyAny>) -> String {
    o.get_type().name().map(|n| n.to_string()).unwrap_or_else(|_| "?".into())
}

/// `_flatbuffer_program._coerce_bytes`.
pub(crate) fn coerce_bytes(o: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(b) = o.cast::<PyBytes>() {
        return Ok(b.as_bytes().to_vec());
    }
    if let Ok(b) = o.cast::<PyByteArray>() {
        return Ok(b.to_vec());
    }
    if let Ok(tobytes) = o.getattr("tobytes") {
        if tobytes.is_callable() {
            return Ok(tobytes.call0()?.cast::<PyBytes>()?.as_bytes().to_vec());
        }
    }
    let b = o.py().get_type::<PyBytes>().call1((o,))?;
    Ok(b.cast::<PyBytes>()?.as_bytes().to_vec())
}

fn scalar_le(s: Scalar, x: Num, path: &str) -> PyResult<Vec<u8>> {
    let bad = |x: &dyn std::fmt::Debug| PyTypeError::new_err(format!("{path}: bad number {x:?} for type {s:?}"));
    Ok(match (s, x) {
        (Scalar::F32, Num::F(f)) => (f as f32).to_le_bytes().to_vec(),
        (Scalar::F64, Num::F(f)) => f.to_le_bytes().to_vec(),
        (Scalar::F32, Num::I(i)) => (i as f32).to_le_bytes().to_vec(),
        (Scalar::F64, Num::I(i)) => (i as f64).to_le_bytes().to_vec(),
        (Scalar::Bool, Num::B(b)) => vec![b as u8],
        (_, Num::I(_)) | (_, Num::B(_)) if !s.is_float() => {
            let i = match x {
                Num::I(i) => i,
                Num::B(b) => b as i128,
                Num::F(_) => unreachable!(),
            };
            let (lo, hi) = s.range();
            if i < lo || i > hi {
                return Err(bad(&i));
            }
            let le = i.to_le_bytes();
            le[..s.size()].to_vec()
        }
        _ => return Err(bad(&x)),
    })
}

fn py_num(o: &Bound<'_, PyAny>, s: Scalar, double_table: bool, path: &str) -> PyResult<Num> {
    if o.is_none() {
        return Err(PyTypeError::new_err(format!("{path}: None for a {s:?} field (issue #423 serializer)")));
    }
    if s == Scalar::Bool {
        return Ok(Num::B(o.is_truthy()?));
    }
    if s.is_float() {
        if double_table && o.is_instance_of::<PyString>() {
            // `_convert_double`: the strings "inf" / "-inf" (exir/schema.py)
            let f = o.py().get_type::<PyFloat>().call1((o,))?;
            return Ok(Num::F(f.extract::<f64>()?));
        }
        return Ok(Num::F(o.extract::<f64>().map_err(|_| {
            PyTypeError::new_err(format!("{path}: bad number of type {} for {s:?}", type_name(o)))
        })?));
    }
    if o.is_instance_of::<PyFloat>() {
        return Err(PyTypeError::new_err(format!("{path}: bad number {o} for type {s:?}")));
    }
    let i = o
        .extract::<i128>()
        .map_err(|_| PyTypeError::new_err(format!("{path}: bad number of type {} for {s:?}", type_name(o))))?;
    Ok(Num::I(i))
}

fn is_ndarray(o: &Bound<'_, PyAny>) -> bool {
    let t = o.get_type();
    t.name().map(|n| n.to_string() == "ndarray").unwrap_or(false)
        && t.module().map(|m| m.to_string() == "numpy").unwrap_or(false)
}

/// `CreateNumpyVector`: element size, alignment and little-endian bytes from
/// the array itself, not from the schema.
fn numpy_raw(o: &Bound<'_, PyAny>, path: &str) -> PyResult<Val> {
    let dtype = o.getattr("dtype")?;
    let kind: String = dtype.getattr("kind")?.extract()?;
    if !["b", "i", "u", "f"].contains(&kind.as_str()) {
        return Err(PyTypeError::new_err(format!("{path}: numpy-ndarray holds elements of unsupported datatype")));
    }
    let ndim: usize = o.getattr("ndim")?.extract()?;
    if ndim > 1 {
        return Err(PyTypeError::new_err(format!("{path}: multidimensional-ndarray passed to CreateNumpyVector")));
    }
    let elem: usize = o.getattr("itemsize")?.extract()?;
    let n: usize = o.getattr("size")?.extract()?;
    let align: usize = dtype.getattr("alignment")?.extract()?;
    let order: String = dtype.getattr("str")?.extract()?;
    let le = if order.starts_with('<') { o.clone() } else { o.call_method1("byteswap", (false,))? };
    let data = le.call_method1("tobytes", ("C",))?.cast::<PyBytes>()?.as_bytes().to_vec();
    Ok(Val::Raw { data, elem, n, align })
}

fn seq_items<'py>(o: &Bound<'py, PyAny>, path: &str) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if let Ok(l) = o.cast::<PyList>() {
        return Ok(l.iter().collect());
    }
    if let Ok(t) = o.cast::<PyTuple>() {
        return Ok(t.iter().collect());
    }
    if o.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(format!("{path}: a str where a vector is expected")));
    }
    o.try_iter()
        .map_err(|_| PyTypeError::new_err(format!("{path}: {} is not a sequence", type_name(o))))?
        .collect()
}

/// `_flatbuffer_program._convert_dataclass` followed by the field reads of the
/// generated `Pack`, for one table.
pub(crate) fn convert(sch: &Schema, t: usize, o: &Bound<'_, PyAny>, a: Aligns, path: &str) -> PyResult<Node> {
    let table = &sch.tables[t];
    let mut v = Vec::with_capacity(table.fields.len());
    for f in &table.fields {
        let fpath = format!("{path}.{}", f.name);
        let present = o.hasattr(f.name.as_str())?;
        let val = if !present {
            match &f.ty {
                Ty::Scalar(_) | Ty::Enum(_) => Val::Num(f.default),
                Ty::Union(_) => Val::Union(0, None),
                _ => Val::Absent,
            }
        } else {
            let x = o.getattr(f.name.as_str())?;
            convert_field(sch, f, &x, a, &fpath, table.name == "Double")?
        };
        v.push(val);
    }
    Ok(Node { t, v })
}

fn convert_field(sch: &Schema, f: &Field, x: &Bound<'_, PyAny>, a: Aligns, path: &str, double_table: bool) -> PyResult<Val> {
    let scalar_of = |ty: &Ty| -> Option<Scalar> {
        match ty {
            Ty::Scalar(s) => Some(*s),
            Ty::Enum(i) => Some(sch.enums[*i].base),
            _ => None,
        }
    };
    match &f.ty {
        Ty::Scalar(_) | Ty::Enum(_) => {
            let s = scalar_of(&f.ty).unwrap();
            Ok(Val::Num(py_num(x, s, double_table, path)?))
        }
        _ if x.is_none() => Ok(match f.ty {
            Ty::Union(_) => Val::Union(0, None),
            _ => Val::Absent,
        }),
        Ty::Str => {
            if let Ok(s) = x.cast::<PyString>() {
                Ok(Val::Str(s.to_str()?.as_bytes().to_vec()))
            } else if let Ok(b) = x.cast::<PyBytes>() {
                Ok(Val::Str(b.as_bytes().to_vec()))
            } else {
                Err(PyTypeError::new_err(format!("{path}: non-string passed to CreateString ({})", type_name(x))))
            }
        }
        Ty::Table(t2) => Ok(Val::Table(Box::new(convert(sch, *t2, x, a, path)?))),
        Ty::Union(u) => {
            let name = type_name(x);
            let un = &sch.unions[*u];
            let i = un.members.iter().position(|m| *m == name).ok_or_else(|| {
                PyTypeError::new_err(format!("{path}: {name} is not a member of the union {}", un.name))
            })?;
            let t2 = sch.table(&name);
            Ok(Val::Union((i + 1) as u8, Some(Box::new(convert(sch, t2, x, a, path)?))))
        }
        Ty::Vec(inner) => match inner.as_ref() {
            Ty::Str => {
                let mut out = Vec::new();
                for (i, e) in seq_items(x, path)?.iter().enumerate() {
                    match convert_field(sch, &Field { name: String::new(), ty: Ty::Str, default: Num::I(0), force_align: None, role: None, slot: 0 }, e, a, &format!("{path}[{i}]"), false)? {
                        Val::Str(s) => out.push(s),
                        _ => return Err(PyTypeError::new_err(format!("{path}[{i}]: None in a vector of strings"))),
                    }
                }
                Ok(Val::Strs(out))
            }
            Ty::Table(t2) => {
                let mut out = Vec::new();
                for (i, e) in seq_items(x, path)?.iter().enumerate() {
                    if e.is_none() {
                        return Err(PyTypeError::new_err(format!("{path}[{i}]: None in a vector of tables")));
                    }
                    out.push(convert(sch, *t2, e, a, &format!("{path}[{i}]"))?);
                }
                Ok(Val::Tables(out))
            }
            other => {
                let s = scalar_of(other).ok_or_else(|| PyTypeError::new_err(format!("{path}: vector of {other:?}")))?;
                if let Some(align) = field_align(f, a) {
                    // the replaced packers `_pack_buffer` /
                    // `_pack_backend_delegate_inline_data`: raw bytes, aligned
                    if s.size() != 1 {
                        return Err(PyTypeError::new_err(format!("{path}: force_align on a vector of {s:?}")));
                    }
                    let data = coerce_bytes(x)?;
                    let n = data.len();
                    return Ok(Val::Raw { data, elem: 1, n, align });
                }
                if is_ndarray(x) {
                    return numpy_raw(x, path);
                }
                if s.size() == 1 {
                    if let Ok(b) = x.cast::<PyBytes>() {
                        let data = b.as_bytes().to_vec();
                        let n = data.len();
                        return Ok(Val::Raw { data, elem: 1, n, align: 1 });
                    }
                    if let Ok(b) = x.cast::<PyByteArray>() {
                        let data = b.to_vec();
                        let n = data.len();
                        return Ok(Val::Raw { data, elem: 1, n, align: 1 });
                    }
                }
                let items = seq_items(x, path)?;
                let mut data = Vec::with_capacity(items.len() * s.size());
                for (i, e) in items.iter().enumerate() {
                    let p = format!("{path}[{i}]");
                    data.extend(scalar_le(s, py_num(e, s, false, &p)?, &p)?);
                }
                Ok(Val::Raw { data, elem: s.size(), n: items.len(), align: s.size() })
            }
        },
    }
}

/// The generated `Pack`, for any table.
pub(crate) fn pack(b: &mut Builder, sch: &Schema, node: &Node) -> PyResult<usize> {
    let table = &sch.tables[node.t];
    let mut offs = vec![0usize; table.fields.len()];
    for (i, val) in node.v.iter().enumerate() {
        offs[i] = match val {
            Val::Str(s) => b.create_string(s)?,
            Val::Raw { data, elem, n, align } => b.raw_vector(data, *elem, *n, *align)?,
            Val::Strs(v) => {
                let mut o = Vec::with_capacity(v.len());
                for s in v {
                    o.push(b.create_string(s)?);
                }
                b.offset_vector(&o)?
            }
            Val::Tables(v) => {
                let mut o = Vec::with_capacity(v.len());
                for n in v {
                    o.push(pack(b, sch, n)?);
                }
                b.offset_vector(&o)?
            }
            Val::Table(n) => pack(b, sch, n)?,
            Val::Union(_, Some(n)) => pack(b, sch, n)?,
            _ => 0,
        };
    }
    b.start_object(table.slots)?;
    for (i, (f, val)) in table.fields.iter().zip(node.v.iter()).enumerate() {
        match val {
            Val::Num(x) => {
                let s = match &f.ty {
                    Ty::Scalar(s) => *s,
                    Ty::Enum(e) => sch.enums[*e].base,
                    other => return Err(err(format!("{}.{}: a number for {other:?}", table.name, f.name))),
                };
                let differs = match (x, f.default) {
                    (Num::I(a), Num::I(d)) => *a != d,
                    (Num::F(a), Num::F(d)) => *a != d,
                    (Num::B(a), Num::B(d)) => *a != d,
                    (Num::I(a), Num::F(d)) => (*a as f64) != d,
                    (Num::B(a), Num::I(d)) => (*a as i128) != d,
                    (Num::I(a), Num::B(d)) => *a != d as i128,
                    _ => true,
                };
                if differs {
                    let le = scalar_le(s, *x, &format!("{}.{}", table.name, f.name))?;
                    b.prepend(&le)?;
                    b.slot(f.slot);
                }
            }
            Val::Union(tag, v) => {
                if *tag != 0 {
                    b.prepend(&[*tag])?;
                    b.slot(f.slot);
                }
                if v.is_some() {
                    b.prepend_uoffset_slot(f.slot + 1, offs[i])?;
                }
            }
            Val::Absent => {}
            _ => b.prepend_uoffset_slot(f.slot, offs[i])?,
        }
    }
    Ok(b.end_object()?)
}

// --------------------------------------------------------------- the PTE

fn padding_required(offset: usize, alignment: usize) -> usize {
    let r = offset % alignment;
    if r != 0 {
        alignment - r
    } else {
        0
    }
}

fn aligned_size(n: usize, alignment: usize) -> usize {
    n + padding_required(n, alignment)
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn lcm(a: usize, b: usize) -> usize {
    if a == 0 || b == 0 {
        0
    } else {
        a / gcd(a, b) * b
    }
}

fn valid_alignment(a: usize) -> bool {
    a > 0 && a & (a - 1) == 0
}

pub(crate) struct PteOptions {
    pub extract_delegate_segments: bool,
    pub segment_alignment: usize,
    pub constant_tensor_alignment: Option<usize>,
    pub delegate_alignment: Option<usize>,
}

/// One named-data entry of `NamedDataStoreOutput.pte_data`, in dict order.
pub(crate) struct NamedEntry {
    pub key: String,
    pub buffer_index: usize,
    pub alignment: usize,
}

fn num_of(v: &Val) -> i128 {
    match v {
        Val::Num(Num::I(i)) => *i,
        Val::Num(Num::B(b)) => *b as i128,
        _ => 0,
    }
}

fn raw_of(v: &Val) -> Option<&Vec<u8>> {
    match v {
        Val::Raw { data, .. } => Some(data),
        _ => None,
    }
}

fn u64_list(xs: &[usize]) -> Val {
    let mut data = Vec::with_capacity(xs.len() * 8);
    for x in xs {
        data.extend((*x as u64).to_le_bytes());
    }
    Val::Raw { data, elem: 8, n: xs.len(), align: 8 }
}

fn mk(sch: &Schema, table: &str, vals: Vec<(&str, Val)>) -> Node {
    let t = sch.table(table);
    let mut v: Vec<Val> = sch.tables[t]
        .fields
        .iter()
        .map(|f| match f.ty {
            Ty::Scalar(_) | Ty::Enum(_) => Val::Num(f.default),
            _ => Val::Absent,
        })
        .collect();
    for (name, val) in vals {
        v[sch.field(t, name)] = val;
    }
    Node { t, v }
}

/// `serialize_pte_binary(pte_file, extract_delegate_segments=...,
/// segment_alignment=..., constant_tensor_alignment=...,
/// delegate_alignment=...)`, returning the bytes of its `Cord`.
pub(crate) fn serialize_pte(
    program: &Bound<'_, PyAny>,
    mutable_data: Option<Vec<Vec<u8>>>,
    named: Option<(Vec<Vec<u8>>, Vec<NamedEntry>)>,
    opt: &PteOptions,
) -> PyResult<Vec<u8>> {
    let sch = program_schema()?;
    let tensor_alignment = opt.constant_tensor_alignment.unwrap_or(16);
    if !valid_alignment(tensor_alignment) {
        return Err(err(format!("Bad constant_tensor_alignment {tensor_alignment}")));
    }
    let schema_delegate = sch
        .tables
        .iter()
        .flat_map(|t| t.fields.iter())
        .find(|f| f.role == Some(AlignRole::Delegate))
        .and_then(|f| f.force_align)
        .ok_or_else(|| err("program.fbs has no @executorch-delegate-alignment field"))?;
    let delegate_alignment = match opt.delegate_alignment {
        Some(d) if !valid_alignment(d) => return Err(err(format!("Bad delegate_alignment {d}"))),
        Some(d) => d,
        None => schema_delegate,
    };
    if opt.segment_alignment == 0 {
        return Err(err("segment_alignment 0"));
    }
    let aligns = Aligns { tensor: tensor_alignment, delegate: delegate_alignment };
    let max_alignment = sch
        .tables
        .iter()
        .flat_map(|t| t.fields.iter())
        .filter_map(|f| field_align(f, aligns).filter(|_| f.force_align.is_some()))
        .max()
        .unwrap_or(0);

    let pt = sch.root;
    let mut prog = convert(sch, pt, program, aligns, "Program")?;
    let fi = |name: &str| sch.field(pt, name);

    // Segments: (data, alignment), in order.
    let mut segments: Vec<(Vec<u8>, usize)> = Vec::new();

    // _extract_constant_segment
    let constant = match &prog.v[fi("constant_buffer")] {
        Val::Tables(v) => v.clone(),
        _ => Vec::new(),
    };
    if !constant.is_empty() {
        let bt = sch.table("Buffer");
        let sf = sch.field(bt, "storage");
        let mut data = Vec::new();
        let mut offsets = Vec::new();
        let mut current = 0usize;
        for (i, buf) in constant.iter().enumerate() {
            let storage = raw_of(&buf.v[sf])
                .ok_or_else(|| PyTypeError::new_err(format!("Program.constant_buffer[{i}].storage is None")))?;
            data.extend_from_slice(storage);
            let pad = padding_required(storage.len(), tensor_alignment);
            if i < constant.len() - 1 {
                data.extend(std::iter::repeat_n(0u8, pad));
            }
            offsets.push(current);
            current += storage.len() + pad;
        }
        prog.v[fi("constant_segment")] = Val::Table(Box::new(mk(
            sch,
            "SubsegmentOffsets",
            vec![("segment_index", Val::Num(Num::I(segments.len() as i128))), ("offsets", u64_list(&offsets))],
        )));
        prog.v[fi("constant_buffer")] = Val::Tables(Vec::new());
        segments.push((data, 1));
    }

    if let Some(mutable) = mutable_data {
        let mut data = Vec::new();
        let mut offsets = Vec::new();
        for m in &mutable {
            offsets.push(data.len());
            data.extend_from_slice(m);
        }
        if !data.is_empty() {
            prog.v[fi("mutable_data_segments")] = Val::Tables(vec![mk(
                sch,
                "SubsegmentOffsets",
                vec![("segment_index", Val::Num(Num::I(segments.len() as i128))), ("offsets", u64_list(&offsets))],
            )]);
            segments.push((data, 1));
        }
    }

    if opt.extract_delegate_segments {
        extract_delegate_segments(sch, &mut prog, &mut segments)?;
    }

    if let Some((buffers, entries)) = named {
        let nf = fi("named_data");
        if let Val::Tables(v) = &prog.v[nf] {
            if !v.is_empty() {
                return Err(err("Program already has named data."));
            }
        }
        let mut map: HashMap<usize, usize> = HashMap::new();
        let mut out = Vec::new();
        for e in entries {
            let idx = match map.get(&e.buffer_index) {
                Some(i) => *i,
                None => {
                    let buf = buffers
                        .get(e.buffer_index)
                        .ok_or_else(|| err(format!("named data {:?}: buffer index {} out of range", e.key, e.buffer_index)))?;
                    let i = segments.len();
                    map.insert(e.buffer_index, i);
                    segments.push((buf.clone(), e.alignment.max(1)));
                    i
                }
            };
            out.push(mk(
                sch,
                "NamedData",
                vec![("key", Val::Str(e.key.into_bytes())), ("segment_index", Val::Num(Num::I(idx as i128)))],
            ));
        }
        prog.v[nf] = Val::Tables(out);
    }

    // The segment table and the aggregate segment data.
    let sf = fi("segments");
    let dst = sch.table("DataSegment");
    let (off_f, size_f) = (sch.field(dst, "offset"), sch.field(dst, "size"));
    let mut table = match std::mem::replace(&mut prog.v[sf], Val::Absent) {
        Val::Tables(v) => v,
        _ => Vec::new(),
    };
    let mut segments_data: Vec<u8> = Vec::new();
    for (data, align) in &segments {
        let prev_end = table.last().map(|s| (num_of(&s.v[off_f]) + num_of(&s.v[size_f])) as usize).unwrap_or(0);
        let alignment = lcm(opt.segment_alignment, *align);
        table.push(mk(
            sch,
            "DataSegment",
            vec![
                ("offset", Val::Num(Num::I(aligned_size(prev_end, alignment) as i128))),
                ("size", Val::Num(Num::I(data.len() as i128))),
            ],
        ));
        let pad = padding_required(segments_data.len(), alignment);
        segments_data.extend(std::iter::repeat_n(0u8, pad));
        segments_data.extend_from_slice(data);
    }
    prog.v[sf] = Val::Tables(table);

    let mut b = Builder::new();
    let root = pack(&mut b, sch, &prog)?;
    let fb = b.finish(root, Some(sch.file_identifier))?;

    if segments_data.is_empty() {
        return Ok(fb);
    }
    const EH_LEN: usize = 32;
    let padded_header = aligned_size(EH_LEN, max_alignment.max(1));
    let program_size = padded_header + fb.len();
    let segment_base = aligned_size(program_size, opt.segment_alignment);
    let mut header = Vec::with_capacity(padded_header);
    header.extend_from_slice(b"eh00");
    header.extend((EH_LEN as u32).to_le_bytes());
    header.extend((program_size as u64).to_le_bytes());
    header.extend((segment_base as u64).to_le_bytes());
    header.extend((segments_data.len() as u64).to_le_bytes());
    header.resize(padded_header, 0);

    let magic = &fb[4..8];
    if !(magic[0] == b'E' && magic[1] == b'T' && magic[2].is_ascii_alphanumeric() && magic[3].is_ascii_alphanumeric()) {
        return Err(err(format!("Flatbuffer data magic bytes {magic:?} does not match pattern /ET[0-9a-zA-Z][0-9a-zA-Z]/")));
    }
    let root_offset = u32::from_le_bytes(fb[0..4].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(segment_base + segments_data.len());
    out.extend(((root_offset + header.len()) as u32).to_le_bytes());
    out.extend_from_slice(&fb[4..8]);
    out.extend_from_slice(&header);
    out.extend_from_slice(&fb[8..]);
    debug_assert_eq!(out.len(), program_size);
    out.resize(segment_base, 0);
    out.extend_from_slice(&segments_data);
    Ok(out)
}

/// `_extract_delegate_segments`.
fn extract_delegate_segments(sch: &Schema, prog: &mut Node, segments: &mut Vec<(Vec<u8>, usize)>) -> PyResult<()> {
    let pt = sch.root;
    let bdd_f = sch.field(pt, "backend_delegate_data");
    let inline_all = match &prog.v[bdd_f] {
        Val::Tables(v) => v.clone(),
        _ => Vec::new(),
    };
    let ept = sch.table("ExecutionPlan");
    let del_f = sch.field(ept, "delegates");
    let bdt = sch.table("BackendDelegate");
    let proc_f = sch.field(bdt, "processed");
    let reft = sch.table("BackendDelegateDataReference");
    let (loc_f, idx_f) = (sch.field(reft, "location"), sch.field(reft, "index"));
    let idt = sch.table("BackendDelegateInlineData");
    let data_f = sch.field(idt, "data");

    let mut remaining: Vec<Node> = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    let mut seg_map: HashMap<Vec<u8>, usize> = HashMap::new();
    if let Val::Tables(plans) = &mut prog.v[sch.field(pt, "execution_plan")] {
        for plan in plans.iter_mut() {
            let Val::Tables(delegates) = &mut plan.v[del_f] else { continue };
            for d in delegates.iter_mut() {
                let Val::Table(processed) = &mut d.v[proc_f] else {
                    return Err(err("BackendDelegate.processed is None"));
                };
                if num_of(&processed.v[loc_f]) != 0 {
                    return Err(err(format!(
                        "Program must only contain inline delegate data, saw a delegate with location {}",
                        num_of(&processed.v[loc_f])
                    )));
                }
                let index = num_of(&processed.v[idx_f]) as usize;
                let inline = inline_all.get(index).ok_or_else(|| {
                    err(format!(
                        "Delegate processed index {index} >= len(Program.backend_delegate_data) {}",
                        inline_all.len()
                    ))
                })?;
                seen.insert(index);
                let data = raw_of(&inline.v[data_f]).cloned().unwrap_or_default();
                if !data.is_empty() {
                    let seg = match seg_map.get(&data) {
                        Some(s) => *s,
                        None => {
                            let s = segments.len();
                            seg_map.insert(data.clone(), s);
                            segments.push((data, 1));
                            s
                        }
                    };
                    **processed = mk(
                        sch,
                        "BackendDelegateDataReference",
                        vec![("location", Val::Num(Num::I(1))), ("index", Val::Num(Num::I(seg as i128)))],
                    );
                } else {
                    let new_index = remaining.len();
                    remaining.push(inline.clone());
                    processed.v[idx_f] = Val::Num(Num::I(new_index as i128));
                }
            }
        }
    }
    let left: Vec<usize> = (0..inline_all.len()).filter(|i| !seen.contains(i)).collect();
    if !left.is_empty() {
        return Err(err(format!("Did not handle all elements of backend_delegate_data; remaining: {left:?}")));
    }
    prog.v[bdd_f] = Val::Tables(remaining);
    Ok(())
}

// ------------------------------------------------------------ XNNPACK

/// `backends/xnnpack/serialization/schema.fbs`, the schema exir's XNNPACK
/// serializer hands to flatc (file identifier XN01).
pub(crate) const XNN_SCHEMA_FBS: &str = include_str!("../../vendor/executorch/backends/xnnpack/serialization/schema.fbs");

/// `serialize_xnnpack_binary`'s framing around the flatbuffer: the `XH00`
/// header, the flatbuffer and the constant data, each padded to 16.
pub(crate) fn xnnpack_frame(flatbuffer: &[u8], constant_data: &[u8]) -> FbResult<Vec<u8>> {
    const ALIGN: usize = 16;
    const HEADER_LEN: usize = 4 + 4 + 2 + 4 + 4 + 4 + 8;
    let padded_fb = aligned_size(flatbuffer.len(), ALIGN);
    let padded_header = aligned_size(HEADER_LEN, ALIGN);
    let fb_offset = padded_header;
    let const_offset = padded_header + padded_fb;
    if flatbuffer.is_empty() {
        return Err(fb_err("Invalid XNNHeader: header failed is_valid() check"));
    }
    let u32_of = |x: usize, what: &str| u32::try_from(x).map_err(|_| fb_err(format!("XNNHeader: {what} {x} does not fit 4 bytes")));
    let mut out = Vec::with_capacity(const_offset + constant_data.len());
    out.extend_from_slice(&[0, 0, 0, 0]);
    out.extend_from_slice(b"XH00");
    out.extend((HEADER_LEN as u16).to_le_bytes());
    out.extend(u32_of(fb_offset, "flatbuffer offset")?.to_le_bytes());
    out.extend(u32_of(flatbuffer.len(), "flatbuffer size")?.to_le_bytes());
    out.extend(u32_of(const_offset, "constant data offset")?.to_le_bytes());
    out.extend((constant_data.len() as u64).to_le_bytes());
    out.resize(padded_header, 0);
    out.extend_from_slice(flatbuffer);
    out.resize(const_offset, 0);
    out.extend_from_slice(constant_data);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn program_schema_parses_with_upstream_facts() {
        let s = parse_schema(&[SCALAR_TYPE_FBS, PROGRAM_FBS]).unwrap();
        assert_eq!(&s.file_identifier, b"ET12");
        assert_eq!(s.tables[s.root].name, "Program");
        let ev = s.table("EValue");
        assert_eq!(s.tables[ev].slots, 2, "a union takes two slots");
        let t = s.table("Tensor");
        let names: Vec<&str> = s.tables[t].fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names[0], "scalar_type");
        assert_eq!(names.len(), 10);
        let buf = s.table("Buffer");
        assert_eq!(s.tables[buf].fields[0].role, Some(AlignRole::Tensor));
        assert_eq!(s.tables[buf].fields[0].force_align, Some(16));
        let inl = s.table("BackendDelegateInlineData");
        assert_eq!(s.tables[inl].fields[0].role, Some(AlignRole::Delegate));
        let kt = s.unions.iter().find(|u| u.name == "KernelTypes").unwrap();
        assert_eq!(kt.members[0], "Null");
        assert_eq!(kt.members.len(), 11);
        let nc = s.table("NonConstBufferDevice");
        assert!(matches!(s.tables[nc].fields[1].default, Num::I(0)));
    }

    #[test]
    fn the_xnnpack_schema_is_the_serializers() {
        assert!(XNN_SCHEMA_FBS.contains("file_identifier \"XN01\""));
    }

    #[test]
    fn an_empty_table_matches_the_python_builder() {
        // flatbuffers 25.12.19 (Python): b = Builder(); b.StartObject(0);
        // o = b.EndObject(); b.Finish(o, b"ET12"); bytes(b.Output())
        let mut b = Builder::new();
        b.start_object(0).unwrap();
        let o = b.end_object().unwrap();
        let out = b.finish(o, Some(*b"ET12")).unwrap();
        assert_eq!(out, vec![12, 0, 0, 0, b'E', b'T', b'1', b'2', 4, 0, 4, 0, 4, 0, 0, 0]);
    }

    #[test]
    fn the_xnnpack_frame_matches_upstreams_layout() {
        let out = xnnpack_frame(&[1, 2, 3], &[9, 9]).unwrap();
        assert_eq!(&out[4..8], b"XH00");
        assert_eq!(u16::from_le_bytes([out[8], out[9]]), 30);
        assert_eq!(u32::from_le_bytes(out[10..14].try_into().unwrap()), 32);
        assert_eq!(u32::from_le_bytes(out[14..18].try_into().unwrap()), 3);
        assert_eq!(u32::from_le_bytes(out[18..22].try_into().unwrap()), 48);
        assert_eq!(out.len(), 50);
        assert_eq!(&out[32..35], &[1, 2, 3]);
        assert_eq!(&out[48..], &[9, 9]);
    }
}
