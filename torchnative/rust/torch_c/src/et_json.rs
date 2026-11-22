//! `json.dumps(obj, cls=_DataclassEncoder)` of exir, re-implemented in Rust
//! (issue #423, step R5-S1; owner 2026-11-21, "Rust로 재구현").
//!
//! exir's XNNPACK serializer (`xnnpack_graph_serialize.convert_to_flatbuffer`,
//! executorch 1.4.1) turns the `XNNGraph` dataclasses into JSON text with
//! `exir/_serialize/_dataclass.py::_DataclassEncoder` and hands that text to
//! flatc. The flatbuffers parser builds nested objects in the order it meets
//! them in the text, so the bytes depend on the text, not only on the values.
//! This module therefore reproduces the text exactly: CPython's C encoder with
//! the default options (`", "` and `": "` separators, `ensure_ascii=True`,
//! `allow_nan=True`), and the encoder's `default`:
//!
//! * a dataclass becomes an object of its `dataclasses.fields`, in order; a
//!   field whose annotation is a string and resolves (`typing.get_type_hints`)
//!   to a `Union` is followed by `"<name>_type": "<class name of the value>"`;
//! * `bytes` becomes the list of its byte values;
//! * anything else is a `TypeError` naming the type, as in Python.
//!
//! The check order is CPython's: `None`, `True`, `False`, `str`, `int`
//! (including `IntEnum`, written as its integer), `float` (its `repr`, with
//! `NaN`, `Infinity`, `-Infinity`), `list` or `tuple`, `dict`, then `default`.

use pyo3::exceptions::{PyAssertionError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use std::collections::HashMap;

struct FieldInfo {
    name: String,
    is_union: bool,
}

#[derive(Default)]
struct Cache {
    fields: HashMap<usize, (Py<PyAny>, std::rc::Rc<Vec<FieldInfo>>)>,
}

fn dataclass_fields(py: Python<'_>, o: &Bound<'_, PyAny>, cache: &mut Cache) -> PyResult<Option<std::rc::Rc<Vec<FieldInfo>>>> {
    let t = o.get_type();
    let key = t.as_ptr() as usize;
    if let Some((_, f)) = cache.fields.get(&key) {
        return Ok(Some(f.clone()));
    }
    if !t.hasattr("__dataclass_fields__")? {
        return Ok(None);
    }
    let dataclasses = PyModule::import(py, "dataclasses")?;
    let typing = PyModule::import(py, "typing")?;
    let hints = typing.getattr("get_type_hints")?.call1((&t,))?;
    let get_origin = typing.getattr("get_origin")?;
    let union = typing.getattr("Union")?;
    let mut out = Vec::new();
    for f in dataclasses.getattr("fields")?.call1((o,))?.try_iter()? {
        let f = f?;
        let name: String = f.getattr("name")?.extract()?;
        let ty_is_str = f.getattr("type")?.is_instance_of::<PyString>();
        let hint = hints.get_item(&name)?;
        let is_union = ty_is_str && get_origin.call1((hint,))?.is(&union);
        out.push(FieldInfo { name, is_union });
    }
    let rc = std::rc::Rc::new(out);
    cache.fields.insert(key, (t.into_any().unbind(), rc.clone()));
    Ok(Some(rc))
}

/// CPython's `float.__repr__` (shortest round-trip digits; exponent form when
/// the decimal exponent is below -4 or at least 16), as `json` writes it.
pub(crate) fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    let s = format!("{x:e}");
    let (mant, exp) = s.split_once('e').expect("LowerExp has an exponent");
    let exp: i32 = exp.parse().expect("LowerExp exponent");
    let (neg, mant) = match mant.strip_prefix('-') {
        Some(m) => (true, m),
        None => (false, mant),
    };
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let decpt = exp + 1;
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if decpt > -4 && decpt <= 16 {
        let n = digits.len() as i32;
        if decpt <= 0 {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', (-decpt) as usize));
            out.push_str(&digits);
        } else if decpt >= n {
            out.push_str(&digits);
            out.extend(std::iter::repeat_n('0', (decpt - n) as usize));
            out.push_str(".0");
        } else {
            out.push_str(&digits[..decpt as usize]);
            out.push('.');
            out.push_str(&digits[decpt as usize..]);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let e = decpt - 1;
        out.push('e');
        out.push(if e < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", e.abs()));
    }
    out
}

/// `json`'s `ensure_ascii` string encoding.
pub(crate) fn json_string(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(ch),
            _ => {
                let mut buf = [0u16; 2];
                for u in ch.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", u));
                }
            }
        }
    }
    out.push('"');
}

fn encode(py: Python<'_>, o: &Bound<'_, PyAny>, out: &mut String, cache: &mut Cache, depth: usize) -> PyResult<()> {
    if depth > 1000 {
        return Err(pyo3::exceptions::PyRecursionError::new_err("maximum recursion depth exceeded while encoding a JSON object"));
    }
    if o.is_none() {
        out.push_str("null");
        return Ok(());
    }
    if let Ok(b) = o.cast::<PyBool>() {
        out.push_str(if b.is_true() { "true" } else { "false" });
        return Ok(());
    }
    if let Ok(s) = o.cast::<PyString>() {
        json_string(&s.to_cow()?, out);
        return Ok(());
    }
    if o.is_instance_of::<PyInt>() {
        match o.extract::<i128>() {
            Ok(i) => out.push_str(&i.to_string()),
            Err(_) => {
                let r = py.get_type::<PyInt>().getattr("__repr__")?.call1((o,))?;
                out.push_str(&r.extract::<String>()?);
            }
        }
        return Ok(());
    }
    if o.is_instance_of::<PyFloat>() {
        out.push_str(&float_repr(o.extract::<f64>()?));
        return Ok(());
    }
    let seq: Option<Vec<Bound<'_, PyAny>>> = if let Ok(l) = o.cast::<PyList>() {
        Some(l.iter().collect())
    } else if let Ok(t) = o.cast::<PyTuple>() {
        Some(t.iter().collect())
    } else {
        None
    };
    if let Some(items) = seq {
        out.push('[');
        for (i, e) in items.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            encode(py, e, out, cache, depth + 1)?;
        }
        out.push(']');
        return Ok(());
    }
    if let Ok(d) = o.cast::<PyDict>() {
        out.push('{');
        for (i, (k, v)) in d.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            let ks = k.cast::<PyString>().map_err(|_| {
                PyTypeError::new_err("keys must be str in torchnative's JSON encoder (issue #423)")
            })?;
            json_string(&ks.to_cow()?, out);
            out.push_str(": ");
            encode(py, &v, out, cache, depth + 1)?;
        }
        out.push('}');
        return Ok(());
    }
    // _DataclassEncoder.default
    if let Some(fields) = dataclass_fields(py, o, cache)? {
        out.push('{');
        for (i, f) in fields.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            let v = o.getattr(f.name.as_str())?;
            json_string(&f.name, out);
            out.push_str(": ");
            encode(py, &v, out, cache, depth + 1)?;
            if f.is_union {
                out.push_str(", ");
                json_string(&format!("{}_type", f.name), out);
                out.push_str(": ");
                json_string(&v.get_type().name()?.to_cow()?, out);
            }
        }
        out.push('}');
        return Ok(());
    }
    if let Ok(b) = o.cast::<PyBytes>() {
        out.push('[');
        for (i, x) in b.as_bytes().iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&x.to_string());
        }
        out.push(']');
        return Ok(());
    }
    Err(PyTypeError::new_err(format!(
        "Object of type {} is not JSON serializable",
        o.get_type().name()?.to_cow()?
    )))
}

/// The JSON text `json.dumps(obj, cls=_DataclassEncoder)` produces.
pub(crate) fn dataclass_json(py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<String> {
    let mut out = String::new();
    let mut cache = Cache::default();
    encode(py, o, &mut out, &mut cache, 0)?;
    Ok(out)
}

/// `xnnpack_graph_serialize.sanity_check_xnngraph_dataclass`: no `Sym*` value
/// (a `SymInt` that leaked out of tracing) anywhere in the graph.
pub(crate) fn sanity_check(py: Python<'_>, o: &Bound<'_, PyAny>, name: &str) -> PyResult<()> {
    let mut cache = Cache::default();
    check(py, o, name, &mut cache)
}

fn cls(o: &Bound<'_, PyAny>) -> String {
    o.get_type().name().map(|n| n.to_string()).unwrap_or_default()
}

fn check_sym(o: &Bound<'_, PyAny>, name: &str) -> PyResult<()> {
    let c = cls(o);
    if c.contains("Sym") {
        return Err(PyAssertionError::new_err(format!("Non serializable type {c} found at type {name}")));
    }
    Ok(())
}

fn check(py: Python<'_>, table: &Bound<'_, PyAny>, name: &str, cache: &mut Cache) -> PyResult<()> {
    let Some(fields) = dataclass_fields(py, table, cache)? else {
        return Err(PyAssertionError::new_err(format!("Expecting a dataclass but got {}", cls(table))));
    };
    let base = if name.is_empty() { cls(table) } else { name.to_string() };
    for f in fields.iter() {
        let o = table.getattr(f.name.as_str())?;
        if o.is_instance_of::<PyString>() || o.is_instance_of::<PyBytes>() {
            continue;
        }
        let field_name = format!("{base}.<{}>{}", cls(&o), f.name);
        if dataclass_fields(py, &o, cache)?.is_some() {
            check(py, &o, &field_name, cache)?;
        } else if let Ok(l) = o.cast::<PyList>() {
            for (i, v) in l.iter().enumerate() {
                let n = format!("{field_name}[{i}]");
                if dataclass_fields(py, &v, cache)?.is_some() {
                    check(py, &v, &n, cache)?;
                } else {
                    check_sym(&v, &n)?;
                }
            }
        } else {
            check_sym(&o, &field_name)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_are_written_as_cpython_repr_writes_them() {
        // repr(x) in CPython 3.13
        for (x, want) in [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (123456789.0, "123456789.0"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.2345e16, "1.2345e+16"),
            (3.4028234663852886e38, "3.4028234663852886e+38"),
            (-2.5, "-2.5"),
            (f64::NAN, "NaN"),
            (f64::INFINITY, "Infinity"),
            (f64::NEG_INFINITY, "-Infinity"),
            (5e-324, "5e-324"),
        ] {
            assert_eq!(float_repr(x), want, "{x:e}");
        }
    }

    #[test]
    fn strings_are_escaped_as_ensure_ascii_does() {
        let mut s = String::new();
        json_string("a\"b\\c\nd\u{7f}é😀\u{1}", &mut s);
        assert_eq!(s, "\"a\\\"b\\\\c\\nd\\u007f\\u00e9\\ud83d\\ude00\\u0001\"");
    }
}
