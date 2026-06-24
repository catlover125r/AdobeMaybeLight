//! Minimal, safe RAW decode via LibRaw (through a C shim).
//!
//! Output is **linear, sRGB-primary, 16-bit interleaved RGB** — scene-referred
//! enough for the GPU develop pipeline to own all tone/exposure decisions.

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_uint, c_ushort};
use std::path::Path;

#[repr(C)]
struct AmlImage {
    data: *mut c_ushort,
    width: c_uint,
    height: c_uint,
    channels: c_uint,
    error: c_int,
}

#[repr(C)]
struct AmlMeta {
    error: c_int,
    width: c_uint,
    height: c_uint,
    flip: c_int,
    timestamp: i64,
    iso: f32,
    shutter: f32,
    aperture: f32,
    focal: f32,
    make: [c_char; 64],
    model: [c_char; 64],
    lens: [c_char; 64],
}

extern "C" {
    fn aml_decode_linear(path: *const c_char) -> AmlImage;
    fn aml_free(data: *mut c_ushort);
    fn aml_probe(path: *const c_char) -> AmlMeta;
}

/// Header-only metadata for catalog import (no demosaic — fast).
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    pub width: u32,
    pub height: u32,
    pub flip: i32,
    pub timestamp: i64,
    pub iso: f32,
    pub shutter: f32,
    pub aperture: f32,
    pub focal: f32,
    pub make: String,
    pub model: String,
    pub lens: String,
}

fn cstr_to_string(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).trim().to_string()
}

/// Probe a RAW file's metadata without decoding pixels.
pub fn probe<P: AsRef<Path>>(path: P) -> Result<Metadata, RawError> {
    let c = CString::new(path.as_ref().to_string_lossy().as_bytes())
        .map_err(|_| RawError::BadPath)?;
    let m = unsafe { aml_probe(c.as_ptr()) };
    if m.error != 0 {
        return Err(RawError::Libraw(m.error));
    }
    Ok(Metadata {
        width: m.width,
        height: m.height,
        flip: m.flip,
        timestamp: m.timestamp,
        iso: m.iso,
        shutter: m.shutter,
        aperture: m.aperture,
        focal: m.focal,
        make: cstr_to_string(&m.make),
        model: cstr_to_string(&m.model),
        lens: cstr_to_string(&m.lens),
    })
}

#[derive(Debug, thiserror::Error)]
pub enum RawError {
    #[error("path contains interior NUL byte")]
    BadPath,
    #[error("libraw failed with code {0}")]
    Libraw(i32),
    #[error("libraw returned a null buffer")]
    NullBuffer,
}

/// A decoded image. Owns its pixel buffer; frees it (via libc free) on drop.
pub struct RawImage {
    ptr: *mut c_ushort,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
}

// The buffer is a plain malloc'd block we exclusively own.
unsafe impl Send for RawImage {}
unsafe impl Sync for RawImage {}

impl RawImage {
    /// 16-bit linear RGB samples, length = width * height * channels.
    pub fn samples(&self) -> &[u16] {
        unsafe {
            std::slice::from_raw_parts(
                self.ptr,
                (self.width * self.height * self.channels) as usize,
            )
        }
    }

    /// Bytes view, handy for direct GPU upload (RGB16 -> expand to RGBA on GPU).
    pub fn as_bytes(&self) -> &[u8] {
        let s = self.samples();
        unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
    }
}

impl Drop for RawImage {
    fn drop(&mut self) {
        unsafe { aml_free(self.ptr) }
    }
}

/// Decode a RAW file (CR2/CR3/NEF/ARW/RAF/DNG/…) to linear 16-bit RGB.
pub fn decode<P: AsRef<Path>>(path: P) -> Result<RawImage, RawError> {
    let c = CString::new(path.as_ref().to_string_lossy().as_bytes())
        .map_err(|_| RawError::BadPath)?;
    let img = unsafe { aml_decode_linear(c.as_ptr()) };
    if img.error != 0 {
        return Err(RawError::Libraw(img.error));
    }
    if img.data.is_null() {
        return Err(RawError::NullBuffer);
    }
    Ok(RawImage {
        ptr: img.data,
        width: img.width,
        height: img.height,
        channels: img.channels,
    })
}
