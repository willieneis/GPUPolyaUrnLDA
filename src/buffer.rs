use libc::{uint32_t, c_void};
use std::ptr;

#[repr(C)]
pub struct Buffer {
    z: *mut uint32_t,
    w: *mut uint32_t,
    d: *mut uint32_t,
    k_d: *mut uint32_t,
    pub n_docs: uint32_t,
    pub n_tokens: uint32_t,
    gpu_z: *mut c_void,
    gpu_w: *mut c_void,
    gpu_d_len: *mut c_void,
    gpu_d_idx: *mut c_void,
    gpu_k_d: *mut c_void,
    gpu_rng: *mut c_void,
    stream: *mut c_void,
}

impl Buffer {
    pub fn new() -> Buffer {
        Buffer {
            z: ptr::null_mut(),
            w: ptr::null_mut(),
            d: ptr::null_mut(),
            k_d: ptr::null_mut(),
            n_docs: 0,
            n_tokens: 0,
            gpu_z: ptr::null_mut(),
            gpu_w: ptr::null_mut(),
            gpu_d_len: ptr::null_mut(),
            gpu_d_idx: ptr::null_mut(),
            gpu_k_d: ptr::null_mut(),
            gpu_rng: ptr::null_mut(),
            stream: ptr::null_mut(),
        }
    }

    pub fn set_n_docs(&mut self, new: u32) {
        self.n_docs = new
    }

    pub fn set_n_tokens(&mut self, new: u32) {
        self.n_tokens = new
    }

    pub fn set_z(&mut self, new: *mut u32) {
        self.z = new
    }

    pub fn set_w(&mut self, new: *mut u32) {
        self.w = new
    }

    pub fn set_d(&mut self, new: *mut u32) {
        self.d = new
    }

    pub fn set_k_d(&mut self, new: *mut u32) {
        self.k_d = new
    }
}
