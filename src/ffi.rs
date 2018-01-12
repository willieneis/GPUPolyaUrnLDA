use libc::{uint32_t, c_float, c_void};
use buffer::Buffer;

#[allow(non_snake_case)]
#[repr(C)]
pub struct Args_FFI {
    pub alpha: c_float,
    pub beta: c_float,
    pub K: uint32_t,
    pub V: uint32_t,
    pub C: *const c_void,
    pub buffer_size: uint32_t,
    pub max_D: uint32_t,
    pub max_N_d: uint32_t,
}

#[link(name = "GPUPolyaUrnLDA"/*, kind = "static"*/)] // static fails to link CUDA runtime
extern {
    #[link_name="initialize"]
    fn unsafe_initialize(args: *mut Args_FFI, buffers: *mut Buffer, n_buffers: uint32_t);

    #[link_name="sample_phi"]
    fn unsafe_sample_phi();

    #[link_name="sample_z_async"]
    fn unsafe_sample_z_async(buffer: *mut Buffer);

    #[link_name="cleanup"]
    fn unsafe_cleanup(buffers: *mut Buffer, n_buffers: uint32_t);

    #[link_name="sync_buffer"]
    fn unsafe_sync_buffer(buffer: *mut Buffer);
}

pub fn initialize(args: &mut Args_FFI, buffers: &mut [Buffer]) {
    unsafe {
        unsafe_initialize(args as *mut Args_FFI, buffers.as_mut_ptr(), buffers.len() as uint32_t);
    }
}

pub fn sample_phi() {
    unsafe {
        unsafe_sample_phi();
    }
}

pub fn sample_z_async(buffer: &mut Buffer) {
    unsafe {
        unsafe_sample_z_async(buffer as *mut Buffer);
    }
}

pub fn cleanup(buffers: &mut [Buffer]) {
    unsafe {
        unsafe_cleanup(buffers.as_mut_ptr(), buffers.len() as uint32_t);
    }
}

pub fn sync_buffer(buffer: &mut Buffer) {
    unsafe {
        unsafe_sync_buffer(buffer as *mut Buffer);
    }
}
