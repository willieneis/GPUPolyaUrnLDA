use libc::{uint32_t, c_float, c_void};
use buffer::Buffer;

#[repr(C)]
pub struct Args_FFI {
    pub alpha: c_float,
    pub beta: c_float,
    pub k: uint32_t,
    pub v: uint32_t,
    pub c: *const c_void,
    pub buffer_size: uint32_t,
    pub max_d: uint32_t,
}

#[link(name = "GPUPolyaUrnLDA"/*, kind = "static"*/)] // static fails to link CUDA runtime
extern {
    #[link_name="initialize"]
    fn unsafe_initialize(args: *const Args_FFI, buffers: *const Buffer, n_buffers: uint32_t);

    #[link_name="sample_phi"]
    fn unsafe_sample_phi();

    #[link_name="sample_z_async"]
    fn unsafe_sample_z_async(buffer: *const Buffer);

    #[link_name="cleanup"]
    fn unsafe_cleanup(buffers: *const Buffer, n_buffers: uint32_t);

    #[link_name="sync_buffer"]
    fn unsafe_sync_buffer(buffer: *const Buffer);
}

pub fn initialize(args: &Args_FFI, buffers: &Vec<Buffer>) {
    unsafe {
        unsafe_initialize(args as *const Args_FFI, buffers.as_ptr(), buffers.len() as uint32_t);
    }
}

pub fn sample_phi() {
    unsafe {
        unsafe_sample_phi();
    }
}

pub fn sample_z_async(buffer: &Buffer) {
    unsafe {
        unsafe_sample_z_async(buffer as *const Buffer);
    }
}

pub fn cleanup(buffers: &Vec<Buffer>) {
    unsafe {
        unsafe_cleanup(buffers.as_ptr(), buffers.len() as uint32_t);
    }
}

pub fn sync_buffer(buffer: &Buffer) {
    unsafe {
        unsafe_sync_buffer(buffer as *const Buffer);
    }
}
