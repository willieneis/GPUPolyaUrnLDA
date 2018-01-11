use libc::{uint32_t, c_float, c_void};

use args::ARGS;
use buffer::Buffer;
use std::ptr;

#[allow(non_snake_case)]
#[repr(C)]
struct Args_FFI {
    alpha: c_float,
    beta: c_float,
    K: uint32_t,
    V: uint32_t,
    C: *const c_void,
    buffer_size: uint32_t,
    max_D: uint32_t,
    max_N_d: uint32_t,
}

#[link(name = "GPUPolyaUrnLDA")]
extern {
    fn initialize(args: *mut Args_FFI, buffers: *mut Buffer, n_buffers: uint32_t) -> c_void;
    fn sample_phi() -> c_void;
    fn sample_z_async(buffer: *mut Buffer) -> c_void;
    fn cleanup(buffers: *mut Buffer, n_buffers: uint32_t) -> c_void;
    fn sync_buffer(buffer: *mut Buffer) -> c_void;
}

#[allow(unused_variables)] // remove later
fn init_cuda() {
    let args_ffi = Args_FFI {
        alpha: ARGS.alpha,
        beta: ARGS.beta,
        K: ARGS.K,
        V: 0, // not yet implemented
        C: ptr::null_mut(), // not yet implemented
        buffer_size: ARGS.buffer_size,
        max_D: ARGS.buffer_max_docs,
        max_N_d: ARGS.buffer_max_docs,
    };
    // unimplemented!();
}

pub fn train() {
    init_cuda()
}
