use args::ARGS;
use buffer::Buffer;
use std::ptr;
use ffi::Args_FFI;

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
}

pub fn train() {
    init_cuda()
}
