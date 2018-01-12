use args::ARGS;
use buffer::Buffer;
use std::iter::repeat;
use std::ptr;
use ffi::{Args_FFI, initialize, sample_phi, sample_z_async, sync_buffer, cleanup};

#[allow(unused_variables)] // remove later
fn init_cuda() {
    let buffers: Vec<Buffer> = repeat(1).map(|x| Buffer::new()).take(5).collect();
    let args = Args_FFI {
        alpha: ARGS.alpha,
        beta: ARGS.beta,
        K: ARGS.K,
        V: 0, // not yet implemented
        C: ptr::null_mut(), // not yet implemented
        buffer_size: ARGS.buffer_size,
        max_D: ARGS.buffer_max_docs,
        max_N_d: ARGS.buffer_max_docs,
    };
    initialize(&args, &buffers);
}

pub fn train() {
    init_cuda();
}
