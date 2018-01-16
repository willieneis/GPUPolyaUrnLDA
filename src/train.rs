use args::ARGS;
use buffer::Buffer;
use std::iter::repeat;
use std::ptr;
use ffi::{Args_FFI, initialize, sample_phi, sample_z_async, sync_buffer, cleanup};
use std::fs::{File};
use std::io::prelude::{BufRead, Write};
use std::io::{Read, BufReader, BufWriter};
use libc::c_void;


fn get_token_counts() -> Vec<u32> {
    let c = BufReader::new(File::open(&ARGS.c_temp_file).unwrap());
    c.lines().map(|l| l.unwrap().split("\t").nth(2).unwrap().parse().unwrap()).collect()
}

pub fn train() {
    let c = get_token_counts();
    let args = Args_FFI {
        alpha: ARGS.alpha,
        beta: ARGS.beta,
        K: ARGS.K,
        V: c.len() as u32,
        C: c.as_ptr() as *const c_void,
        buffer_size: ARGS.buffer_size,
        max_D: ARGS.buffer_max_docs,
        max_N_d: ARGS.buffer_max_docs,
    };
    let buffers: Vec<Buffer> = repeat(1).map(|x| Buffer::new()).take(1).collect();

    initialize(&args, &buffers);

    // sample z

    // sample Phi

    cleanup(&buffers);
}
