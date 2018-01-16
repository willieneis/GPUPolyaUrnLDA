use args::ARGS;
use buffer::Buffer;
use std::iter::repeat;
use ffi::{Args_FFI, initialize, cleanup};
use std::fs::{File};
use std::io::prelude::{BufRead};
use std::io::{BufReader};
use libc::c_void;


fn get_token_counts() -> Vec<u32> {
    let c = BufReader::new(File::open(&ARGS.c_temp_file).unwrap());
    c.lines().map(|l| l.unwrap().split("\t").nth(2).unwrap().parse().unwrap()).collect()
}

pub fn train() {
    println!("training");
    let c = get_token_counts();
    let args = Args_FFI {
        alpha: ARGS.alpha,
        beta: ARGS.beta,
        k: ARGS.k,
        v: c.len() as u32,
        c: c.as_ptr() as *const c_void,
        buffer_size: ARGS.buffer_size,
        max_d: ARGS.max_d,
        max_n_d: ARGS.max_n_d,
    };
    let buffers: Vec<Buffer> = repeat(1).map(|_x| Buffer::new()).take(1).collect();

    initialize(&args, &buffers);

    // sample z

    // sample Phi

    cleanup(&buffers);
}
