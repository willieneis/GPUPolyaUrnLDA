use args::ARGS;
use buffer::Buffer;
use std::iter::repeat;
use ffi::{Args_FFI, initialize, cleanup};
use std::fs::{File};
use std::io::prelude::{BufRead};
use std::io::{Read, BufReader};
use libc::c_void;
use std::slice;
use std::mem;


fn get_token_counts() -> Vec<u32> {
    let c = BufReader::new(File::open(&ARGS.c_temp_file).unwrap());
    c.lines().map(|l| l.unwrap().split("\t").nth(2).unwrap().parse().unwrap()).collect()
}

fn as_u32<'a>(s: &'a [u8]) -> &'a [u32] {
    unsafe { // integer division rounds down
        slice::from_raw_parts(s.as_ptr() as *const u32, (s.len() * mem::size_of::<u8>()) / mem::size_of::<u32>())
    }
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
    let mut buffers: Vec<Buffer> = repeat(0).map(|_| Buffer::new()).take(1).collect();
    initialize(&args, &buffers);

    let mut d_reader = BufReader::new(File::open(&ARGS.d_temp_file).unwrap());
    let mut z_reader = BufReader::new(File::open(&ARGS.z_temp_file).unwrap());
    let mut w_reader = BufReader::new(File::open(&ARGS.w_temp_file).unwrap());
    let mut k_d_reader = BufReader::new(File::open(&ARGS.d_temp_file).unwrap());

    // sample z
    loop {
        println!("starting read");

        let buffer_idx = 0;
        let mut d_num_read = 0;
        let mut d_count = 0;
        {
            let d_buffer = as_u32(d_reader.fill_buf().unwrap());
            for d_size in d_buffer {
                d_num_read += 1;
                d_count += d_size;
                if d_num_read > ARGS.max_d || d_count > ARGS.buffer_size {
                    d_count -= d_size;
                    d_num_read -= 1;
                    break;
                };
            }
        };

        if d_num_read == 0 { break };

        { d_reader.consume(0) }; // ensure buffer can be used again

        { buffers[buffer_idx].set_n_docs(d_num_read); } // ensure scope of borrow ends here

        let read_buffer = &buffers[buffer_idx];
        // let write_buffer = &buffers[(buffer_idx + 1) % buffers.len()];

        let mut z_buffer_bytes = unsafe { slice::from_raw_parts_mut(read_buffer.z as *mut u8, (d_count as usize * mem::size_of::<u32>()) / mem::size_of::<u8>()) };
        let mut w_buffer_bytes = unsafe { slice::from_raw_parts_mut(read_buffer.w as *mut u8, (d_count as usize * mem::size_of::<u32>()) / mem::size_of::<u8>()) };
        let mut d_buffer_bytes = unsafe { slice::from_raw_parts_mut(read_buffer.d as *mut u8, (d_num_read as usize * mem::size_of::<u32>()) / mem::size_of::<u8>()) };
        let mut k_d_buffer_bytes = unsafe { slice::from_raw_parts_mut(read_buffer.k_d as *mut u8, (d_num_read as usize * mem::size_of::<u32>()) / mem::size_of::<u8>()) };

        z_reader.read_exact(&mut z_buffer_bytes).unwrap();
        w_reader.read_exact(&mut w_buffer_bytes).unwrap();
        d_reader.read_exact(&mut d_buffer_bytes).unwrap();
        k_d_reader.read_exact(&mut k_d_buffer_bytes).unwrap();
    }

    // sample Phi

    cleanup(&buffers);
}
