use args::ARGS;
use buffer::Buffer;
use std::iter::repeat;
use ffi::{Args_FFI, initialize, sample_phi, sample_z_async, sync_buffer, cleanup};
use std::fs::{File};
use std::io::prelude::{BufRead, Write};
use std::io::{Read, BufReader, BufWriter};
use libc::c_void;
use std::slice;
use std::mem;


fn get_token_counts() -> Vec<u32> {
    let c = BufReader::new(File::open(&ARGS.c_temp_file).unwrap());
    c.lines().map(|l| l.unwrap().split("\t").nth(2).unwrap().parse().unwrap()).collect()
}

fn fill_buffer(buffer: &mut Buffer, z_reader: &mut BufReader<File>, w_reader: &mut BufReader<File>, d_reader: &mut BufReader<File>, k_d_reader: &mut BufReader<File>) {
    let mut d_num_read = 0;
    let mut d_count = 0;
    {
        let d_buffer_bytes = d_reader.fill_buf().unwrap();
        let d_buffer = unsafe { slice::from_raw_parts(d_buffer_bytes.as_ptr() as *const u32, (d_buffer_bytes.len() * mem::size_of::<u8>()) / mem::size_of::<u32>()) };
        for d_size in d_buffer {
            d_num_read += 1;
            d_count += d_size;
            if d_num_read > ARGS.max_d || d_count > ARGS.buffer_size {
                d_count -= d_size;
                d_num_read -= 1;
                break
            };
        }
    };

    { d_reader.consume(0) }; // ensure buffer can be used again

    { buffer.set_n_docs(d_num_read); }

    { buffer.set_n_tokens(d_count); }

    let mut z_buffer_bytes = unsafe { slice::from_raw_parts_mut(buffer.z as *mut u8, (d_count as usize * mem::size_of::<u32>()) / mem::size_of::<u8>()) };
    let mut w_buffer_bytes = unsafe { slice::from_raw_parts_mut(buffer.w as *mut u8, (d_count as usize * mem::size_of::<u32>()) / mem::size_of::<u8>()) };
    let mut d_buffer_bytes = unsafe { slice::from_raw_parts_mut(buffer.d as *mut u8, (d_num_read as usize * mem::size_of::<u32>()) / mem::size_of::<u8>()) };
    let mut k_d_buffer_bytes = unsafe { slice::from_raw_parts_mut(buffer.k_d as *mut u8, (d_num_read as usize * mem::size_of::<u32>()) / mem::size_of::<u8>()) };

    z_reader.read_exact(&mut z_buffer_bytes).unwrap();
    w_reader.read_exact(&mut w_buffer_bytes).unwrap();
    d_reader.read_exact(&mut d_buffer_bytes).unwrap();
    k_d_reader.read_exact(&mut k_d_buffer_bytes).unwrap();
}

fn empty_buffer(buffer: &mut Buffer, z_writer: &mut BufWriter<File>, d_writer: &mut BufWriter<File>, k_d_writer: &mut BufWriter<File>) {
    let z_buffer_bytes = unsafe { slice::from_raw_parts(buffer.z as *const u8, (buffer.n_tokens as usize * mem::size_of::<u32>()) / mem::size_of::<u8>()) };
    let d_buffer_bytes = unsafe { slice::from_raw_parts(buffer.d as *const u8, (buffer.n_docs as usize * mem::size_of::<u32>()) / mem::size_of::<u8>()) };
    let k_d_buffer_bytes = unsafe { slice::from_raw_parts(buffer.k_d as *const u8, (buffer.n_docs as usize * mem::size_of::<u32>()) / mem::size_of::<u8>()) };

    z_writer.write_all(z_buffer_bytes).unwrap();
    d_writer.write_all(d_buffer_bytes).unwrap();
    k_d_writer.write_all(k_d_buffer_bytes).unwrap();

    { buffer.set_n_docs(0); }
    { buffer.set_n_tokens(0); }
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

    let mut z_writer = BufWriter::new(File::open(&ARGS.z_temp_file).unwrap());
    let mut d_writer = BufWriter::new(File::open(&ARGS.d_temp_file).unwrap());
    let mut k_d_writer = BufWriter::new(File::open(&ARGS.k_d_temp_file).unwrap());

    for _i in 0..ARGS.n_mc {
        'iteration: loop {
            for buffer in buffers.iter_mut() {
                fill_buffer(buffer, &mut z_reader, &mut w_reader, &mut d_reader, &mut k_d_reader);
                if buffer.n_docs == 0 { break 'iteration }

                sample_z_async(buffer);

                sync_buffer(buffer);

                empty_buffer(buffer, &mut z_writer, &mut d_writer, &mut k_d_writer);
            }
        }
    }

    sample_phi();

    cleanup(&buffers);
}
