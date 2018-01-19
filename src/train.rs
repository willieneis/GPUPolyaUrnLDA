use args::ARGS;
use buffer::Buffer;
use std::iter::repeat;
use ffi::{Args_FFI, initialize, sample_phi, sample_z_async, sync_buffer, cleanup};
use std::fs::{File, OpenOptions};
use std::io::prelude::BufRead;
use std::io::{Cursor, BufReader};
use libc::c_void;
use std::mem;
use memmap::{Mmap, MmapMut};
use byteorder::{LittleEndian, ReadBytesExt};


fn get_token_counts() -> Vec<u32> {
    let c = BufReader::new(File::open(&ARGS.c_temp_file).unwrap());
    c.lines().map(|l| l.unwrap().split("\t").nth(2).unwrap().parse().unwrap()).collect()
}

fn fill_buffer(buffer: &mut Buffer, z: &mut Cursor<MmapMut>, w: &mut Cursor<Mmap>, d: &mut Cursor<Mmap>, k_d: &mut Cursor<Mmap>) {
    let original_position = d.position();
    let mut n_docs = 0;
    let mut n_tokens = 0;
    loop {
        let d_size = d.read_u32::<LittleEndian>().unwrap_or(0);
        n_docs += 1;
        n_tokens += d_size;
        if n_tokens > ARGS.max_n_d {
            panic!("n_tokens exceeded max_N_d: {}", n_tokens)
        }
        if d_size == 0 || n_docs > ARGS.max_d || n_tokens > ARGS.buffer_size {
            n_docs -= 1;
            n_tokens -= d_size;
            break
        }
    }
    d.set_position(original_position);

    buffer.set_n_docs(n_docs);
    buffer.set_n_tokens(n_tokens);

    buffer.set_z(unsafe { z.get_mut().as_mut_ptr().offset(z.position() as isize) } as *mut u32);
    buffer.set_w(unsafe { w.get_ref().as_ptr().offset(w.position() as isize) } as *mut u32);
    buffer.set_d(unsafe { d.get_ref().as_ptr().offset(d.position() as isize) } as *mut u32);
    buffer.set_k_d(unsafe { k_d.get_ref().as_ptr().offset(k_d.position() as isize) } as *mut u32);
}

fn empty_buffer(buffer: &mut Buffer, z: &mut Cursor<MmapMut>, w: &mut Cursor<Mmap>, d: &mut Cursor<Mmap>, k_d: &mut Cursor<Mmap>) {
    let z_position = z.position();
    let w_position = w.position();
    let d_position = d.position();
    let k_d_position = k_d.position();
    z.set_position(z_position + ((buffer.n_tokens as usize * mem::size_of::<u32>() / mem::size_of::<u8>()) as u64));
    w.set_position(w_position + ((buffer.n_tokens as usize * mem::size_of::<u32>() / mem::size_of::<u8>()) as u64));
    d.set_position(d_position + ((buffer.n_docs as usize * mem::size_of::<u32>() / mem::size_of::<u8>()) as u64));
    k_d.set_position(k_d_position + ((buffer.n_docs as usize * mem::size_of::<u32>() / mem::size_of::<u8>()) as u64));
    buffer.set_n_docs(0);
    buffer.set_n_tokens(0);
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

    let z_file = OpenOptions::new().read(true).write(true).open(&ARGS.z_temp_file).unwrap();
    let w_file = OpenOptions::new().read(true).write(false).open(&ARGS.w_temp_file).unwrap();
    let d_file = OpenOptions::new().read(true).write(false).open(&ARGS.d_temp_file).unwrap();
    let k_d_file = OpenOptions::new().read(true).write(false).open(&ARGS.k_d_temp_file).unwrap();

    let mut z = Cursor::new(unsafe { MmapMut::map_mut(&z_file).unwrap() });
    let mut w = Cursor::new(unsafe { Mmap::map(&w_file).unwrap() });
    let mut d = Cursor::new(unsafe { Mmap::map(&d_file).unwrap() });
    let mut k_d = Cursor::new(unsafe { Mmap::map(&k_d_file).unwrap() });

    for _i in 0..ARGS.n_mc {
        sample_phi();
        'iteration: loop {
            for buffer in buffers.iter_mut() {
                fill_buffer(buffer, &mut z, &mut w, &mut d, &mut k_d);
                if buffer.n_docs == 0 { break 'iteration }

                sample_z_async(buffer);

                sync_buffer(buffer);

                empty_buffer(buffer, &mut z, &mut w, &mut d, &mut k_d);
            }
        }
        z.get_mut().flush().unwrap();
        println!("finished iteration: {}", _i);
    }

    cleanup(&buffers);
}
