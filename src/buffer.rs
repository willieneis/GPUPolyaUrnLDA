use args::ARGS;
use libc::{uint32_t, c_void};
use std::iter::repeat;
use std::mem;
use std::slice;
use std::ptr;

#[repr(C)]
pub struct Buffer {
    pub z: *const uint32_t,
    pub w: *const uint32_t,
    pub d: *const uint32_t,
    pub k_d: *const uint32_t,
    pub n_docs: uint32_t,
    pub n_tokens: uint32_t,
    gpu_z: *mut c_void,
    gpu_w: *mut c_void,
    gpu_d_len: *mut c_void,
    gpu_d_idx: *mut c_void,
    gpu_k_d: *mut c_void,
    gpu_hash: *mut c_void,
    gpu_temp: *mut c_void,
    gpu_rng: *mut c_void,
    stream: *mut c_void,
}

impl Buffer {
    pub fn new() -> Buffer {
        // allocate arrays
        let z = repeat(0).take(ARGS.buffer_size as usize).collect::<Vec<u32>>().into_boxed_slice();
        let w = repeat(0).take(ARGS.buffer_size as usize).collect::<Vec<u32>>().into_boxed_slice();
        let d = repeat(0).take(ARGS.max_d as usize).collect::<Vec<u32>>().into_boxed_slice();
        let k_d = repeat(0).take(ARGS.max_d as usize).collect::<Vec<u32>>().into_boxed_slice();
        // create buffer
        let b = Buffer {
            z: z.as_ptr(),
            w: w.as_ptr(),
            d: d.as_ptr(),
            k_d: k_d.as_ptr(),
            n_docs: 0,
            n_tokens: 0,
            gpu_z: ptr::null_mut(),
            gpu_w: ptr::null_mut(),
            gpu_d_len: ptr::null_mut(),
            gpu_d_idx: ptr::null_mut(),
            gpu_k_d: ptr::null_mut(),
            gpu_hash: ptr::null_mut(),
            gpu_temp: ptr::null_mut(),
            gpu_rng: ptr::null_mut(),
            stream: ptr::null_mut(),
        };
        // as_ptr doesn't take ownership, so we need to be sure not to deallocate any arrays
        mem::forget(z);
        mem::forget(w);
        mem::forget(d);
        mem::forget(k_d);
        // return the buffer
        b
    }

    pub fn set_n_docs(&mut self, new: u32) {
        self.n_docs = new
    }

    pub fn set_n_tokens(&mut self, new: u32) {
        self.n_tokens = new
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            let _z = Box::new(slice::from_raw_parts(self.z, ARGS.buffer_size as usize));
            let _w = Box::new(slice::from_raw_parts(self.w, ARGS.buffer_size as usize));
            let _d = Box::new(slice::from_raw_parts(self.d, ARGS.max_d as usize));
            let _k_d = Box::new(slice::from_raw_parts(self.k_d, ARGS.max_d as usize));
            // at this point, z,w,d,K_d are dropped and deallocated
        }
    }
}
