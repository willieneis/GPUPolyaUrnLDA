#pragma once

#include "types.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h> // need to add -lcurand to nvcc flags

#include "poisson.cuh"
#include "spalias.cuh"

namespace gpulda {

struct Args {
  f32 alpha;
  f32 beta;
  u32 K;
  u32 V;
  u32* C;
  u32 buffer_size;
  u32 max_D;
};

struct Buffer {
  u32* z;
  u32* w;
  u32* d;
  u32* K_d;
  u32 n_docs;
  u32 n_tokens;
  u32* gpu_z;
  u32* gpu_w;
  u32* gpu_d_len;
  u32* gpu_d_idx;
  u32* gpu_K_d;
  curandStatePhilox4_32_10_t* gpu_rng;
  cudaStream_t* stream;
};

extern Args* args;

extern "C" void initialize(Args* args, Buffer* buffers, u32 n_buffers);
extern "C" void sample_phi();
extern "C" void sample_z_async(Buffer* buffer);
extern "C" void cleanup(Buffer* buffers, u32 n_buffers);
extern "C" void sync_buffer(Buffer* buffer);

}
