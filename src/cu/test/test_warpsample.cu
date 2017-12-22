#include "test_warpsample.cuh"
#include "../warpsample.cuh"
#include "../error.cuh"
#include "assert.h"

using gpulda::FileLine;
using gpulda::f32;
using gpulda::i32;
using gpulda::u32;
using gpulda::u64;

namespace gpulda_test {

void test_compute_d_idx() {
  u32 size = 4*GPULDA_COMPUTE_D_IDX_BLOCKDIM;
  u32 d_len[4*GPULDA_COMPUTE_D_IDX_BLOCKDIM];
  u32 d_idx[4*GPULDA_COMPUTE_D_IDX_BLOCKDIM];
  u32 n_docs = 2*GPULDA_COMPUTE_D_IDX_BLOCKDIM + 15;

  for(i32 i = 0; i < size; ++i) {
    d_len[i] = i+1;
  }

  u32* gpu_d_len;
  u32* gpu_d_idx;
  cudaMalloc(&gpu_d_len, size*sizeof(u32)) >> GPULDA_CHECK;
  cudaMalloc(&gpu_d_idx, size*sizeof(u32)) >> GPULDA_CHECK;

  cudaMemcpy(gpu_d_len, d_len, size*sizeof(u32), cudaMemcpyHostToDevice) >> GPULDA_CHECK;

  gpulda::compute_d_idx<<<1,GPULDA_COMPUTE_D_IDX_BLOCKDIM>>>(gpu_d_len, gpu_d_idx, n_docs);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  cudaMemcpy(d_idx, gpu_d_idx, size*sizeof(u32), cudaMemcpyDeviceToHost) >> GPULDA_CHECK;

  assert(d_idx[0] == 0);
  u32 j = d_len[0];
  for(i32 i = 1; i < n_docs; ++i) {
     assert(d_idx[i] == j);
     j = j + d_len[i];
   }

  cudaFree(gpu_d_len);
  cudaFree(gpu_d_idx);
}

void test_warp_sample_topics() {

}

}
