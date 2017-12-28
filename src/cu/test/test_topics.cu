#include "test_topics.cuh"
#include "../topics.cuh"
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

__global__ void test_draw_wary_search(u32* error) {
  i32 lane_idx = threadIdx.x % warpSize;
  f32 u = 0.1;
  constexpr i32 size = 10;
  __shared__ gpulda::HashMap m[1];
  __shared__ u64 data[size];
  m->size_1 = size;
  m->data_1 = data;
  m->state = 2;
  __shared__ f32 mPhi[size];
  f32 sigma_b = 75.0f;

  u64 empty = m->entry(0, 0, m->null_pointer(), 0, 0);
  for(i32 offset = 0; offset < size / warpSize + 1; ++offset) {
    i32 i = offset * warpSize + lane_idx;
    if(i<size) {
      data[i] = m->with_key(i, empty);
      mPhi[i] = ((float)i) * sigma_b / ((float)size);
    }
  }

  u32 topic = gpulda::draw_wary_search(u, m, mPhi, sigma_b, lane_idx);

  if(lane_idx==0 && topic!=9){
    error[0] = 1;
  }
}

void test_sample_topics() {
  constexpr u32 warpSize = 32;

  u32* out;
  cudaMalloc(&out, sizeof(u32)) >> GPULDA_CHECK;
  u32 out_host = 0;

  // test draw_wary_search
  test_draw_wary_search<<<1,warpSize>>>(out);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  cudaMemcpy(&out_host, out, sizeof(u32), cudaMemcpyDeviceToHost) >> GPULDA_CHECK;
  assert(out_host == 0);

}

}
