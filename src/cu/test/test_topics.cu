#include "test_topics.cuh"
#include "../topics.cuh"
#include "../random.cuh"
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
  f32 u = 0.2f;
  constexpr i32 size = 96; // Need: 16 * something?
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

__global__ void test_count_topics(u32* error, curandStatePhilox4_32_10_t* rng) {
  // compute constants
  i32 lane_idx = threadIdx.x % warpSize;
  i32 half_lane_idx = lane_idx % (warpSize/2);
  curandStatePhilox4_32_10_t warp_rng = rng[0];
  constexpr u32 cutoff = 25;
  __shared__ u32 count[cutoff];

  // declare arguments
  constexpr u32 size = 100;
  __shared__ u32 z[size];
  __shared__ gpulda::HashMap m[1];
  __shared__ u64 data[2*size];
  constexpr u32 ring_buffer_size = 4*3; // number of concurrent elements * 96 bits per concurrent element
  __shared__ u32 ring_buffer[ring_buffer_size];
  m->init(data, 2*size, size, ring_buffer, ring_buffer_size, &warp_rng, warpSize);
  __syncthreads();

  // prepare state
  for(i32 offset = 0; offset < size / warpSize + 1; ++offset) {
    i32 i = offset * warpSize + lane_idx;
    if(i < size) {
      z[i] = i % cutoff;
    }
  }

  // test count_topics
  gpulda::count_topics(z, size, m, lane_idx);

  // // retrieve values
  for(i32 offset = 0; offset < cutoff / 2 + 1; ++offset) {
    i32 i = offset * 2 + (lane_idx < warpSize/2 ? 0 : 1);
    if(i < cutoff) {
      u32 ct = m->get2(i);
      if(half_lane_idx == 0) {
        count[i] = ct;
      }
    }
  }


  // check correctness
  if(lane_idx == 0) {
    for(i32 i = 0; i < cutoff; ++i) {
      if(count[i] != size/cutoff) {
        error[0] = i;
        break;
      }
    }
  }
}

void test_sample_topics() {
  constexpr u32 warpSize = 32;

  curandStatePhilox4_32_10_t* rng;
  cudaMalloc(&rng, sizeof(curandStatePhilox4_32_10_t)) >> GPULDA_CHECK;
  gpulda::rng_init<<<1,1>>>(0,0,rng);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  u32* out;
  cudaMalloc(&out, sizeof(u32)) >> GPULDA_CHECK;
  u32 out_host = 0;

  // draw topic via wary search
  test_draw_wary_search<<<1,warpSize>>>(out);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  cudaMemcpy(&out_host, out, sizeof(u32), cudaMemcpyDeviceToHost) >> GPULDA_CHECK;
  assert(out_host == 0);

  // count topics
  test_count_topics<<<1,warpSize>>>(out, rng);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  cudaMemcpy(&out_host, out, sizeof(u32), cudaMemcpyDeviceToHost) >> GPULDA_CHECK;
  assert(out_host == 0);

}

}
