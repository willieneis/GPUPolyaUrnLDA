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

__global__ void test_draw_alias(u32* error) {
  // compute constants
  i32 lane_idx = threadIdx.x % warpSize;
  constexpr u32 size = 5;
  __shared__ f32 prob[size];
  __shared__ u32 alias[size];

  // build alias table
  for(i32 offset = 0; offset < size / warpSize + 1; ++offset) {
    i32 i = offset * warpSize + lane_idx;
    if(i<size) {
      prob[i] = 0.5;
      alias[i] = 1;
    }
  }

  // draw from prob
  u32 topic = gpulda::draw_alias(0.6, prob, alias, size, lane_idx);
  if(lane_idx==0 && topic!=3){
    error[0] = 1;
  }

  // draw from alias
  topic = gpulda::draw_alias(0.75, prob, alias, size, lane_idx);
  if(lane_idx==0 && topic!=1){
    error[0] = 2;
  }
}

__global__ void test_draw_wary_search(u32* error) {
  i32 lane_idx = threadIdx.x % warpSize;
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
      /*printf("mPhi val: %0.6f\n", ((float)i) * sigma_b / ((float)size));*/
      mPhi[i] = ((float)i) * sigma_b / ((float)size);
    }
  }

  // test standard case: first entry in first slot
  u32 topic = gpulda::draw_wary_search(0.0f, m, mPhi, sigma_b, lane_idx);
  if(lane_idx==0 && topic!=0){
    error[0] = 1;
  }

  // test standard case: second entry in first slot
  topic = gpulda::draw_wary_search(0.02f, m, mPhi, sigma_b, lane_idx);
  if(lane_idx==0 && topic!=1){
    error[0] = 2;
  }

  // test edge case 1: last entry in first slot, search ends in second slot
  topic = gpulda::draw_wary_search(0.16f, m, mPhi, sigma_b, lane_idx);
  if(lane_idx==0 && topic!=15){
    error[0] = 3;
  }

  // test standard case: value in middle of slot
  topic = gpulda::draw_wary_search(0.4f, m, mPhi, sigma_b, lane_idx);
  if(lane_idx==0 && topic!=38){
    error[0] = 4;
  }

  // test standard case: second-to-last entry in last slot
  topic = gpulda::draw_wary_search(0.985f, m, mPhi, sigma_b, lane_idx);
  if(lane_idx==0 && topic!=94){
    error[0] = 5;
  }

  // test edge case 2: last entry in last slot
  topic = gpulda::draw_wary_search(1.0f, m, mPhi, sigma_b, lane_idx);
  if(lane_idx==0 && topic!=95){
    error[0] = 6;
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

  // initialize hashmap
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

  // retrieve values
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
        error[0] = i+1;
        break;
      }
    }
  }
}

__global__ void test_compute_product_cumsum(u32* error) {
  // compute constants
  i32 lane_idx = threadIdx.x % warpSize;
  typedef cub::BlockScan<i32, GPULDA_COMPUTE_D_IDX_BLOCKDIM> BlockScan;
  __shared__ typename cub::WarpScan<f32>::TempStorage warp_scan_temp[1];
  f32 tolerance = 0.0001f; // large to allow for randomness

  // declare arguments
  constexpr u32 size = 100;
  __shared__ f32 Phi_dense[size];
  __shared__ f32 mPhi[size];
  __shared__ f32 check[size];

  // populate hashmap data
  __shared__ gpulda::HashMap m[1];
  __shared__ u64 data[size];
  m->size_1 = size;
  m->data_1 = data;
  m->state = 2;

  // prepare state
  u64 empty = m->entry(0, 0, m->null_pointer(), 0, 0);
  for(i32 offset = 0; offset < size / warpSize + 1; ++offset) {
    i32 i = offset * warpSize + lane_idx;
    if(i < size) {
      Phi_dense[i] = 6.0f * (float) i;
      data[i] = m->with_key(i, m->with_value(i, empty));
      check[i] = (i == 0) ? 0.0f : ((float) (i-1)) * (((float) (i-1))+1.0f) * ((2.0f*((float) (i-1)))+1.0f);
    }
  }

  // test count_topics
  f32 total = gpulda::compute_product_cumsum(mPhi, m, Phi_dense, lane_idx, warp_scan_temp);

  // check correctness
  if(lane_idx == 0) {
    for(i32 i = 0; i < size; ++i) {
      if(abs(mPhi[i] - check[i]) > tolerance) {
        error[0] = i+1;
        break;
      }
    }
    f32 expected_total = ((float) (size-1)) * (((float) (size-1))+1.0f) * ((2.0f*((float) (size-1)))+1.0f);
    if(total != expected_total) {
      error[0] = size+1;
    }
  }
}

void test_sample_topics_functions() {
  constexpr u32 warpSize = 32;

  curandStatePhilox4_32_10_t* rng;
  cudaMalloc(&rng, sizeof(curandStatePhilox4_32_10_t)) >> GPULDA_CHECK;
  gpulda::rng_init<<<1,1>>>(0,0,rng);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  u32* out;
  cudaMalloc(&out, sizeof(u32)) >> GPULDA_CHECK;
  u32 out_host = 0;

  // draw topic via Alias table
  test_draw_alias<<<1,warpSize>>>(out);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  cudaMemcpy(&out_host, out, sizeof(u32), cudaMemcpyDeviceToHost) >> GPULDA_CHECK;
  assert(out_host == 0);

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

  // compute sparse vector product
  test_compute_product_cumsum<<<1,warpSize>>>(out);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  cudaMemcpy(&out_host, out, sizeof(u32), cudaMemcpyDeviceToHost) >> GPULDA_CHECK;
  assert(out_host == 0);

}

void test_sample_topics() {

}

}
