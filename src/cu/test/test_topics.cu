#include "test_topics.cuh"
#include "../topics.cuh"
#include "../train.cuh"
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
  constexpr u32 size = 5;
  __shared__ f32 prob[size];
  __shared__ u32 alias[size];

  // build alias table
  for(i32 offset = 0; offset < size / blockDim.x + 1; ++offset) {
    i32 i = offset * blockDim.x + threadIdx.x;
    if(i<size) {
      prob[i] = 0.5;
      alias[i] = 1;
    }
  }
  __syncthreads();

  // draw from prob
  u32 topic = gpulda::draw_alias(0.6, prob, alias, size);
  if(threadIdx.x==0 && topic!=3){
    error[0] = 1;
  }

  // draw from alias
  topic = gpulda::draw_alias(0.75, prob, alias, size);
  if(threadIdx.x==0 && topic!=1){
    error[0] = 2;
  }
}

__global__ void test_draw_wary_search(u32* error) {
  constexpr i32 size = 96; // hashmap size must be multiple of 16
  __shared__ gpulda::HashMap m;
  __shared__ u64 data[size];
  m.capacity = size;
  m.data = data;
  __shared__ f32 mPhi[size];
  f32 sigma_b = 50.0f;

  for(i32 offset = 0; offset < size / blockDim.x + 1; ++offset) {
    i32 i = offset * blockDim.x + threadIdx.x;
    if(i<size) {
      data[i] = m.entry(i,0);
      mPhi[i] = ((float)i) * sigma_b / ((float)size);
    }
  }
  __syncthreads();

  // test standard case: first entry in first slot
  u32 topic = gpulda::draw_wary_search(0.0f, &m, mPhi, sigma_b);
  if(threadIdx.x==0 && topic!=0){
    error[0] = 1;
  }

  // test standard case: second entry in first slot
  topic = gpulda::draw_wary_search(0.02f, &m, mPhi, sigma_b);
  if(threadIdx.x==0 && topic!=1){
    error[0] = 2;
  }

  // test edge case 1: last entry in first slot, search ends in second slot
  topic = gpulda::draw_wary_search(0.16f, &m, mPhi, sigma_b);
  if(threadIdx.x==0 && topic!=15){
    error[0] = 3;
  }

  // test standard case: value in middle of slot
  topic = gpulda::draw_wary_search(0.4f, &m, mPhi, sigma_b);
  if(threadIdx.x==0 && topic!=38){
    error[0] = 4;
  }

  // test standard case: second-to-last entry in last slot
  topic = gpulda::draw_wary_search(0.985f, &m, mPhi, sigma_b);
  if(threadIdx.x==0 && topic!=94){
    error[0] = 5;
  }

  // test edge case 2: last entry in last slot
  topic = gpulda::draw_wary_search(1.0f, &m, mPhi, sigma_b);
  if(threadIdx.x==0 && topic!=95){
    error[0] = 6;
  }
}

__global__ void test_count_topics(u32* error) {
  // compute constants
  __shared__ curandStatePhilox4_32_10_t rng;
  if(threadIdx.x == 0) {
    curand_init((unsigned long long) 0, (unsigned long long) 0, (unsigned long long) 0, &rng);
  }
  __syncthreads();
  constexpr u32 cutoff = 25;
  __shared__ u32 count[cutoff];

  // declare arguments
  constexpr u32 size = 100;
  __shared__ u32 z[size];

  // initialize hashmap
  __shared__ gpulda::HashMap m;
  m.init(size, &rng);
  __syncthreads();

  // prepare state
  for(i32 offset = 0; offset < size / blockDim.x + 1; ++offset) {
    i32 i = offset * blockDim.x + threadIdx.x;
    if(i < size) {
      z[i] = i % cutoff;
    }
  }
  __syncthreads();

  // test count_topics
  gpulda::count_topics(z, size, &m);
  __syncthreads();

  // retrieve values
  for(i32 offset = 0; offset < blockDim.x + 1; ++offset) {
    i32 i = offset * blockDim.x + threadIdx.x;

    // retreive from hashmap, two half-lanes at a time
    for(i32 j = 0; j < warpSize/2; ++j) {
      i32 warp_i = __shfl(i, j, warpSize/2);
      u32 warp_ct = m.get2(warp_i);
      if(i == warp_i && i < cutoff) {
        count[i] = warp_ct;
      }
    }
  }

  // check correctness
  if(threadIdx.x == 0) {
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
  __shared__ f32 block_scan_temp[GPULDA_SAMPLE_TOPICS_BLOCKDIM / GPULDA_BLOCK_SCAN_WARP_SIZE];
  f32 tolerance = 0.0001f; // large to allow for randomness

  // declare arguments
  constexpr u32 size = 100;
  __shared__ f32 Phi_dense[size];
  __shared__ f32 mPhi[size];
  __shared__ f32 check[size];

  // populate hashmap data
  __shared__ gpulda::HashMap m;
  __shared__ u64 data[size];
  m.capacity = size;
  m.data = data;

  // prepare state
  for(i32 offset = 0; offset < size / blockDim.x + 1; ++offset) {
    i32 i = offset * blockDim.x + threadIdx.x;
    if(i < size) {
      Phi_dense[i] = 6.0f * (float) i;
      data[i] = m.entry(i, i);
      check[i] = (i == 0) ? 0.0f : ((float) (i-1)) * (((float) (i-1))+1.0f) * ((2.0f*((float) (i-1)))+1.0f);
    }
  }
  __syncthreads();

  // test count_topics
  f32 total = gpulda::compute_product_cumsum(mPhi, &m, Phi_dense, block_scan_temp);
  __syncthreads();

  // check correctness
  if(threadIdx.x == 0) {
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
  u32* out;
  cudaMalloc(&out, sizeof(u32)) >> GPULDA_CHECK;
  u32 out_host = 0;

  // draw topic via Alias table
  test_draw_alias<<<1,GPULDA_SAMPLE_TOPICS_BLOCKDIM>>>(out);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  cudaMemcpy(&out_host, out, sizeof(u32), cudaMemcpyDeviceToHost) >> GPULDA_CHECK;
  assert(out_host == 0);

  // draw topic via wary search
  test_draw_wary_search<<<1,GPULDA_SAMPLE_TOPICS_BLOCKDIM>>>(out);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  cudaMemcpy(&out_host, out, sizeof(u32), cudaMemcpyDeviceToHost) >> GPULDA_CHECK;
  assert(out_host == 0);

  // count topics
  test_count_topics<<<1,GPULDA_SAMPLE_TOPICS_BLOCKDIM>>>(out);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  cudaMemcpy(&out_host, out, sizeof(u32), cudaMemcpyDeviceToHost) >> GPULDA_CHECK;
  assert(out_host == 0);

  // compute sparse vector product
  test_compute_product_cumsum<<<1,GPULDA_SAMPLE_TOPICS_BLOCKDIM>>>(out);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  cudaMemcpy(&out_host, out, sizeof(u32), cudaMemcpyDeviceToHost) >> GPULDA_CHECK;
  assert(out_host == 0);

  // cleanup
  cudaFree(out);
}

void test_sample_topics() {
  constexpr f32 alpha = 0.1f;
  constexpr f32 beta = 0.1f;
  constexpr u32 V = 3;
  constexpr u32 K = 5;
  u32 C[V] = {1,1,1};
  constexpr u32 buffer_size = 5;
  constexpr u32 max_D = 2;

  gpulda::Args args = {alpha,beta,K,V,C,buffer_size,max_D};
  u32 z[buffer_size] = {4,1,0,4,0};
  u32 w[buffer_size] = {0,0,0,0,0};
  u32 d[max_D] = {3,2};
  u32 K_d[max_D] = {1,1};
  u32 n_docs = max_D;
  u32 n_tokens = buffer_size;
  gpulda::Buffer buffer = {z, w, d, K_d, n_docs, n_tokens, NULL, NULL, NULL, NULL, NULL, NULL, NULL};

  gpulda::initialize(&args, &buffer, 1);

  // initialize test-specific Phi
  f32 Phi_host[K*V] = { 0.98f, 0.02f, 0.02f, 0.02f, 0.02f,
                        0.01f, 0.49f, 0.49f, 0.49f, 0.49f,
                        0.01f, 0.49f, 0.49f, 0.49f, 0.49f };
  f32* Phi_dense;
  cudaMalloc(&Phi_dense, K*V*sizeof(f32)) >> GPULDA_CHECK;
  cudaMemcpy(Phi_dense, Phi_host, K*V*sizeof(f32), cudaMemcpyHostToDevice) >> GPULDA_CHECK;

  // initialize test-specific sigma_a
  f32 sigma_a_host[V] = { 0.0f, 0.0f, 0.0f };
  f32* sigma_a;
  cudaMalloc(&sigma_a, V*sizeof(f32)) >> GPULDA_CHECK;
  cudaMemcpy(sigma_a, sigma_a_host, V*sizeof(f32), cudaMemcpyHostToDevice) >> GPULDA_CHECK;

  // copy z,w,d to buffer
  cudaMemcpy(buffer.gpu_z, z, buffer_size*sizeof(u32), cudaMemcpyHostToDevice) >> GPULDA_CHECK;
  cudaMemcpy(buffer.gpu_w, w, buffer_size*sizeof(u32), cudaMemcpyHostToDevice) >> GPULDA_CHECK;
  cudaMemcpy(buffer.gpu_d_len, d, n_docs*sizeof(u32), cudaMemcpyHostToDevice) >> GPULDA_CHECK;
  cudaMemcpy(buffer.gpu_K_d, K_d, n_docs*sizeof(u32), cudaMemcpyHostToDevice) >> GPULDA_CHECK;
  gpulda::compute_d_idx<<<1,GPULDA_COMPUTE_D_IDX_BLOCKDIM>>>(buffer.gpu_d_len, buffer.gpu_d_idx, n_docs);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  // sample a topic indicator
  gpulda::sample_topics<<<n_docs,GPULDA_SAMPLE_TOPICS_BLOCKDIM>>>(args.buffer_size, buffer.gpu_z, buffer.gpu_w, buffer.gpu_d_len, buffer.gpu_d_idx, buffer.gpu_K_d, Phi_dense, sigma_a, NULL, NULL, 0, buffer.gpu_rng);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  cudaMemcpy(z, buffer.gpu_z, buffer_size*sizeof(u32), cudaMemcpyDeviceToHost) >> GPULDA_CHECK;
  for(i32 i = 0; i < buffer_size; ++i) {
    assert(z[i] == 0);
  }

  // cleanup
  cudaFree(Phi_dense);
  cudaFree(sigma_a);
  gpulda::cleanup(&buffer, 1);
}

}
