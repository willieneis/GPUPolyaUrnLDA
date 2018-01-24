#include "test_hashmap_sync.cuh"
#include "../hashmap_sync.cuh"
#include "../random.cuh"
#include "../error.cuh"
#include "assert.h"

using gpulda::FileLine;
using gpulda::f32;
using gpulda::i32;
using gpulda::u32;
using gpulda::u64;

namespace gpulda_test {

__global__ void test_hash_map_init(i32 initial_capacity, i32* map_returned_size, curandStatePhilox4_32_10_t* rng) {
  // allocate map
  __shared__ gpulda::HashMap m[1];
  m->init(initial_capacity, rng);
  __syncthreads();

  // check returned capacity
  if(threadIdx.x == 0) {
    map_returned_size[0] = m->capacity;
  }

  // deallocate map
  m->deallocate();
}




__global__ void test_hash_map_insert2(i32 num_unique_elements, i32 num_elements, i32 max_size, i32* out, curandStatePhilox4_32_10_t* rng, i32 rebuild) {
  __shared__ gpulda::HashMap m[1];
  u32 initial_size = rebuild ? num_elements : max_size;
  m->init(initial_size, rng);
  i32 dim = blockDim.x / (warpSize / 2);
  i32 half_warp_idx = threadIdx.x / (warpSize / 2);
  i32 half_lane_idx = threadIdx.x % (warpSize / 2);
  __syncthreads();

  // accumulate elements
  for(i32 offset = 0; offset < num_elements / dim + 1; ++offset) {
    u32 i = offset * dim + half_warp_idx;
    m->insert2(i % num_unique_elements, i < num_elements ? 1 : 0);
  }

  // ensure insertion finished
  __syncthreads();

  // rebuild if needed
  if(rebuild == true) {
    m->resize(0,0);
  }

  // ensure rebuild finished
  __syncthreads();

  // retrieve elements
  for(i32 offset = 0; offset < num_elements / dim + 1; ++offset) {
    i32 i = offset * dim + half_warp_idx;
    if(i < num_unique_elements) {
      u32 element = m->get2(i);
      if(half_lane_idx == 0) {
        out[i] = element;
      }
    }
  }

  // deallocate map
  m->deallocate();
}







void test_hash_map_sync() {
  constexpr i32 max_size = 100; // will round down to 96 for cache alignment
  constexpr i32 num_elements = 90; // large contention to ensure collisions occur
  constexpr i32 num_unique_elements = 9;
  constexpr i32 warpSize = 32;

  curandStatePhilox4_32_10_t* rng;
  cudaMalloc(&rng, sizeof(curandStatePhilox4_32_10_t)) >> GPULDA_CHECK;
  gpulda::rng_init<<<1,1>>>(0,0,rng);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  i32* out;
  cudaMalloc(&out, num_elements * sizeof(i32)) >> GPULDA_CHECK;
  i32* out_host = new i32[num_elements];

  // init<warp>
  test_hash_map_init<<<1,warpSize>>>(max_size, out, rng);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  cudaMemcpy(out_host, out, sizeof(i32), cudaMemcpyDeviceToHost) >> GPULDA_CHECK;

  assert(out_host[0] == (max_size / GPULDA_HASH_LINE_SIZE) * GPULDA_HASH_LINE_SIZE);
  out_host[0] = 0;

  // init<block>
  test_hash_map_init<<<1,GPULDA_POLYA_URN_SAMPLE_BLOCKDIM>>>(max_size, out, rng);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  cudaMemcpy(out_host, out, sizeof(i32), cudaMemcpyDeviceToHost) >> GPULDA_CHECK;

  assert(out_host[0] == (max_size / GPULDA_HASH_LINE_SIZE) * GPULDA_HASH_LINE_SIZE);
  out_host[0] = 0;

  // insert2: warp, no rebuild
  test_hash_map_insert2<<<1,warpSize>>>(num_unique_elements, num_elements, max_size, out, rng, false);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  cudaMemcpy(out_host, out, num_elements * sizeof(i32), cudaMemcpyDeviceToHost) >> GPULDA_CHECK;

  for(i32 i = 0; i < num_unique_elements; ++i) {
    assert(out_host[i] == num_elements / num_unique_elements);
    out_host[i] = 0;
  }

  // insert2: block, no rebuild
  test_hash_map_insert2<<<1,GPULDA_POLYA_URN_SAMPLE_BLOCKDIM>>>(num_unique_elements, num_elements, max_size, out, rng, false);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  cudaMemcpy(out_host, out, num_elements * sizeof(i32), cudaMemcpyDeviceToHost) >> GPULDA_CHECK;

  for(i32 i = 0; i < num_unique_elements; ++i) {
    assert(out_host[i] == num_elements / num_unique_elements);
    out_host[i] = 0;
  }

  // insert2: warp, rebuild
  test_hash_map_insert2<<<1,warpSize>>>(num_unique_elements, num_elements, max_size, out, rng, true);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  cudaMemcpy(out_host, out, num_elements * sizeof(u32), cudaMemcpyDeviceToHost) >> GPULDA_CHECK;

  for(i32 i = 0; i < num_unique_elements; ++i) {
    assert(out_host[i] == num_elements / num_unique_elements);
    out_host[i] = 0;
  }

  // insert2: block, rebuild
  test_hash_map_insert2<<<1,GPULDA_POLYA_URN_SAMPLE_BLOCKDIM>>>(num_unique_elements, num_elements, max_size, out, rng, true);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  cudaMemcpy(out_host, out, num_elements * sizeof(u32), cudaMemcpyDeviceToHost) >> GPULDA_CHECK;

  for(i32 i = 0; i < num_unique_elements; ++i) {
    assert(out_host[i] == num_elements / num_unique_elements);
    out_host[i] = 0;
  }




  // cleanup
  cudaFree(out);
  delete[] out_host;
  cudaFree(rng);
}


}
