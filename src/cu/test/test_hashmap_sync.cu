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

__global__ void test_hash_map_init(i32 initial_capacity, i32* map_returned_size) {
  // allocate map
  __shared__ curandStatePhilox4_32_10_t rng;
  if(threadIdx.x == 0) {
    curand_init((unsigned long long) 0, (unsigned long long) 0, (unsigned long long) 0, &rng);
  }
  __syncthreads();
  __shared__ gpulda::HashMap m;
  m.init(initial_capacity, &rng);
  __syncthreads();

  // check returned capacity
  if(threadIdx.x == 0) {
    map_returned_size[0] = m.capacity;
  }

  // deallocate map
  m.deallocate();
}




__global__ void test_hash_map_insert_print_steps() {
  #ifdef GPULDA_HASH_DEBUG
  // initialize
  __shared__ curandStatePhilox4_32_10_t rng;
  if(threadIdx.x == 0) {
    curand_init((unsigned long long) 0, (unsigned long long) 0, (unsigned long long) 0, &rng);
  }
  __syncthreads();
  __shared__ gpulda::HashMap m;
  m.init(96, &rng);
  m.a=26; m.b=1; m.c=30; m.d=13;
  __syncthreads();

  if(threadIdx.x < warpSize) {
    if(threadIdx.x == 0) { m.debug_print_slot(0); }
    if(threadIdx.x == 0) { m.debug_print_slot(16); }
    if(threadIdx.x == 0) { m.debug_print_slot(32); }
    if(threadIdx.x == 0) { m.debug_print_slot(48); }
    if(threadIdx.x == 0) { m.debug_print_slot(64); }
    if(threadIdx.x == 0) { m.debug_print_slot(80); }
    if(threadIdx.x == 0) { printf("------------------------------------------------------------\n"); }

    // 16
    m.insert2(threadIdx.x < 16 ? 0 : 3, 1); if(threadIdx.x == 0) { m.debug_print_slot(16); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 6 : 9, 1); if(threadIdx.x == 0) { m.debug_print_slot(16); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 12 : 15, 1); if(threadIdx.x == 0) { m.debug_print_slot(16); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 18 : 21, 1);if(threadIdx.x == 0) { m.debug_print_slot(16); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 24 : 27, 1); if(threadIdx.x == 0) { m.debug_print_slot(16); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 30 : 33, 1); if(threadIdx.x == 0) { m.debug_print_slot(16); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 36 : 39, 1); if(threadIdx.x == 0) { m.debug_print_slot(16); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 42 : 45, 1); if(threadIdx.x == 0) { m.debug_print_slot(16); printf("------------------------------------------------------------\n"); }

    // 16->32
    m.insert2(threadIdx.x < 16 ? 48 : 51, 1); if(threadIdx.x == 0) { m.debug_print_slot(32); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 54 : 57, 1); if(threadIdx.x == 0) { m.debug_print_slot(32); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 60 : 63, 1); if(threadIdx.x == 0) { m.debug_print_slot(32); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 66 : 69, 1); if(threadIdx.x == 0) { m.debug_print_slot(32); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 72 : 75, 1); if(threadIdx.x == 0) { m.debug_print_slot(32); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 78 : 81, 1); if(threadIdx.x == 0) { m.debug_print_slot(32); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 84 : 87, 1); if(threadIdx.x == 0) { m.debug_print_slot(32); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 90 : 93, 1); if(threadIdx.x == 0) { m.debug_print_slot(32); printf("------------------------------------------------------------\n"); }

    // 48
    m.insert2(threadIdx.x < 16 ? 1 : 4, 1); if(threadIdx.x == 0) { m.debug_print_slot(48); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 7 : 10, 1); if(threadIdx.x == 0) { m.debug_print_slot(48); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 13 : 16, 1); if(threadIdx.x == 0) { m.debug_print_slot(48); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 19 : 22, 1); if(threadIdx.x == 0) { m.debug_print_slot(48); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 25 : 28, 1); if(threadIdx.x == 0) { m.debug_print_slot(48); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 31 : 34, 1); if(threadIdx.x == 0) { m.debug_print_slot(48); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 37 : 40, 1); if(threadIdx.x == 0) { m.debug_print_slot(48); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 43 : 46, 1); if(threadIdx.x == 0) { m.debug_print_slot(48); printf("------------------------------------------------------------\n"); }

    // 16->48 evict
    m.insert2(threadIdx.x < 16 ? 96 : 99, 1); if(threadIdx.x == 0) { m.debug_print_slot(48); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 102 : 105, 1); if(threadIdx.x == 0) { m.debug_print_slot(48); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 108 : 111, 1); if(threadIdx.x == 0) { m.debug_print_slot(48); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 114 : 117, 1); if(threadIdx.x == 0) { m.debug_print_slot(48); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 120 : 123, 1); if(threadIdx.x == 0) { m.debug_print_slot(48); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 126 : 129, 1); if(threadIdx.x == 0) { m.debug_print_slot(48); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 132 : 135, 1); if(threadIdx.x == 0) { m.debug_print_slot(48); printf("------------------------------------------------------------\n"); }
    m.insert2(threadIdx.x < 16 ? 138 : 141, 1); if(threadIdx.x == 0) { m.debug_print_slot(48); printf("------------------------------------------------------------\n"); }

    if(threadIdx.x == 0) { m.debug_print_slot(0); }
    if(threadIdx.x == 0) { m.debug_print_slot(16); }
    if(threadIdx.x == 0) { m.debug_print_slot(32); }
    if(threadIdx.x == 0) { m.debug_print_slot(48); }
    if(threadIdx.x == 0) { m.debug_print_slot(64); }
    if(threadIdx.x == 0) { m.debug_print_slot(80); }
  }

  m.deallocate();
  #endif
}




__global__ void test_hash_map_insert2(i32 num_unique_elements, i32 num_elements, i32 max_size, i32* out, i32 rebuild) {
  __shared__ curandStatePhilox4_32_10_t rng;
  if(threadIdx.x == 0) {
    curand_init((unsigned long long) 0, (unsigned long long) 0, (unsigned long long) 0, &rng);
  }
  __syncthreads();
  __shared__ gpulda::HashMap m;
  u32 initial_size = rebuild ? num_elements : max_size;
  m.init(initial_size, &rng);
  i32 dim = blockDim.x / (warpSize / 2);
  i32 half_warp_idx = threadIdx.x / (warpSize / 2);
  i32 half_lane_idx = threadIdx.x % (warpSize / 2);
  __syncthreads();

  // accumulate elements
  for(i32 offset = 0; offset < num_elements / dim + 1; ++offset) {
    u32 i = offset * dim + half_warp_idx;
    m.insert2(i % num_unique_elements, i < num_elements ? 1 : 0);
  }

  // ensure insertion finished
  __syncthreads();

  // rebuild if needed
  if(rebuild == true) {
    m.resize(0,0);
  }

  // ensure rebuild finished
  __syncthreads();

  // retrieve elements
  for(i32 offset = 0; offset < num_elements / dim + 1; ++offset) {
    i32 i = offset * dim + half_warp_idx;
    if(i < num_unique_elements) {
      u32 element = m.get2(i);
      if(half_lane_idx == 0) {
        out[i] = element;
      }
    }
  }

  // deallocate map
  m.deallocate();
}







void test_hash_map_sync() {
  constexpr i32 max_size = 100; // will round down to 96 for cache alignment
  constexpr i32 num_elements = 90; // large contention to ensure collisions occur
  constexpr i32 num_unique_elements = 9;
  constexpr i32 warpSize = 32;

  i32* out;
  cudaMalloc(&out, num_elements * sizeof(i32)) >> GPULDA_CHECK;
  i32* out_host = new i32[num_elements];

  // init<warp>
  test_hash_map_init<<<1,warpSize>>>(max_size, out);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  cudaMemcpy(out_host, out, sizeof(i32), cudaMemcpyDeviceToHost) >> GPULDA_CHECK;

  assert(out_host[0] == (max_size / GPULDA_HASH_LINE_SIZE) * GPULDA_HASH_LINE_SIZE);
  out_host[0] = 0;

  // init<block>
  test_hash_map_init<<<1,GPULDA_POLYA_URN_SAMPLE_BLOCKDIM>>>(max_size, out);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  cudaMemcpy(out_host, out, sizeof(i32), cudaMemcpyDeviceToHost) >> GPULDA_CHECK;

  assert(out_host[0] == (max_size / GPULDA_HASH_LINE_SIZE) * GPULDA_HASH_LINE_SIZE);
  out_host[0] = 0;

  // print steps
  // #ifdef GPULDA_HASH_DEBUG
  // test_hash_map_insert_print_steps<<<1,warpSize>>>();
  // cudaDeviceSynchronize() >> GPULDA_CHECK;
  // assert(false);
  // #endif

  // insert2: warp, no rebuild
  test_hash_map_insert2<<<1,warpSize>>>(num_unique_elements, num_elements, max_size, out, false);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  cudaMemcpy(out_host, out, num_elements * sizeof(i32), cudaMemcpyDeviceToHost) >> GPULDA_CHECK;

  for(i32 i = 0; i < num_unique_elements; ++i) {
    assert(out_host[i] == num_elements / num_unique_elements);
    out_host[i] = 0;
  }

  // insert2: block, no rebuild
  test_hash_map_insert2<<<1,GPULDA_POLYA_URN_SAMPLE_BLOCKDIM>>>(num_unique_elements, num_elements, max_size, out, false);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  cudaMemcpy(out_host, out, num_elements * sizeof(i32), cudaMemcpyDeviceToHost) >> GPULDA_CHECK;

  for(i32 i = 0; i < num_unique_elements; ++i) {
    assert(out_host[i] == num_elements / num_unique_elements);
    out_host[i] = 0;
  }

  // insert2: warp, rebuild
  test_hash_map_insert2<<<1,warpSize>>>(num_unique_elements, num_elements, max_size, out, true);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  cudaMemcpy(out_host, out, num_elements * sizeof(u32), cudaMemcpyDeviceToHost) >> GPULDA_CHECK;

  for(i32 i = 0; i < num_unique_elements; ++i) {
    assert(out_host[i] == num_elements / num_unique_elements);
    out_host[i] = 0;
  }

  // insert2: block, rebuild
  test_hash_map_insert2<<<1,GPULDA_POLYA_URN_SAMPLE_BLOCKDIM>>>(num_unique_elements, num_elements, max_size, out, true);
  cudaDeviceSynchronize() >> GPULDA_CHECK;

  cudaMemcpy(out_host, out, num_elements * sizeof(u32), cudaMemcpyDeviceToHost) >> GPULDA_CHECK;

  for(i32 i = 0; i < num_unique_elements; ++i) {
    assert(out_host[i] == num_elements / num_unique_elements);
    out_host[i] = 0;
  }




  // cleanup
  cudaFree(out);
  delete[] out_host;
}


}
