#pragma once

#include "types.cuh"
#include <curand_kernel.h> // need to add -lcurand to nvcc flags
#include "tuning.cuh"
#include "hashmap.cuh"

namespace gpulda {

__global__ void compute_d_idx(u32* d_len, u32* d_idx, u32 n_docs);

__global__ void sample_topics(u32 size,
    u32* z, u32* w, u32* d_len, u32* d_idx, u32* K_d,
    u32 V, u32* n_dense, f32* Phi_dense, f32* sigma_a,
    f32** prob, u32** alias, u32 table_size, curandStatePhilox4_32_10_t* rng);



template<class T>
__device__ __forceinline__ T block_scan_sum(T& thread_value, T* temp) {
  T current_value = thread_value;

  // first, compute a warp scan
  for(i32 i = 1; i < warpSize; i<<=1) {
    T add_value = __shfl_up(current_value, i, warpSize);
    add_value = (i <= threadIdx.x % warpSize) ? add_value : 0;
    current_value += add_value;
  }

  // final thread: write to shared array
  if((threadIdx.x+1) % warpSize == 0) {
    temp[threadIdx.x / warpSize] = current_value;
  }

  // ensure warp totals have been written
  __syncthreads();

  // read from shared array
  T warp_total = 0;
  if(threadIdx.x % warpSize < blockDim.x / warpSize) {
    warp_total = temp[threadIdx.x % warpSize];
  }

  // compute warp scan
  for(i32 i = 1; i < blockDim.x / warpSize; i<<=1) {
    T add_value = __shfl_up(warp_total, i, warpSize);
    add_value = (i <= threadIdx.x % warpSize) ? add_value : 0;
    warp_total += add_value;
  }

  // add to current value
  if(threadIdx.x >= warpSize) {
    current_value += __shfl(warp_total, (threadIdx.x / warpSize) - 1, warpSize);
  }

  // adjust to make scan exclusive, write output, and return
  thread_value = current_value - thread_value;
  return __shfl(warp_total, (blockDim.x / warpSize) - 1, warpSize);
}



__device__ __forceinline__ u32 draw_alias(f32 u, f32* prob, u32* alias, u32 table_size) {
  u32 ret = 0;
  if(threadIdx.x == 0) {
    // determine the slot and update random number
    f32 ts = (f32) table_size;
    u32 slot = (u32) (u * ts);
    u = fmodf(u, __frcp_rz(ts)) * ts;

    // load table elements from global memory
    f32 thread_prob = prob[slot];
    u32 thread_alias = alias[slot];

    // return the resulting draw
    if(u < thread_prob) {
      ret = slot;
    } else {
      ret = thread_alias;
    }
  }
  return __shfl(ret, 0);
}




__device__ __forceinline__ u32 draw_wary_search(f32 u, HashMap* m, f32 sigma_b) {
  // determine size and key array
  i32 size = m->capacity;
  u64* data = m->data;
  f32* mPhi = m->temp_data; // length MUST be multiple of GPULDA_HASH_LINE_SIZE

  u32 thread_key;
  i32 lane_idx = threadIdx.x; // TODO: vectorize
  if(lane_idx < warpSize/2) {
    // perform search
    i32 left = 0;
    i32 right = (size-1)/16;
    f32 target = u * sigma_b;
    u32 up;
    u32 down;
    i32 index = (left + right)/2;
    f32 thread_mPhi = mPhi[(16*index) + lane_idx];
    do {
      up = __ballot(target > thread_mPhi) & 0x0000ffff;
      down = __ballot(target <= thread_mPhi) & 0x0000ffff;
      if(__popc(up) == warpSize/2) {
        left = index + 1;
      } else if(__popc(down) == warpSize/2) {
        right = index - 1;
      } else {
        left = index;
        right = index;
      }
      index = (left + right) / 2;
      thread_mPhi = mPhi[(16*index) + lane_idx];
    } while(left < right); // don't use != because of possible edge case

    // retreive keys and determine value
    u64 thread_data = data[(16*index) + lane_idx];
    u32 lane_found = __ballot(target <= thread_mPhi) & 0x0000ffff;
    i32 read_idx = __ffs(lane_found)-2; // -2 because 1-based, and we want the last thread below threshold, not first one above

    if(lane_found == 0) {
      // edge case 1: read from last thread
      read_idx = 15;
    } else if(lane_found & 1 == 1) {
      // edge case 2: go back a slot, read from last thread
      thread_data = data[16*(index - 1) + lane_idx]; // no need for min, slot 0 doesn't go into this edge case
      read_idx = 15;
    }
    thread_key = __shfl(m->key(thread_data), read_idx);
  }

  return __shfl(thread_key, 0);
}




__device__ __forceinline__ void count_topics(u32* z, u32 document_size, HashMap* m) {
  // loop over z, add to m
  for(i32 offset = 0; offset < document_size / blockDim.x + 1; ++offset) {
    i32 i = offset * blockDim.x + threadIdx.x;

    // retreive z from global memory
    u32 lane_z = (i < document_size) ? z[i] : 0;
    i32 lane_K = (i < document_size) ? 1 : 0;

    // insert to hashmap two half lanes at a time
    for(i32 j = 0; j < warpSize/2; ++j) {
      u32 half_warp_z = __shfl(lane_z, j, warpSize/2);
      i32 half_warp_K = __shfl(lane_K, j, warpSize/2);
      m->insert2(half_warp_z, half_warp_K);
    }
  }
}




__device__ __forceinline__ f32 compute_product_cumsum(HashMap* m, f32* Phi_dense, f32* temp) {
  i32 m_size = m->capacity;
  u64* m_data = m->data;
  f32* mPhi = m->temp_data;

  f32 initial_value = 0;
  for(i32 offset = 0; offset < m_size / blockDim.x + 1; ++offset) {
    i32 i = offset * blockDim.x + threadIdx.x;
    u64 m_i = (i < m_size) ? m_data[i] : 0;
    u32 token = (i < m_size) ? m->key(m_i) : 0xffffffff;
    f32 m_count = (float) m->value(m_i);
    f32 Phi_count = (token == 0xffffffff) ? 0.0f : Phi_dense[token];
    f32 thread_mPhi = m_count * Phi_count;
    f32 total_value;

    // compute scan
    total_value = block_scan_sum<f32>(thread_mPhi, temp);
    __syncthreads();

    // apply offset
    thread_mPhi = thread_mPhi + initial_value;
    initial_value = total_value + initial_value;

    // write output to array
    if(i < m_size) {
      mPhi[i] = thread_mPhi;
    }
  }
  return initial_value;
}

}
