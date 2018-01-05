#pragma once

#include "types.cuh"
#include <curand_kernel.h> // need to add -lcurand to nvcc flags
#include "tuning.cuh"
#include "hashmap.cuh"
#include <thrust/system/cuda/detail/cub/block/block_scan.cuh>
#include <thrust/system/cuda/detail/cub/warp/warp_scan.cuh>

namespace gpulda {

__global__ void compute_d_idx(u32* d_len, u32* d_idx, u32 n_docs);

__global__ void sample_topics(u32 size, u32 n_docs,
    u32* z, u32* w, u32* d_len, u32* d_idx, u32* K_d, u64* hash, f32* mPhi,
    u32 K, u32 V, u32 max_K_d,
    f32* Phi_dense,
    f32** prob, u32** alias, curandStatePhilox4_32_10_t* rng);



__device__ __forceinline__ u32 draw_alias(f32 u, f32* prob, u32* alias, u32 table_size, i32 lane_idx) {
  u32 ret = 0;
  if(lane_idx == 0) {
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




__device__ __forceinline__ u32 draw_wary_search(f32 u, HashMap* m, f32* mPhi, f32 sigma_b, i32 lane_idx) {
  // determine size and key array
  u32 size;
  u64* data;
  if(m->state < 3) {
    size = m->size_1;
    data = m->data_1;
  } else {
    size = m->size_2;
    data = m->data_2;
  }

  u32 thread_key;
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
      up = __ballot(target > thread_mPhi);
      down = __ballot(target < thread_mPhi);
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
    } while(left != right);

    // retreive keys and determine value
    u64 thread_data = data[(16*left) + lane_idx];
    u32 lane_found = __ballot(thread_mPhi > target);
    i32 read_idx = __ffs(lane_found)-2; // -2 because 1-based, and we want the last thread below threshold, not first one above

    if(lane_found == 0) {
      // edge case 1: read from last thread
      read_idx = 15;
    } else if(lane_found & 1 == 1) {
      // edge case 2: go back a slot, read from last thread
      thread_data = data[16*(left - 1) + lane_idx]; // no need for min, slot 0 doesn't go into this edge case
      read_idx = 15;
    }
    thread_key = __shfl(m->key(thread_data), read_idx);
  }

  return __shfl(thread_key, 0);
}




__device__ __forceinline__ void count_topics(u32* z, u32 document_size, HashMap* m, i32 lane_idx) {
  // loop over z, add to m
  for(i32 offset = 0; offset < document_size / warpSize + 1; ++offset) {
    i32 i = offset * warpSize + lane_idx;
    u32 lane_z;
    u32 lane_K;
    if(i < document_size) {
      lane_z = z[i];
      lane_K = 1;
    } else {
      lane_z = 0;
      lane_K = 0;
    }
    for(i32 j = 0; j < warpSize/2; ++j) {
      u32 half_warp_z = __shfl(lane_z, j, warpSize/2);
      u32 half_warp_K = __shfl(lane_K, j, warpSize/2);
      m->insert2(half_warp_z, half_warp_K);
    }
  }
}




__device__ __forceinline__ f32 compute_product_cumsum(f32* mPhi, HashMap* m, f32* Phi_dense, i32 lane_idx, cub::WarpScan<f32>::TempStorage* temp) {
  typedef cub::WarpScan<f32> WarpScan;
  u32 m_size;
  u64* m_data;
  if(m->state < 3) {
    m_size = m->size_1;
    m_data = m->data_1;
  } else {
    m_size = m->size_2;
    m_data = m->data_2;
  }

  f32 initial_value = 0;
  for(i32 offset = 0; offset < m_size / warpSize + 1; ++offset) {
    i32 i = offset * warpSize + lane_idx;
    u64 m_i = (i < m_size) ? m_data[i] : 0;
    u32 token = (i < m_size) ? m->key(m_i) : m->empty_key();
    f32 m_count = (token == m->empty_key()) ? 0.0f : (float) m->value(m_i);
    f32 Phi_count = (token == m->empty_key()) ? 0.0f : Phi_dense[token];
    f32 thread_mPhi = m_count * Phi_count;
    f32 total_value;

    // compute scan
    WarpScan(*temp).ExclusiveScan(thread_mPhi, thread_mPhi, 0, cub::Sum(), total_value);

    // workaround for CUB bug: apply offset manually
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
