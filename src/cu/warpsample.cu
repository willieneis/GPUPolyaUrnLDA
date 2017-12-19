#include "hashmap.cuh"
#include "warpsample.cuh"
#include <thrust/system/cuda/detail/cub/block/block_scan.cuh>
#include <thrust/system/cuda/detail/cub/warp/warp_scan.cuh>

namespace gplda {

__global__ void compute_d_idx(u32* d_len, u32* d_idx, u32 n_docs) {
  typedef cub::BlockScan<i32, GPLDA_COMPUTE_D_IDX_BLOCKDIM> BlockScan;
  __shared__ typename BlockScan::TempStorage temp;

  if(blockIdx.x == 0) {
    i32 thread_d;
    i32 initial_value = 0;
    i32 total_value;
    for(i32 offset = 0; offset < n_docs / blockDim.x + 1; ++offset) {
      i32 i = threadIdx.x + offset * blockDim.x;
      if(i < n_docs) {
        thread_d = d_len[i];
      } else {
        thread_d = 0;
      }

      BlockScan(temp).ExclusiveScan(thread_d, thread_d, 0, cub::Sum(), total_value);

      // workaround for CUB bug: apply offset manually
      __syncthreads();
      thread_d = thread_d + initial_value;
      initial_value = total_value + initial_value;

      if(i < n_docs) {
        d_idx[i] = thread_d;
      }
    }
  }
}

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
  return ret;
}

__device__ __forceinline__ u32 draw_wary_search(f32 u) {
  return 0;
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

__device__ __forceinline__ f32 compute_product_cumsum(u32* mPhi, HashMap* m, f32* Phi_dense, i32 warp_idx, cub::WarpScan<i32>::TempStorage* temp) {
  i32 thread_mPhi = 0;
  cub::WarpScan<i32>(temp[warp_idx]).ExclusiveSum(thread_mPhi, thread_mPhi);
  return 0.0f;
}

__global__ void warp_sample_topics(u32 size, u32 n_docs,
    u32* z, u32* w, u32* d_len, u32* d_idx, u32* K_d, void* temp,
    u32 K, u32 V, u32 max_K_d,
    f32* Phi_dense,
    f32** prob, u32** alias, curandStatePhilox4_32_10_t* rng) {
  // initialize variables
  i32 lane_idx = threadIdx.x % warpSize;
  i32 warp_idx = threadIdx.x / warpSize;
  curandStatePhilox4_32_10_t warp_rng = rng[0];
  __shared__ HashMap m[1];
  u32* mPhi;
  __shared__ typename cub::WarpScan<i32>::TempStorage warp_scan_temp[1];

  // loop over documents
  for(i32 i = 0; i < n_docs; ++i) {
    // count topics in document
    u32 warp_d_len = d_len[i];
    u32 warp_d_idx = d_idx[i];
    // m->init();
    count_topics(z + warp_d_idx * sizeof(u32), warp_d_len, &m[1], lane_idx);

    // loop over words
    for(i32 j = 0; j < warp_d_len; ++j) {
      // load z,w from global memory
      u32 warp_z = z[warp_d_idx + j];
      u32 warp_w = 0;//w[warp_d_idx + j]; // why is this broken?

      // remove current z from sufficient statistic
      m->insert2(warp_z, lane_idx < 16 ? -1 : 0); // don't branch

      // compute m*phi and sigma_b
      f32 warp_sigma_a = 0.0f;
      f32 sigma_b = compute_product_cumsum(mPhi, &m[1], Phi_dense, warp_idx, warp_scan_temp);

      // update z
      f32 u1 = curand_uniform(&warp_rng);
      f32 u2 = curand_uniform(&warp_rng);
      if(u1 * (warp_sigma_a + sigma_b) > warp_sigma_a) {
        // sample from m*Phi
        warp_z = draw_wary_search(u2);
      } else {
        // sample from alias table
        warp_z = draw_alias(u2, prob[warp_w], alias[warp_w], /*table_size =*/ 1, lane_idx);
      }

      // add new z to sufficient statistic
      m->insert2(warp_z, lane_idx < 16 ? -1 : 0); // don't branch
      if(lane_idx == 0) {
        z[warp_d_idx + j] = warp_z;
      }
    }
  }
}

}
