#include "topics.cuh"

namespace gpulda {

__global__ void compute_d_idx(u32* d_len, u32* d_idx, u32 n_docs) {
  typedef cub::BlockScan<i32, GPULDA_COMPUTE_D_IDX_BLOCKDIM> BlockScan;
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
      __syncthreads();

      // workaround for CUB bug: apply offset manually
      thread_d = thread_d + initial_value;
      initial_value = total_value + initial_value;

      if(i < n_docs) {
        d_idx[i] = thread_d;
      }
    }
  }
}




__global__ void sample_topics(u32 size, u32 n_docs,
    u32* z, u32* w, u32* d_len, u32* d_idx, u32* K_d, u64* hash, f32* mPhi,
    u32 K, u32 V, u32 max_N_d,
    f32* Phi_dense, f32* sigma_a,
    f32** prob, u32** alias, u32 table_size, curandStatePhilox4_32_10_t* rng) {
  // initialize variables
  curandStatePhilox4_32_10_t block_rng = rng[0];
  __shared__ HashMap m[1];
  constexpr u32 ring_buffer_size = (GPULDA_SAMPLE_TOPICS_BLOCKDIM/16)*2;
  __shared__ u64 ring_buffer[ring_buffer_size];
  __shared__ u32 ring_buffer_queue[ring_buffer_size];
  __shared__ typename cub::BlockScan<f32, GPULDA_SAMPLE_TOPICS_BLOCKDIM>::TempStorage block_scan_temp[1];

  // initialize
  u32 block_d_len = d_len[blockIdx.x];
  u32 block_d_idx = d_idx[blockIdx.x];
  m->init(hash, 2*max_N_d, max_N_d, ring_buffer, ring_buffer_queue, ring_buffer_size, &block_rng, blockDim.x);
  __syncthreads();

  // count topics in document
  count_topics(z + block_d_idx * sizeof(u32), block_d_len, m);
  __syncthreads();

  // loop over words
  for(i32 j = 0; j < block_d_len; ++j) {
    // load z,w from global memory
    u32 block_z = z[block_d_idx + j];
    u32 block_w = w[block_d_idx + j];

    // remove current z from sufficient statistic
    if(threadIdx.x < warpSize) {
      m->insert2(block_z, threadIdx.x < warpSize/2 ? -1 : 0); // don't branch
    }
    __syncthreads();

    // compute m*phi and sigma_b
    f32 block_sigma_a = sigma_a[block_w];
    f32 sigma_b = compute_product_cumsum(mPhi, m, Phi_dense, block_scan_temp);
    __syncthreads();

    // update z
    f32 u1 = curand_uniform(&block_rng);
    f32 u2 = curand_uniform(&block_rng);
    if(u1 * (block_sigma_a + sigma_b) > block_sigma_a) {
      // sample from m*Phi
      block_z = draw_wary_search(u2, m, mPhi, sigma_b);
    } else {
      // sample from alias table
      block_z = draw_alias(u2, prob[block_w], alias[block_w], table_size);
    }

    // add new z to sufficient statistic
    if(threadIdx.x < warpSize) {
      m->insert2(block_z, threadIdx.x < warpSize/2 ? 1 : 0); // don't branch
    }

    // write output
    if(threadIdx.x == 0) {
      z[block_d_idx + j] = block_z;
    }
  }
}

}
