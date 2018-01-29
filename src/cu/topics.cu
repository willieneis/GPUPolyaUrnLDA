#include "topics.cuh"

namespace gpulda {

__global__ void compute_d_idx(u32* d_len, u32* d_idx, u32 n_docs) {
  __shared__ i32 temp[GPULDA_COMPUTE_D_IDX_BLOCKDIM / GPULDA_BLOCK_SCAN_WARP_SIZE];

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

      total_value = block_scan_sum<i32>(thread_d, temp);
      __syncthreads();

      // apply offset
      thread_d = thread_d + initial_value;
      initial_value = total_value + initial_value;

      if(i < n_docs) {
        d_idx[i] = thread_d;
      }
    }
  }
}



__global__ void sample_topics(u32 size,
    u32* z, u32* w, u32* d_len, u32* d_idx, u32* K_d,
    f32* Phi_dense, f32* sigma_a,
    f32** prob, u32** alias, u32 table_size, curandStatePhilox4_32_10_t* rng) {
  // initialize
  __shared__ curandStatePhilox4_32_10_t block_rng;
  __shared__ f32 u[2];
  __shared__ HashMap m;
  __shared__ f32 block_scan_temp[GPULDA_SAMPLE_TOPICS_BLOCKDIM / GPULDA_BLOCK_SCAN_WARP_SIZE];
  u32 block_d_len = d_len[blockIdx.x];
  u32 block_d_idx = d_idx[blockIdx.x];
  if(threadIdx.x == 0) {
    block_rng = rng[0];
    skipahead((unsigned long long int) block_d_idx, &block_rng);
  }
  m.init(threadIdx.x == 0 ? K_d[blockIdx.x] : 0, &block_rng, true);
  __syncthreads();

  // count topics in document
  count_topics(z + block_d_idx * sizeof(u32), block_d_len, &m);
  __syncthreads();

  // loop over words
  for(i32 i = 0; i < block_d_len; ++i) {
    // load z,w from global memory
    u32 block_z = z[block_d_idx + i];
    u32 block_w = w[block_d_idx + i];

    // remove current z from sufficient statistic
    m.insert2(block_z, threadIdx.x < warpSize/2 ? -1 : 0); // don't branch: might need to resize


    // compute random numbers
    if(threadIdx.x == 0) {
      u[0] = curand_uniform(&block_rng);
      u[1] = curand_uniform(&block_rng);
    }

    // compute m*phi and sigma_b
    f32 block_sigma_a = sigma_a[block_w];
    f32 sigma_b = compute_product_cumsum(&m, Phi_dense, block_scan_temp);
    __syncthreads();

    // update z
    if(u[0] * (block_sigma_a + sigma_b) > block_sigma_a) {
      // sample from m*Phi
      block_z = draw_wary_search(u[1], &m, sigma_b);
    } else {
      // sample from alias table
      block_z = draw_alias(u[1], prob[block_w], alias[block_w], table_size);
    }

    // add new z to sufficient statistic
    m.insert2(block_z, threadIdx.x < warpSize/2 ? 1 : 0); // don't branch: might need to resize

    // write output
    if(threadIdx.x == 0) {
      // atomicAdd(&n, ..)
      z[block_d_idx + i] = block_z;
    }
  }

  // update topic count and deallocate hashmap
  if(threadIdx.x == 0) {
    K_d[blockIdx.x] = m.size;
  }
  m.deallocate();
}

}
