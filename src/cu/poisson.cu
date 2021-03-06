#include "error.cuh"
#include "poisson.cuh"
#include "spalias.cuh"
#include "train.cuh"


namespace gpulda {

__global__ void build_poisson_prob(f32** prob, f32 beta, u32 table_size) {
  // determine constants
  f32 lambda = blockIdx.x + beta; // each block builds one table
  // populate PMF
  for(i32 offset = 0; offset < table_size / blockDim.x + 1; ++offset) {
    i32 i = threadIdx.x + offset * blockDim.x;
    f32 x = i;
    if(i < table_size) {
      prob[blockIdx.x][i] = expf(x*logf(lambda) - lambda - lgammaf(x + 1));
    }
  }
}

Poisson::Poisson(u32 ml, u32 mv, f32 b) {
  // assign class parameters
  max_lambda = ml;
  max_value = mv;
  beta = b;
  // allocate alias table
  pois_alias = new SpAlias(max_lambda, max_value);
  // launch kernel to build the alias tables
  build_poisson_prob<<<max_lambda,96>>>(pois_alias->prob, beta, max_value);
  cudaDeviceSynchronize() >> GPULDA_CHECK;
  build_alias<<<max_lambda,32,2*next_pow2(max_value)*sizeof(i32)>>>(pois_alias->prob, pois_alias->alias, max_value);
  cudaDeviceSynchronize() >> GPULDA_CHECK;
}

Poisson::~Poisson() {
  delete pois_alias;
}

}
