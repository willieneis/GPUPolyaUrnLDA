#include <stdlib.h>
#include <iostream>
#include "types.cuh"
#include "train.cuh"
#include "test/run_tests.cuh"


using gpulda::f32;
using gpulda::i32;
using gpulda::u32;
using gpulda::u64;

int main(void) {
  gpulda_test::run_tests();

  constexpr f32 alpha = 0.1f;
  constexpr f32 beta = 0.1f;
  constexpr u32 V = 5;
  constexpr u32 K = 10;
  u32 C[V] = {1,1,1,1,1};
  constexpr u32 buffer_size = 5;
  constexpr u32 max_D = 2;
  constexpr u32 hashmap_size = 96;
  constexpr u32 max_N_d = hashmap_size;

  gpulda::Args args = {alpha,beta,K,V,C,buffer_size,max_D,max_N_d};
  u32 z[buffer_size] = {0,0,0,0,0};
  u32 w[buffer_size] = {0,1,2,3,4};
  u32 d[max_D] = {3,2};
  u32 K_d[max_D] = {1,1};
  u32 n_docs = max_D;
  gpulda::Buffer buffer = {z, w, d, K_d, n_docs, NULL, NULL, NULL, NULL, NULL, NULL, NULL};

  std::cout << "initializing" << std::endl;
  gpulda::initialize(&args, &buffer, 1);

  std::cout << "launching poisson polya urn sampler" << std::endl;
  gpulda::sample_phi();

  std::cout << "launching topic indicator sampler" << std::endl;
  gpulda::sample_z_async(&buffer);
  gpulda::sync_buffer(&buffer);

  std::cout << "cleanup" << std::endl;
  gpulda::cleanup(&buffer, 1);

  std::cout << "finished" << std::endl;
  return 0;
}
