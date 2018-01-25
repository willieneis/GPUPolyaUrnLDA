#include <stdlib.h>
#include <iostream>
#include "test_hashmap.cuh"
#include "test_polyaurn.cuh"
#include "test_spalias.cuh"
#include "test_topics.cuh"

namespace gpulda_test {

void run_tests() {
  std::cout << "running tests" << std::endl;

  std::cout << "testing hashmap" << std::endl;
  test_hashmap();

  std::cout << "testing polya_urn_init" << std::endl;
  test_polya_urn_init();
  std::cout << "testing polya_urn_sample" << std::endl;
  test_polya_urn_sample();
  std::cout << "testing polya_urn_transpose" << std::endl;
  test_polya_urn_transpose();
  std::cout << "testing polya_urn_reset" << std::endl;
  test_polya_urn_reset();
  std::cout << "testing polya_urn_colsums" << std::endl;
  test_polya_urn_colsums();

  std::cout << "testing build_alias" << std::endl;
  test_build_alias();

  std::cout << "testing compute_d_idx" << std::endl;
  test_compute_d_idx();
  std::cout << "testing sample_topics_functions" << std::endl;
  test_sample_topics_functions();
  std::cout << "testing sample_topics" << std::endl;
  test_sample_topics();

  std::cout << "tests completed" << std::endl;
}

}
