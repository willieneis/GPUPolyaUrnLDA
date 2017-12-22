#include "dsmatrix.cuh"
#include "error.cuh"
#include "train.cuh"

namespace gpulda {

template<class T>
DSMatrix<T>::DSMatrix<T>() {
  cudaMalloc/*Pitch*/(&dense, args->K * args->V * sizeof(T)) >> GPULDA_CHECK;
}

template<class T>
DSMatrix<T>::~DSMatrix<T>() {
  cudaFree(dense) >> GPULDA_CHECK;
}

template class DSMatrix<f32>;
template class DSMatrix<u32>;

}
