#ifndef REDUCTION_REDUCE_IDLE_THREADS_CUH
#define REDUCTION_REDUCE_IDLE_THREADS_CUH

#include <cuda_runtime.h>

/*
 * reduce idle threads
 */

#define REDUCTION_SEQUENTIAL_SIZE 16

__global__ void reductionReduceIdleThreads(float *v, float *v_r) {
  __shared__ float psum[REDUCTION_SEQUENTIAL_SIZE];

  int tid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  psum[threadIdx.x] = v[tid] + v[tid + blockDim.x];
  __syncthreads();

  for (int s = blockDim.x/2; s > 0; s >>= 1) { // iterate strides in block

    if (threadIdx.x < s) {
      psum[threadIdx.x] += psum[threadIdx.x + s];
    }
    __syncthreads(); // wait for this step to be done
  }

  if (threadIdx.x == 0) { // set the first thread in this block
    v_r[blockIdx.x] = psum[0];
  }
}

#endif
