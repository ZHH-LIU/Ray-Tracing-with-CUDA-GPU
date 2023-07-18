#ifndef CUDARAND_H
#define CUDARAND_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include <stdio.h>

__global__ void setup_kernel(curandState* state, unsigned long seed)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int index = idx + idy * blockDim.x * gridDim.x;
	curand_init(seed, index, 0, &state[index]);// initialize the state
}

class cudaRand
{
public:
	cudaRand()=default;
	cudaRand(int numThread, unsigned long _seed)
		:seed(_seed)
	{
		cudaMalloc(&devStates, numThread * sizeof(curandState));
		setup_kernel<<<grid,block>>>(devStates, seed);
	}
	curandState* devStates;
	unsigned long seed;

};

#endif 
