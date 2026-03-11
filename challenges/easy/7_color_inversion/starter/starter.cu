#include <cuda_runtime.h>
#include <iostream>

__global__ void invert_kernel(float* image, int width, int height) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < width * height && idx % 4 != 3) {
        image[idx] = 255.0f - image[idx];
    }
}

extern "C" void solve(float* image, int width, int height) {

    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}