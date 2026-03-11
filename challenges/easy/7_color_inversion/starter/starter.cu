#include <cuda_runtime.h>
#include <iostream>

__global__ void invert_kernel(unsigned char* image, int width, int height) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < width * height && idx % 4 != 3) {
        image[idx] = 255 - image[idx];
    }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
