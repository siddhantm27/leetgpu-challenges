#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    int row = blockDim.y*blockIdx.y + threadIdx.y;

    if (row < height && col < width){
        image[row*width + col] = 255 - image[row*width + col];
    }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
