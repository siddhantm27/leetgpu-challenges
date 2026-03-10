#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
        int idx = blockDim.x*blockIdx.x + threadIdx.x;
        int idy = blockDim.y*blockIdx.y + threadIdx.y;
        if (idy < M && idx < K)
        {
            C[idy*M + idx] = 0;
            for (int i = 0;i<N;i++){
                    C[idy*K + idx] += A[idy*N + i]*B[i*K + idx];
            }
        }
        
    }

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
