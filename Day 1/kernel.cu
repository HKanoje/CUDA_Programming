#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 10;
    float A[N], B[N], C[N];

    // Initialize arrays
    for (int i = 0; i < N; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(i * 2);
    }

    float* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    cudaMemcpy(d_a, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, N * sizeof(float), cudaMemcpyHostToDevice);

    int blocksize = 256;
    int gridsize = (N + blocksize - 1) / blocksize; // Ensure proper grid size

    // Get GPU Name
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Assuming the first GPU (device 0) is used
    std::cout << "Running on GPU: " << prop.name << std::endl;

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    vectorAdd << <gridsize, blocksize >> > (d_a, d_b, d_c, N);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(C, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Print result
    for (int i = 0; i < N; i++) {
        std::cout << "C[" << i << "] = " << C[i] << std::endl;
    }

    // Print execution time
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    return 0;
}
