#include <iostream>
#include <cuda_runtime.h>
#include <chrono> // For measuring computation time

// Kernel function to add matrices (each thread computes one element of the matrix)
__global__ void MatrixAdd_B(const float* A, const float* B, float* C, int N) {
    // Compute global row and column indices
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Row index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Column index

    // Boundary check to ensure threads don't access out-of-bound memory
    if (i < N && j < N) {
        C[i * N + j] = A[i * N + j] + B[i * N + j];
    }
}

int main() {
    const int N = 3; // Matrix size (N x N)

    float* A, * B, * C; // Host matrices

    // Allocate memory for matrices on the host
    A = (float*)malloc(N * N * sizeof(float));
    B = (float*)malloc(N * N * sizeof(float));
    C = (float*)malloc(N * N * sizeof(float));

    // Initialize matrices A and B with specific values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = i + j; // Example values: sum of row and column indices
            B[i * N + j] = (i + 1) * (j + 1); // Example values: product of row+1 and column+1
            C[i * N + j] = 0.0f; // Initialize matrix C to 0
        }
    }

    float* d_a, * d_b, * d_c; // Device matrices

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_a, N * N * sizeof(float));
    cudaMalloc((void**)&d_b, N * N * sizeof(float));
    cudaMalloc((void**)&d_c, N * N * sizeof(float));

    // Copy matrices A and B from host to device
    cudaMemcpy(d_a, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Configure grid and block dimensions
    dim3 dimBlock(16, 16); // 16x16 threads per block
    dim3 dimGrid((N + 15) / 16, (N + 15) / 16); // Grid size to cover the matrix

    // Measure computation time
    auto start = std::chrono::high_resolution_clock::now();

    // Launch the kernel
    MatrixAdd_B << <dimGrid, dimBlock >> > (d_a, d_b, d_c, N);

    // Synchronize to ensure kernel execution is complete
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;

    // Copy the result matrix C from device to host
    cudaMemcpy(C, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Display which GPU is being used
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    printf("Using GPU: %s\n", prop.name);

    // Print computation time
    printf("Computation Time: %.2f ms\n\n", duration.count());

    // Print the matrices
    printf("Matrix A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", A[i * N + j]);
        }
        printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", B[i * N + j]);
        }
        printf("\n");
    }

    printf("\nMatrix C (Result of A + B):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", C[i * N + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(A);
    free(B);
    free(C);

    return 0;
}