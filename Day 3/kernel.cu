#include <iostream>
#include <cuda_runtime.h>

// CUDA Kernel for Matrix-Vector Multiplication
__global__ void VectorMatrixMul(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {  // Corrected loop condition
            sum += A[j * N + i] * B[j]; // Multiplying matrix column with vector
        }
        C[i] = sum;
    }
}

int main() {
    // Define Matrix/Vector Size
    const int N = 3;
    float* A, * B, * C;

    // Allocate Host Memory
    A = (float*)malloc(N * N * sizeof(float));
    B = (float*)malloc(N * sizeof(float));
    C = (float*)malloc(N * sizeof(float));

    // Initialize Matrix A (All 1s) and Vector B (All 2s)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = 1.0f;
        }
        B[i] = 2.0f;
        C[i] = 0.0f;
    }

    // Allocate Device Memory
    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // Copy Data from Host to Device
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Configure Kernel Launch
    int blocksize = 256;
    int gridsize = (N + blocksize - 1) / blocksize;

    // Measure Execution Time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);  // Start Timing

    // Launch Kernel
    VectorMatrixMul << <gridsize, blocksize >> > (d_A, d_B, d_C, N);

    cudaEventRecord(stop);   // Stop Timing
    cudaEventSynchronize(stop);

    // Copy Result Back to Host
    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate Time Taken
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);



    // Print GPU Details
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU Used: %s\n", prop.name);
    // Print Execution Time
    printf("Kernel Execution Time: %.5f ms\n", milliseconds);
    printf("\n");

    // Print Matrix A
    printf("Matrix A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", A[i * N + j]);
        }
        printf("\n");
    }

    // Print Vector B
    printf("\nMatrix B:\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", B[i]);
    }
    printf("\n");

    // Print Result Vector C
    printf("\nResult Vector C (A * B):\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", C[i]);
    }
    printf("\n");



    // Free Allocated Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}
