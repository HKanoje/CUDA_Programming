#include <stdio.h>
#include <cuda_runtime.h>

//------------------------------------------//
// KERNEL 1: Per-block prefix scan
//------------------------------------------//
__global__ void blockScanKernel(const int* input, int* output, int* blockSums, int n)
{
    extern __shared__ int sdata[];  // dynamic shared memory

    int tid = threadIdx.x;
    int blockId = blockIdx.x;
    int offset = blockDim.x * blockId;  // each block processes "blockDim.x" elements
    int index = offset + tid;

    // 1) Load data into shared memory
    sdata[tid] = (index < n) ? input[index] : 0;
    __syncthreads();

    // 2) In-place inclusive scan in sdata
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int temp = 0;
        if (tid >= stride) {
            temp = sdata[tid - stride];
        }
        __syncthreads();
        sdata[tid] += temp;
        __syncthreads();
    }

    // 3) Write the scanned values back to output
    if (index < n) {
        output[index] = sdata[tid];
    }

    // 4) The last thread in this block writes the sum of this block to blockSums
    if (tid == blockDim.x - 1) {
        blockSums[blockId] = sdata[tid];
    }
}

//------------------------------------------//
// KERNEL 2: Scan the block-sums array
//------------------------------------------//
__global__ void blockSumScanKernel(int* blockSums, int numBlocks)
{
    extern __shared__ int sdata[];  // dynamic shared memory for block sums

    int tid = threadIdx.x;
    if (tid < numBlocks) {
        sdata[tid] = blockSums[tid];
    }
    else {
        sdata[tid] = 0;
    }
    __syncthreads();

    // Inclusive scan over blockSums
    for (int stride = 1; stride < numBlocks; stride *= 2) {
        int temp = 0;
        if (tid >= stride) {
            temp = sdata[tid - stride];
        }
        __syncthreads();
        sdata[tid] += temp;
        __syncthreads();
    }

    // Write back
    if (tid < numBlocks) {
        blockSums[tid] = sdata[tid];
    }
}

//------------------------------------------//
// KERNEL 3: Add block offsets to get the true prefix sum
//------------------------------------------//
__global__ void addBlockOffsetsKernel(int* output, const int* blockSums, int n)
{
    int blockId = blockIdx.x;
    if (blockId == 0) return;  // first block has no offset
    int offset = blockSums[blockId - 1];

    int tid = threadIdx.x;
    int index = blockDim.x * blockId + tid;
    if (index < n) {
        output[index] += offset;
    }
}

int main()
{


    // 1) Setup problem size
    const int N = 16;
    // We'll use 2 blocks, each processes 8 elements
    int blockSize = 8;
    int gridSize = N / blockSize; // 16/8=2

    // Host data
    int h_input[N];
    for (int i = 0; i < N; i++) {
        h_input[i] = i + 1; // 1..16
    }
    int h_output[N] = { 0 };

    // Device pointers
    int* d_input, * d_output;
    // Array to store block sums (size = gridSize)
    int* d_blockSums;

    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));
    cudaMalloc(&d_blockSums, gridSize * sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // 2) CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    //--------------------------------------
    // KERNEL 1: per-block prefix sum
    //--------------------------------------
    blockScanKernel << <gridSize, blockSize, blockSize * sizeof(int) >> > (
        d_input, d_output, d_blockSums, N
        );

    //--------------------------------------
    // KERNEL 2: scan the block sums
    //--------------------------------------
    // We'll just launch 1 block of "gridSize" threads
    blockSumScanKernel << <1, gridSize, gridSize * sizeof(int) >> > (d_blockSums, gridSize);

    //--------------------------------------
    // KERNEL 3: add offsets to each block
    //--------------------------------------
    addBlockOffsetsKernel << <gridSize, blockSize >> > (d_output, d_blockSums, N);

    // Record the stop event
    cudaEventRecord(stop);

    // Wait for all to finish
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy final prefix sums back
    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    // 0) Print which GPU we're using
    int deviceId = 0; // or pick another device if you have multiple GPUs
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    printf("Using GPU: %s\n", prop.name);
    cudaSetDevice(deviceId);
    // Print measured time
    printf("Kernel execution time %.3f ms\n", milliseconds);
 

    // Print the result
    printf("\nInput:  ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_input[i]);
    }
    printf("\n");
    printf("Output (true prefix sum): ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");



    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_blockSums);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
