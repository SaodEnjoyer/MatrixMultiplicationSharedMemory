#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define BLOCK_SIZE 32  

// GPU kernel for float matrix multiplication
__global__ void kernel_smem_float(float* a, float* b, int n, float* c) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int aBegin = n * BLOCK_SIZE * by, aEnd = aBegin + n - 1;
    int bBegin = BLOCK_SIZE * bx, aStep = BLOCK_SIZE, bStep = BLOCK_SIZE * n;
    float sum = 0.0f;
    __shared__ float as[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float bs[BLOCK_SIZE][BLOCK_SIZE + 1];
    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep) {
        as[tx][ty] = a[ia + n * ty + tx];
        bs[tx][ty] = b[ib + n * ty + tx];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++) sum += as[k][ty] * bs[tx][k];
        __syncthreads();
    }
    c[aBegin + bBegin + ty * n + tx] = sum;
}

// GPU kernel for double matrix multiplication
__global__ void kernel_smem_double(double* a, double* b, int n, double* c) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int aBegin = n * BLOCK_SIZE * by, aEnd = aBegin + n - 1;
    int bBegin = BLOCK_SIZE * bx, aStep = BLOCK_SIZE, bStep = BLOCK_SIZE * n;
    double sum = 0.0;
    __shared__ double as[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ double bs[BLOCK_SIZE][BLOCK_SIZE + 1];
    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep) {
        as[tx][ty] = a[ia + n * ty + tx];
        bs[tx][ty] = b[ib + n * ty + tx];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++) sum += as[k][ty] * bs[tx][k];
        __syncthreads();
    }
    c[aBegin + bBegin + ty * n + tx] = sum;
}

int main() {
    int N = 2048;
    int m, n, k;
    float timerValueGPU_float, timerValueCPU_float;
    float timerValueGPU_double, timerValueCPU_double;

    // Allocate memory for float type data
    int numBytes = N * N * sizeof(float);
    float* adev, * bdev, * cdev, * a, * b, * c, * cc, * bT;

    // Allocate memory for double type data
    int numBytes_double = N * N * sizeof(double);
    double* adev_double, * bdev_double, * cdev_double, * a_double, * b_double, * c_double, * cc_double;

    // Allocate memory on CPU (host)
    a = (float*)malloc(numBytes);
    b = (float*)malloc(numBytes);
    bT = (float*)malloc(numBytes);
    c = (float*)malloc(numBytes);
    cc = (float*)malloc(numBytes);
    a_double = (double*)malloc(numBytes_double);
    b_double = (double*)malloc(numBytes_double);
    c_double = (double*)malloc(numBytes_double);
    cc_double = (double*)malloc(numBytes_double);

    // Initialize matrices A, B, and transpose of B
    for (n = 0; n < N; n++) {
        for (m = 0; m < N; m++) {
            a[m + n * N] = 2.0f * m + n;
            b[m + n * N] = m - n;
            bT[m + n * N] = n - m;
            a_double[m + n * N] = 2.0 * m + n;
            b_double[m + n * N] = m - n;
        }
    }

    // Set up grid and block sizes for CUDA
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(N / threads.x, N / threads.y);

    // Allocate memory on GPU for float type
    cudaMalloc((void**)&adev, numBytes);
    cudaMalloc((void**)&bdev, numBytes);
    cudaMalloc((void**)&cdev, numBytes);

    // Allocate memory on GPU for double type
    cudaMalloc((void**)&adev_double, numBytes_double);
    cudaMalloc((void**)&bdev_double, numBytes_double);
    cudaMalloc((void**)&cdev_double, numBytes_double);

    // ---------------- GPU calculation for float -------------------
    // Copy matrices A and B to GPU (float)
    cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Run the kernel for matrix multiplication (float)
    kernel_smem_float << <blocks, threads >> > (adev, bdev, N, cdev);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueGPU_float, start, stop);
    printf("\nGPU calculation time for float: %f msec\n", timerValueGPU_float);

    // Copy the result back to host (float)
    cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);

    // ---------------- GPU calculation for double -------------------
    // Copy matrices A and B to GPU (double)
    cudaMemcpy(adev_double, a_double, numBytes_double, cudaMemcpyHostToDevice);
    cudaMemcpy(bdev_double, b_double, numBytes_double, cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);

    // Run the kernel for matrix multiplication (double)
    kernel_smem_double << <blocks, threads >> > (adev_double, bdev_double, N, cdev_double);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueGPU_double, start, stop);
    printf("\nGPU calculation time for double: %f msec\n", timerValueGPU_double);

    // Copy the result back to host (double)
    cudaMemcpy(c_double, cdev_double, numBytes_double, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(adev);
    cudaFree(bdev);
    cudaFree(cdev);
    cudaFree(adev_double);
    cudaFree(bdev_double);
    cudaFree(cdev_double);
    free(a);
    free(b);
    free(bT);
    free(c);
    free(cc);
    free(a_double);
    free(b_double);
    free(c_double);
    free(cc_double);

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
