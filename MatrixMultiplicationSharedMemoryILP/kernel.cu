#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <chrono> 

#define BLOCK_SIZE 32  

// Ядро для GPU (матрица умножения)
__global__ void kernel_smem_3(float* a, float* b, int n, float* c)
{
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int aBegin = n * BLOCK_SIZE * by, aEnd = aBegin + n - 1;
    int bBegin = BLOCK_SIZE * bx, aStep = BLOCK_SIZE, bStep = BLOCK_SIZE * n;
    float sum = 0.0f;
    __shared__ float as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float bs[BLOCK_SIZE][BLOCK_SIZE];
    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep)
    {
        as[ty][tx] = a[ia + n * ty + tx];
        bs[ty][tx] = b[ib + n * ty + tx];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++) sum += as[ty][k] * bs[k][tx];
        __syncthreads();
    }
    c[aBegin + bBegin + ty * n + tx] = sum;
}

// Ядро для GPU (матрица умножения)
__global__ void kernel_smem_4(float* a, float* b, int n, float* c)
{
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int aBegin = n * BLOCK_SIZE * by, aEnd = aBegin + n - 1;
    int bBegin = BLOCK_SIZE * bx, aStep = BLOCK_SIZE, bStep = BLOCK_SIZE * n;
    float sum1 = 0.0f, sum2 = 0.0f;
    __shared__ float as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float bs[BLOCK_SIZE][BLOCK_SIZE];
    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep)
    {
        as[ty][tx] = a[ia + n * ty + tx];
        bs[ty][tx] = b[ib + n * ty + tx];
        as[ty + 16][tx] = a[ia + n * (ty + 16) + tx];
        bs[ty + 16][tx] = b[ib + n * (ty + 16) + tx];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum1 += as[ty][k] * bs[k][tx];
            sum2 += as[ty + 16][k] * bs[k][tx];
        }

        __syncthreads();
    }
    c[aBegin + bBegin + ty * n + tx] = sum1;
    c[aBegin + bBegin + (ty + 16) * n + tx] = sum2;
}

// Ядро для GPU (матрица умножения)
__global__ void kernel_smem_5(float* a, float* b, int n, float* c)
{
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int aBegin = n * BLOCK_SIZE * by, aEnd = aBegin + n - 1;
    int bBegin = BLOCK_SIZE * bx, aStep = BLOCK_SIZE, bStep = BLOCK_SIZE * n;
    float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;
    __shared__ float as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float bs[BLOCK_SIZE][BLOCK_SIZE];
    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep)
    {
        as[ty][tx] = a[ia + n * ty + tx];
        bs[ty][tx] = b[ib + n * ty + tx];
        as[ty + 8][tx] = a[ia + n * (ty + 8) + tx];
        bs[ty + 8][tx] = b[ib + n * (ty + 8) + tx];
        as[ty + 16][tx] = a[ia + n * (ty + 16) + tx];
        bs[ty + 16][tx] = b[ib + n * (ty + 16) + tx];
        as[ty + 24][tx] = a[ia + n * (ty + 24) + tx];
        bs[ty + 24][tx] = b[ib + n * (ty + 24) + tx];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum1 += as[ty][k] * bs[k][tx];
            sum2 += as[ty + 8][k] * bs[k][tx];
            sum3 += as[ty + 16][k] * bs[k][tx];
            sum4 += as[ty + 24][k] * bs[k][tx];
        }

        __syncthreads();
    }
    c[aBegin + bBegin + ty * n + tx] = sum1;
    c[aBegin + bBegin + (ty + 8) * n + tx] = sum2;
    c[aBegin + bBegin + (ty + 16) * n + tx] = sum3;
    c[aBegin + bBegin + (ty + 24) * n + tx] = sum4;
}

int main()
{
    int N = 2048;
    int m, n, k;
    float timerValueGPU, timerValueCPU;

    // Создание переменных для хранения данных
    int numBytes = N * N * sizeof(float);
    float* adev, * bdev, * cdev, * a, * b, * c, * cc, * bT;

    // Выделение памяти на CPU (host)
    a = (float*)malloc(numBytes);  // Матрица A
    b = (float*)malloc(numBytes);  // Матрица B
    bT = (float*)malloc(numBytes); // Транспонированная матрица B
    c = (float*)malloc(numBytes);  // Матрица C для GPU
    cc = (float*)malloc(numBytes); // Матрица C для CPU

    // Заполнение матриц A, B и транспонированной матрицы B
    for (n = 0; n < N; n++)
    {
        for (m = 0; m < N; m++)
        {
            a[m + n * N] = 2.0f * m + n;
            b[m + n * N] = m - n;
            bT[m + n * N] = n - m;
        }
    }

    // Настройка сетки нитей и блоков
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(N / threads.x, N / threads.y);

    // Выделение памяти на GPU
    cudaMalloc((void**)&adev, numBytes);
    cudaMalloc((void**)&bdev, numBytes);
    cudaMalloc((void**)&cdev, numBytes);

    dim3 threads_4(BLOCK_SIZE, BLOCK_SIZE / 2);

    dim3 threads_5(BLOCK_SIZE, BLOCK_SIZE / 4);

    // ---------------- GPU-вариант ------------------------
    // Копирование матриц A и B с host на device
    cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);

    // Таймер для GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // Запуск функции-ядра для умножения матриц
    //kernel_smem_3 << <blocks, threads >> > (adev, bdev, N, cdev);
    //kernel_smem_4 << < blocks, threads_4 >> > (adev, bdev, N, cdev);
    kernel_smem_5 << < blocks, threads_5 >> > (adev, bdev, N, cdev);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueGPU, start, stop);

    printf("\nGPU calculation time: %f msec\n", timerValueGPU);

    // Копирование матрицы C с device на host
    cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);

    // Освобождение памяти на GPU и CPU
    cudaFree(adev);
    cudaFree(bdev);
    cudaFree(cdev);
    free(a);
    free(b);
    free(bT);
    free(c);
    free(cc);

    // Уничтожение переменных-событий
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
