#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define BLOCK_SIZE 32  

// Ядро для GPU (матрица умножения) для типа float
__global__ void kernel_smem_1_float(float* a, float* b, int n, float* c)
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
        as[tx][ty] = a[ia + n * ty + tx]; bs[tx][ty] = b[ib + n * ty + tx];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++) sum += as[k][ty] * bs[tx][k];
        __syncthreads();
    }
    c[aBegin + bBegin + ty * n + tx] = sum;
}

// Ядро для GPU (матрица умножения) для типа double
__global__ void kernel_smem_1_double(double* a, double* b, int n, double* c)
{
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int aBegin = n * BLOCK_SIZE * by, aEnd = aBegin + n - 1;
    int bBegin = BLOCK_SIZE * bx, aStep = BLOCK_SIZE, bStep = BLOCK_SIZE * n;
    double sum = 0.0;
    __shared__ double as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double bs[BLOCK_SIZE][BLOCK_SIZE];
    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep)
    {
        as[tx][ty] = a[ia + n * ty + tx]; bs[tx][ty] = b[ib + n * ty + tx];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++) sum += as[k][ty] * bs[tx][k];
        __syncthreads();
    }
    c[aBegin + bBegin + ty * n + tx] = sum;
}

int main()
{
    int N = 2048;
    int m, n, k;
    float timerValueGPU, timerValueCPU;
    float timerValueGPU_double, timerValueCPU_double;

    // Создание переменных для хранения данных
    int numBytes = N * N * sizeof(float);
    int numBytesDouble = N * N * sizeof(double);
    float* adev, * bdev, * cdev, * a, * b, * c, * cc, * bT;
    double* adev_double, * bdev_double, * cdev_double, * a_double, * b_double, * c_double, * cc_double, * bT_double;

    // Выделение памяти на CPU (host)
    a = (float*)malloc(numBytes);  // Матрица A
    b = (float*)malloc(numBytes);  // Матрица B
    bT = (float*)malloc(numBytes); // Транспонированная матрица B
    c = (float*)malloc(numBytes);  // Матрица C для GPU
    cc = (float*)malloc(numBytes); // Матрица C для CPU

    a_double = (double*)malloc(numBytesDouble);  // Матрица A (double)
    b_double = (double*)malloc(numBytesDouble);  // Матрица B (double)
    bT_double = (double*)malloc(numBytesDouble); // Транспонированная матрица B (double)
    c_double = (double*)malloc(numBytesDouble);  // Матрица C (double) для GPU
    cc_double = (double*)malloc(numBytesDouble); // Матрица C (double) для CPU

    // Заполнение матриц A, B и транспонированной матрицы B
    for (n = 0; n < N; n++)
    {
        for (m = 0; m < N; m++)
        {
            a[m + n * N] = 2.0f * m + n;
            b[m + n * N] = m - n;
            bT[m + n * N] = n - m;

            a_double[m + n * N] = 2.0 * m + n;
            b_double[m + n * N] = m - n;
            bT_double[m + n * N] = n - m;
        }
    }

    // Настройка сетки нитей и блоков
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(N / threads.x, N / threads.y);

    // Выделение памяти на GPU
    cudaMalloc((void**)&adev, numBytes);
    cudaMalloc((void**)&bdev, numBytes);
    cudaMalloc((void**)&cdev, numBytes);

    cudaMalloc((void**)&adev_double, numBytesDouble);
    cudaMalloc((void**)&bdev_double, numBytesDouble);
    cudaMalloc((void**)&cdev_double, numBytesDouble);

    // ---------------- GPU-вариант для float ------------------------
    // Копирование матриц A и B с host на device для float
    cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);

    // Таймер для GPU (float)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // Запуск функции-ядра для умножения матриц для float
    kernel_smem_1_float << <blocks, threads >> > (adev, bdev, N, cdev);

    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueGPU, start, stop);

    printf("\nGPU calculation time (float): %f msec\n", timerValueGPU);

    // Копирование матрицы C с device на host для float
    cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);

    // ---------------- GPU-вариант для double ------------------------
    // Копирование матриц A и B с host на device для double
    cudaMemcpy(adev_double, a_double, numBytesDouble, cudaMemcpyHostToDevice);
    cudaMemcpy(bdev_double, b_double, numBytesDouble, cudaMemcpyHostToDevice);

    // Таймер для GPU (double)
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // Запуск функции-ядра для умножения матриц для double
    kernel_smem_1_double << <blocks, threads >> > (adev_double, bdev_double, N, cdev_double);

    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueGPU_double, start, stop);

    printf("\nGPU calculation time (double): %f msec\n", timerValueGPU_double);

    // Копирование матрицы C с device на host для double
    cudaMemcpy(c_double, cdev_double, numBytesDouble, cudaMemcpyDeviceToHost);

   

    // Освобождение памяти на GPU и CPU
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
    free(bT_double);
    free(c_double);
    free(cc_double);

    // Уничтожение переменных-событий
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
