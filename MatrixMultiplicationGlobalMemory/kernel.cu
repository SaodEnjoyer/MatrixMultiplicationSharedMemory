#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define BLOCK_SIZE 32  

// Ядро для GPU (матрица умножения) для float
__global__ void kernel_global_float(float* a, float* b, int n, float* c)
{
    int bx = blockIdx.x;  // Номер блока по X
    int by = blockIdx.y;  // Номер блока по Y
    int tx = threadIdx.x; // Номер нити в блоке по X
    int ty = threadIdx.y; // Номер нити в блоке по Y
    float sum = 0.0f;
    int ia = n * (BLOCK_SIZE * by + ty);  // Номер строки из A
    int ib = BLOCK_SIZE * bx + tx;
    int ic = ia + ib;
    // Вычисление элемента матрицы C
    for (int k = 0; k < n; k++)
        sum += a[ia + k] * b[ib + k * n];
    c[ic] = sum;
}

// Ядро для GPU (матрица умножения) для double
__global__ void kernel_global_double(double* a, double* b, int n, double* c)
{
    int bx = blockIdx.x;  // Номер блока по X
    int by = blockIdx.y;  // Номер блока по Y
    int tx = threadIdx.x; // Номер нити в блоке по X
    int ty = threadIdx.y; // Номер нити в блоке по Y
    double sum = 0.0;
    int ia = n * (BLOCK_SIZE * by + ty);  // Номер строки из A
    int ib = BLOCK_SIZE * bx + tx;
    int ic = ia + ib;
    // Вычисление элемента матрицы C
    for (int k = 0; k < n; k++)
        sum += a[ia + k] * b[ib + k * n];
    c[ic] = sum;
}

int main()
{
    int N = 2048;
    int m, n, k;
    float timerValueGPU_float, timerValueGPU_double;

    // Создание переменных для хранения данных
    int numBytes = N * N * sizeof(float);
    float* adev, * bdev, * cdev, * a, * b, * c, * bT;
    double* adev_double, * bdev_double, * cdev_double, * a_double, * b_double, * c_double, * bT_double;

    // Выделение памяти на CPU (host)
    a = (float*)malloc(numBytes);  // Матрица A
    b = (float*)malloc(numBytes);  // Матрица B
    bT = (float*)malloc(numBytes); // Транспонированная матрица B
    c = (float*)malloc(numBytes);  // Матрица C для GPU

    a_double = (double*)malloc(N * N * sizeof(double));  // Матрица A для double
    b_double = (double*)malloc(N * N * sizeof(double));  // Матрица B для double
    bT_double = (double*)malloc(N * N * sizeof(double)); // Транспонированная матрица B для double
    c_double = (double*)malloc(N * N * sizeof(double));  // Матрица C для GPU (double)

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

    cudaMalloc((void**)&adev_double, N * N * sizeof(double));
    cudaMalloc((void**)&bdev_double, N * N * sizeof(double));
    cudaMalloc((void**)&cdev_double, N * N * sizeof(double));

    // ---------------- GPU-вариант для float ------------------------
    cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // Запуск функции-ядра для умножения матриц (для float)
    kernel_global_float << <blocks, threads >> > (adev, bdev, N, cdev);

    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueGPU_float, start, stop);

    printf("\nGPU calculation time for float: %f msec\n", timerValueGPU_float);

    cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);

    // ---------------- GPU-вариант для double ------------------------
    cudaMemcpy(adev_double, a_double, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(bdev_double, b_double, N * N * sizeof(double), cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);

    // Запуск функции-ядра для умножения матриц (для double)
    kernel_global_double << <blocks, threads >> > (adev_double, bdev_double, N, cdev_double);

    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueGPU_double, start, stop);

    printf("\nGPU calculation time for double: %f msec\n", timerValueGPU_double);

    cudaMemcpy(c_double, cdev_double, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    // Освобождение памяти на GPU и CPU
    cudaFree(adev);
    cudaFree(bdev);
    cudaFree(cdev);
    free(a);
    free(b);
    free(bT);
    free(c);

    cudaFree(adev_double);
    cudaFree(bdev_double);
    cudaFree(cdev_double);
    free(a_double);
    free(b_double);
    free(bT_double);
    free(c_double);

    // Уничтожение переменных-событий
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
