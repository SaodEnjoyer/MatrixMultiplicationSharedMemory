#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <chrono> 

#define BLOCK_SIZE 32  

// Ядро для GPU (матрица умножения)
__global__ void kernel_global(double* a, double* b, int n, double* c)
{
    int bx = blockIdx.x;  // Номер блока по X
    int by = blockIdx.y;  // Номер блока по Y
    int tx = threadIdx.x; // Номер нити в блоке по X
    int ty = threadIdx.y; // Номер нити в блоке по Y
    double sum = 0.0f;
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
    float timerValueGPU, timerValueCPU;

    // Создание переменных для хранения данных
    int numBytes = N * N * sizeof(double);
    double* adev, * bdev, * cdev, * a, * b, * c, * cc, * bT;

    // Выделение памяти на CPU (host)
    a = (double*)malloc(numBytes);  // Матрица A
    b = (double*)malloc(numBytes);  // Матрица B
    bT = (double*)malloc(numBytes); // Транспонированная матрица B
    c = (double*)malloc(numBytes);  // Матрица C для GPU
    cc = (double*)malloc(numBytes); // Матрица C для CPU

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
    kernel_global << <blocks, threads >> > (adev, bdev, N, cdev);

    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueGPU, start, stop);

    printf("\nGPU calculation time: %f msec\n", timerValueGPU);

    // Копирование матрицы C с device на host
    cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);

    // -------------------- CPU-вариант --------------------
    auto cpu_start = std::chrono::high_resolution_clock::now();

    // Вычисление матрицы C на CPU
    for (n = 0; n < N; n++)
    {
        for (m = 0; m < N; m++)
        {
            cc[m + n * N] = 0.f;
            for (k = 0; k < N; k++) cc[m + n * N] += a[k + n * N] * bT[k + m * N];
        }
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
    timerValueCPU = cpu_duration.count() * 1000;  // Перевод в миллисекунды

    printf("\nCPU calculation time: %f msec\n", timerValueCPU);
    printf("\nSpeedup: %f x\n", timerValueCPU / timerValueGPU);

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
