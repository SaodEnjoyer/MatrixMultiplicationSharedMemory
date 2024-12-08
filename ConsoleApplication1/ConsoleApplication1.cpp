#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define N 2048  // Размер матрицы

int main()
{
    int m, n, k;
    float timerValueCPU;

    // Создание переменных для хранения данных
    int numBytes = N * N * sizeof(float);
    float* a, * b, * c, * cc, * bT;

    // Выделение памяти на CPU (host)
    a = (float*)malloc(numBytes);  // Матрица A
    b = (float*)malloc(numBytes);  // Матрица B
    bT = (float*)malloc(numBytes); // Транспонированная матрица B
    c = (float*)malloc(numBytes);  // Матрица C для CPU
    cc = (float*)malloc(numBytes); // Матрица C для хранения результата

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

    // -------------------- CPU-вариант с типом float --------------------
    auto cpu_start = std::chrono::high_resolution_clock::now();

    // Вычисление матрицы C на CPU
    for (n = 0; n < N; n++)
    {
        for (m = 0; m < N; m++)
        {
            cc[m + n * N] = 0.0f;
            for (k = 0; k < N; k++)
                cc[m + n * N] += a[k + n * N] * bT[k + m * N];
        }
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> cpu_duration = cpu_end - cpu_start;
    timerValueCPU = cpu_duration.count() * 1000;  // Перевод в миллисекунды

    printf("\nCPU calculation time (float): %f msec\n", timerValueCPU);

    // Освобождение памяти на CPU
    free(a);
    free(b);
    free(bT);
    free(c);
    free(cc);

    return 0;
}
