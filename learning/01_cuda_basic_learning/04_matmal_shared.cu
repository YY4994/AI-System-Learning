#include <stdio.h>
#include <vector>
#include <random>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

// const int TILE_SIZE = 16; //tile太小内存效率低

__global__ void matmal_shared_kernel(float *a, float *b, float *c, int n)
{
    __shared__ float ds_a[TILE_SIZE][TILE_SIZE];
    __shared__ float ds_b[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float Cvalue = 0;

    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t)
    {
        int A_tile_col = t * TILE_SIZE + tx;
        int B_tile_row = t * TILE_SIZE + ty;

        if (row < n && A_tile_col < n)
        {
            ds_a[ty][tx] = a[row * n + A_tile_col];
        }
        else
        {
            ds_a[ty][tx] = 0;
        }

        if (col < n && B_tile_row < n)
        {
            ds_b[ty][tx] = b[B_tile_row * n + col];
        }
        else
        {
            ds_b[ty][tx] = 0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
        {
            Cvalue += ds_a[ty][k] * ds_b[k][tx];
        }
        __syncthreads();
    }

    if (row < n && col < n)
    {
        c[row * n + col] = Cvalue;
    }

    return;
}

int main()
{
    // Allocate memory on the host
     // const vector<int> matrix_sizes = {256 , 512 , 1024 , 2048 }; //矩阵太小，测不出时间差异
    const vector<int> matrix_sizes = {2048, 4096, 8192, 16384};
    for (int N : matrix_sizes)
    {
        vector<float> h_A(N * N);
        vector<float> h_B(N * N);
        vector<float> h_C(N * N);

        // Initialize matrices with random values
        // seed = chrono::system_clock::now().time_since_epoch().count();
        // default_random_engine generator(seed);
        // uniform_real_distribution<float> distribution(0.0, 10.0);
        // for (int i = 0; i < N * N; i++)
        // {
        //     h_A[i] = distribution(generator);
        //     h_B[i] = distribution(generator);
        // }
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(0.0, 10.0);
        for (int i = 0; i < N * N; i++)
        {
            h_A[i] = dis(gen);
            h_B[i] = dis(gen);
        }

        // Allocate memory on the device
        float *d_A, *d_B, *d_C;
        cudaMalloc((void **)&d_A, N * N * sizeof(float));
        cudaMalloc((void **)&d_B, N * N * sizeof(float));
        cudaMalloc((void **)&d_C, N * N * sizeof(float));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Copy data from host to device
        cudaEventRecord(start);
        if (cudaMemcpy(d_A, h_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy matrix A from host to device\n");
            return -1;
        }
        if (cudaMemcpy(d_B, h_B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy matrix B from host to device\n");
            return -1;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float cpu2gpu_milliseconds = 0;
        cudaEventElapsedTime(&cpu2gpu_milliseconds, start, stop);

        // Launch the kernel
        cudaEventRecord(start);
        dim3 blockSize(TILE_SIZE, TILE_SIZE, 1);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y, 1);
        matmal_shared_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float gpu_time = 0.0f;
        cudaEventElapsedTime(&gpu_time, start, stop);

        // Copy the result back to host
        cudaEventRecord(start);
        if (cudaMemcpy(h_C.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy matrix C from device to host\n");
            return -1;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float gpu2cpu_milliseconds = 0;
        cudaEventElapsedTime(&gpu2cpu_milliseconds, start, stop);

        printf("Time to copy data from CPU to GPU: %f ms\n", cpu2gpu_milliseconds);
        printf("Time to execute kernel on GPU: %f ms\n", gpu_time);
        printf("Time to copy data from GPU to CPU: %f ms\n", gpu2cpu_milliseconds);
        printf("Total time: %f ms\n", cpu2gpu_milliseconds + gpu_time + gpu2cpu_milliseconds);

        // Free device memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    return 0;
}