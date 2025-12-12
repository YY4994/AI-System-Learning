#include <stdio.h>
#include <vector>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

//CUDA for vector addition
__global__ void gpu_vec_add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void gpu_vec_add(const vector<float>& h_a, const vector<float>& h_b, vector<float>& h_c) {
    int n = h_a.size() < h_b.size() ? h_a.size() : h_b.size();
    h_c.resize(n);

    // Allocate memory on the device
    float *d_a, *d_b, *d_c;
    if(cudaMalloc((void**)&d_a, n * sizeof(float)) != cudaSuccess ||
       cudaMalloc((void**)&d_b, n * sizeof(float)) != cudaSuccess ||
       cudaMalloc((void**)&d_c, n * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory\n");
        return;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    // Copy data from host to device
    if(cudaMemcpy(d_a,h_a.data(),n*sizeof(float),cudaMemcpyHostToDevice)!= cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device\n");
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        return;
    }
    if(cudaMemcpy(d_b,h_b.data(),n*sizeof(float),cudaMemcpyHostToDevice)!= cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device\n");
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        return;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float cpugpu_milliseconds = 0;
    cudaEventElapsedTime(&cpugpu_milliseconds, start, stop);

    // Launch the kernel and record time
    dim3 block_size(256,1,1);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,1,1);
    cudaEventRecord(start, 0);
    gpu_vec_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaEventRecord(start, 0);
    // Copy data from device to host
    if(cudaMemcpy(h_c.data(),d_c,n*sizeof(float),cudaMemcpyDeviceToHost)!= cudaSuccess){
        fprintf(stderr, "Failed to copy vector C from device to host\n");
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        return;       
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpucpu_milliseconds = 0;
    cudaEventElapsedTime(&gpucpu_milliseconds, start, stop);

    // Print the results
    printf("total time: %f ms\n", cpugpu_milliseconds + gpu_time + gpucpu_milliseconds);
    printf("CPU-GPU time: %f ms\n", cpugpu_milliseconds);
    printf("GPU time: %f ms\n", gpu_time);
    printf("GPU-CPU time: %f ms\n", gpucpu_milliseconds);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return;
}

//CPU for vector addition
void cpu_vec_add(const vector<float>& a, const vector<float>& b, vector<float>& c) {
    int n = a.size() < b.size() ? a.size() : b.size();
    c.resize(n);

    clock_t start = clock();
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
    clock_t end = clock();
    double cpu_time = double(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU add finish time: %f ms\n", cpu_time);
}

//compare results
bool compare_results(const vector<float>& gpu_res, const vector<float>& cpu_res, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(gpu_res[i] - cpu_res[i]) > 1e-5) {
            printf("Results do not match at index %d: GPU result = %f, CPU result = %f\n", i, gpu_res[i], cpu_res[i]);
            return false;
        }
    }
    return true;
}

int main() {
    vector<int> sizes = {1000, 10000, 100000, 1000000, 10000000};
    for (int size : sizes) {
        printf("------- Vector Addition of size %d -------\n", size);
        //randomly initialize vectors
        vector<float> a(size), b(size), gpu_c(size), cpu_c(size);
        for (int i = 0; i < size; i++) {
            a[i] = static_cast<float>(rand()) / RAND_MAX;
            b[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        //call gpu and cpu functions
        cpu_vec_add(a, b, cpu_c);
        gpu_vec_add(a, b, gpu_c);

        //compare results
        if (compare_results(gpu_c, cpu_c, size)) {
            printf("Results match!\n");
        } else {
            printf("Results do not match!\n");
        }
    }

    return 0;
}