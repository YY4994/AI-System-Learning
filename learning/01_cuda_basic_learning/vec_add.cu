#include <stdio.h>
#include <vector>

using namespace std;

//CUDA for vector addition
__glaboal__ gpu_vec_add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void gpu_vec_add(const vector<float>& h_a, const vector<float>& h_b, vector<float>& h_c) {
    int n = a.size() < b.size() ? a.size() : b.size();
    c.resize(n);

    // Allocate memory on the device
    float *d_a, *d_b, *d_c;
    if(!cudaMalloc((void**)&d_a, n * sizeof(float)) ||
       !cudaMalloc((void**)&d_b, n * sizeof(float)) ||
       !cudaMalloc((void**)&d_c, n * sizeof(float))) {
        fprintf(stderr, "Failed to allocate device memory\n");
        return;
    }

    // Copy data from host to device
    if(!cudaMemcpy(d_a,h_a.data(),n*sizeof(float),cudaMemcpyHostToDevice))
    {
        fprintf(stderr, "Failed to copy vector A from host to device\n");
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        return;
    }

    if(!cudaMemcpy(d_b,h_b.data(),n*sizeof(float),cudaMemcpyHostToDevice))
    {
        fprintf(stderr, "Failed to copy vector B from host to device\n");
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        return;
    }

    // Launch the kernel and record time
    dim3 block_size(256,1,1);
    dim3 grid_size((n + block_size.x - 1) / block_size.x,1,1);
    cudaEvent_t start, stop;
    if(!cudaEventCreate(&start) || !cudaEventCreate(&stop)) {
        fprintf(stderr, "Failed to create CUDA events\n");
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        return;
    }
    cudaEventRecord(start, 0);
    gpu_vec_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU add finish time: %f ms\n", gpu_time);

    // Copy data from device to host
    if(!cudaMemcpy(h_c.data(),d_c,n*sizeof(float),cudaMemcpyDeviceToHost)){
        fprintf(stderr, "Failed to copy vector C from device to host\n");
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        return;       
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

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
        if (gpu_res[i] != cpu_res[i]) {
            printf("Results do not match at index %d: GPU result = %f, CPU result = %f\n", i, gpu_res[i], cpu_res[i]);
            return false;
        }
    }
    return true;
}

void main() {
    vector<int> sizes = {1000, 10000, 100000, 1000000, 10000000};
    for (int size : sizes) {
        printf("------- Vector Addition of size %d -------\n", size);
        //randomly initialize vectors
        vector<flaot> a(size), b(size), gpu_c(size), cpu_c(size);
        for (int i = 0; i < size; i++) {
            a[i] = static_cast<float>(rand()) / RAND_MAX;
            b[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        //call gpu and cpu functions
        gpu_add(a, b, gpu_c, size);
        cpu_add(a, b, cpu_c, size);

        //compare results
        if (compare_results(gpu_c, cpu_c, size)) {
            printf("Results match!\n");
        } else {
            printf("Results do not match!\n");
        }
    }
}