#include <stdio.h>

__global__ void helloWorldKernel()
{
    printf("Hello World from GPU %d in block %d!\n", threadIdx.x, blockIdx.x);
}
int main()
{
    dim3 grid_size(2, 1, 1);
    dim3 block_size(256, 1, 1);
    helloWorldKernel<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();
    return 0;
}