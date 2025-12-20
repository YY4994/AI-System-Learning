#include "swish_cuda_kernel.cuh"
template <typename T>
__global__ void swish_forward_kernel(const T *input, T *output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    T one = static_cast<T>(1.0); // 1.0f or 1.0
    T sigmod = one / (one + exp(-input[idx]));
    output[idx] = input[idx] * sigmod;
}
template <typename T>
__global__ void swish_backward_kernel(const T *grad_output, const T *input, T *grad_input, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    T one = static_cast<T>(1.0); // 1.0f or 1.0
    T sigmod = one / (one + exp(-input[idx]));
    grad_input[idx] = sigmod * (one + input[idx] * (one - sigmod)) * grad_output[idx];
}

template <typename T>
void swish_forward(const T *input, T *output, int n, cudaStream_t stream)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    swish_forward_kernel<T><<<numBlocks, blockSize, 0, stream>>>(input, output, n);
}

template <typename T>
void swish_backward(const T *grad_output, const T *input, T *grad_input, int n, cudaStream_t stream)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    swish_backward_kernel<T><<<numBlocks, blockSize, 0, stream>>>(grad_output, input, grad_input, n);
}

template void swish_forward<float>(const float *, float *, int, cudaStream_t);
template void swish_backward<float>(const float *, const float *, float *, int, cudaStream_t);

template void swish_forward<double>(const double *, double *, int, cudaStream_t);
template void swish_backward<double>(const double *, const double *, double *, int, cudaStream_t);