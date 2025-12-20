#ifndef SWISH_CUDA_KERNEL_CUH
#define SWISH_CUDA_KERNEL_CUH

#include <cuda.h>
#include <cuda_runtime.h>

// template <typename T>
// __global__ void swish_forward_kernel(const T *input, T *output, int n);
// template <typename T>
// __global__ void swish_backward_kernel(const T *grad_output,
//                                       const T *input,
//                                       // const T *saved_sigmod,
//                                       T *grad_input,
//                                       int n);

template <typename T>
void swish_forward(const T *input, T *output, int n, cudaStream_t stream);
template <typename T>
void swish_backward(const T *grad_output,
                    const T *input,
                    // const T *saved_sigmod,
                    T *grad_input,
                    int n,
                    cudaStream_t stream);
#endif // SWISH_CUDA_KERNEL_CUH