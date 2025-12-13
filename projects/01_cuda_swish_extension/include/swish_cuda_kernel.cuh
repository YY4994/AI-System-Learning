#ifndef SWISH_CUDA_KERNEL_CUH
#define SWISH_CUDA_KERNEL_CUH

#include <cuda_fp16.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // 前向核函数声明
    void swish_forward_kernel_fp32(const float *x, float *y, float *saved_s, int64_t n, cudaStream_t stream);
    void swish_forward_kernel_fp16(const __half *x, __half *y, __half *saved_s, int64_t n, cudaStream_t stream);

    // 反向核函数声明
    void swish_backward_kernel_fp32(const float *grad_y, const float *x, const float *saved_s, float *grad_x, int64_t n, cudaStream_t stream);
    void swish_backward_kernel_fp16(const __half *grad_y, const __half *x, const __half *saved_s, __half *grad_x, int64_t n, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif