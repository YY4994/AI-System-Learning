#ifndef SWISH_CUDA_LAUNCHER_H
#define SWISH_CUDA_LAUNCHER_H

#include <ATen/ATen.h>

class SwishCudaLauncher
{
public:
    // 禁用拷贝和赋值
    SwishCudaLauncher(const SwishCudaLauncher &) = delete;
    SwishCudaLauncher &operator=(const SwishCudaLauncher &) = delete;

    // 静态方法，无需实例化即可调用
    static void forward(at::Tensor &x, at::Tensor &y, at::Tensor &saved_s);
    static void backward(at::Tensor &grad_y, at::Tensor &x, at::Tensor &saved_s, at::Tensor &grad_x);

private:
    // 辅助函数：检查Tensor是否满足条件（设备、形状、类型）
    static void check_tensor_conditions(const at::Tensor &t, const std::string &name);
};

#endif