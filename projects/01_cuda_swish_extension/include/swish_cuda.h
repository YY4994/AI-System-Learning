// 第一行必须是这个！
#define TORCH_EXTENSION_NAME my_swish
#pragma once
#include <torch/extension.h>
#include "swish_cuda_kernel.cuh"

class MySwishFunction : public torch::autograd::Function<MySwishFunction>
{
public:
    // 注意：返回类型和方法签名需要精确
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        const torch::Tensor &input); // 添加const引用

    static torch::autograd::tensor_list backward( // 返回tensor_list，不是Tensor
        torch::autograd::AutogradContext *ctx,
        const torch::autograd::tensor_list &grad_outputs); // 添加const引用, tensor_list类型
};