#include "swish_cuda.h"
#include <cuda.h>
#include <ATen/cuda/CUDAContext.h> // 添加这个头文件以使用getCurrentCUDAStream

torch::Tensor MySwishFunction::forward(torch::autograd::AutogradContext *ctx, const torch::Tensor &input)
{
    // 参数验证
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    // 确保输入tensor是连续的
    auto input_contiguous = input.is_contiguous() ? input : input.contiguous();
    // 资源分配阶段：
    // 1.根据输入tensor的形状和类型分配输出tensor
    auto output = torch::empty_like(input);
    // 2.计算数据总量和CUDA配置
    int64_t total_elements = input.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // 调用CUDA核函数
    switch (input.scalar_type())
    {
    case torch::kFloat32:
        swish_forward<float>(input_contiguous.data_ptr<float>(), output.data_ptr<float>(), total_elements, stream);
        break;
    case torch::kFloat64:
        swish_forward<double>(input_contiguous.data_ptr<double>(), output.data_ptr<double>(), total_elements, stream);
        break;
    default:
        TORCH_CHECK(false, "my_swish only supports float32 and float64");
    }
    // 检查核函数是否调用成功
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // 保存上下文以备反向传播使用
    ctx->save_for_backward({input_contiguous});
    return output;
}

torch::autograd::tensor_list MySwishFunction::backward(torch::autograd::AutogradContext *ctx, const torch::autograd::tensor_list &grad_outputs)
{
    // 获取保存的输入tensor
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto grad_output = grad_outputs[0];
    // 参数验证
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    //确保tensor是连续的
    auto grad_output_contiguous = grad_output.is_contiguous() 
        ? grad_output 
        : grad_output.contiguous();
    
    // 同样检查input的连续性（前向应该已保证，但这里双重检查）
    auto input_contiguous = input.is_contiguous()
        ? input
        : input.contiguous();
    // 资源分配阶段：
    // 1.根据输入tensor的形状和类型分配输出tensor
    auto grad_input = torch::empty_like(input);
    // 2.计算数据总量和CUDA配置
    int64_t total_elements = input.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // 调用CUDA核函数
    switch (input.scalar_type())
    {
    case torch::kFloat32:
        swish_backward<float>(grad_output_contiguous.data_ptr<float>(), input.data_ptr<float>(), grad_input.data_ptr<float>(), total_elements, stream);
        break;
    case torch::kFloat64:
        swish_backward<double>(grad_output_contiguous.data_ptr<double>(), input.data_ptr<double>(), grad_input.data_ptr<double>(), total_elements, stream);
        break;
    default:
        TORCH_CHECK(false, "my_swish only supports float32 and float64");
    }
    // 检查核函数是否调用成功
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_input};
}

torch::Tensor my_swish(const torch::Tensor &input)
{
    return MySwishFunction::apply(input);
}

// TORCH_LIBRARY(my_swish, m)
// {
//     m.def("my_swish", &my_swish); // 注册独立函数
// }

// 旧API - 简单明了
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_swish", &my_swish);
}