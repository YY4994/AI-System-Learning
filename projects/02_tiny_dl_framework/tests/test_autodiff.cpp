#include <iostream>
#include <vector>
#include "tensor.hpp"
#include "function.hpp"

using namespace tiny_dl;

int main()
{
    std::cout << "=== 自动求导核心功能测试 ===\n"
              << std::endl;

    try
    {
        // ===== 测试1: 基础加法求导 =====
        std::cout << "测试1: 基础加法求导" << std::endl;
        {
            // 创建需要梯度的张量
            Tensor<float, CPU> a(float(2.0), {2, 2}, true);
            Tensor<float, CPU> b(float(3.0), {2, 2}, true);

            // 前向传播
            auto c = a + b;

            std::cout << "c = a + b 前向传播完成" << std::endl;
            std::cout << "c.shape: (";
            for (auto dim : c.shape())
                std::cout << dim << " ";
            std::cout << ")" << std::endl;

            // 反向传播
            c.backward();

            // 检查梯度
            auto grad_a = a.grad();
            auto grad_b = b.grad();

            std::cout << "梯度检查:" << std::endl;
            std::cout << "grad_a 是否有效: " << (grad_a.data() != nullptr ? "是" : "否") << std::endl;
            std::cout << "grad_b 是否有效: " << (grad_b.data() != nullptr ? "是" : "否") << std::endl;

            if (grad_a.data() && grad_b.data())
            {
                bool correct = true;
                for (int i = 0; i < 4; ++i)
                {
                    if (std::abs(grad_a.data()[i] - 1.0f) > 1e-6 ||
                        std::abs(grad_b.data()[i] - 1.0f) > 1e-6)
                    {
                        correct = false;
                        break;
                    }
                }
                std::cout << "梯度计算: " << (correct ? "正确 ✓" : "错误 ✗") << std::endl;
            }
            else
            {
                std::cout << "梯度计算: 错误 ✗ (梯度为空)" << std::endl;
            }
        }

        std::cout << "\n"
                  << std::endl;

        // ===== 测试2: 乘法求导 =====
        std::cout << "测试2: 乘法求导" << std::endl;
        {
            Tensor<float, CPU> a(float(2.0), {2, 2}, true);
            Tensor<float, CPU> b(float(3.0), {2, 2}, true);

            // 前向传播
            auto c = a * b;

            std::cout << "c = a * b 前向传播完成" << std::endl;

            // 反向传播
            c.backward();

            // 检查梯度
            auto grad_a = a.grad();
            auto grad_b = b.grad();

            std::cout << "梯度检查:" << std::endl;
            std::cout << "grad_a 是否有效: " << (grad_a.data() != nullptr ? "是" : "否") << std::endl;
            std::cout << "grad_b 是否有效: " << (grad_b.data() != nullptr ? "是" : "否") << std::endl;

            if (grad_a.data() && grad_b.data())
            {
                bool correct = true;
                // d(c)/d(a) = b = 3.0
                // d(c)/d(b) = a = 2.0
                for (int i = 0; i < 4; ++i)
                {
                    if (std::abs(grad_a.data()[i] - 3.0f) > 1e-6 ||
                        std::abs(grad_b.data()[i] - 2.0f) > 1e-6)
                    {
                        correct = false;
                        break;
                    }
                }
                std::cout << "梯度计算: " << (correct ? "正确 ✓" : "错误 ✗") << std::endl;
            }
            else
            {
                std::cout << "梯度计算: 错误 ✗ (梯度为空)" << std::endl;
            }
        }

        std::cout << "\n"
                  << std::endl;

        // ===== 测试3: 链式法则（复合函数求导） =====
        std::cout << "测试3: 链式法则（复合函数求导）" << std::endl;
        {
            Tensor<float, CPU> a(float(2.0), {2}, true);
            Tensor<float, CPU> b(float(3.0), {2}, true);

            // 复合运算：y = (a + b) * b
            auto temp = a + b; // temp = a + b = 5.0
            auto y = temp * b; // y = temp * b = 5.0 * 3.0 = 15.0

            std::cout << "y = (a + b) * b 前向传播完成" << std::endl;

            // 反向传播
            y.backward();

            // 检查梯度
            auto grad_a = a.grad();
            auto grad_b = b.grad();

            std::cout << "梯度检查:" << std::endl;
            std::cout << "grad_a 是否有效: " << (grad_a.data() != nullptr ? "是" : "否") << std::endl;
            std::cout << "grad_b 是否有效: " << (grad_b.data() != nullptr ? "是" : "否") << std::endl;

            if (grad_a.data() && grad_b.data())
            {
                // 理论梯度：
                // dy/da = b = 3.0
                // dy/db = a + 2b = 2.0 + 2*3.0 = 8.0
                bool correct = true;
                if (std::abs(grad_a.data()[0] - 3.0f) > 1e-6)
                {
                    correct = false;
                }
                if (std::abs(grad_b.data()[0] - 8.0f) > 1e-6)
                {
                    correct = false;
                }
                std::cout << "梯度计算: " << (correct ? "正确 ✓" : "错误 ✗") << std::endl;
                std::cout << "理论梯度:dy/da = 3.0, dy/db = 8.0" << std::endl;
                std::cout
                    << "实际梯度:dy/da = " << grad_a.data()[0] << std::endl;
                std::cout << "实际梯度:dy/db = " << grad_b.data()[0] << std::endl;
            }
            else
            {
                std::cout << "梯度计算: 错误 ✗ (梯度为空)" << std::endl;
            }
        }

        std::cout << "\n"
                  << std::endl;

        // ===== 测试4: 梯度累积 =====
        std::cout << "测试4: 梯度累积" << std::endl;
        {
            Tensor<float, CPU> x(float(2.0), {1}, true);
            Tensor<float, CPU> y(float(3.0), {1}, true);
            // 多次使用同一个变量
            auto y1 = x * y;
            auto y2 = x * y;

            // 求和
            auto z = y1 + y2;

            std::cout << "z = (x*3) + (x*3) 前向传播完成" << std::endl;

            // 反向传播
            z.backward();

            auto grad_x = x.grad();

            std::cout << "梯度检查:" << std::endl;
            std::cout << "grad_x 是否有效: " << (grad_x.data() != nullptr ? "是" : "否") << std::endl;

            if (grad_x.data())
            {
                // dz/dx = 3 + 4 = 7
                bool correct = std::abs(grad_x.data()[0] - 6.0f) < 1e-6;
                std::cout << "梯度计算: " << (correct ? "正确 ✓" : "错误 ✗") << std::endl;
                std::cout << "理论梯度: 6.0, 实际梯度: " << grad_x.data()[0] << std::endl;
            }
            else
            {
                std::cout << "梯度计算: 错误 ✗ (梯度为空)" << std::endl;
            }
        }

        // std::cout << "\n"
        //           << std::endl;

        // // ===== 测试5: 广播梯度 =====
        // std::cout << "测试5: 广播梯度" << std::endl;
        // {
        //     Tensor<float, CPU> a(float(2.0), {2, 3}, true);
        //     Tensor<float, CPU> b(float(3.0), {1, 3}, true); // 可以广播到 {2, 3}

        //     auto c = a + b;

        //     std::cout << "c = a + b (广播) 前向传播完成" << std::endl;
        //     std::cout << "c.shape: (";
        //     for (auto dim : c.shape())
        //         std::cout << dim << " ";
        //     std::cout << ")" << std::endl;

        //     // 反向传播
        //     c.backward();

        //     auto grad_a = a.grad();
        //     auto grad_b = b.grad();

        //     std::cout << "梯度检查:" << std::endl;
        //     std::cout << "grad_a 形状: (";
        //     if (grad_a.data())
        //     {
        //         for (auto dim : grad_a.shape())
        //             std::cout << dim << " ";
        //     }
        //     else
        //     {
        //         std::cout << "null";
        //     }
        //     std::cout << ")" << std::endl;

        //     std::cout << "grad_b 形状: (";
        //     if (grad_b.data())
        //     {
        //         for (auto dim : grad_b.shape())
        //             std::cout << dim << " ";
        //     }
        //     else
        //     {
        //         std::cout << "null";
        //     }
        //     std::cout << ")" << std::endl;

        //     if (grad_a.data() && grad_b.data())
        //     {
        //         // 广播梯度应该正确求和
        //         std::cout << "广播梯度测试完成" << std::endl;
        //     }
        // }

        std::cout << "\n=== 自动求导测试完成 ===" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n测试失败! 异常信息: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}