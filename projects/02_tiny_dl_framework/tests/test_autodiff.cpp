// test_copy_constructor.cpp
#include <iostream>
#include <vector>
#include "tensor.hpp"

using namespace tiny_dl;

void test_tensorimpl_copy()
{
    std::cout << "========== 测试TensorImpl复制 ==========" << std::endl;

    try
    {
        // 创建一个TensorImpl
        std::cout << "\n1. 创建原始TensorImpl:" << std::endl;
        auto impl1 = std::make_shared<TensorImpl<float>>(5.0f, std::vector<size_t>{1}, true);

        std::cout << "impl1->data(): " << impl1->data() << std::endl;
        if (impl1->data())
        {
            std::cout << "impl1->data()[0]: " << impl1->data()[0] << std::endl;
        }
        std::cout << "impl1->shape()[0]: " << impl1->shape()[0] << std::endl;

        // 尝试复制（通过set_grad模拟）
        std::cout << "\n2. 尝试复制TensorImpl:" << std::endl;
        auto impl2 = std::make_shared<TensorImpl<float>>(*impl1); // 使用复制构造函数

        std::cout << "impl2->data(): " << impl2->data() << std::endl;
        if (impl2->data())
        {
            std::cout << "impl2->data()[0]: " << impl2->data()[0] << std::endl;
        }
        std::cout << "impl2->shape().size(): " << impl2->shape().size() << std::endl;
        if (!impl2->shape().empty())
        {
            std::cout << "impl2->shape()[0]: " << impl2->shape()[0] << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cout << "测试异常: " << e.what() << std::endl;
    }

    std::cout << "======================================\n"
              << std::endl;
}

void test_simple_fix()
{
    std::cout << "========== 测试简单修复 ==========" << std::endl;

    try
    {
        // 创建Tensor
        Tensor<float> a(2.0f, std::vector<size_t>{1}, true);
        Tensor<float> b(3.0f, std::vector<size_t>{1}, true);

        std::cout << "创建Tensor成功" << std::endl;

        // 执行加法
        Tensor<float> c = a + b;
        std::cout << "c = " << c.data()[0] << std::endl;

        // 在执行反向传播之前，先手动检查一些东西
        std::cout << "\n检查c的shape: [";
        for (auto dim : c.shape())
            std::cout << dim << " ";
        std::cout << "]" << std::endl;

        // 尝试一个不同的方法：不使用backward，而是手动计算梯度
        std::cout << "\n手动计算梯度:" << std::endl;

        // 对于c = a + b，梯度应该是1
        // 我们可以手动创建梯度Tensor
        Tensor<float> a_grad_manual(1.0f, std::vector<size_t>{1});
        Tensor<float> b_grad_manual(1.0f, std::vector<size_t>{1});

        std::cout << "手动创建的梯度Tensor:" << std::endl;
        std::cout << "a_grad_manual.data(): " << a_grad_manual.data() << std::endl;
        if (a_grad_manual.data())
        {
            std::cout << "a_grad_manual.data()[0]: " << a_grad_manual.data()[0] << std::endl;
        }

        // 现在尝试反向传播
        std::cout << "\n执行反向传播..." << std::endl;
        c.backward();

        // 检查结果
        std::cout << "\n检查梯度:" << std::endl;
        auto a_grad = a.grad();
        std::cout << "a.grad()获取成功" << std::endl;
        std::cout << "a_grad.shape(): [";
        for (auto dim : a_grad.shape())
            std::cout << dim << " ";
        std::cout << "]" << std::endl;

        // 检查shape是否有效
        if (!a_grad.shape().empty() && a_grad.shape()[0] > 0)
        {
            std::cout << "a_grad.data(): " << a_grad.data() << std::endl;
            if (a_grad.data())
            {
                std::cout << "a梯度值: " << a_grad.data()[0] << std::endl;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cout << "测试异常: " << e.what() << std::endl;
    }

    std::cout << "======================================\n"
              << std::endl;
}

int main()
{
    std::cout << "TensorImpl复制问题测试" << std::endl;
    std::cout << std::endl;

    // test_tensorimpl_copy();  // 这个可能需要修改TensorImpl为public

    test_simple_fix();

    std::cout << "\n测试完成！" << std::endl;

    return 0;
}