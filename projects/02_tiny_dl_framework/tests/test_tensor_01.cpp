#include "tensor_01.hpp"
#include <iostream>

int main()
{
    using namespace tiny_dl;

    std::cout << "=== 基础构造测试 ===" << std::endl;
    // 1. 从标量构造
    Tensor<float> t1(3.14f, {2, 3});
    std::cout << "标量构造 (2x3):" << std::endl;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            std::cout << t1(i, j) << " ";
        }
        std::cout << std::endl;
    }

    // 2. 从一维列表构造 - 修正：只有一个参数
    Tensor<float> t2{1.0f, 2.0f, 3.0f, 4.0f};
    std::cout << "\n一维列表构造 (size=" << t2.size() << "):" << std::endl;
    for (size_t i = 0; i < t2.size(); ++i)
    {
        std::cout << t2.data()[i] << " ";
    }
    std::cout << std::endl;

    // 3. 从二维列表构造
    Tensor<float> t3{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
    std::cout << "\n二维列表构造 (3x2):" << std::endl;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            std::cout << t3(i, j) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\n=== 视图操作测试 ===" << std::endl;
    // 4. reshape视图（共享数据）
    std::vector<size_t> original_shape = {2, 4};
    Tensor<float> original(original_shape); // 2x4矩阵
    for (size_t i = 0; i < original.size(); ++i)
    {
        original.data()[i] = static_cast<float>(i); // 填充0-7
    }
    // std::cout << "原始 (2x4):" << std::endl;
    // std::vector<size_t> shape = original.shape();
    // std::cout << "Shape: " << shape.size() << " dimensions: " << std::endl;
    // for (size_t i = 0; i < shape.size(); ++i)
    // {
    //     std::cout << shape[i] << " ";
    // }

    auto reshaped = original.reshape({4, 2}); // 变为4x2
    std::cout << "原始 (2x4):" << std::endl;
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            std::cout << original(i, j) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\n重塑后 (4x2):" << std::endl;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            std::cout << reshaped(i, j) << " ";
        }
        std::cout << std::endl;
    }

    // 测试数据共享
    original(0, 0) = 100.0f;
    std::cout << "\n修改原始[0,0]为100后，重塑视图[0,0]的值: "
              << reshaped(0, 0) << std::endl;

    std::cout << "\n=== 深拷贝测试 ===" << std::endl;
    Tensor<float> copied = original.clone();
    copied(0, 1) = 200.0f;
    std::cout << "深拷贝后修改[0,1]为200:" << std::endl;
    std::cout << "原始[0,1]: " << original(0, 1) << std::endl;
    std::cout << "拷贝[0,1]: " << copied(0, 1) << std::endl;

    std::cout << "\n=== 运算测试 ===" << std::endl;
    // 修正：使用正确的构造函数
    // 方法1: 使用形状构造函数 + 手动填充
    std::vector<size_t> shape = {3};
    Tensor<float> a(shape);
    a(0) = 1.0f;
    a(1) = 2.0f;
    a(2) = 3.0f;

    Tensor<float> b(shape);
    b(0) = 4.0f;
    b(1) = 5.0f;
    b(2) = 6.0f;

    // 方法2: 使用标量构造函数（如果需要相同值）
    // Tensor<float> a(1.0f, {3}); // 全部为1.0
    // Tensor<float> b(2.0f, {3}); // 全部为2.0

    try
    {
        auto c = a + b;
        std::cout << "向量a: ";
        for (size_t i = 0; i < a.size(); ++i)
            std::cout << a.data()[i] << " ";
        std::cout << "\n向量b: ";
        for (size_t i = 0; i < b.size(); ++i)
            std::cout << b.data()[i] << " ";
        std::cout << "\n加法结果c: ";
        for (size_t i = 0; i < c.size(); ++i)
        {
            std::cout << c.data()[i] << " ";
        }
        std::cout << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "运算错误: " << e.what() << std::endl;
    }

    std::cout << "\n=== 错误处理测试 ===" << std::endl;
    try
    {
        // 形状不匹配的reshape
        original.reshape({3, 3});
    }
    catch (const std::exception &e)
    {
        std::cout << "预期错误(reshape形状不匹配): " << e.what() << std::endl;
    }

    try
    {
        // 越界访问
        std::cout << "尝试访问越界元素: " << original(10, 10) << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "预期错误(越界访问): " << e.what() << std::endl;
    }

    try
    {
        // 形状不匹配的加法
        Tensor<float> m1({2, 2});
        Tensor<float> m2({3, 3});
        auto result = m1 + m2;
    }
    catch (const std::exception &e)
    {
        std::cout << "预期错误(加法形状不匹配): " << e.what() << std::endl;
    }

    std::cout << "\n=== 性能特性验证 ===" << std::endl;
    std::cout << "原始数据指针: " << static_cast<void *>(original.data()) << std::endl;
    std::cout << "视图数据指针: " << static_cast<void *>(reshaped.data()) << std::endl;
    std::cout << "指针相同? " << (original.data() == reshaped.data() ? "是" : "否")
              << " (应为是，表示数据共享)" << std::endl;

    std::cout << "\n=== 多维索引测试 ===" << std::endl;
    std::vector<size_t> shape3d = {2, 3, 4};
    Tensor<float> t4(shape3d); // 2x3x4张量
    for (size_t i = 0; i < t4.size(); ++i)
    {
        t4.data()[i] = static_cast<float>(i);
    }

    std::cout << "三维张量 t4(1, 2, 3) = " << t4(1, 2, 3) << std::endl;

    std::cout << "\n=== 测试完成 ===" << std::endl;
    return 0;
}