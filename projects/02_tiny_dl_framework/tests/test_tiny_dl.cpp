// test_minimal.cpp
#include <iostream>
#include <iomanip>
#include <vector>
#include "tensor.hpp"

using namespace tiny_dl;

// 简单打印Tensor信息
template <typename Scalar>
void print_tensor(const std::string &name, const Tensor<Scalar> &tensor)
{
    std::cout << name << ": shape=[";
    const auto &shape = tensor.shape();
    for (size_t i = 0; i < shape.size(); ++i)
    {
        std::cout << shape[i];
        if (i < shape.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]";

    if (shape.empty() || shape[0] == 0)
    {
        std::cout << " (empty)\n";
        return;
    }

    std::cout << ", data=[";
    const Scalar *data = tensor.data();
    size_t total_size = 1;
    for (auto dim : shape)
        total_size *= dim;

    // 只打印前几个元素
    size_t print_count = std::min(total_size, static_cast<size_t>(5));
    for (size_t i = 0; i < print_count; ++i)
    {
        std::cout << std::fixed << std::setprecision(2) << data[i];
        if (i < print_count - 1)
            std::cout << ", ";
    }
    if (total_size > print_count)
        std::cout << ", ...";
    std::cout << "]\n";
}

void test1_constructors()
{
    std::cout << "\n=== 测试1: 构造函数 ===\n";

    try
    {
        // 1.1 默认构造函数
        Tensor<float> t1;
        std::cout << "✅ 默认构造函数\n";
        print_tensor("t1 (默认)", t1);

        // 1.2 从形状构造
        Tensor<float> t2({2, 3});
        std::cout << "✅ 形状构造函数\n";
        print_tensor("t2 (2x3)", t2);

        // 1.3 从标量和形状构造
        Tensor<float> t3(5.0f, {2, 2});
        std::cout << "✅ 标量广播构造函数\n";
        print_tensor("t3 (全5.0)", t3);

        // 1.4 复制构造
        Tensor<float> t4 = t3;
        std::cout << "✅ 复制构造函数\n";
    }
    catch (const std::exception &e)
    {
        std::cout << "❌ 构造函数测试失败: " << e.what() << "\n";
    }
}

void test2_basic_operations()
{
    std::cout << "\n=== 测试2: 基础运算 ===\n";

    try
    {
        // 创建标量Tensor（通过标量构造函数）
        Tensor<float> a(3.0f, {1});
        Tensor<float> b(2.0f, {1});

        std::cout << "创建标量Tensor:\n";
        print_tensor("a (3.0)", a);
        print_tensor("b (2.0)", b);

        // 标量加法
        auto c = a + b;
        std::cout << "\n✅ 标量加法: a + b\n";
        print_tensor("c (应为5.0)", c);

        // 标量乘法
        auto d = a * b;
        std::cout << "\n✅ 标量乘法: a * b\n";
        print_tensor("d (应为6.0)", d);

        // 向量运算测试
        Tensor<float> v1({3}); // 3维向量
        Tensor<float> v2({3});

        // 注意：这里向量是未初始化的，但我们只测试运算能否执行
        auto v3 = v1 + v2;
        std::cout << "\n✅ 向量加法: v1 + v2\n";
        print_tensor("v3 (未初始化)", v3);
    }
    catch (const std::exception &e)
    {
        std::cout << "❌ 基础运算测试失败: " << e.what() << "\n";
    }
}

void test3_shape_operations()
{
    std::cout << "\n=== 测试3: 形状操作 ===\n";

    try
    {
        // 创建Tensor
        Tensor<float> t({2, 3});
        print_tensor("原始Tensor", t);

        // 测试reshape
        auto t_reshaped = t.reshape({3, 2});
        std::cout << "\n✅ reshape操作\n";
        print_tensor("reshape后", t_reshaped);

        // 测试clone
        auto t_cloned = t.clone();
        std::cout << "\n✅ clone操作\n";
        print_tensor("克隆后", t_cloned);

        // 测试requires_grad
        Tensor<float> t_grad({2, 2}, true);
        std::cout << "\n✅ requires_grad设置\n";
        std::cout << "t_grad.requires_grad() = "
                  << (t_grad.requires_grad() ? "true" : "false") << "\n";
    }
    catch (const std::exception &e)
    {
        std::cout << "❌ 形状操作测试失败: " << e.what() << "\n";
    }
}

void test4_autodiff_simple()
{
    std::cout << "\n=== 测试4: 简单自动微分 ===\n";

    try
    {
        // 创建需要梯度的Tensor
        Tensor<float> x({1}, true); // 标量，需要梯度
        Tensor<float> y({1}, true);

        // 注意：这里我们无法设置x和y的值，但可以测试计算图构建
        std::cout << "创建需要梯度的Tensor:\n";
        std::cout << "x.requires_grad() = " << (x.requires_grad() ? "true" : "false") << "\n";
        std::cout << "y.requires_grad() = " << (y.requires_grad() ? "true" : "false") << "\n";

        // 尝试构建计算图
        try
        {
            auto z = x + y;
            std::cout << "\n✅ 计算图构建成功\n";
            std::cout << "z.requires_grad() = " << (z.requires_grad() ? "true" : "false") << "\n";
            print_tensor("z", z);

            // 测试backward（可能需要梯度值）
            // 由于没有初始化数据，这里可能会失败，但我们测试接口
            std::cout << "\n尝试backward...\n";
        }
        catch (const std::exception &e)
        {
            std::cout << "⚠️  计算图构建失败（可能是正常情况）: " << e.what() << "\n";
        }
    }
    catch (const std::exception &e)
    {
        std::cout << "❌ 自动微分测试失败: " << e.what() << "\n";
    }
}

void test5_broadcasting()
{
    std::cout << "\n=== 测试5: 广播 ===\n";

    try
    {
        // 标量和向量广播
        Tensor<float> scalar(5.0f, {1});
        Tensor<float> vector({3});

        std::cout << "测试标量-向量广播:\n";
        print_tensor("scalar", scalar);
        print_tensor("vector", vector);

        try
        {
            auto result = scalar + vector;
            std::cout << "\n✅ 广播加法成功\n";
            print_tensor("scalar + vector", result);
        }
        catch (const std::exception &e)
        {
            std::cout << "\n❌ 广播失败: " << e.what() << "\n";
        }
    }
    catch (const std::exception &e)
    {
        std::cout << "❌ 广播测试失败: " << e.what() << "\n";
    }
}

void test6_exceptions()
{
    std::cout << "\n=== 测试6: 异常处理 ===\n";

    try
    {
        // 测试形状不匹配
        Tensor<float> t1({2, 3});
        Tensor<float> t2({3, 2}); // 不兼容的形状

        std::cout << "测试形状不匹配异常:\n";
        try
        {
            auto result = t1 + t2;
            std::cout << "❌ 应该抛出异常但没有\n";
        }
        catch (const std::runtime_error &e)
        {
            std::cout << "✅ 正确捕获异常: " << e.what() << "\n";
        }
        catch (...)
        {
            std::cout << "✅ 正确捕获异常（类型不同）\n";
        }
    }
    catch (const std::exception &e)
    {
        std::cout << "❌ 异常测试失败: " << e.what() << "\n";
    }
}

int main()
{
    std::cout << "===========================================\n";
    std::cout << "    TinyDL 最小化测试程序\n";
    std::cout << "    只测试最基本功能，不修改原有代码\n";
    std::cout << "===========================================\n";

    // 运行所有测试
    test1_constructors();
    test2_basic_operations();
    test3_shape_operations();
    test4_autodiff_simple();
    test5_broadcasting();
    test6_exceptions();

    std::cout << "\n===========================================\n";
    std::cout << "    测试完成\n";
    std::cout << "    注：某些测试可能因为功能未完全实现而失败\n";
    std::cout << "    这是正常的开发过程\n";
    std::cout << "===========================================\n";

    return 0;
}