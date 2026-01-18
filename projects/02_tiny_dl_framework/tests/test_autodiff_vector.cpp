#include <iostream>
#include <vector>
#include <cmath>
#include "tensor.hpp"
#include "function.hpp"

using namespace tiny_dl;

// 工具函数：打印向量信息
template <typename Scalar, typename Device>
void print_vector(const std::string &name, const Tensor<Scalar, Device> &vec)
{
    std::cout << name << ": [";
    if (vec.data() != nullptr && vec.shape().size() > 0)
    {
        size_t size = vec.shape()[0];
        for (size_t i = 0; i < size; ++i)
        {
            std::cout << vec.data()[i];
            if (i < size - 1)
                std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

// 工具函数：验证向量值
template <typename Scalar, typename Device>
bool verify_vector(const Tensor<Scalar, Device> &vec,
                   const std::vector<Scalar> &expected,
                   Scalar tolerance = Scalar(1e-6))
{
    if (vec.data() == nullptr)
        return false;
    if (vec.shape().size() != 1)
        return false;
    if (vec.shape()[0] != expected.size())
        return false;

    for (size_t i = 0; i < expected.size(); ++i)
    {
        if (std::abs(vec.data()[i] - expected[i]) > tolerance)
        {
            return false;
        }
    }
    return true;
}

// 辅助函数：创建标量张量并广播
template <typename Scalar, typename Device>
Tensor<Scalar, Device> scalar_tensor(Scalar value, const std::vector<size_t> &shape)
{
    return Tensor<Scalar, Device>(value, shape);
}

int main()
{
    std::cout << "=== 向量自动求导测试（不使用数乘） ===\n"
              << std::endl;

    try
    {
        // ===== 测试1: 基础向量运算 =====
        std::cout << "测试1: 向量加法和乘法" << std::endl;
        {
            // 创建向量 [2, 2, 2] 和 [3, 3, 3]
            Tensor<float, CPU> a(float(2.0), {3}, true);
            Tensor<float, CPU> b(float(3.0), {3}, true);

            print_vector("a", a);
            print_vector("b", b);

            // 测试加法
            auto c = a + b;
            print_vector("c = a + b", c);

            // 验证加法结果：2+3=5
            std::vector<float> expected_c = {5.0f, 5.0f, 5.0f};
            if (verify_vector(c, expected_c))
            {
                std::cout << "加法前向传播: 正确 ✓" << std::endl;
            }
            else
            {
                std::cout << "加法前向传播: 错误 ✗" << std::endl;
            }

            // 测试乘法
            auto d = a * b;
            print_vector("d = a * b", d);

            // 验证乘法结果：2*3=6
            std::vector<float> expected_d = {6.0f, 6.0f, 6.0f};
            if (verify_vector(d, expected_d))
            {
                std::cout << "乘法前向传播: 正确 ✓" << std::endl;
            }
            else
            {
                std::cout << "乘法前向传播: 错误 ✗" << std::endl;
            }

            // 反向传播测试：先测试乘法的梯度
            d.backward();

            auto grad_a = a.grad();
            auto grad_b = b.grad();

            std::cout << "乘法梯度检查:" << std::endl;
            print_vector("grad_a (d(a*b)/da = b)", grad_a);
            print_vector("grad_b (d(a*b)/db = a)", grad_b);

            // 验证梯度: d(a*b)/da = b = 3, d(a*b)/db = a = 2
            std::vector<float> expected_grad_a = {3.0f, 3.0f, 3.0f};
            std::vector<float> expected_grad_b = {2.0f, 2.0f, 2.0f};
            bool grad_correct = verify_vector(grad_a, expected_grad_a) &&
                                verify_vector(grad_b, expected_grad_b);
            std::cout << "乘法反向传播: " << (grad_correct ? "正确 ✓" : "错误 ✗") << std::endl;
        }

        std::cout << "\n"
                  << std::endl;

        // ===== 测试2: 链式法则 =====
        std::cout << "测试2: 链式法则 (x*(x+y))" << std::endl;
        {
            Tensor<float, CPU> x(float(2.0), {3}, true);
            Tensor<float, CPU> y(float(3.0), {3}, true);

            print_vector("x", x);
            print_vector("y", y);

            // 计算 z = x * (x + y)
            auto sum = x + y; // x + y = 2 + 3 = 5
            auto z = x * sum; // x * (x + y) = 2 * 5 = 10

            print_vector("x + y", sum);
            print_vector("z = x * (x + y)", z);

            // 验证前向传播：2 * (2+3) = 10
            std::vector<float> expected_z = {10.0f, 10.0f, 10.0f};
            if (verify_vector(z, expected_z))
            {
                std::cout << "前向传播: 正确 ✓" << std::endl;
            }
            else
            {
                std::cout << "前向传播: 错误 ✗" << std::endl;
            }

            // 反向传播
            z.backward();

            auto grad_x = x.grad();
            auto grad_y = y.grad();

            std::cout << "梯度检查:" << std::endl;
            print_vector("grad_x", grad_x);
            print_vector("grad_y", grad_y);

            // 理论梯度计算：
            // z = x*(x+y) = x^2 + x*y
            // dz/dx = 2x + y = 2*2 + 3 = 7
            // dz/dy = x = 2
            std::vector<float> expected_grad_x = {7.0f, 7.0f, 7.0f};
            std::vector<float> expected_grad_y = {2.0f, 2.0f, 2.0f};
            bool grad_correct = verify_vector(grad_x, expected_grad_x) &&
                                verify_vector(grad_y, expected_grad_y);
            std::cout << "反向传播: " << (grad_correct ? "正确 ✓" : "错误 ✗") << std::endl;

            std::cout << "\n理论分析:" << std::endl;
            std::cout << "z = x * (x + y)" << std::endl;
            std::cout << "dz/dx = 2x + y = 2*2 + 3 = 7" << std::endl;
            std::cout << "dz/dy = x = 2" << std::endl;
        }

        std::cout << "\n"
                  << std::endl;

        // ===== 测试3: 多变量复合运算 =====
        std::cout << "测试3: 多变量复合运算 (a*b + a*c)" << std::endl;
        {
            Tensor<float, CPU> a(float(1.0), {2}, true);
            Tensor<float, CPU> b(float(2.0), {2}, true);
            Tensor<float, CPU> c(float(3.0), {2}, true);

            print_vector("a", a);
            print_vector("b", b);
            print_vector("c", c);

            // 计算 d = a*b + a*c
            auto ab = a * b;  // 1*2 = 2
            auto ac = a * c;  // 1*3 = 3
            auto d = ab + ac; // 2 + 3 = 5

            print_vector("a*b", ab);
            print_vector("a*c", ac);
            print_vector("d = a*b + a*c", d);

            // 验证前向传播
            std::vector<float> expected_d = {5.0f, 5.0f};
            if (verify_vector(d, expected_d))
            {
                std::cout << "前向传播: 正确 ✓" << std::endl;
            }
            else
            {
                std::cout << "前向传播: 错误 ✗" << std::endl;
            }

            // 反向传播
            d.backward();

            auto grad_a = a.grad();
            auto grad_b = b.grad();
            auto grad_c = c.grad();

            std::cout << "梯度检查:" << std::endl;
            print_vector("grad_a", grad_a);
            print_vector("grad_b", grad_b);
            print_vector("grad_c", grad_c);

            // 理论梯度：
            // d = a*b + a*c = a*(b+c)
            // dd/da = b + c = 2 + 3 = 5
            // dd/db = a = 1
            // dd/dc = a = 1
            std::vector<float> expected_grad_a = {5.0f, 5.0f};
            std::vector<float> expected_grad_b = {1.0f, 1.0f};
            std::vector<float> expected_grad_c = {1.0f, 1.0f};
            bool grad_correct = verify_vector(grad_a, expected_grad_a) &&
                                verify_vector(grad_b, expected_grad_b) &&
                                verify_vector(grad_c, expected_grad_c);
            std::cout << "反向传播: " << (grad_correct ? "正确 ✓" : "错误 ✗") << std::endl;

            std::cout << "\n理论分析:" << std::endl;
            std::cout << "d = a*b + a*c" << std::endl;
            std::cout << "dd/da = b + c = 2 + 3 = 5" << std::endl;
            std::cout << "dd/db = a = 1" << std::endl;
            std::cout << "dd/dc = a = 1" << std::endl;
        }

        std::cout << "\n"
                  << std::endl;

        // ===== 测试4: 深度链式法则 =====
        std::cout << "测试4: 深度链式法则 ((a+b)*(b+c))" << std::endl;
        {
            Tensor<float, CPU> a(float(1.0), {3}, true);
            Tensor<float, CPU> b(float(2.0), {3}, true);
            Tensor<float, CPU> c(float(3.0), {3}, true);

            print_vector("a", a);
            print_vector("b", b);
            print_vector("c", c);

            // 计算 e = (a+b) * (b+c)
            auto ab = a + b;  // 1+2 = 3
            auto bc = b + c;  // 2+3 = 5
            auto e = ab * bc; // 3*5 = 15

            print_vector("a+b", ab);
            print_vector("b+c", bc);
            print_vector("e = (a+b)*(b+c)", e);

            // 验证前向传播
            std::vector<float> expected_e = {15.0f, 15.0f, 15.0f};
            if (verify_vector(e, expected_e))
            {
                std::cout << "前向传播: 正确 ✓" << std::endl;
            }
            else
            {
                std::cout << "前向传播: 错误 ✗" << std::endl;
            }

            // 反向传播
            e.backward();

            auto grad_a = a.grad();
            auto grad_b = b.grad();
            auto grad_c = c.grad();

            std::cout << "梯度检查:" << std::endl;
            print_vector("grad_a", grad_a);
            print_vector("grad_b", grad_b);
            print_vector("grad_c", grad_c);

            // 理论梯度：
            // e = (a+b)*(b+c)
            // de/da = (b+c) = 5
            // de/db = (a+b) + (b+c) = 3 + 5 = 8
            // de/dc = (a+b) = 3
            std::vector<float> expected_grad_a = {5.0f, 5.0f, 5.0f};
            std::vector<float> expected_grad_b = {8.0f, 8.0f, 8.0f};
            std::vector<float> expected_grad_c = {3.0f, 3.0f, 3.0f};
            bool grad_correct = verify_vector(grad_a, expected_grad_a) &&
                                verify_vector(grad_b, expected_grad_b) &&
                                verify_vector(grad_c, expected_grad_c);
            std::cout << "反向传播: " << (grad_correct ? "正确 ✓" : "错误 ✗") << std::endl;

            std::cout << "\n理论分析:" << std::endl;
            std::cout << "e = (a+b)*(b+c)" << std::endl;
            std::cout << "de/da = (b+c) = 2+3 = 5" << std::endl;
            std::cout << "de/db = (a+b) + (b+c) = (1+2) + (2+3) = 3+5 = 8" << std::endl;
            std::cout << "de/dc = (a+b) = 1+2 = 3" << std::endl;
        }

        std::cout << "\n"
                  << std::endl;

        // // ===== 测试5: 梯度累积测试 =====
        // std::cout << "测试5: 梯度累积测试" << std::endl;
        // {
        //     Tensor<float, CPU> p(float(2.0), {2}, true);
        //     Tensor<float, CPU> q(float(3.0), {2}, true);

        //     print_vector("p", p);
        //     print_vector("q", q);

        //     // 创建与p形状相同的张量，用于模拟常数
        //     Tensor<float, CPU> three(float(3.0), {2}, false); // 不需要梯度
        //     Tensor<float, CPU> four(float(4.0), {2}, false);
        //     Tensor<float, CPU> five(float(5.0), {2}, false);

        //     // 多个使用p的计算路径（使用相同形状的张量相乘）
        //     auto r1 = p * three; // p * 3
        //     auto r2 = p * four;  // p * 4
        //     auto r3 = p * five;  // p * 5

        //     // 合并结果
        //     auto s = r1 + r2 + r3;

        //     print_vector("r1 = p * [3,3]", r1);
        //     print_vector("r2 = p * [4,4]", r2);
        //     print_vector("r3 = p * [5,5]", r3);
        //     print_vector("s = r1 + r2 + r3", s);

        //     // 验证前向传播
        //     std::vector<float> expected_s = {
        //         2.0f * 3.0f + 2.0f * 4.0f + 2.0f * 5.0f, // 6+8+10=24
        //         2.0f * 3.0f + 2.0f * 4.0f + 2.0f * 5.0f  // 6+8+10=24
        //     };
        //     if (verify_vector(s, expected_s))
        //     {
        //         std::cout << "前向传播: 正确 ✓" << std::endl;
        //     }
        //     else
        //     {
        //         std::cout << "前向传播: 错误 ✗" << std::endl;
        //     }

        //     // 反向传播
        //     s.backward();

        //     auto grad_p = p.grad();

        //     std::cout << "梯度检查:" << std::endl;
        //     print_vector("grad_p", grad_p);

        //     // 理论梯度：ds/dp = 3 + 4 + 5 = 12
        //     std::vector<float> expected_grad_p = {12.0f, 12.0f};
        //     bool grad_correct = verify_vector(grad_p, expected_grad_p);
        //     std::cout << "反向传播: " << (grad_correct ? "正确 ✓" : "错误 ✗") << std::endl;
        // }

        std::cout << "\n=== 所有向量测试完成 ===" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n测试失败! 异常信息: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}