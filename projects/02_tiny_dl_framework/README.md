# tiny_dl - 一个轻量级的深度学习框架

## 概述

tiny_dl 是一个用 C++17 编写的轻量级深度学习框架，实现了张量运算和自动求导功能。该项目旨在提供一个简洁、高效且易于理解的深度学习基础框架。

## 核心特性

### 1. 张量运算

- 支持任意维度的张量
- 支持元素级加法和乘法
- 支持广播机制
- 支持视图（view）和重塑（reshape）

### 2. 自动求导系统

- 基于计算图的反向传播
- 支持链式法则和梯度累积
- 动态构建计算图
- 内存高效的梯度计算

### 3. 架构设计

- 清晰的张量实现与函数实现分离
- 访问控制机制（ImplAccessToken）
- 版本控制和状态管理
- 支持不同的设备后端（CPU/GPU）

# 

## 快速开始

### 编译要求

- C++17 兼容编译器（g++ 7+ 或 clang++ 6+）
- 标准模板库

### 基本使用

```cpp
#include "tensor.hpp"
#include "function.hpp"

using namespace tiny_dl;

int main() {
    // 创建需要梯度的张量
    Tensor<float, CPU> x({2, 3}, true);
    Tensor<float, CPU> y({2, 3}, true);

    // 前向传播
    auto z = x + y;
    auto w = z * x;

    // 反向传播
    w.backward();

    // 获取梯度
    auto grad_x = x.grad();
    auto grad_y = y.grad();

    return 0;
}
```

## API 文档

### 张量类（Tensor）

#### 构造函数

```cpp
// 空张量
Tensor<Scalar, Device>();

// 从形状构造
Tensor(const std::vector<size_t>& shape, bool requires_grad = false);

// 从标量构造（广播）
Tensor(Scalar value, const std::vector<size_t>& shape, bool requires_grad = false);
```

#### 主要方法

```cpp
// 数据访问
const Scalar* data() const;              // 只读访问
Scalar* data(ImplAccessToken token);     // 需要令牌的写访问

// 形状信息
const std::vector<size_t>& shape() const;
size_t size() const;

// 自动求导
void set_requires_grad(bool requires_grad);
bool requires_grad() const;
void backward();
Tensor grad() const;

// 运算
Tensor clone() const;
Tensor view(const std::vector<size_t>& new_shape);
Tensor reshape(const std::vector<size_t>& new_shape);

// 运算符重载
Tensor operator+(const Tensor& other) const;
Tensor operator*(const Tensor& other) const;
```

### 函数基类（Function）

#### 核心接口

```cpp
// 前向传播
virtual std::vector<TensorType> forward(std::vector<TensorType> inputs) = 0;

// 反向传播
virtual std::vector<TensorType> backward(std::vector<TensorType> grad_outputs) = 0;

// 工具方法
bool is_differentiable() const;
void release_resources();
```

#### 已实现的函数

- `AddFunction`: 加法运算
- `MulFunction`: 乘法运算

## 设计模式

### 1. 访问控制模式

使用 `ImplAccessToken` 限制对张量内部实现的访问，确保只有 `Function` 和 `Tensor` 类能修改内部状态。

### 2. 计算图构建

- 前向传播时保存输入快照
- 建立输出与函数的关联
- 动态构建计算图

### 3. 版本控制

- 每个张量有唯一ID和版本号
- 防止使用过期的张量状态
- 支持视图和别名检测

## 示例代码

### 线性回归

```cpp
#include "tensor.hpp"
#include "function.hpp"

void linear_regression() {
    // 初始化参数
    Tensor<float, CPU> W({10, 1}, true);  // 权重
    Tensor<float, CPU> b({1}, true);      // 偏置

    // 训练循环
    for (int epoch = 0; epoch < 100; ++epoch) {
        // 前向传播
        auto predictions = X * W + b;
        auto loss = (predictions - y) * (predictions - y);

        // 反向传播
        loss.backward();

        // 参数更新（手动）
        // W = W - learning_rate * W.grad()
        // b = b - learning_rate * b.grad()
    }
}
```

### 多层感知机

```cpp
#include "tensor.hpp"
#include "function.hpp"

class MLP {
private:
    std::vector<Tensor<float, CPU>> layers;

public:
    Tensor<float, CPU> forward(Tensor<float, CPU> x) {
        for (auto& layer : layers) {
            x = x * layer;  // 简化版，实际需要加偏置和激活函数
        }
        return x;
    }
};
```

## 性能特点

### 内存管理

- 使用共享指针管理存储
- 支持视图，避免不必要的内存复制
- 及时释放中间变量

### 计算优化

- 广播优化
- 原地操作支持
- 缓存友好的内存布局

## 扩展指南

### 添加新的运算

1. 继承 `Function` 基类
2. 实现 `forward` 和 `backward` 方法
3. 在 `Tensor` 类中添加对应的运算符重载

### 添加新的设备后端

1. 实现设备特定的存储分配器
2. 提供设备特定的运算实现
3. 更新张量类以支持新设备

## 测试

### 运行测试

```bash
# 编译测试
g++ -std=c++17 -I./include tests/test_autodiff.cpp -o test_autodiff

# 运行测试
./test_autodiff
```

### 测试覆盖率

- [x] 张量基本运算
- [x] 自动求导
- [x] 广播机制
- [ ] GPU支持
- [ ] 优化器实现

## 限制和未来工作

### 当前限制

- 仅支持CPU后端
- 基础运算有限（加法和乘法）
- 缺乏高级优化器
- 不支持动态图优化

### 未来计划

- 实现更多运算函数（减法、除法、矩阵乘法）
- 添加激活函数
- 实现优化器（SGD、Adam等）
- 支持GPU计算
- 添加序列化支持
