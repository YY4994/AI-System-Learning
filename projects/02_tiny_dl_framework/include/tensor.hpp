#pragma once

#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <type_traits>
#include <utility>
#include <iterator>
#include <initializer_list>
#include <iostream>
#include "function.hpp

namespace tiny_dl
{
    // 定义Device的参数
    struct CPU
    {
        template <typename T>
        using DefaultAllocator = std::allocator<T>;
    };
#ifdef USE_CUDA
    struct GPU
    {
        template <typename T>
        using DefaultAllocator = cuda::device_allocator<T>;
    };
#endif

    // 前向声明
    class Function;

    // 访问令牌 - 只有Function能创建
    class ImplAccessToken
    {
    private:
        ImplAccessToken() = default; // 私有构造函数
        friend class Function;       // 只有Function能构造
    };

    template <typename Scalar, typename Allocator = CPU::template DefaultAllocator<Scalar>>
    class Storage // 独占所有权
    {
    private:
        Scalar *data_;
        size_t capacity_ = 0;
        Allocator allocator_;

    public:
        // 构造函数,析构函数
        Storage(size_t capacity = 0) : capacity_(capacity)
        {
            if (capacity > 0)
            {
                data_ = allocator_.allocate(capacity);
            }
            else
            {
                data_ = nullptr;
            }
        }
        ~Storage()
        {
            if (data_)
            {
                allocator_.deallocate(data_, capacity_);
                // allocator_.deallocate(data_);
                data_ = nullptr;
            }
        }
        void release()
        {
            if (data_)
            {
                allocator_.deallocate(data_, capacity_);
                // allocator_.deallocate(data_);
                data_ = nullptr;
            }
        }
        // 获取数据
        Scalar *data() { return data_; }
        const Scalar *data() const { return data_; }
        size_t capacity() const { return capacity_; }
        // 通过data_数据长度获取数据实际长度
        void resize(size_t new_capacity)
        {
            if (new_capacity == capacity_)
            {
                return;
            }
            if (new_capacity == 0)
            {
                release();
                return;
            }
            Scalar *new_data = allocator_.allocate(new_capacity);
            size_t copy_count = std::min(capacity_, new_capacity);
            if (data_ && copy_count)
            {
                std::copy(data_, data_ + copy_count, new_data);
            }
            if (data_)
            {
                allocator_.deallocate(data_);
            }
            data_ = new_data;
            capacity_ = new_capacity;
        }

        // 拷贝构造函数和赋值运算符
        Storage(Storage &&other) noexcept : data_(other.data_), capacity_(other.capacity_)
        {
            // 使用移动语义
            other.data_ = nullptr;
            other.capacity_ = 0;
        }; // 在Storage类中添加：
        Storage(const Storage &other)
            : capacity_(other.capacity_), allocator_(other.allocator_)
        {
            if (capacity_ > 0)
            {
                data_ = allocator_.allocate(capacity_);
                std::copy(other.data_, other.data_ + capacity_, data_);
            }
            else
            {
                data_ = nullptr;
            }
        }
        Storage &operator=(Storage &) = delete;
    };

    template <typename Scalar, typename Device = CPU>
    class TensorImpl
    {
    private:
        using AllocatorType = typename Device::template DefaultAllocator<Scalar>;
        // === 第一部分：数据与视图（不可变核心）===
        std::shared_ptr<Storage<Scalar, AllocatorType>> storage_;
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        size_t offset_ = 0;

        // === 第二部分：自动求导状态（可变）===
        bool requires_grad_ = false;
        std::weak_ptr<Function> grad_fn_;                  // 生成该Tensor的Function，用于反向传播
        std::shared_ptr<TensorImpl<Scalar, Device>> grad_; // 该Tensor的梯度

        // === 第三部分：版本与元数据 ===
        size_t version_ = 0;
        size_t unique_id_ = 0; // 全局唯一ID，用于标识不同Tensor实例
    public:
        // 构造函数
        TensorImpl() : storage_(nullptr), shape_({0}), strides_({0}), offset_(0), version_(0), grad_fn_(), grad_(nullptr)
        {
            static size_t global_id = 0x10000;
            unique_id_ = global_id++;
        }
        // 构造函数：从形状构造
        TensorImpl(const std::vector<size_t> &shape, bool requires_grad = false)
        {
            storage_ = std::make_shared<Storage<Scalar, AllocatorType>>(shape);
            shape_ = shape;
            strides_ = compute_strides(shape);
            offset_ = 0;
            requires_grad_ = requires_grad;
        }
        // 构造函数：从数据构造
        TensorImpl(const std::shared_ptr<Storage<Scalar, AllocatorType>> &storage,
                   const std::vector<size_t> &shape,
                   const std::vector<size_t> &strides,
                   size_t offset)
            : storage_(storage), shape_(shape), strides_(strides), offset_(offset), version_(0), grad_fn_(), grad_(nullptr)
        {
            static size_t global_id = 0xf000;
            unique_id_ = global_id++;
        }
        // 构造函数：从标量构造（广播）
        TensorImpl(Scalar scalar, const std::vector<size_t> &shape, bool requires_grad = false)
        {
            storage_ = std::make_shared<Storage<Scalar, AllocatorType>>(shape);
            std::fill(storage_->data(), storage_->data() + storage_->capacity(), scalar);
            shape_ = shape;
            strides_ = compute_strides(shape);
            offset_ = 0;
            requires_grad_ = requires_grad;
            static size_t global_id = 0x5000;
            unique_id_ = global_id++;
        }
        // 构造函数：创建视图
        TensorImpl(const TensorImpl &other,
                   const std::vector<size_t> &new_shape,
                   const std::vector<size_t> &new_strides,
                   size_t new_offset)
            : storage_(other.storage_), shape_(new_shape), strides_(new_strides), offset_(new_offset), version_(0), grad_fn_(), grad_(nullptr)
        {
            static size_t global_id = 0xa000;
            unique_id_ = global_id++;
        }
        // 析构函数
        ~TensorImpl() = default;

        // 获取数据指针
        Scalar *data()
        {
            if (!storage_)
            {
                throw std::runtime_error("Tensor has no storage");
            }
            return storage_->data() + offset_;
        }
        const Scalar *data() const
        {
            if (!storage_)
            {
                throw std::runtime_error("Tensor has no storage");
            }
            return storage_->data() + offset_;
        }

        // 通用函数
        const size_t size() const
        {
            if (shape_.empty())
                return 0;
            return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>());
        }
        const std::vector<size_t> &shape() const { return shape_; }
        const size_t version() const { return version_; }
        // 自动求导相关
        void set_requires_grad(bool requires_grad)
        {
            requires_grad_ = requires_grad;
        }
        bool requires_grad() const { return requires_grad_; }
        void set_grad_fn(const std::shared_ptr<Function> &grad_fn)
        {
            grad_fn_ = grad_fn;
        }
        void set_grad(const std::shared_ptr<TensorImpl<Scalar, Device>> &grad)
        {
            grad_ = grad;
        }
        std::shared_ptr<TensorImpl<Scalar, Device>> grad() const { return grad_; }
        std::shared_ptr<Function> grad_fn() const { return grad_fn_; }

        void make_modified()
        {
            version_++;
        }
        // 检查输入是否有效（用于Function的backward）
        bool is_valid_input(size_t saved_version) const
        {
            return version_ == saved_version; // 版本号匹配说明未被修改
        }
        // 计算步长
        std::vector<size_t> compute_strides(const std::vector<size_t> &shape)
        {
            if (shape.empty())
                return {};
            std::vector<size_t> strides(shape.size());
            strides.back() = 1;
            for (int i = shape.size() - 2; i >= 0; --i)
            {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            return strides;
        }
        // 计算索引
        template <typename... Indices>
        size_t compute_index(Indices... indices) const
        {
            // 1. 检查维度数量匹配（编译期）
            if (sizeof...(indices) != shape_.size())
            {
                throw std::runtime_error("Number of indices does not match number of dimensions");
            }

            // 2. 展开参数包计算偏移
            size_t index = offset_;
            size_t dim = 0;
            ((index += strides_[dim++] * indices), ...); // 折叠表达式

            return index;
        }
        // 创建视图（工厂方法）
        std::shared_ptr<TensorImpl<Scalar, Device>> view(const std::vector<size_t> &shape) const
        {
            size_t new_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
            if (new_size != this->size())
            {
                throw std::runtime_error("New shape size does not match tensor size");
            }
            auto new_strides = compute_strides(shape);
            return std::make_shared<TensorImpl<Scalar, Device>>(*this, shape, new_strides, offset_);
        }
    };

    template <typename Scalar, typename Device = CPU>
    class Tensor
    {
    private:
        std::shared_ptr<TensorImpl<Scalar, Device>> impl_;

        // 广播两个形状，返回广播后的形状
        vector<size_t> broadcast_shape(const vector<size_t> &shape1, const vector<size_t> &shape2) const
        {
            size_t shape1_size = shape1.size();
            size_t shape2_size = shape2.size();
            size_t max_size = std::max(shape1_size, shape2_size);
            // 结果形状
            vector<size_t> result_shape(max_size);
            // 遍历两个形状，将较小的形状用1填充
            for (size_t i = 0; i < max_size; ++i)
            {
                size_t dim1 = (i < shape1_size) ? shape1[shape1_size - 1 - i] : 1;
                size_t dim2 = (i < shape2_size) ? shape2[shape2_size - 1 - i] : 1;
                if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
                {
                    throw std::runtime_error("Shapes cannot be broadcast together");
                }
                result_shape[i] = std::max(dim1, dim2);
            }
        }
        // 源索引到目标索引的映射
        size_t map_index(const std::vector<size_t> &idx,
                         const std::vector<size_t> &source_shape,
                         const std::vector<size_t> &target_shape) const
        {
            size_t source_dims = source_shape.size();
            size_t target_dims = target_shape.size();
            size_t source_index = 0;
            size_t stride = 1;
            for (size_t dim_offset = 0; dim_offset < source_dims; ++dim_offset)
            {
                size_t target_dim_idx = target_dims - 1 - dim_offset;
                size_t input_dim_size = source_shape[source_dim_idx];
                size_t target_dim_size = target_shape[target_dim_idx];
                if (input_dim_size == target_dim_size)
                {
                    source_index += target_shape[target_dim_idx] * stride;
                }
                else if (input_dim_size == 1)
                {
                    // 广播维度，索引始终为0
                    source_index += 0;
                }
                else
                {
                    throw std::runtime_error("Incompatible shapes for broadcasting");
                }
                stride *= target_dim_size;
            }
            return source_index;
        }

    public:
        static_assert(std::is_arithmetic_v<Scalar>, "Scalar must be arithmetic type");
        using ImplType = TensorImpl<Scalar, Device>;
        using AllocatorType = typename Device::template DefaultAllocator<Scalar>;
        // === 构造函数 ===
        Tensor()
        {
            impl_ = std::make_shared<ImplType>();
        }
        explicit Tensor(const std::vector<size_t> &shape, bool requires_grad = false)
        {
            impl_ = std::make_shared<ImplType>(shape, requires_grad);
        }
        // 从标量构造（广播）
        Tensor(Scalar value, const std::vector<size_t> &shape, bool requires_grad = false) noexcept
        {

            impl_ = std::make_shared<ImplType>(value, shape, requires_grad);
        };
        // 从已有Impl构造（内部使用）
        Tensor(ImplAccessToken, const std::shared_ptr<ImplType> &impl) : impl_(impl) {}
        ~Tensor() = default;

        // === 数据访问，委托给impl_对象 ===
        // 数据访问（公开，但只读指针）
        const Scalar *data() const { return impl_->data(); }
        // 可写数据访问需要令牌
        Scalar *data(ImplAccessToken) { return impl_->data(); }
        template <typename... Indices>
        Scalar &operator()(Indices... indices)
        {
            if (!impl_ || !impl_->data())
            {
                throw std::runtime_error("Tensor has no implementation or data");
            }
            size_t index = impl_->compute_index(indices...);
            if (index >= impl_->size())
            {
                throw std::out_of_range("Index out of bounds");
            }
            return impl_->data()[index];
        }
        // === 受限的内部接口（需要令牌）===
        // 获取TensorImpl指针
        std::shared_ptr<TensorImpl<Scalar, Device>> impl(ImplAccessToken) const
        {
            return impl_;
        }

        // 创建弱引用用于SavedTensor
        std::weak_ptr<void> weak_impl(ImplAccessToken) const
        {
            return std::static_pointer_cast<void>(impl_);
        }

        // 标记修改（增加版本号）
        void mark_modified(ImplAccessToken)
        {
            impl_->make_modified();
        }

        // 获取当前版本号
        size_t version(ImplAccessToken) const
        {
            // 需要在TensorImpl中添加版本号获取方法
            return impl_->version();
        }
        // === 自动求导接口 ===
        void set_requires_grad(bool requires_grad)
        {
            impl_->set_requires_grad(requires_grad);
        }
        bool requires_grad() const
        {
            return impl_->requires_grad();
        }
        Tensor grad() const
        {
            if (!impl_->requires_grad())
            {
                throw std::runtime_error("This tensor does not require grad");
            }
            auto grad_impl = impl_->grad();
            if (!grad_impl)
            {
                return Tensor(); // 返回空Tensor，表示梯度尚未计算
            }
            return Tensor(grad_impl); // 用grad_impl构造Tensor句柄
        }
        // 创建新形状视图
        Tensor view(const std::vector<size_t> &new_shape)
        {
            return Tensor(impl_->view(new_shape));
        }
        Tensor reshape(const std::vector<size_t> &new_shape)
        {
            // 验证新形状是否合法
            size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());
            if (new_size != impl_->size())
            {
                throw std::runtime_error("New shape size does not match tensor size");
            }
            return view(new_shape);
        }
        // 深拷贝
        Tensor clone() const
        {
            Tensor new_tensor(impl_->shape());
            std::copy(data(), data() + impl_->size(), new_tensor.data());
            return new_tensor;
        }

        // 移动赋值运算符
        Tensor &operator=(Tensor &&other) noexcept
        {
            if (this != &other)
            {
                impl_ = std::move(other.impl_);
                other.impl_ = nullptr;
            }
            return *this;
        }
        // === 运算符重载 ===
        template <typename otherScalar, typename otherDevice>
        Tensor operator+(const Tensor<otherScalar, otherDevice> &other) const
        {
            static_assert(std::is_same_v<Device, otherDevice>, "Incompatible devices for + operation");
            if (this->impl_->shape() == other.impl_->shape())
            {
                Tensor<Scalar, Device> result(this->impl_->shape());
                for (size_t i = 0; i < this->impl_->size(); ++i)
                {
                    result.impl_->data()[i] = this->impl_->data()[i] + other.impl_->data()[i];
                }
                return result;
            }
            else if (this->impl_->shape().size() == 1)
            {
                for (size_t i = 0; i < other.impl_->size(); ++i)
                {
                    Tensor<Scalar, Device> result(other.impl_->shape());
                    result.impl_->data()[i] = this->impl_->data()[0] + other.impl_->data()[i];
                }
                return result;
            }
            else if (other.impl_->shape().size() == 1)
            {
                for (size_t i = 0; i < this->impl_->size(); ++i)
                {
                    Tensor<Scalar, Device> result(this->impl_->shape());
                    result.impl_->data()[i] = this->impl_->data()[i] + other.impl_->data()[0];
                }
                return result;
            }
            else
            {
                // 获取广播后形状
                auto result_shape = broadcast_shape(this->impl_->shape(), other.impl_->shape());
                Tensor<Scalar, Device> result(result_shape);
                auto total_dim = result_shape.size();
                auto total_size = std::accumulate(result_shape.begin(), result_shape.end(), 1, std::multiplies<size_t>());

                std::vector<size_t> idx(total_dim, 0); // 初始化索引为0
                // 遍历每一个元素，将一维索引映射为多维索引
                for (size_t i = 0; i < total_size; ++i)
                {
                    size_t temp = i;
                    for (size_t j = total_dim - 1; j >= 0; --j)
                    {
                        idx[j] = temp % result_shape[j];
                        temp /= result_shape[j];
                    }
                    // 计算当前索引对应的元素在源张量中索引
                    size_t this_index = map_index(idx, this->impl_->shape(), result_shape);
                    size_t other_index = map_index(idx, other.impl_->shape(), result_shape);

                    result.impl_->data()[i] = this->impl_->data()[this_index] + other.impl_->data()[other_index];
                }
                return result;
            }
        }
        template <typename otherScalar, typename otherDevice>
        Tensor operator*(const Tensor<otherScalar, otherDevice> &other) const
        {
            static_assert(std::is_same_v<Device, otherDevice>, "Incompatible devices for + operation");
            if (this->impl_->shape() == other.impl_->shape())
            {
                Tensor<Scalar, Device> result(this->impl_->shape());
                for (size_t i = 0; i < this->impl_->size(); ++i)
                {
                    result.impl_->data()[i] = this->impl_->data()[i] * other.impl_->data()[i];
                }
                return result;
            }
            else if (this->impl_->shape().size() == 1)
            {
                for (size_t i = 0; i < other.impl_->size(); ++i)
                {
                    Tensor<Scalar, Device> result(other.impl_->shape());
                    result.impl_->data()[i] = this->impl_->data()[0] * other.impl_->data()[i];
                }
                return result;
            }
            else if (other.impl_->shape().size() == 1)
            {
                for (size_t i = 0; i < this->impl_->size(); ++i)
                {
                    Tensor<Scalar, Device> result(this->impl_->shape());
                    result.impl_->data()[i] = this->impl_->data()[i] * other.impl_->data()[0];
                }
                return result;
            }
            else
            {
                // 获取广播后形状
                auto result_shape = broadcast_shape(this->impl_->shape(), other.impl_->shape());
                Tensor<Scalar, Device> result(result_shape);
                auto total_dim = result_shape.size();
                auto total_size = std::accumulate(result_shape.begin(), result_shape.end(), 1, std::multiplies<size_t>());

                std::vector<size_t> idx(total_dim, 0); // 初始化索引为0
                // 遍历每一个元素，将一维索引映射为多维索引
                for (size_t i = 0; i < total_size; ++i)
                {
                    size_t temp = i;
                    for (size_t j = total_dim - 1; j >= 0; --j)
                    {
                        idx[j] = temp % result_shape[j];
                        temp /= result_shape[j];
                    }
                    // 计算当前索引对应的元素在源张量中索引
                    size_t this_index = map_index(idx, this->impl_->shape(), result_shape);
                    size_t other_index = map_index(idx, other.impl_->shape(), result_shape);

                    result.impl_->data()[i] = this->impl_->data()[this_index] * other.impl_->data()[other_index];
                }
                return result;
            }
        }
        Tensor matmul(const Tensor &other) const
        {
            if (impl_->shape().size() != 2 || other.impl_->shape().size() != 2)
            {
                throw std::runtime_error("Both tensors must be 2D for matmul");
            }
            size_t m = impl_->shape()[0];
            size_t n = impl_->shape()[1];
            size_t p = other.impl_->shape()[1];
            if (n != other.impl_->shape()[0])
            {
                throw std::runtime_error("Inner dimensions must match for matmul");
            }
            Tensor result({m, p});
            for (size_t i = 0; i < m; ++i)
            {
                for (size_t j = 0; j < p; ++j)
                {
                    Scalar sum = 0;
                    for (size_t k = 0; k < n; ++k)
                    {
                        sum += (*this)(i, k) * other(k, j);
                    }
                    result(i, j) = sum;
                }
            }
            return result;
        }
    };
} // namespace tiny_dl