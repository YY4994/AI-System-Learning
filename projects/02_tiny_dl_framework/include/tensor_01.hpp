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

    template <typename Derived>
    class TensorBase
    {
    protected:
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        size_t offset_ = 0;

    public:
        TensorBase() = default;
        ~TensorBase() = default;

        // 静态多态：静态转换到派生类
        Derived &derived() { return static_cast<Derived &>(*this); }
        const Derived &derived() const { return static_cast<const Derived &>(*this); }
        // 通用函数
        const std::vector<size_t> &shape() const { return shape_; }
        const size_t size() const
        {
            if (shape_.empty())
                return 0;
            return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>());
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
    };

    template <typename Scalar, typename Device = CPU>
    class Tensor : public TensorBase<Tensor<Scalar, Device>>
    {
    private:
        using Base = TensorBase<Tensor<Scalar, Device>>;
        using Base::offset_;
        using Base::shape_;
        using Base::strides_;
        using AllocatorType = typename Device::template DefaultAllocator<Scalar>;

        std::shared_ptr<Storage<Scalar, AllocatorType>> storage_;
        // 视图构造函数
        Tensor(const Tensor &other,
               const std::vector<size_t> &new_shape,
               const std::vector<size_t> &new_strides,
               const size_t new_offset)
            : storage_(other.storage_)
        {
            shape_ = new_shape;
            strides_ = new_strides;
            offset_ = new_offset;
        }

    public:
        using value_type = Scalar;
        using device_type = Device;
        using allocator_type = typename Device::template DefaultAllocator<Scalar>;
        static_assert(std::is_arithmetic_v<Scalar>, "Scalar must be arithmetic type");
        // 构造函数
        Tensor()
        {
            shape_ = {0}; // 1维，长度为0
            strides_ = {1};
            offset_ = 0;
            storage_ = std::make_shared<Storage<Scalar, AllocatorType>>(0); // 分配0容量
        }
        explicit Tensor(const std::vector<size_t> &shape)
        {
            size_t total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
            storage_ = std::make_shared<Storage<Scalar, AllocatorType>>(total_size); // 接受参数为size_t，返回shared_ptr<Storage<Scalar,AllocatorType>>
            std::fill(storage_->data(), storage_->data() + total_size, 0);
            // std::cout << "从形状构造Tensor，total_size: " << total_size << std::endl;
            shape_ = shape;
            strides_ = this->compute_strides(shape); // 计算strides
            offset_ = 0;
        }
        // 1. 从标量构造（广播）
        Tensor(Scalar value, const std::vector<size_t> &shape) noexcept
        {
            size_t total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
            storage_ = std::make_shared<Storage<Scalar, allocator_type>>(total_size);
            std::fill(storage_->data(), storage_->data() + total_size, value);
            shape_ = shape;
            strides_ = this->compute_strides(shape); // 计算strides
            offset_ = 0;
        };
        // 2. 通过一维向量构造
        Tensor(std::initializer_list<Scalar> data)
        {
            // std::cout << "通过一维向量构造，data size: " << data.size() << std::endl;
            // std::vector<size_t> data(data_elements.begin(), data_elements.end());
            if (data.size() == 0)
            {
                throw std::runtime_error("data is empty");
            }
            storage_ = std::make_shared<Storage<Scalar, allocator_type>>(data.size());
            std::copy(data.begin(), data.end(), storage_->data());
            shape_ = {data.size()};
            strides_ = this->compute_strides(shape_);
            offset_ = 0;
        }
        // 3. 通过二维向量构造
        Tensor(std::initializer_list<std::initializer_list<Scalar>> data)
        {
            if (data.size() <= 0 || data.begin()->size() <= 0)
            {
                throw std::invalid_argument("data size is zero");
            }
            shape_ = {data.size(), data.begin()->size()};
            storage_ = std::make_shared<Storage<Scalar, allocator_type>>(data.size() * data.begin()->size());
            size_t i = 0;
            for (auto &row : data)
            {
                if (row.size() != shape_[1]) // 添加行长度检查
                    throw std::invalid_argument("inconsistent row sizes");
                std::copy(row.begin(), row.end(), storage_->data() + i);
                i += row.size();
            }
            strides_ = this->compute_strides(shape_);
            offset_ = 0;
        }
        // 4. 拷贝构造函数（深拷贝）
        Tensor(Tensor &&other) noexcept
            : storage_(std::make_shared<Storage<Scalar, AllocatorType>>(other.storage_->capacity()))
        {
            shape_ = other.shape_;
            strides_ = other.strides_;
            offset_ = other.offset_;
            if (other.storage_ && other.storage_->data())
            {
                std::copy(other.data(), other.data() + other.size(), this->data());
            }
        }
        // 析构函数
        ~Tensor() = default;

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

        // 创建新形状视图
        Tensor reshape(const std::vector<size_t> &new_shape)
        {
            size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());
            if (new_size != this->size())
            {
                std::cout << "New shape size " << new_size << " does not match tensor size " << std::endl;
                throw std::runtime_error("New shape size does not match tensor size");
            }
            return Tensor(*this, new_shape, this->compute_strides(new_shape), offset_);
        }
        // 深拷贝
        Tensor clone() const
        {
            Tensor new_tensor(shape_);
            std::copy(data(), data() + this->size(), new_tensor.data());
            return new_tensor;
        }

        // 移动赋值运算符
        Tensor &operator=(Tensor &&other) noexcept
        {
            if (this != &other)
            {
                shape_ = std::move(other.shape_);
                strides_ = std::move(other.strides_);
                offset_ = other.offset_;
                storage_ = std::move(other.storage_);
                other.offset_ = 0;
                other.shape_.clear();
                other.strides_.clear();
            }
            return *this;
        }
        // 加法运算符
        template <typename otherScalar, typename otherDevice>
        Tensor operator+(const Tensor<otherScalar, otherDevice> &other) const
        {
            static_assert(std::is_same_v<Scalar, otherScalar>, "Incompatible types for + operation");
            static_assert(std::is_same_v<Device, otherDevice>, "Incompatible devices for + operation");
            if (shape_ != other.shape_ || shape_.empty())
            {
                throw std::runtime_error("Incompatible shapes for + operation");
            }
            Tensor result(shape_);
            for (size_t i = 0; i < this->size(); i++)
            {
                result(i) = data()[i] + other.data()[i];
            }
            return result;
        }

        template <typename... Indices>
        Scalar &operator()(Indices... indices)
        {
            if (!storage_ || !storage_->data())
                throw std::runtime_error("Tensor data is null");
            size_t index = this->compute_index(indices...);
            if (index >= this->size())
            {
                throw std::out_of_range("Index out of bounds");
            }
            return storage_->data()[index];
        }
    };
} // namespace tiny_dl