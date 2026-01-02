#pragma once

#include "tensor.hpp"

namespace tiny_dl
{
    template <typename Scalar, typename Device>
    // 安全地保存输入Tensor的“快照”，用于反向传播。
    class SavedTensor
    {
        using TensorType = Tensor<Scalar, Device>;
        using ImplType = TensorImpl<Scalar, Device>;

    private:
        std::weak_ptr<void> weak_impl_; // 使用void指针以支持不同类型的Tensor
        std::vector<size_t> saved_shape_;
        std::vector<size_t> saved_strides_;
        size_t saved_offset_;
        size_t saved_version_;
        bool saved_requires_grad_;

    public:
        SavedTensor(ImplAccessToken token, const TensorType &tensor)
        {
            weak_impl_ = tensor.weak_impl(token);
            saved_shape_ = tensor.impl(token)->shape();
            saved_strides_ = tensor.impl(token)->strides();
            saved_offset_ = tensor.impl(token)->offset();
            saved_requires_grad_ = tensor.impl(token)->requires_grad();
            saved_version_ = tensor.impl(token)->version();
        }

        // 恢复Tensor：检查版本，如果有效则返回强引用
        TensorType recover(ImplAccessToken token)
        {
            auto impl = weak_impl_.lock();
            if (!impl || impl->version() != saved_version_)
                return TensorType(); // 返回空Tensor，表示无效
            return TensorType(impl);
        }

        bool is_valid(ImplAccessToken token)
        {
            auto impl = weak_impl_.lock();
            return impl && impl->version() == saved_version_;
        }
    };

    // 基类，所有函数的基类
    template <typename Scalar, typename Device>
    class Function
    {
        using TensorType = Tensor<Scalar, Device>;
        using ImplType = TensorImpl<Scalar, Device>;

    protected:
        std::vector<SavedTensor<Scalar, Device>> saved_inputs_; // 保存的中间结果
        std::vector<std::weak_ptr<ImplType>> output_impls_;

        size_t sequence_number_;      // 在计算图中的顺序（用于调试） = 0; // 需要的输入数量
        bool isdifferentable_ = true; // 是否可微分

        ImplAccessToken get_token() { return ImplAccessToken(); }

        // 保存中间结果（用于反向传播，foreward中调用）
        void save_inputs(ImplAccessToken token, const TensorType &inputs)
        {
            saved_inputs_.clear();
            for (auto &input : inputs)
            {
                saved_inputs.push_back(SavedTensor(token, input));
            }
        }
        // 标记输出（建立grad_fn关联）
        void mark_outputs(ImplAccessToken token, std::shared_ptr<TensorType> outputs)
        {
            output_impls_.clear();
            for (auto &output : outputs)
            {
                output.impl(token)->set_grad_fn(shared_from_this()); // Tensor c “知道”了自己是由哪个Function计算产生的
                output_impls_.push_back(output.weak_impl(token));    // 让Function对象（父）也能知道它生成了哪些Tensor（子）
            }
        }
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
                size_t source_dim_idx = source_dims - 1 - dim_offset;
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
        Function()
        {
            static size_t global_seq = 0;
            sequence_number_ = global_seq++;
        }
        virtual ~Function() = default;
        // === 核心接口 ===

        // 前向传播：接收Tensor句柄，返回Tensor句柄
        virtual std::vector<TensorType> forward(std::vector<TensorType> inputs) = 0;

        // 反向传播：接收输出梯度，返回输入梯度
        virtual std::vector<TensorType> backward(std::vector<TensorType> grad_outputs) = 0;

        // 恢复输入（在backward()中调用）
        std::vector<TensorType> get_saved_inputs(ImplAccessToken token)
        {
            std::vector<TensorType> inputs;
            for (auto &saved_input : saved_inputs_)
            {
                auto input = saved_input.recover(token);
                if (!input)
                {
                    throw std::runtime_error("Saved input is no longer valid");
                }
                inputs.push_back(*input);
            }
            return inputs;
        }

        // 获取输出的弱引用（用于构建计算图）
        const auto &get_output_impls() const { return output_impls_; }

        // 是否可微分（某些操作如reshape不可微）
        virtual bool is_differentiable() const { return is_differentiable_; }

        // 释放保存的中间状态（内存优化）
        virtual void release_resources()
        {
            saved_inputs_.clear();
            // 但保留output_impls_（需要知道哪些Tensor是自己创建的）
        }
    };

    template <typename Scalar, typename Device = CPU>
    class AddFunction : public Function<Scalar, Device>
    {
    public:
        using TensorType = typename Function<Scalar, Device>::TensorType;
        std::vector<TensorType> forward(std::vector<TensorType> inputs) override
        {
            if (inputs.size() != 2)
            {
                throw std::runtime_error("AddFunction requires exactly 2 inputs");
            }

            ImplAccessToken token = this->get_token();
            auto input1 = inputs[0];
            auto input2 = inputs[1];
            this->save_inputs(token, {input1, input2}); // 保存输入
            auto input_shape1 = input1.shape();
            auto input_shape2 = input2.shape();
            auto output_shape = broadcast_shape(input_shape1, input_shape2);
            TensorType output(output_shape); // 创建输出Tensor

            if (input_shape1 == input_shape2)
            {
                for (size_t i = 0; i < input_shape1; ++i)
                {
                    output.impl(token)->data()[i] = input1.data()[i] + input2.data()[i];
                }
            }
            else if (input_shape1.size() == 1)
            {
                for (size_t i = 0; i < input_shape2.size(); ++i)
                {
                    output.impl(token)->data()[i] = input1.data()[0] + input2.data()[i];
                }
            }
            else if (input_shape2.size() == 1)
            {
                for (size_t i = 0; i < input_shape1.size(); ++i)
                {
                    output.impl(token)->data()[i] = input1.data()[i] + input2.data()[0];
                }
            }
            else
            {
                // 获取广播后形状
                auto total_dim = output_shape.size();
                auto total_size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<size_t>());

                std::vector<size_t> idx(total_dim, 0); // 初始化索引为0
                // 遍历每一个元素，将一维索引映射为多维索引
                for (size_t i = 0; i < total_size; ++i)
                {
                    size_t temp = i;
                    for (size_t j = total_dim - 1; j >= 0; --j)
                    {
                        idx[j] = temp % output_shape[j];
                        temp /= output_shape[j];
                    }
                    // 计算当前索引对应的元素在源张量中索引
                    size_t idx1 = map_index(idx, input_shape1, output_shape);
                    size_t idx2 = map_index(idx, input_shape2, output_shape);

                    output.impl(token)->data()[i] = input1.data()[idx1] + input2.data()[idx2];
                }
            }

            if (input1.requires_grad() || input2.requires_grad())
            {
                this->mark_outputs(token, {output});
                output.set_requires_grad(true);
            }

            return {output};
        }

        std::vector<TensorType> backward(std::vector<TensorType> grad_outputs) override
        {
            if (grad_outputs.size() != 1)
            {
                throw std::runtime_error("AddFunction requires exactly 1 grad_output");
            }
            TensorType grad_c = grad_outputs[0];

            ImplAccessToken token = this->get_token();
            auto inputs = this->get_saved_inputs(token);
            if (inputs.size() != 2)
            {
                throw std::runtime_error("Saved inputs size mismatch in AddFunction backward");
            }
            TensorType input1 = inputs[0];
            TensorType input2 = inputs[1];
            auto shape1 = input1.shape();
            auto shape2 = input2.shape();
            if (shape1 == shape2)
            {
                TensorType grad_input1 = grad_c.clone();
                TensorType grad_input2 = grad_c.clone();
                return {grad_input1, grad_input2};
            }
            else
            {
                // 处理广播情况下的梯度
                auto c_shape = grad_c.shape();
                TensorType grad_input1 = Tensor(0, shape1);
                TensorType grad_input2 = Tensor(0, shape2);
                size_t total_dim = c_shape.size();
                std::vector<size_t> idx(total_dim, 0);

                // 对每个维度进行求和还原
                for (size_t i = 0; i < total_dim; ++i)
                {
                    size_t temp = i;
                    for (size_t j = total_dim - 1; j >= 0; --j)
                    {
                        idx[j] = temp % c_shape[j];
                        temp /= c_shape[j];
                    }
                    size_t idx1 = this->map_index(idx, shape1, c_shape);
                    size_t idx2 = this->map_index(idx, shape2, c_shape);
                    grad_input1.data()[idx1] += grad_c.data()[i];
                    grad_input2.data()[idx2] += grad_c.data()[i];
                }
                return {grad_input1, grad_input2};
            }
        }

        bool is_differentiable() const override { return true; }
    };

    class MulFunction : public Function<Scalar, Device>
    {
    public:
        using TensorType = typename Function<Scalar, Device>::TensorType;
        std::vector<TensorType> forward(std::vector<TensorType> inputs) override
        {
            if (inputs.size() != 2)
            {
                throw std::runtime_error("AddFunction requires exactly 2 inputs");
            }

            ImplAccessToken token = this->get_token();
            auto input1 = inputs[0];
            auto input2 = inputs[1];
            this->save_inputs(token, {input1, input2}); // 保存输入
            auto input_shape1 = input1.shape();
            auto input_shape2 = input2.shape();
            auto output_shape = broadcast_shape(input_shape1, input_shape2);
            TensorType output(output_shape); // 创建输出Tensor

            if (input_shape1 == input_shape2)
            {
                for (size_t i = 0; i < input_shape1; ++i)
                {
                    output.impl(token)->data()[i] = input1.data()[i] * input2.data()[i];
                }
            }
            else if (input_shape1.size() == 1)
            {
                for (size_t i = 0; i < input_shape2.size(); ++i)
                {
                    output.impl(token)->data()[i] = input1.data()[0] * input2.data()[i];
                }
            }
            else if (input_shape2.size() == 1)
            {
                for (size_t i = 0; i < input_shape1.size(); ++i)
                {
                    output.impl(token)->data()[i] = input1.data()[i] * input2.data()[0];
                }
            }
            else
            {
                // 获取广播后形状
                auto total_dim = output_shape.size();
                auto total_size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<size_t>());

                std::vector<size_t> idx(total_dim, 0); // 初始化索引为0
                // 遍历每一个元素，将一维索引映射为多维索引
                for (size_t i = 0; i < total_size; ++i)
                {
                    size_t temp = i;
                    for (size_t j = total_dim - 1; j >= 0; --j)
                    {
                        idx[j] = temp % output_shape[j];
                        temp /= output_shape[j];
                    }
                    // 计算当前索引对应的元素在源张量中索引
                    size_t idx1 = map_index(idx, input_shape1, output_shape);
                    size_t idx2 = map_index(idx, input_shape2, output_shape);

                    output.impl(token)->data()[i] = input1.data()[idx1] * input2.data()[idx2];
                }
            }

            if (input1.requires_grad() || input2.requires_grad())
            {
                this->mark_outputs(token, {output});
                output.set_requires_grad(true);
            }

            return {output};
        }

        std::vector<TensorType> backward(const std::vector<TensorType> &grad_outputs) override
        {
            if (grad_outputs.size() != 1)
            {
                throw std::runtime_error("MulFunction requires exactly 1 grad_output");
            }
            ImplAccessToken token = this->get_token();
            auto grad_output = grad_outputs[0];
            auto inputs = this->load_inputs(token);
            auto input1 = inputs[0];
            auto input2 = inputs[1];
            auto shape1 = input1.shape();
            auto shape2 = input2.shape();
            if (shape1 == shape2)
            {
                TensorType grad_input1 = grad_output * input2;
                TensorType grad_input2 = grad_output * input1;
                return {grad_input1, grad_input2};
            }
            else
            {
                auto c_shape = grad_output.shape();
                TensorType grad_input1 = Tensor(0, shape1);
                TensorType grad_input2 = Tensor(0, shape2);
                size_t total_dim = c_shape.size();
                std::vector<size_t> idx(total_dim, 0);

                for (int i = 0; i < total_dim; i++)
                {
                    size_t temp = i;
                    for (size_t j = total_dim - 1; j >= 0; --j)
                    {
                        idx[j] = temp % c_shape[j];
                        temp /= c_shape[j];
                    }
                    size_t idx1 = this->map_index(idx, shape1, c_shape);
                    size_t idx2 = this->map_index(idx, shape2, c_shape);
                    grad_input1.data()[idx1] += grad_output.data()[i] * input2.data()[idx2];
                    grad_input2.data()[idx2] += grad_output.data()[i] * input1.data()[idx1];
                }
                return {grad_input1, grad_input2};
            }
        }
    };
}