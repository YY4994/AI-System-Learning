# Tensor类重构方案：引入显式TensorImpl

## 🎯 重构的核心目标与原理

### **为什么必须引入显式TensorImpl？**

当前融合设计的根本矛盾：**Tensor同时承担了太多职责**，导致自动求导时出现**无法解决的角色冲突**。

**具体冲突**：

1. **身份冲突**：Tensor既是“数据容器”又是“计算图节点”
2. **所有权循环风险**：`Tensor A` 持有 `Tensor B` 作为梯度，`Tensor B` 又可能引用其他Function...
3. **版本控制缺失**：无法检测in-place操作（如 `x.relu_()` 会修改原数据，破坏反向传播）

### **核心设计原则：单一职责分离**

```
重构前：
Tensor = 数据存储 + 视图参数 + 自动求导元数据 + 用户接口

重构后：
Tensor (用户句柄) → 仅提供用户接口
    ↓ 持有
TensorImpl (实现层) → 数据 + 视图 + 自动求导状态
    ↓ 持有  
Storage (纯数据块) → 原始内存
```

## 🏗️ 新架构详细设计

### **1. Storage层（不变，保持现有）**

```cpp
template <typename Scalar, typename Allocator>
class Storage {
    // 保持你现有的设计，纯数据容器
    Scalar* data_;
    size_t capacity_;
    Allocator allocator_;
    // ... 分配/释放方法
};
```

### **2. TensorImpl层（新增，核心实现层）**

这是**自动求导系统的核心承载者**，每个唯一的（数据+视图+梯度状态）对应一个TensorImpl实例。

```cpp
template <typename Scalar, typename Device>
class TensorImpl {
private:
    // === 第一部分：数据与视图（不可变核心）===
    std::shared_ptr<Storage<Scalar>> storage_;  // 数据所有权
    std::vector<size_t> shape_;                 // 形状（创建后应不变）
    std::vector<size_t> strides_;               // 步长（创建后应不变）
    size_t offset_ = 0;                         // 存储中的偏移

    // === 第二部分：自动求导状态（可变）===
    std::weak_ptr<Function> grad_fn_;           // 关键：弱引用到创建者Function
    std::shared_ptr<TensorImpl> grad_;          // 梯度本身是另一个TensorImpl
    bool requires_grad_ = false;

    // === 第三部分：版本与元数据 ===
    size_t version_ = 0;                        // 用于检测in-place修改
    size_t unique_id_;                          // 唯一标识符，用于调试

public:
    // 构造函数：从数据创建
    TensorImpl(std::shared_ptr<Storage<Scalar>> storage, 
               std::vector<size_t> shape,
               std::vector<size_t> strides,
               size_t offset);

    // 构造函数：创建视图（共享数据）
    TensorImpl(std::shared_ptr<TensorImpl> other,  // 共享storage_
               std::vector<size_t> new_shape,
               std::vector<size_t> new_strides,
               size_t new_offset);

    // === 关键方法 ===

    // 数据访问
    Scalar* data() { return storage_->data() + offset_; }

    // 自动求导相关
    void set_gradient(std::shared_ptr<TensorImpl> grad) { 
        grad_ = grad; 
    }

    void set_grad_fn(std::shared_ptr<Function> fn) { 
        grad_fn_ = fn;  // 弱引用！不增加Function引用计数
    }

    void mark_modified() { version_++; }  // in-place操作时调用

    // 检查输入是否有效（用于Function的backward）
    bool is_valid_input(size_t saved_version) const {
        return version_ == saved_version;  // 版本号匹配说明未被修改
    }

    // 创建视图（工厂方法）
    std::shared_ptr<TensorImpl> view(std::vector<size_t> new_shape);
};
```

### **3. Tensor层（用户句柄，轻量级）**

这是**用户唯一直接接触的类**，非常轻量，拷贝成本低。

```cpp
template <typename Scalar, typename Device>
class Tensor {
private:
    // 唯一的核心数据成员：指向实现的共享指针
    std::shared_ptr<TensorImpl<Scalar, Device>> impl_;

public:
    using ImplType = TensorImpl<Scalar, Device>;

    // === 构造函数 ===

    // 1. 从现有实现创建
    explicit Tensor(std::shared_ptr<ImplType> impl) : impl_(impl) {}

    // 2. 从形状创建（分配新存储）
    explicit Tensor(std::vector<size_t> shape) {
        size_t total_size = compute_total_size(shape);
        auto storage = std::make_shared<Storage<Scalar>>(total_size);
        auto strides = compute_strides(shape);
        impl_ = std::make_shared<ImplType>(storage, shape, strides, 0);
    }

    // 3. 从标量值创建（广播）
    Tensor(Scalar value, std::vector<size_t> shape) {
        *this = Tensor(shape);
        std::fill(data(), data() + size(), value);
    }

    // === 数据访问（委托给impl_）===

    Scalar* data() { return impl_->data(); }
    const std::vector<size_t>& shape() const { return impl_->shape(); }
    size_t size() const { return impl_->size(); }

    // 索引访问（关键：返回新Tensor句柄，而不是引用）
    Tensor operator()(std::initializer_list<size_t> indices) {
        // 计算元素偏移
        size_t elem_index = impl_->compute_offset(indices);

        // 创建标量Tensor的视图（形状[1]，偏移到具体元素）
        // 注意：这是视图，共享数据
        return Tensor(impl_->view_single_element(elem_index));
    }

    // === 自动求导接口 ===

    Tensor grad() const {
        if (!impl_->grad()) {
            return Tensor();  // 返回空Tensor
        }
        return Tensor(impl_->grad());  // 包装成Tensor句柄
    }

    void set_grad(const Tensor& grad) {
        // 重要：grad.impl_ 是 grad的TensorImpl
        impl_->set_gradient(grad.impl_);
    }

    bool requires_grad() const { return impl_->requires_grad(); }
    void requires_grad(bool requires) { impl_->set_requires_grad(requires); }

    // === 视图操作（返回新Tensor句柄）===

    Tensor view(std::vector<size_t> new_shape) {
        return Tensor(impl_->view(new_shape));
    }

    Tensor reshape(std::vector<size_t> new_shape) {
        // 验证大小匹配，然后创建视图
        return view(new_shape);
    }

    // === 运算符重载（示例：加法）===

    Tensor operator+(const Tensor& other) const {
        // 1. 执行实际计算（分配新存储）
        Tensor result(this->shape());
        elementwise_add(this->data(), other.data(), result.data(), size());

        // 2. 如果需要梯度，创建Function记录计算
        if (this->requires_grad() || other.requires_grad()) {
            auto add_fn = std::make_shared<AddFunction>();

            // 关键：设置新Tensor的grad_fn
            result.impl_->set_grad_fn(add_fn);

            // Function需要记录输入信息
            add_fn->save_inputs({
                InputInfo{this->impl_, this->impl_->version()},
                InputInfo{other.impl_, other.impl_->version()}
            });

            // 设置输出
            add_fn->set_output(result.impl_);

            // 新Tensor需要梯度
            result.requires_grad(true);
        }

        return result;
    }

    // === 反向传播入口 ===

    void backward(Tensor grad = Tensor()) {
        if (!grad.impl_) {
            // 创建全1的梯度，形状与this相同
            grad = Tensor(1.0, this->shape());
        }

        // 委托给impl_的反向传播引擎
        AutogradEngine::backward(impl_, grad.impl_);
    }
};
```

## 🔗 关键关系与内存管理

### **1. 所有权关系（无循环引用）**

```
Tensor A ──持有──> TensorImpl_A ──持有──> Storage
    ↑                      ↓
    |                weak_ptr<Function>（不增加计数）
    |                      ↓
Tensor grad_A ←─持有── TensorImpl_grad_A
```

### **2. 版本控制的工作流程**

```cpp
// in-place操作示例：ReLU原地激活
void relu_(Tensor& input) {
    // 修改数据
    for (size_t i = 0; i < input.size(); ++i) {
        if (input.data()[i] < 0) input.data()[i] = 0;
    }

    // 关键：标记已修改
    input.impl_->mark_modified();  // version_++
}

// 反向传播时检测
bool Function::check_inputs() {
    for (auto& input_info : saved_inputs_) {
        // 提升弱引用为shared_ptr
        auto impl = input_info.impl_weak.lock();

        if (!impl) {
            // TensorImpl已被释放，计算图无效
            return false;
        }

        if (impl->version() != input_info.saved_version) {
            // 版本不匹配，输入被in-place修改过
            throw std::runtime_error("One of the inputs has been modified in-place");
        }
    }
    return true;
}
```

### **3. 梯度累加模式**

```cpp
// 当多个Function贡献给同一个Tensor的梯度时：
// TensorImpl内部：
void accumulate_gradient(std::shared_ptr<TensorImpl> new_grad) {
    if (!grad_) {
        grad_ = new_grad;  // 第一次，直接赋值
    } else {
        // 后续，累加：grad_ = grad_ + new_grad
        // 需要创建新的TensorImpl来存储累加结果
        grad_ = elementwise_add(grad_, new_grad);
    }
}
```

## 🚀 重构的优势

1. **解决循环引用**：Tensor句柄轻量，TensorImpl有清晰的所有权链
2. **版本控制**：支持in-place操作检测
3. **调试友好**：每个TensorImpl有唯一ID，便于跟踪计算图
4. **性能优化**：可在TensorImpl层做梯度累加等优化，对用户透明
5. **扩展性**：易于添加新特性（如设备迁移、序列化）

# 
