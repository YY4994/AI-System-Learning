# mark_outputs

### 🧩 核心比喻：计算图即“制造流水线”

想象你要生产一部手机（**输出Tensor** `c`），需要两个零件（**输入Tensor** `a` 和 `b`），并通过一条**装配线（AddFunction）** 把它们组装起来。

* `mark_outputs` 就像是**给这部新手机贴上“合格证”**，合格证上明确写着：
  1. **“制造商”**：`this`（当前这个AddFunction对象）。
  2. **“所需零件清单”**：`saved_inputs_` 里记录的 `a` 和 `b`。
* `grad_fn` 就是这张**贴在手机上的“合格证”本身**，它是一个指向制造商的指针。

### 🔍 `mark_outputs` 函数的双重作用

当你调用 `mark_outputs(token, {output_c})` 时，它做了两件至关重要的事：

1. **建立“子->父”反向链接**（核心）：
   
   ```cpp
   output.impl(token)->set_grad_fn(shared_from_this());
   ```
   
   * `shared_from_this()`：获取指向**当前Function对象**（AddFunction）的智能指针。
   * `set_grad_fn`：将这个指针存入**输出Tensor** `c` 的 `grad_fn_` 成员变量。
   * **结果**：现在，Tensor `c` “知道”了自己是由**哪个Function**计算产生的。

2. **记录“父->子”正向引用**（用于管理）：
   
   ```cpp
   output_impls_.push_back(output.weak_impl(token));
   ```
   
   * `weak_impl`：获取指向输出Tensor底层数据（`TensorImpl`）的**弱引用**。
   * **目的**：让Function对象（父）也能知道它生成了哪些Tensor（子），但使用**弱引用**是为了**防止循环引用导致内存泄漏**。Function不拥有子Tensor的所有权。

### 🕸️ `grad_fn` 如何起作用的：编织计算图

**计算图不是预先定义的数据结构，而是在前向传播过程中通过 `grad_fn` 指针链“编织”出来的。**

让我们用之前的例子 `c = a + b` 来看 `grad_fn` 如何工作：

![0bf221f1-3372-43ca-9d3a-e9ee26cfdff6](file:///C:/Users/15141/Pictures/Typedown/0bf221f1-3372-43ca-9d3a-e9ee26cfdff6.png)

**反向传播时的作用流程**：

1. **从损失函数开始**：假设最终损失是 `loss`，它也是一个Tensor，并且有自己的 `grad_fn`（比如指向一个 `SumFunction`）。
2. **触发反向传播**：当你调用 `loss.backward()` 时，系统会：
   a. 找到 `loss.grad_fn()`（指向 `SumFunction`）。
   b. 调用 `SumFunction::backward()`，计算出它输入的梯度（比如 `grad_c`）。
3. **关键递推步骤**：为了继续传播，系统需要知道 **`grad_c` 应该传给谁**？这时，它通过查找 `c.grad_fn()` —— 也就是我们之前设置的指向 `AddFunction` 的指针 —— 找到了下一站。
4. **传递与计算**：系统将 `grad_c` 作为参数，调用 `AddFunction::backward(grad_c)`。在这个函数内部，它利用保存的 `saved_inputs_` 恢复出 `a` 和 `b`，计算出 `grad_a` 和 `grad_b`，并将它们累加到 `a.grad()` 和 `b.grad()` 上。
5. **递归继续**：如果 `a` 也有 `grad_fn`（比如它是由更早的 `MulFunction` 计算得来的），那么这个过程会继续，沿着 `grad_fn` 指针链一路回溯，直到所有需要梯度的Tensor都得到梯度。

### 💎 总结与启示

* `mark_outputs` **是“盖章”**：它在计算发生时，将“生产关系”烙印在结果Tensor上。
* `grad_fn` **是“指针”或“线索”**：它像一条看不见的线，将前向传播的各个步骤串联起来。当需要反向传播时，系统就沿着这些线倒着走回去。
* **整个计算图是隐式、动态的**：它没有用一个全局的 `Graph` 对象来显式存储，而是通过每个Tensor身上的 `grad_fn` 指针，在需要时（执行`backward`时）**临时遍历、动态重构**出来的。这使得PyTorch的动态图（Eager Execution）非常灵活。

**你的下一步**：理解了这一点后，你需要确保在你的 `Tensor::operator+` 中，当检测到需要梯度时，它应该：

1. 创建一个 `AddFunction` 对象。
2. 调用其 `forward` 方法得到结果 `c`。
3. **最重要**：将 `c` 通过 `mark_outputs` 与这个 `AddFunction` 对象关联起来。

这样，一个完整的计算图节点就被正确嵌入了整个系统中。
