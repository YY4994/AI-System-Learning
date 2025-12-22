# CUDA基础

## 1.CUDA 的核心概念

- GPU 和 CPU 的协同计算
  CPU：负责运行主程序（主机代码，Host Code），将任务分配给 GPU。
  GPU：负责运行数据并行的任务（设备代码，Device Code），执行大量并行计算。

- CUDA 核函数 (Kernel)
  Kernel 是 CUDA 程序中的核心函数，它运行在 GPU 上的所有线程中。
  核函数有以下要求：
  
  1. 核函数只能访问GPU内存
  2. 核函数不能使用变长参数
  3. 核函数不能使用静态变量
  4. 核函数不能使用函数指针
  5. 核函数具有异步性
  
  通过 __global__ 关键字定义：

```cpp
__global__ void myKernel(int *a) {
    int i = threadIdx.x; // 当前线程索引
    a[i] = a[i] * 2;     // 并行计算
}
```

- 线程模型
  
  ```
  网格（Grid） → 线程块（Thread Block） → 线程（Thread）
         │               │                │
     顶层并行           中层并行          底层并行
  （多任务级别）        （协作级别）      （执行单元）
  ```
  
  * 线程 (Thread)：执行计算的基本单元。
  * 线程块 (Thread Block)：由一组线程组成，可以共享数据。
  * 网格 (Grid)：由多个线程块组成。

```
        ┌───────────────────────────────────── 网格（Grid） ────────────────────────────────────┐
        │                                                                                      │
        │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐           │
        │  │ 线程块     │  │ 线程块     │  │ 线程块     │  │ 线程块     │  │ 线程块     │  ...           │
        │  │ Block 0   │  │ Block 1   │  │ Block 2   │  │ Block 3   │  │ Block 4   │           │
        │  └───────────┘  └───────────┘  └───────────┘  └───────────┘  └───────────┘           │
        │                                                                                      │
        └──────────────────────────────────────────────────────────────────────────────────────┘

        每个线程块内部：
        ┌───────────┐
        │ Block X   │
        ├───────────┤
        │ Thread 0  │ ← 线程（最小执行单元）
        │ Thread 1  │
        │ Thread 2  │
        │ Thread 3  │
        │ ...       │
        │ Thread 255│
        └───────────┘
```

```cpp
dim3 gridSize(4, 1, 1);   // 网格：4个线程块（一维）
dim3 blockSize(256, 1, 1); // 每个块：256个线程（一维）
// 总线程数 = gridSize.x * blockSize.x = 4 * 256 = 1

// 处理512×512的图像
dim3 gridSize(32, 32, 1);     // 32×32网格 = 1024个线程块
dim3 blockSize(16, 16, 1);    // 每个块16×16=256个线程
// 总线程数 = 32×32×16×16 = 262,144个线程
```

- 内存模型

        全局内存（Global Memory）：GPU 所有线程都可以访问，访问速度慢但容量大。
        共享内存（Shared Memory）：线程块内的线程共享的数据，速度快但容量小。
        寄存器（Registers）：线程私有的内存，速度最快。
        常量内存（Constant Memory）：只读内存，所有线程共享。
        纹理内存 / 表面内存：用于特殊优化的数据访问场景。

## 2.CUDA 程序结构

### 2.1简化版本

一个典型的 CUDA 程序分为以下几个步骤：

1. 主机代码初始化（CPU）：分配和初始化内存。从CPU内存中拷贝数据到GPU内存

2. 从CPU内存中拷贝数据到GPU：内存将数据传输到 GPU（cudaMemcpy）。

3. 内核函数执行（GPU）：调用核函数（<<<gridSize, blockSize>>>），GPU 执行并行任务。

4. 结果传回主机：将 GPU 上的数据传回 CPU。

5. 释放资源：释放主机和设备内存。

### 2.2详细版本

#### 2.2.1 初始化与环境设置

选择设备：通过调用cudaSetDevice选择要使用的GPU设备。 

分配内存：使用cudaMalloc为设备端变量分配内存，使用cudaMemcpy将数据从主机复制到设备。

#### 2.2.2 编写核函数

定义核函数：使用__global__关键字声明核函数，指定输入输出参数和执行逻辑。

 启动核函数：通过<<<...>>>语法配置网格和块尺寸，并启动核函数。例如，kernel<<<gridDim, blockDim>>>(args)。

#### 2.2.3 执行与同步

 异步执行：可选地使用流（Stream）来并发执行多个内核或拷贝操作，提升效率。

 同步操作：使用cudaDeviceSynchronize等待所有先前启动的内核执行完毕，确保结果可用。

#### 2.2.4 结果回收与清理

 获取结果：使用cudaMemcpy将计算结果从设备复制回主机。

 释放资源：调用cudaFree释放设备端分配的内存，调用cudaDeviceReset重置设备状态。

## 3. CUDA编程模型

CUDA采用的是**单指令多线程（SIMT）**结构，在这种架构下，一组线程（通常称为warp）会同时执行相同的指令，但作用于不同的数据。

为了使每个线程知道它应该处理的数据位置，CUDA提供了几个内置变量：

- blockIdx：当前线程块在整个网格中的索引。

- threadIdx：当前线程在其所属块内的索引。

- blockDim：当前块的维度大小。

- gridDim：整个网格的维度大小。
  通过组合这些变量，我们可以计算出每个线程的唯一ID，进而确定该线程应处理的数据位置

```cpp
int tid = blockIdx.x * blockDim.x + threadIdx.x; // 一维线程ID
int tid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + 
          threadIdx.y * blockDim.x + threadIdx.x; // 二维线程ID
```

## 4.helloworld

```cpp
#include<stdio.h>
__global__ void helloWorldFromGPU(){
    printf("hello world from GPU!\n\n");
}

int main(int argc, char const *argv[])
{
    helloWorldFromGPU<<<1,10>>>();
    return 0;
```

执行：

```shell-session
nvcc helloworld.cu -o main
main
```



tips:

- 数据并行化思维：将问题分解为可以并行执行的小任务。例如，向量加法可以通过让每个线程负责一对元素的加法操作来实现。

- 内存管理：需要手动管理设备内存（通过cudaMalloc和cudaFree），并且要考虑到主机与设备之间的数据传输成本。

- 同步机制：由于并行执行的特点，线程间的同步变得至关重要。例如，块内的线程可能需要使用__syncthreads()确保它们在继续执行之前完成某些关键步骤。


