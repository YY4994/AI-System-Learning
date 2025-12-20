import torch
import my_swish  # 你编译的模块名
import torch.nn.functional as F

def test_dtypes():
    """测试不同数据类型"""
    print("=== 测试数据类型支持 ===")
    
    # 测试 float32
    x_f32 = torch.randn(4, 4, device='cuda', dtype=torch.float32, requires_grad=True)
    y_custom_f32 = my_swish.my_swish(x_f32)
    y_ref_f32 = F.silu(x_f32)  # PyTorch的Swish叫SiLU
    print(f"float32 前向匹配: {torch.allclose(y_custom_f32, y_ref_f32, rtol=1e-5)}")
    
    # 测试 float64  
    x_f64 = torch.randn(4, 4, device='cuda', dtype=torch.float64, requires_grad=True)
    y_custom_f64 = my_swish.my_swish(x_f64)
    y_ref_f64 = F.silu(x_f64)
    print(f"float64 前向匹配: {torch.allclose(y_custom_f64, y_ref_f64, rtol=1e-5)}")
    
    return x_f32, x_f64

def test_gradients():
    """测试梯度计算"""
    print("\n=== 测试梯度计算 ===")
    
    # 使用 torch.autograd.gradcheck（严格的数值梯度检查）
    test_input = torch.randn(3, 3, device='cuda', dtype=torch.float64, requires_grad=True)
    test_input = test_input.double()  # gradcheck需要高精度
    
    test_passed = torch.autograd.gradcheck(
        lambda x: my_swish.my_swish(x),
        (test_input,),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3
    )
    print(f"gradcheck 通过: {test_passed}")
    
    return test_passed

def test_integration():
    """测试在真实网络中的集成"""
    print("\n=== 测试网络集成 ===")
    
    # 创建一个简单网络，使用你的Swish作为激活函数
    class SimpleNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
            
        def forward(self, x):
            x = self.linear(x)
            x = my_swish.my_swish(x)  # 使用你的自定义算子
            return x
    
    net = SimpleNet().cuda()
    x = torch.randn(5, 10, device='cuda', requires_grad=True)
    
    try:
        output = net(x)
        loss = output.sum()
        loss.backward()
        print("网络前向传播和反向传播成功!")
        print(f"输出形状: {output.shape}")
        if x.grad is not None:
            print(f"梯度计算成功，梯度形状: {x.grad.shape}")
    except Exception as e:
        print(f"网络测试失败: {e}")

def benchmark():
    """简单性能测试"""
    print("\n=== 简单性能测试 ===")
    
    # 与PyTorch原生实现对比
    x = torch.randn(1024, 1024, device='cuda')
    
    # 预热
    for _ in range(10):
        _ = my_swish.my_swish(x)
        _ = F.silu(x)
    
    # 计时
    torch.cuda.synchronize()
    import time
    
    # 自定义实现
    start = time.time()
    for _ in range(100):
        y_custom = my_swish.my_swish(x)
    torch.cuda.synchronize()
    custom_time = time.time() - start
    
    # PyTorch原生
    start = time.time()
    for _ in range(100):
        y_native = F.silu(x)
    torch.cuda.synchronize()
    native_time = time.time() - start
    
    print(f"自定义Swish 100次平均时间: {custom_time/100*1000:.2f} ms")
    print(f"PyTorch原生SiLU 100次平均时间: {native_time/100*1000:.2f} ms")
    print(f"速度比 (自定义/原生): {custom_time/native_time:.2f}x")

if __name__ == "__main__":
    print("开始测试 Swish CUDA 扩展...")
    
    # 基础测试
    x_f32, x_f64 = test_dtypes()
    
    # 梯度测试（可选，但推荐）
    # gradcheck可能较慢，但能验证反向传播完全正确
    gradient_ok = test_gradients()
    
    # 集成测试
    test_integration()
    
    # 性能测试
    benchmark()
    
    print("\n=== 测试完成 ===")