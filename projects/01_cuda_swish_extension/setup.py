from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='my_swish',
    version='0.1.0',
    description='CUDA extension for Swish activation function',
    author='Huang Yuyao',
    
    ext_modules=[
        CUDAExtension(
            name='my_swish',
            sources=[
                'src/swish_cuda.cpp',
                'src/swish_cuda_kernel.cu'
            ],
            include_dirs=[
                os.path.join(current_dir, 'include'),  # 确保路径正确
            ],
            extra_compile_args={
                'cxx': ['-O2', '-fPIC'],
                'nvcc': [
                    '-O2',
                    '-Xcompiler', '-fPIC',
                    '--expt-relaxed-constexpr',  # 重要：允许constexpr
                    '--use_fast_math',           # 快速数学（可选）
                ]
            }
        ),
    ],
    
    cmdclass={'build_ext': BuildExtension},
)