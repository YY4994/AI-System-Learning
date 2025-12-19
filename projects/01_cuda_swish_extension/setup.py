from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension(
        name='swish_cuda',
        sources=[
            'src/swish_cuda.cpp',
            'swish_cuda_kernel.cu'
        ],
        include_dirs=['include'],
    ),
]

setup(
    name='swish_cuda',
    version='0.1.0',
    description='CUDA extension for Swish activation function',
    author='Huang Yuyao',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
)