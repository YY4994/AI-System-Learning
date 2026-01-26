cmake_minimum_required(VERSION 3.16)
project(inference_app)

# 1. 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. 查找LibTorch包（自动处理路径和依赖）
set(Torch_DIR "/root/miniconda3/lib/python3.12/site-packages/torch/share/cmake/Torch")
find_package(Torch REQUIRED)

# 3. 添加可执行文件
add_executable(inference_app src/inference.cpp)

# 4. 链接LibTorch（单行解决所有库依赖）
target_link_libraries(inference_app "${TORCH_LIBRARIES}")