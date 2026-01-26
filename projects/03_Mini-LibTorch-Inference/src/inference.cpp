#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>

std::vector<char> read_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Failed to read file: " + filename);
    }
    
    return buffer;
}

int main()
{
    //加载模型
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("./models/resnet50.pt"); //加载模型
        std::cout << "Load module success!\n"<< std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model"<<e.what() << std::endl;
        return -1;
    }

    //加载example_input
    torch::Tensor example_input;
    try{
        // 读取文件
        std::vector<char> input_data = read_file("./models/example_input.pt");
        
        // 使用 pickle_load 加载张量
        torch::IValue input_ivalue = torch::pickle_load(input_data);
        example_input = input_ivalue.toTensor();
        std::cout << "Load example_input success!\n" << std::endl;
    }catch (...) {
        example_input = torch::ones({1, 3, 224, 224}, torch::kFloat);
        std::cout << "Load example_input failed, use default input!\n"<< std::endl;
    }

    //推理
    module.eval();
    torch::NoGradGuard no_grad;
    auto start = std::chrono::system_clock::now();
    torch::Tensor output = module.forward({example_input}).toTensor();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Inference Success!" << std::endl;
    std::cout << "Inference time: " << elapsed_seconds.count() << "s\n"<< std::endl;

    //查看结果
    torch::Tensor example_output;
    try{
        // 读取文件
        std::vector<char> output_data = read_file("./models/example_output.pt");
        
        // 使用 pickle_load 加载张量
        torch::IValue output_ivalue = torch::pickle_load(output_data);
        example_output = output_ivalue.toTensor();
        std::cout << "Load example_output success!\n" << std::endl;
    }
    catch (const c10::Error& e) {
        std::cout << "Load example_output failed!"<<e.what()<< std::endl;
        return 0;
    }

    const float max_diff = 1e-5;
    if (torch::abs(output - example_output).max().item<float>() < max_diff) {
        std::cout << "Inference result is correct!\n"<< std::endl;
    }
    else {
        std::cout << "Inference result is wrong!\n"<< std::endl;
    }
}