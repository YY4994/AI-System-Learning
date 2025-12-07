import time 
import torch

#cpu matrix multiplication
def cpu_matmul(mat1,mat2):
    start = time.perf_counter()
    result = torch.matmul(mat1,mat2)
    end = time.perf_counter()
    print(f"CPU matrix multiplication took {end-start} seconds")
    return result

#gpu matrix multiplication without data transfer time
def gpu_matmul_sync(mat1,mat2,device):
    start = time.perf_counter()
    mat1 = mat1.to(device)
    mat2 = mat2.to(device)
    end = time.perf_counter()
    cpugpu_time = end - start
    
    torch.cuda.synchronize()  # Ensure all previous GPU operations are complete
    
    start = time.perf_counter()
    result = torch.matmul(mat1,mat2)
    end = time.perf_counter()
    torch.cuda.synchronize()  # Ensure all previous GPU operations are complete
    matmul_time = end - start
    
    # Transfer result back to CPU
    start = time.perf_counter()
    result = result.to('cpu')
    end = time.perf_counter()
    gpucpu_time = end - start
    
    
    print(f"GPU matrix multiplication total time (including data transfer) took {cpugpu_time + matmul_time + gpucpu_time} seconds")
    print(f"GPU matrix multiplication (excluding data transfer) took {matmul_time} seconds")
    print(f"Data transfer CPU to GPU took {cpugpu_time} seconds")   
    print(f"Data transfer GPU to CPU took {gpucpu_time} seconds")
    return result

def gpu_matmul_async_compute(mat1,mat2,device):
    stream = torch.cuda.Stream()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start_event.record()
        result = torch.matmul(mat1,mat2)
        end_event.record()
    torch.cuda.synchronize(stream)
    compute_time = start_event.elapsed_time(end_event) / 1000.0  # Convert milliseconds to seconds
    print(f"GPU matrix multiplication with asynchronous compute took {compute_time} seconds")
    return result

def gpu_matmul_async(mat1,mat2,device):
    stream_transfer = torch.cuda.Stream()
    stream_compute = torch.cuda.Stream()
    
    total_start = time.perf_counter()
    with torch.cuda.stream(stream_transfer):
        mat1 = mat1.to(device)
        mat2 = mat2.to(device)

    with torch.cuda.stream(stream_compute):
        stream_transfer.synchronize()  # Ensure data transfer is complete before computation
        result = torch.matmul(mat1,mat2)

    with torch.cuda.stream(stream_transfer):
        result = result.to('cpu')
    
    torch.cuda.synchronize()  # Ensure all operations are complete
    total_end = time.perf_counter()
    total_time = total_end - total_start
    print(f"GPU matrix multiplication with asynchronous transfer and compute took {total_time} seconds")
    return result

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    if(device.type == 'cpu'):
        print("GPU is not available. Exiting.")
        exit(1)

    mat1 = torch.randn((1000, 1000))
    mat2 = torch.randn((1000, 1000))
    
    cpu_result = cpu_matmul(mat1, mat2)
    gpu_result_sync = gpu_matmul_sync(mat1, mat2, device)
    gpu_result_async_compute = gpu_matmul_async_compute(mat1.to(device), mat2.to(device), device)
    gpu_result_async_transfer = gpu_matmul_async(mat1, mat2, device)
    
    # Verify that results are the same
    assert torch.allclose(cpu_result, gpu_result_sync), "Results do not match for synchronous GPU multiplication!"
    assert torch.allclose(cpu_result, gpu_result_async_compute), "Results do not match for async compute GPU multiplication!"
    assert torch.allclose(cpu_result, gpu_result_async_transfer), "Results do not match for async transfer GPU multiplication!"
    print("All results match!") 
    
    
        
    