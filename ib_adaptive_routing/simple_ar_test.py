import os
import time
import torch
import torch.distributed as dist

def benchmark_allreduce(rank, world_size, tensor_size, num_iters=20):
    x = torch.ones(tensor_size, device='cuda', dtype=torch.float32)

    dist.barrier()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    end = time.time()

    avg_time_ms = (end - start) * 1000 / num_iters
    if rank == 0:
        print(f"AllReduce tensor size: {tensor_size} elements, Avg time: {avg_time_ms:.3f} ms")
    return avg_time_ms

def main():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())

    tensor_sizes = [2**13, 2**15, 2**17, 2**19]  # 不同大小的数据测试
    num_iters = 20

    print(f"[Rank {rank}] Testing with NCCL_IB_AR_THRESHOLD = {os.environ.get('NCCL_IB_AR_THRESHOLD')}")

    for size in tensor_sizes:
        benchmark_allreduce(rank, world_size, size, num_iters)

    dist.destroy_process_group()

if __name__ == "__main__":
    # 可以通过环境变量控制NCCL IB AR行为
    # NCCL_IB_AR_THRESHOLD 默认 8192 bytes
    # 设置很大可以关闭AR: e.g., 2**30
    # os.environ['NCCL_IB_AR_THRESHOLD'] = '8192'  # 开启AR
    # os.environ['NCCL_IB_AR_THRESHOLD'] = str(2**30)  # 禁用AR
    main()
