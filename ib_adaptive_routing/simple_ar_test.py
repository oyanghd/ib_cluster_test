import os
import time
import torch
import torch.distributed as dist

def benchmark_allreduce(size=256*1024*1024, iters=20):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    numel = size // 4  # FP32 = 4 bytes
    x = torch.ones(numel, device='cuda', dtype=torch.float32) * rank

    for _ in range(5):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    end = time.time()

    elapsed = end - start
    avg_time = elapsed / iters

    bytes_per_iter = 2 * size * (world_size - 1) / world_size
    total_bytes = bytes_per_iter * iters

    bandwidth = total_bytes / elapsed / 1e9  # GB/s

    if rank == 0:
        print(f"allreduce: "
              f"[World {world_size}] size={size/1e6:.1f} MB, iters={iters}, "
              f"avg_time={avg_time*1e3:.3f} ms, bw={bandwidth:.2f} GB/s")

def benchmark_alltoall(size=256*1024*1024, iters=10):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    numel = size // 4
    input_tensor = torch.ones(numel, device="cuda") * rank
    output_tensor = torch.empty_like(input_tensor)

    split_size = numel // world_size
    input_splits = [split_size] * world_size
    output_splits = [split_size] * world_size

    for _ in range(5):
        dist.all_to_all_single(output_tensor, input_tensor, output_splits, input_splits)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        dist.all_to_all_single(output_tensor, input_tensor, output_splits, input_splits)
    torch.cuda.synchronize()
    end = time.time()

    bytes_per_iter = size
    total_bytes = bytes_per_iter * iters

    elapsed = end - start
    bandwidth = total_bytes / elapsed / 1e9  # GB/s

    if rank == 0:
        print(f"alltoall: "
              f"[World {world_size}] size={size/1e6:.1f} MB, iters={iters}, "
              f"time={elapsed:.3f}s, bw={bandwidth:.2f} GB/s")

def init_process(backend='nccl'):
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    # os.environ['MASTER_ADDR'] = "127.0.0.1"
    # os.environ['MASTER_PORT'] = "28388"
    init_device = torch.device(f"cuda:{local_rank}")
    # torch.distributed.init_process_group(backend, world_size=world_size, rank=rank)
    torch.distributed.init_process_group(backend, world_size=world_size, rank=rank, device_id=init_device)


if __name__ == "__main__":
    # set NCCL_IB_AR_THRESHOLD to control Adaptive Routing (AR) behavior
    # NCCL_IB_AR_THRESHOLD default 8192 bytes
    # can set a large value to disable AR: e.g., 2**30
    # os.environ['NCCL_IB_AR_THRESHOLD'] = '0'  # always enable AR
    # os.environ['NCCL_IB_AR_THRESHOLD'] = '8192'  # enable AR for messages larger than 8KB
    # os.environ['NCCL_IB_AR_THRESHOLD'] = str(2**30)  # disable AR
    # ref https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-ar-threshold
    init_process()
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    torch.cuda.set_device(rank % torch.cuda.device_count())

    num_iters = 20

    if rank == 0:
        print(f"Testing with NCCL_IB_AR_THRESHOLD = {os.environ.get('NCCL_IB_AR_THRESHOLD')}")

    all_reduce_tensor_sizes = [256, 2*1024, 16*1024, 128*1024, 1*1024*1024, 8*1024*1024, 64*1024*1024, 256*1024*1024, 512*1024*1024, 1*1024*1024*1024]
    for size in all_reduce_tensor_sizes:
        benchmark_allreduce(size=size, iters=num_iters)

    alltoall_tensor_sizes = [world_size * i for i in [256, 1024, 8*1024, 32*1024, 256*1024, 1*1024*1024, 4*1024*1024, 16*1024*1024]]
    for size in alltoall_tensor_sizes:
        benchmark_alltoall(size=size, iters=num_iters)

    dist.destroy_process_group()