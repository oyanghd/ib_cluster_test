import os
import torch
import nvtx
import time

from typing import Any, Tuple
from torch import Tensor
import torch.distributed as dist

import contextlib
import functools

try:
    import nvtx
except ModuleNotFoundError:
    class nvtx:
        @staticmethod
        def push_range(msg, color):
            return torch.cuda.nvtx.range_push(msg)

        @staticmethod
        def pop_range():
            return torch.cuda.nvtx.range_pop()


DEFAULT_COLOR = 0xBFBFBF
FORWARD_COLOR = "purple"
BACKWARD_COLOR = "orange"
BACKWARD_RANGE_DEPTH = 0


PERF_MODEL = False
GLOBAL_EVENTS = list()


def record_event(arg):
    if PERF_MODEL:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        GLOBAL_EVENTS.append((arg, ev))

class WrapInputsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, msg, *args):
        ctx.msg = msg
        nvtx.push_range(msg, color=FORWARD_COLOR)
        record_event((msg, "forward", "begin"))
        return args

    @staticmethod
    def backward(ctx, *grads):
        record_event((ctx.msg, "backward", "end"))
        nvtx.pop_range()
        global BACKWARD_RANGE_DEPTH
        BACKWARD_RANGE_DEPTH -= 1
        return None, *grads


class WrapOutputsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, msg, *args):
        ctx.msg = msg
        record_event((msg, "forward", "end"))
        nvtx.pop_range()
        return args

    @staticmethod
    def backward(ctx, *grads):
        nvtx.push_range(ctx.msg, color=BACKWARD_COLOR)
        record_event((ctx.msg, "backward", "begin"))
        global BACKWARD_RANGE_DEPTH
        BACKWARD_RANGE_DEPTH += 1
        return None, *grads


def annotate_forward_backward(forward_msg, backward_msg):
    def decorator(forward_orig):
        @functools.wraps(forward_orig)
        def wrapper(*args, **kwargs):
            kwargs_items = list(kwargs.items())
            inputs_list = list(args) + [t[1] for t in kwargs_items]
            inputs_mask = [torch.is_tensor(x) and not isinstance(x, torch.nn.Parameter) for x in inputs_list]  # mask=True means should be applied
            inputs_list_masked = [x if mask else None for mask, x in zip(inputs_mask, inputs_list)]
            inputs_list_masked_applied = WrapInputsFunction.apply(forward_msg, *inputs_list_masked)
            inputs_list_applied = [x_applied if mask else x for mask, x, x_applied in zip(inputs_mask, inputs_list, inputs_list_masked_applied)]
            args = inputs_list_applied[:len(args)]
            kwargs = {t[0]: v for t, v in zip(kwargs_items, inputs_list_applied[len(args):])}
            outputs = forward_orig(*args, **kwargs)
            if isinstance(outputs, tuple):
                outputs = WrapOutputsFunction.apply(backward_msg, *outputs)
            else:
                outputs, = WrapOutputsFunction.apply(backward_msg, outputs)
            return outputs
        return wrapper
    return decorator

cp_group = None
cp_intra_group = None
cp_inter_group = None
self_cp_group = None
self_cp_intra_group = None
self_cp_inter_group = None

def get_context_parallel_group(cp_size: int):
    global cp_group
    global self_cp_group
    if cp_group == None:
        cp_group = dist.new_group(list(range(0, cp_size)))
        if dist.get_rank() < cp_size:
            self_cp_group = cp_group
    return self_cp_group

def get_context_parallel_intra_group(cp_size: int):
    global cp_intra_group
    global self_cp_intra_group
    if cp_intra_group == None:
        cp_intra_group = []
        for i in range(cp_size // 8):
            cp_intra_group.append(dist.new_group(list(range(i * 8, (i + 1) * 8))))
            if dist.get_rank() >= i * 8 and dist.get_rank() < (i + 1) * 8:
                self_cp_intra_group = cp_intra_group[-1]
    return self_cp_intra_group

def get_context_parallel_inter_group(cp_size: int):
    global cp_inter_group
    global self_cp_inter_group
    if cp_inter_group == None:
        cp_inter_group = []
        for i in range(8):
            cp_inter_group.append(dist.new_group(list(range(i, cp_size, 8))))
            if dist.get_rank() % 8 == i:
                self_cp_inter_group = cp_inter_group[-1]
    return self_cp_inter_group

def init_parallel_group():
    global self_cp_group, self_cp_intra_group, self_cp_inter_group
    cp_size = dist.get_world_size()
    get_context_parallel_group(cp_size)
    get_context_parallel_intra_group(cp_size)
    get_context_parallel_inter_group(cp_size)

@annotate_forward_backward("all_to_all_4D", "all_to_all_4D")
def all_to_all_4D(
    input: torch.tensor, scatter_idx: int = 2, gather_idx: int = 1, group=None, slow_group=None, async_op = False,
) -> torch.tensor:
    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)
    if slow_group is not None:
        seq_world_size = seq_world_size * dist.get_world_size(slow_group)

    if scatter_idx == 2 and gather_idx == 1:
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        input_t = (
            input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .contiguous()
        )

        output = torch.empty_like(input_t)
        if slow_group is not None:
            intra_a2a_size = dist.get_world_size(group)
            inter_a2a_size = dist.get_world_size(slow_group)
            input_t = input_t.view(inter_a2a_size, intra_a2a_size, shard_seqlen, bs, shard_hc, hs).contiguous()
            input_t = input_t.transpose(0, 1).contiguous()
            dist.all_to_all_single(output, input_t, group=group)
            output = output.view(intra_a2a_size, inter_a2a_size, shard_seqlen, bs, shard_hc, hs).contiguous()
            output = output.transpose(0, 1).contiguous()
            output_dst = torch.empty_like(output)
            dist.all_to_all_single(output_dst, output, group=slow_group)
            output = output_dst
        else:
            dist.all_to_all_single(output, input_t, group=group)

        output = output.reshape(seqlen, bs, shard_hc, hs)

        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size

        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        output = torch.empty_like(input_t)
        if slow_group is not None:
            intra_a2a_size = dist.get_world_size(group)
            inter_a2a_size = dist.get_world_size(slow_group)
            input_t = input_t.view(inter_a2a_size, intra_a2a_size, shard_hc, shard_seqlen, bs, hs).contiguous()
            input_t = input_t.transpose(0, 1).contiguous()
            dist.all_to_all_single(output, input_t, group=group)
            output = output.view(intra_a2a_size, inter_a2a_size, shard_hc, shard_seqlen, bs, hs).contiguous()
            output = output.transpose(0, 1).contiguous()
            output_dst = torch.empty_like(output)
            dist.all_to_all_single(output_dst, output, group=slow_group)
            output = output_dst
        else:
            dist.all_to_all_single(output, input_t, group=group)

        output = output.reshape(hc, shard_seqlen, bs, hs)

        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        input: Tensor,
        scatter_idx: int,
        gather_idx: int,
        cp_size: int,
    ) -> Tensor:
        group = get_context_parallel_group(cp_size)
        intra_group = get_context_parallel_intra_group(cp_size)
        inter_group = get_context_parallel_inter_group(cp_size)
        intra_group_size = dist.get_world_size(intra_group)
        inter_group_size = dist.get_world_size(inter_group)
        ctx.group = group
        ctx.intra_group = intra_group
        ctx.inter_group = inter_group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.cp_size = cp_size
        if intra_group_size == 1 or inter_group_size == 1:
            output = all_to_all_4D(input, scatter_idx, gather_idx, group=group)
        else:
            output = all_to_all_4D(
                input, scatter_idx, gather_idx, group=intra_group, slow_group=inter_group
            )
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        return (
            SeqAllToAll4D.apply(
                *grad_output, ctx.gather_idx, ctx.scatter_idx, ctx.cp_size
            ),
            None,
            None,
            None,
        )


def init_process(backend='nccl'):
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    # os.environ['MASTER_ADDR'] = "127.0.0.1"
    # os.environ['MASTER_PORT'] = "28388"
    init_device = torch.device(f"cuda:{local_rank}")
    # torch.distributed.init_process_group(backend, world_size=world_size, rank=rank)
    torch.distributed.init_process_group(backend, world_size=world_size, rank=rank, device_id=init_device)

def cleanup():
    if torch.distributed.is_initialized():
        for group in cp_intra_group:
            torch.distributed.destroy_process_group(group)
        for group in cp_inter_group:
            torch.distributed.destroy_process_group(group)
        torch.distributed.destroy_process_group(cp_group)
        torch.distributed.destroy_process_group()

def measure_bandwidth(func, tensor, **kwargs):
    torch.cuda.synchronize()
    start = time.time()
    output = func(tensor, **kwargs)
    torch.cuda.synchronize()
    end = time.time()

    elapsed = end - start
    num_bytes = tensor.numel() * tensor.element_size()
    bw = num_bytes / elapsed / 1e9  # GB/s
    return output, elapsed, bw

if __name__ == "__main__":
    init_process()
    init_parallel_group()
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    cp_size = dist.get_world_size()
    group = get_context_parallel_group(cp_size)
    intra_group = get_context_parallel_intra_group(cp_size)
    inter_group = get_context_parallel_inter_group(cp_size)

    tensor = torch.randn(2, 49152, 192, 128, device="cuda")

    for i in range(10):
        print(f"Rank {dist.get_rank()} has data {tensor.shape}")
        output, t1, bw1 = measure_bandwidth(
            all_to_all_4D, tensor,
            scatter_idx=1, gather_idx=2, group=intra_group, slow_group=inter_group
        )
        output_baseline, t2, bw2 = measure_bandwidth(
            all_to_all_4D, tensor,
            scatter_idx=1, gather_idx=2, group=group
        )
        print(
            f"Rank {dist.get_rank()} alltoall: "
            f"intra/inter {output.shape}, time {t1:.4f}s, bw {bw1:.2f} GB/s | "
            f"baseline {output_baseline.shape}, time {t2:.4f}s, bw {bw2:.2f} GB/s"
        )
        assert output.shape == output_baseline.shape, f"Output shape mismatch: {output.shape} vs {output_baseline.shape}"
        dist.barrier()
        print("")
    assert torch.allclose(output, output_baseline), "Output tensors are not close enough!"
    dist.barrier()
    print(f"Rank {dist.get_rank()} finished successfully.")
    cleanup()