import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear

def main():
    dist.init_process_group("nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model = Linear(1024, 1024).cuda()
    fsdp_model = FSDP(model)

    x = torch.randn(8, 1024, device="cuda")
    y = fsdp_model(x)

    if dist.get_rank() == 0:
        print("FSDP forward pass OK")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
