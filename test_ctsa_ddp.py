"""Two-process CPU/Gloo smoke test for warmup -> CTSA and unequal gate coverage.

Run: python test_ctsa_ddp.py
No dataset, CUDA or downloaded model weights are needed.
"""
import copy
import socket
from datetime import timedelta
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from model.semseg.dpt import DPT, DPTHead
from model.semseg.ctsa import CTSA
from util.training_runtime import update_ema


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 24, 14, 14)

    def get_intermediate_layers(self, x, indices):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return tuple(x for _ in indices)


def make_model():
    model = DPT.__new__(DPT)
    nn.Module.__init__(model)
    model.encoder_size = 'small'
    model.intermediate_layer_idx = {'small': [2, 5, 8, 11]}
    model.backbone = TinyBackbone()
    model.head = DPTHead(3, 24, features=8, out_channels=[8]*4)
    model.ctsa = CTSA(24, dim=16, heads=4, gate_statistics=True)
    return model


def worker(rank, port):
    torch.set_num_threads(1)
    torch.manual_seed(12 + rank)
    store = dist.TCPStore('127.0.0.1', port, 2, rank == 0,
                          timeout=timedelta(seconds=45), use_libuv=False)
    dist.init_process_group('gloo', store=store, rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        model = nn.parallel.DistributedDataParallel(make_model(), find_unused_parameters=True)
        teacher = copy.deepcopy(model.module); teacher.ctsa = None
        teacher.eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        for step in range(3):
            x = torch.randn(3,3,28,28)
            optimizer.zero_grad(set_to_none=True)
            if step == 0:
                loss = model(x).square().mean()  # globally skipped CTSA warmup
            else:
                with torch.no_grad():
                    _, features = teacher(x[1:], return_last_feature=True)
                gate = torch.ones(2,4,1) if rank == 0 else torch.zeros(2,4,1)
                ordinary, aux = model(x, teacher_feature=features, ctsa_gate=gate, num_labeled=1)
                active = (gate.sum((1,2)) > 0).float()
                loss = ordinary.square().mean() + (aux.square().mean((1,2,3))*active).mean()
            loss.backward()
            optimizer.step()
            update_ema(model,teacher,.9)
            assert torch.isfinite(loss)
        value=model.module.ctsa.q.weight.detach()
        peers=[torch.empty_like(value) for _ in range(2)]
        dist.all_gather(peers,value)
        torch.testing.assert_close(peers[0],peers[1],rtol=0,atol=0)
        if rank==0:
            print('PASS: two-rank warmup, CTSA backward, zero-gate rank, EMA, synchronized weights')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0))
        port=listener.getsockname()[1]
    mp.spawn(worker,args=(port,),nprocs=2,join=True)
