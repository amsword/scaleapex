import os
import torch
from torch import nn


def test_scaleapex():
    from scaleapex import dynamic_scaler, optim_constructor
    from fairscale.optim.oss import OSS
    from functools import partial
    dim = 128

    init_param = {
        'backend': 'nccl',
        'init_method': 'tcp://localhost:12345',
        'rank': int(os.environ.get('OMPI_COMM_WORLD_RANK', '0')),
        'world_size': int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1')),
    }
    device_id = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
    torch.cuda.set_device(device_id)
    torch.distributed.init_process_group(**init_param)

    model = nn.Sequential(
        *[nn.Linear(dim, dim) for i in range(12)]
    )
    model.half().cuda()

    parameters = model.parameters()
    extra_param = {
        'lr': 1.e-7,
        'weight_decay': 0.01,
    }
    optimizer = OSS(
            parameters,
            optim=partial(optim_constructor, torch.optim.AdamW),
            **extra_param,
            )

    scaler = dynamic_scaler()
    max_iter = 10
    batch_size = 16
    for i in range(max_iter):
        data = torch.randn(batch_size, dim).half().cuda()
        y = model(data)
        loss = y.sum(dim=1).abs().mean().float()
        loss.backward()
        _, overflow = optimizer.step(scaler=scaler)

if __name__ == '__main__':
    test_scaleapex()
