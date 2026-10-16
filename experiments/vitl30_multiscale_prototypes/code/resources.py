"""Bound CUDA workspace when sharing a GPU with other projects."""
from __future__ import annotations

import os

import torch


def configure_device(device):
    if not str(device).startswith('cuda'):
        return {'device': str(device), 'memory_limit_mib': None}
    limit_mib = int(os.environ.get('UVG_PROTOTYPE_MEMORY_MIB', '1280'))
    if limit_mib <= 0:
        raise ValueError('UVG_PROTOTYPE_MEMORY_MIB must be positive')
    # Set this before the first CUDA query: otherwise a worker for GPU 2 can
    # create an unnecessary default-device context on GPU 0 as well.
    torch.cuda.set_device(device)
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(limit_mib * 2**20 / total, 1.), device)
    # Autotuning may try a workspace larger than the final algorithm needs.
    torch.backends.cudnn.benchmark = False
    return {'device': str(device), 'memory_limit_mib': limit_mib,
            'current_device': torch.cuda.current_device(),
            'cudnn_benchmark': False, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
