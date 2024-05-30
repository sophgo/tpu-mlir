import torch
if torch.cuda.is_available():
    from . import fpemu as fpemu_cuda