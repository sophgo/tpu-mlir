from functools import partial

import torch

class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a layer/block
    """
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


class PerChannelLoadHook:
    def __init__(self, module, hook_param=["scale", "zero_point"]):
        self.hook = module._register_load_state_dict_pre_hook(partial(self.hook_fn, module=module))
        self.hook_param = hook_param

    def hook_fn(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
                module):
        if module.ch_axis == -1:
            # no per-channel parameters
            return
        for module_key, param in module._parameters.items():
            if module_key not in self.hook_param:
                continue
            candidate = prefix + module_key
            if candidate in state_dict:
                input_param = state_dict[candidate]
                if param.shape != input_param.shape:
                    param.data = torch.ones_like(input_param, dtype=param.dtype, device=param.device)
        for module_key, param in module._buffers.items():
            if module_key not in self.hook_param:
                continue
            candidate = prefix + module_key
            if candidate in state_dict:
                input_param = state_dict[candidate]
                if param.shape != input_param.shape:
                    param.data = torch.ones_like(input_param, dtype=param.dtype, device=param.device)

    def close(self):
        self.hook.remove()
