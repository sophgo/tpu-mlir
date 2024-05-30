from typing import List, Tuple

from torch.nn import Module

from sophgo_mq.mix_precision.hessian_per_layer import hessian_per_layer
from sophgo_mq.prepare_by_platform import BackendType, prepare_by_platform
from sophgo_mq.utils import is_symmetric_quant
from sophgo_mq.utils.logger import logger
from sophgo_mq.utils.state import disable_all


def mixprecision_profiling(model: Module, quantized_model: Module, bitwidth_list: List, data: Tuple, criterion, algo='naive'):
    """
    Get layer sensitive index under a list of bitwidth.
    A lot of algorithms can do the same thing.
    HAWQ is the most useful one.
    Naive is the most straight forward one.
    """
    layer_parameters_dict = model_size_analysis(model)
    sensetive_dict = {}
    if algo == 'hawq_eigen':
        eigen_values_dict = hawq(model, data, criterion, type='eigenvalues')
        # Do normalize.
        for layer, eigen_val in eigen_values_dict.items():
            eigen_values_dict[layer] = eigen_val / layer_parameters_dict[layer]
        for name, max_eignevalues in eigen_values_dict.items():
            logger.info("Layer {} with max eigen values: {}".format(name, max_eignevalues))
        delta_w = get_delta_w(quantized_model, bitwidth_list)
        for layer, max_eignevalues in eigen_values_dict.items():
            # max_eigne_val: Float
            # delta_w: List shape = bitwidth_list
            sensetive_dict[layer] = max_eignevalues * delta_w[layer]
    elif algo == 'hawq_trace':
        trace_value_dict = hawq(model, data, criterion, type='trace')
        # Do normalize.
        for layer, trace in trace_value_dict.items():
            trace_value_dict[layer] = trace / layer_parameters_dict[layer]
        for name, trace in trace_value_dict.items():
            logger.info("Layer {} with trace: {}".format(name, trace))
        delta_w = get_delta_w(quantized_model, bitwidth_list)
        for layer, trace in trace_value_dict.items():
            # max_eigne_val: Float
            # delta_w: List shape = bitwidth_list
            sensetive_dict[layer] = trace * delta_w[layer]
    elif algo == 'naive':
        sensetive_dict = prec_degradation_by_layer(model, quantized_model, bitwidth_list, data, criterion)
    else:
        logger.info("Unknown algorithm!")
    return sensetive_dict


def get_delta_w(quantized_model: Module, bitwidth_list: List):
    def get_new_qrange(bits, qscheme):
        if is_symmetric_quant(qscheme):
            return -2 ** (bits - 1), 2 ** (bits - 1) - 1
        return 0, 2 ** bits - 1

    def square_mean(ta, tb):
        return torch.pow((ta - tb), 2.0).mean().detach().cpu().numpy()

    delta_w = {}
    for name, mod in quantized_model.named_modules():
        logger.setLevel('CRITICAL')
        disable_all(quantized_model)
        logger.setLevel('INFO')
        if hasattr(mod, 'weight_fake_quant'):
            delta_w[name] = []
            mod.weight_fake_quant.enable_observer()
            mod.weight_fake_quant.enable_fake_quant()
            for bits in bitwidth_list:
                qscheme = mod.weight_fake_quant.activation_post_process.qscheme
                new_quant_min, new_quant_max = get_new_qrange(bits, qscheme)
                mod.weight_fake_quant.activation_post_process.quant_min = new_quant_min
                mod.weight_fake_quant.activation_post_process.quant_max = new_quant_max
                mod.weight_fake_quant.quant_min = new_quant_min
                mod.weight_fake_quant.quant_max = new_quant_max
                delta_w[name].append(square_mean(mod.weight, mod.weight_fake_quant(mod.weight)))
            delta_w[name] = np.array(delta_w[name])
            mod.weight_fake_quant.disable_observer()
            mod.weight_fake_quant.disable_fake_quant()

    return delta_w


def model_size_analysis(model):
    layer_parameters_dict = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
            layer_parameters_dict[name] = mod.weight.numel()
    return layer_parameters_dict


def model_latency_analysis(model):
    pass


def model_flops_analyze(model):
    pass


def mp_model_size(model: Module):
    """
    Calcualte model size in different bitwidth.
    """
    mp_size = 0
    for mod in model.modules():
        if hasattr(mod, 'weight_fake_quant'):
            bitwidth = mod.weight_fake_quant.bitwidth
            mp_size += mod.weight.numel() * bitwidth
        elif hasattr(mod, 'weight'):
            mp_size += mod.weight.numel() * 32
    return mp_size / 8 / 1024 / 1024


def prec_degradation_by_layer(model: Module, quantized_model: Module, bitwidth_list: List, data: Tuple, creterion):
    """
    Calculate degradation of each layer in different bitwidth.
    """
    def get_new_qrange(bits, qscheme):
        if is_symmetric_quant(qscheme):
            return -2 ** (bits - 1), 2 ** (bits - 1) - 1
        return 0, 2 ** bits - 1

    input_data, label_data = data
    sensetive_dict = {}
    output_data = model(input_data)
    fp_loss = creterion(output_data, label_data)

    for name, mod in quantized_model.named_modules():
        logger.setLevel('CRITICAL')
        disable_all(quantized_model)
        logger.setLevel('INFO')
        if hasattr(mod, 'weight_fake_quant'):
            sensetive_dict[name] = []
            mod.weight_fake_quant.enable_observer()
            mod.weight_fake_quant.enable_fake_quant()
            for bits in bitwidth_list:
                qscheme = mod.weight_fake_quant.activation_post_process.qscheme
                new_quant_min, new_quant_max = get_new_qrange(bits, qscheme)
                mod.weight_fake_quant.activation_post_process.quant_min = new_quant_min
                mod.weight_fake_quant.activation_post_process.quant_max = new_quant_max
                mod.weight_fake_quant.quant_min = new_quant_min
                mod.weight_fake_quant.quant_max = new_quant_max
                with torch.no_grad():
                    output_data = quantized_model(input_data)
                loss = creterion(output_data, label_data)
                sensetive_dict[name].append(loss - fp_loss)
                logger.info("Layer {} under bit {} with sensetive {}".format(name, bits, loss - fp_loss))
            mod.weight_fake_quant.disable_observer()
            mod.weight_fake_quant.disable_fake_quant()

    return sensetive_dict


def hawq(model: Module, data: Tuple, criterion, type='trace'):
    """
    HAWQ layer sensetive indicator. Using extend PyHessian to calculate.
    """
    inputs, targets = data
    hessian_comp = hessian_per_layer(model, criterion, data=(inputs, targets), cuda=True)
    if type == 'eigenvalues':
        return hessian_comp.layer_eigenvalues()
    elif type == 'trace':
        return hessian_comp.layer_trace()
    else:
        raise(NotImplementedError, "{} is not supported, only trace and eigenvalues.".format(type))


def mixprecision_bit_selection(bitwidth_list, sensetive_dict, layer_parameters_dict, model_size_constraints, latency_constraints):
    """
    Resolute bitwidth by layer sensetive index / model size / accuracy.
    """
    # preato_frontier(model)
    ILP_bit_selection(bitwidth_list, sensetive_dict, layer_parameters_dict, model_size_constraints, latency_constraints)


def ILP_bit_selection(bitwidth_list, sensetive_dict, layer_parameters_dict, model_size_constraints: int, latency_constraints: int):
    """
    Bit selection process using ILP.
    """
    import pulp
    from pulp import (GLPK_CMD, LpInteger, LpMinimize, LpStatus, LpVariable,
                      value)

    assert model_size_constraints or latency_constraints

    prob = pulp.LpProblem("Min model size with best acc", LpMinimize)
    variable = {}
    for layer_name in sensetive_dict:
        for bit in bitwidth_list:
            variable[f"x_{layer_name}_{bit}"] = LpVariable(f"x_{layer_name}_{bit}", 0, 1, cat=LpInteger)

    # Model acc constrains
    senseitve_contrains = []
    for name, params in layer_parameters_dict.items():
        for idx, bit in enumerate(bitwidth_list):
            senseitve_contrains.append(variable[f"x_{name}_{bit}"] * sensetive_dict[name][idx])
    prob += sum(senseitve_contrains)

    # Every Layer can only be assigned to one bitwidth.
    for layer_name in sensetive_dict:
        prob += sum([variable[f"x_{layer_name}_{bit}"] for bit in bitwidth_list]) == 1

    # Model size constrains
    total_size = []
    for name, params in layer_parameters_dict.items():
        for bit in bitwidth_list:
            total_size.append(variable[f"x_{name}_{bit}"] * bit * params)
    prob += sum(total_size) <= model_size_constraints * 8 * 1024 * 1024

    status = prob.solve(GLPK_CMD(msg=1, options=["--tmlim", "10000", "--simplex"]))
    LpStatus[status]
    for layer_name in sensetive_dict:
        for bit in bitwidth_list:
            if value(variable[f"x_{layer_name}_{bit}"]) == 1:
                logger.info("Layer {} with {} bits".format(layer_name, bit))
    total_size = []
    for name, params in layer_parameters_dict.items():
        for bit in bitwidth_list:
            total_size.append(value(variable[f"x_{name}_{bit}"]) * bit * params)
    logger.info("Result model size {} MB.".format(sum(total_size) / 8 / 1024 / 1024))

    senseitve_contrains = []
    for name, params in layer_parameters_dict.items():
        for idx, bit in enumerate(bitwidth_list):
            senseitve_contrains.append(value(variable[f"x_{name}_{bit}"]) * sensetive_dict[name][idx])
    logger.info("Result model sensetive is {}".format(sum(senseitve_contrains)))


if __name__ == '__main__':
    import numpy as np
    import torch
    import torchvision

    model = torchvision.models.resnet18(pretrained=True).eval()

    inputs = torch.rand(2, 3, 224, 224).cuda()
    model = model.cuda()
    with torch.no_grad():
        targets = model(inputs)
    targets = (targets == targets.max(dim=1, keepdim=True)[0]).to(dtype=torch.float32)

    test_bitwidth_list = [2, 4, 8, 16]

    quantized_model = prepare_by_platform(model, BackendType.Tensorrt)
    layer_parameters_dict = model_size_analysis(model)
    model_size = sum(list(layer_parameters_dict.values())) * 32 / 8 / 1024 / 1024
    logger.info("FP model size: {:.2f} MB".format(model_size))
    # naive_sensetive_dict = mixprecision_profiling(model, quantized_model, test_bitwidth_list,
    #                                               data=(inputs, targets), criterion=torch.nn.CrossEntropyLoss(), algo='naive')
    # maxeigen_sensetive_dict = mixprecision_profiling(model, quantized_model, test_bitwidth_list,
    #                                                  data=(inputs, targets), criterion=torch.nn.CrossEntropyLoss(), algo='hawq_eigen')
    trace_sensetive_dict = mixprecision_profiling(model, quantized_model, test_bitwidth_list,
                                                  data=(inputs, targets), criterion=torch.nn.CrossEntropyLoss(), algo='hawq_trace')

    mixprecision_bit_selection(test_bitwidth_list, 
                               # naive_sensetive_dict,
                               # maxeigen_sensetive_dict,
                               trace_sensetive_dict,
                               layer_parameters_dict,
                               model_size_constraints=3, latency_constraints=None)
