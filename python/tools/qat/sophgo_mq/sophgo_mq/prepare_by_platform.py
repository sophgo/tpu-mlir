from copy import deepcopy
from enum import Enum
from typing import Any, Dict
import types
import inspect

import torch
from torch.fx import Tracer
from torch.fx.graph_module import GraphModule
from torch.quantization.quantize_fx import _swap_ff_with_fxff
from torch.quantization import QConfig
import torch.nn.intrinsic as nni
import sophgo_mq.nn.intrinsic as qnni

from sophgo_mq.fake_quantize import (
    LearnableFakeQuantize,
    NNIEFakeQuantize,
    FixedFakeQuantize,
    DoReFaFakeQuantize,
    DSQFakeQuantize,
    PACTFakeQuantize,
    TqtFakeQuantize,
    AdaRoundFakeQuantize,
    QDropFakeQuantize,
    E4M3FakeQuantize,
    E5M2FakeQuantize,
    GPTQFakeQuantize,
    FP4FakeQuantize,
    GPTQFP4FakeQuantize,
    FP4GROUPFakeQuantize,
    FP4GROUPFakeQuantize1,
    Fp16FakeQuantize,
    BF16FakeQuantize
)
from sophgo_mq.observer import (
    ClipStdObserver,
    LSQObserver,
    MinMaxFloorObserver,
    MinMaxObserver,
    EMAMinMaxObserver,
    PoTModeObserver,
    EMAQuantileObserver,
    MSEObserver,
    EMAMSEObserver,
    KLDObserver,
)
import sophgo_mq
from sophgo_mq.fuser_method_mappings import fuse_custom_config_dict
from sophgo_mq.utils.logger import logger
from sophgo_mq.utils.registry import DEFAULT_MODEL_QUANTIZER
from sophgo_mq.scheme import QuantizeScheme
import sophgo_mq.nn.intrinsic.qat as qnniqat

__all__ = ['prepare_by_platform']

ParamsTable = {
    'BM1688':                 dict(qtype='affine',
                                 w_qscheme=QuantizeScheme(symmetry=True, per_channel=True, pot_scale=False, bit=8),
                                 a_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8),
                                 default_weight_quantize=E4M3FakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=EMAMinMaxObserver),    
    'BM1684X':                dict(qtype='affine',
                                 w_qscheme=QuantizeScheme(symmetry=True, per_channel=True, pot_scale=False, bit=8),
                                 a_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8),
                                 default_weight_quantize=LearnableFakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=EMAMinMaxObserver),
    'MARS3':                dict(qtype='affine',
                                 w_qscheme=QuantizeScheme(symmetry=True, per_channel=True, pot_scale=False, bit=8, symmetric_range=True),
                                 a_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8),
                                 default_weight_quantize=LearnableFakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=EMAMinMaxObserver),
    'CV183X':                dict(qtype='affine',
                                 w_qscheme=QuantizeScheme(symmetry=True, per_channel=True, pot_scale=False, bit=8, symmetric_range=True),
                                 a_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8),
                                 default_weight_quantize=LearnableFakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=EMAMinMaxObserver),
    'CV182X':                dict(qtype='affine',
                                 w_qscheme=QuantizeScheme(symmetry=True, per_channel=True, pot_scale=False, bit=8, symmetric_range=True),
                                 a_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8),
                                 default_weight_quantize=LearnableFakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=EMAMinMaxObserver),
    'CV181X':                dict(qtype='affine',
                                 w_qscheme=QuantizeScheme(symmetry=True, per_channel=True, pot_scale=False, bit=8, symmetric_range=True),
                                 a_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8),
                                 default_weight_quantize=LearnableFakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=EMAMinMaxObserver),
    'CV180X':                dict(qtype='affine',
                                 w_qscheme=QuantizeScheme(symmetry=True, per_channel=True, pot_scale=False, bit=8, symmetric_range=True),
                                 a_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8),
                                 default_weight_quantize=LearnableFakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=EMAMinMaxObserver),
    'CV186X':                dict(qtype='affine',
                                 w_qscheme=QuantizeScheme(symmetry=True, per_channel=True, pot_scale=False, bit=8, symmetric_range=True),
                                 a_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8),
                                 default_weight_quantize=LearnableFakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=EMAMinMaxObserver),
    'BM1690':                 dict(qtype='affine',
                                 w_qscheme=QuantizeScheme(symmetry=True, per_channel=True, pot_scale=False, bit=8),
                                 a_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8),
                                 default_weight_quantize=LearnableFakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=EMAMinMaxObserver),
    'Academic':               dict(qtype='affine',
                                 w_qscheme=QuantizeScheme(symmetry=True, per_channel=True, pot_scale=False, bit=8),
                                 a_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8),
                                 default_weight_quantize=E4M3FakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=EMAMinMaxObserver)
}

ObserverDict = {
    'MinMaxObserver':           MinMaxObserver,                                    # noqa: E241
    'EMAMinMaxObserver':        EMAMinMaxObserver,        # More general choice.   # noqa: E241
    'MinMaxFloorObserver':      MinMaxFloorObserver,      # For Vitis HW           # noqa: E241
    'PoTModeObserver':          PoTModeObserver,   # For Vitis HW           # noqa: E241
    'EMAQuantileObserver':      EMAQuantileObserver,      # Quantile observer.     # noqa: E241
    'ClipStdObserver':          ClipStdObserver,          # Usually used for DSQ.  # noqa: E241
    'LSQObserver':              LSQObserver,              # Usually used for LSQ.  # noqa: E241
    'MSEObserver':              MSEObserver,                                       # noqa: E241
    'EMAMSEObserver':           EMAMSEObserver,                                    # noqa: E241
    'KLDObserver':              KLDObserver,
}

FakeQuantizeDict = {
    'FixedFakeQuantize': FixedFakeQuantize,      # Unlearnable scale/zeropoint  # noqa: E241
    'LearnableFakeQuantize': LearnableFakeQuantize,  # Learnable scale/zeropoint    # noqa: E241
    'NNIEFakeQuantize':      NNIEFakeQuantize,       # Quantize function for NNIE   # noqa: E241
    'DoReFaFakeQuantize':    DoReFaFakeQuantize,     # Dorefa                       # noqa: E241
    'DSQFakeQuantize':       DSQFakeQuantize,        # DSQ                          # noqa: E241
    'PACTFakeQuantize':      PACTFakeQuantize,       # PACT                         # noqa: E241
    'TqtFakeQuantize':       TqtFakeQuantize,        # TQT                          # noqa: E241
    'AdaRoundFakeQuantize':  AdaRoundFakeQuantize,   # AdaRound                     # noqa: E241
    'QDropFakeQuantize':     QDropFakeQuantize,      # BRECQ & QDrop                # noqa: E241
    'E4M3FakeQuantize':      E4M3FakeQuantize,
    'E5M2FakeQuantize':      E5M2FakeQuantize,
    'GPTQFakeQuantize':      GPTQFakeQuantize, 
    'FP4FakeQuantize':       FP4FakeQuantize,
    'GPTQFP4FakeQuantize':   GPTQFP4FakeQuantize,
    'FP4GROUPFakeQuantize':  FP4GROUPFakeQuantize,
    'FP4GROUPFakeQuantize1': FP4GROUPFakeQuantize1,
    'Fp16FakeQuantize':      Fp16FakeQuantize,
    'BF16FakeQuantize':      BF16FakeQuantize,
}

FakeQuantizeDict_Chip = {
    'FixedFakeQuantize': FixedFakeQuantize,      # Unlearnable scale/zeropoint  # noqa: E241
    'LearnableFakeQuantize': LearnableFakeQuantize,  # Learnable scale/zeropoint    # noqa: E241
    'DoReFaFakeQuantize':    DoReFaFakeQuantize,     # Dorefa                       # noqa: E241
    'DSQFakeQuantize':       DSQFakeQuantize,        # DSQ                          # noqa: E241
    'PACTFakeQuantize':      PACTFakeQuantize,       # PACT                         # noqa: E241
    'TqtFakeQuantize':       TqtFakeQuantize,        # TQT                          # noqa: E241
    'AdaRoundFakeQuantize':  AdaRoundFakeQuantize,   # AdaRound                     # noqa: E241
    'QDropFakeQuantize':     QDropFakeQuantize,      # BRECQ & QDrop                # noqa: E241
    'E4M3FakeQuantize':      E4M3FakeQuantize,
    'E5M2FakeQuantize':      E5M2FakeQuantize,
    'Fp16FakeQuantize':      Fp16FakeQuantize,
    'BF16FakeQuantize':      BF16FakeQuantize,
}

def get_qconfig_by_platform(quant_dict:Dict,extra_qparams: Dict):
    """

    Args:
        quant_dict (dict):
        extra_qparams (dict):

    >>> extra params format: {
            'w_observer': str, weight observer name,
            'a_observer': str, activation observer name,
            'w_fakequantize': str, weight fake quantize function name,
            'w_fakeq_params": dict, params for weight quantize function,
            'a_fakequantize': str, activation fake quantize function name,
            'a_fakeq_params': dict, params for act quantize function,
            if deploy_backend == BackendType.Academic keys below will be used:
            'w_qscheme': {
                'bit': bitwidth,
                'symmetry': whether quantize scheme is symmetric,
                'per_channel': whether quantize scheme is perchannel,
                'pot_scale': whether scale is power of two.
            }
            'a_qscheme': {
                same with w_qscheme.
            }
            "object_type": [
                (torch.add, qconfig)
            ],
            "module_name": [
                ("conv1", qconfig)
            ]
        }
    """
    chip=quant_dict['chip'] #["BM1688","BM1684X","BM1690"]
    if chip=="BM1688":
        chip_params,w_observer,a_observer,w_fakequantize,a_fakequantize=chipparams(chip,extra_qparams,FakeQuantizeDict_Chip)
    elif chip=="BM1684X":
        chip_params,w_observer,a_observer,w_fakequantize,a_fakequantize=chipparams(chip,extra_qparams,FakeQuantizeDict_Chip)
    elif chip=="CV183X":
        chip_params,w_observer,a_observer,w_fakequantize,a_fakequantize=chipparams(chip,extra_qparams,FakeQuantizeDict_Chip)
    elif chip=="MARS3":
        chip_params,w_observer,a_observer,w_fakequantize,a_fakequantize=chipparams(chip,extra_qparams,FakeQuantizeDict_Chip)
    elif chip=="CV182X":
        chip_params,w_observer,a_observer,w_fakequantize,a_fakequantize=chipparams(chip,extra_qparams,FakeQuantizeDict_Chip)
    elif chip=="CV181X":
        chip_params,w_observer,a_observer,w_fakequantize,a_fakequantize=chipparams(chip,extra_qparams,FakeQuantizeDict_Chip)
    elif chip=="CV180X":
        chip_params,w_observer,a_observer,w_fakequantize,a_fakequantize=chipparams(chip,extra_qparams,FakeQuantizeDict_Chip)
    elif chip=="CV186X":
        chip_params,w_observer,a_observer,w_fakequantize,a_fakequantize=chipparams(chip,extra_qparams,FakeQuantizeDict_Chip)
    elif chip=="BM1690":
        chip_params,w_observer,a_observer,w_fakequantize,a_fakequantize=chipparams(chip,extra_qparams,FakeQuantizeDict_Chip)
    elif chip=="Academic":
        chip_params,w_observer,a_observer,w_fakequantize,a_fakequantize=chipparams(chip,extra_qparams,FakeQuantizeDict)
    else:
        logger.info("The chip is currently not supported")

    #scheme
    w_qscheme = extra_qparams.get('w_qscheme', None)
    if w_qscheme is None:
        w_qscheme = chip_params['w_qscheme']
    else:
        if chip=="BM1688":
            assert (w_qscheme['bit']==4  or w_qscheme['bit']==8), 'unsupported data type'
        if chip=="BM1684X" or chip=="BM1690":
            assert (w_qscheme['bit']==8), 'unsupported data type'
        logger.info("Weight Quant Scheme is overrided!")
        w_qscheme = QuantizeScheme(**w_qscheme)
    a_qscheme = extra_qparams.get('a_qscheme', None)
    if a_qscheme is None:
        a_qscheme = chip_params['a_qscheme']
    else:
        if chip=="BM1688":
            assert (a_qscheme['bit']==4  or a_qscheme['bit']==8),'unsupported data type'
        if chip=="BM1684X" or chip=="BM1690":
            assert (a_qscheme['bit']==8),'unsupported data type'
        logger.info("Activation Quant Scheme is overrided!")
        a_qscheme = QuantizeScheme(**a_qscheme)

    # Set extra args for observers.
    w_observer_extra_args = extra_qparams.get('w_observer_extra_args', {})
    a_observer_extra_args = extra_qparams.get('a_observer_extra_args', {})
    w_qscheme.kwargs.update(w_observer_extra_args)
    a_qscheme.kwargs.update(a_observer_extra_args)
    # Get weight / act fake quantize function and params. And bias fake quantizer if needed(Vitis)
    if not w_fakequantize:
        w_fakequantize = chip_params['default_weight_quantize']
    w_fakeq_params = extra_qparams.get('w_fakeq_params', {})
    if not a_fakequantize:
        a_fakequantize = chip_params['default_act_quantize']
    a_fakeq_params = extra_qparams.get('a_fakeq_params', {})
    # Get default observer type.
    if not w_observer:
        w_observer = chip_params['default_weight_observer']
    if not a_observer:
        a_observer = chip_params['default_act_observer']

    # Create qconfig.
    # here, rewrited by with_args
    w_qconfig = w_fakequantize.with_args(observer=w_observer, **w_fakeq_params, **w_qscheme.to_observer_params())
    a_qconfig = a_fakequantize.with_args(observer=a_observer, **a_fakeq_params, **a_qscheme.to_observer_params())
    assert not(quant_dict["quantmode"]=="weight_only"and quant_dict["strategy"]=="CNN") ,"unsupport this combination"
    if quant_dict["quantmode"]=="weight_activation":
        logger.info('Weight Qconfig:\n    FakeQuantize: {} Params: {}\n'
                    '    Oberver:      {} Params: {}'.format(w_fakequantize.__name__, w_fakeq_params,
                                                            w_observer.__name__, str(w_qscheme)))
        logger.info('Activation Qconfig:\n    FakeQuantize: {} Params: {}\n'
                    '    Oberver:      {} Params: {}'.format(a_fakequantize.__name__, a_fakeq_params,
                                                            a_observer.__name__, str(a_qscheme)))
        logger.info('Bias will also be quantified')
    elif quant_dict["quantmode"]=="weight_only":
        logger.info('Weight Qconfig:\n    FakeQuantize: {} Params: {}\n'
                    '    Oberver:      {} Params: {}'.format(w_fakequantize.__name__, w_fakeq_params,
                                                            w_observer.__name__, str(w_qscheme)))
    else:
        logger.info("The quantmode is currently not supported")
    
    qconfig = {'': QConfig(activation=a_qconfig, weight=w_qconfig)}

    # qconfig["object_type"] = {torch.nn.Linear:createQConfigForSophgo_weight()} #int8 qat, Sophgo_TPU use sym per-layer
    object_type = extra_qparams.get('object_type', None)
    if object_type:
        if "object_type" not in qconfig:
            qconfig["object_type"] = {}
        for type_name,type_data in object_type.items():
            mode=object_type.get(type_name,{}).get("mode")
            bit=object_type.get(type_name,{}).get("bit")
            if mode=="activation":
                afq=object_type.get(type_name,{}).get("afakequantize")
                aob=object_type.get(type_name,{}).get("aobserver")
                qconfig['object_type'][type_name]=createQConfigForSophgo_activation(bit_num = bit, a_fakequantize = afq, a_observer = aob)
            elif mode=="weight":
                wfq=object_type.get(type_name,{}).get("wfakequantize")
                wob=object_type.get(type_name,{}).get("wobserver")
                qconfig['object_type'][type_name]=createQConfigForSophgo_weight(chip, bit_num = bit, w_fakequantize = wfq, w_observer = wob)
            else:
                raise ValueError(f'无效的模式: {mode}。模式应该是 "activation" 或 "weight"。')
    # if object_type is not None:
    #     if "object_type" in qconfig:
    #         qconfig["object_type"].update(object_type)
    #     else:
    #         qconfig["object_type"] = object_type
            
    module_name = extra_qparams.get('module_name', None)
    if module_name:
        if "module_name" not in qconfig:
            qconfig["module_name"] = {}
        for type_name,type_data in module_name.items():
            mode=module_name.get(type_name,{}).get("mode")
            bit=module_name.get(type_name,{}).get("bit")
            if mode=="activation":
                afq=module_name.get(type_name,{}).get("afakequantize")
                aob=module_name.get(type_name,{}).get("aobserver")
                qconfig["module_name"][type_name]=createQConfigForSophgo_activation(bit_num = bit, a_fakequantize = afq, a_observer = aob)
            elif mode=="weight":
                wfq=module_name.get(type_name,{}).get("wfakequantize")
                wob=module_name.get(type_name,{}).get("wobserver")
                qconfig["module_name"][type_name]=createQConfigForSophgo_weight(chip, bit_num = bit, w_fakequantize = wfq, w_observer = wob)
            else:
                raise ValueError(f'无效的模式: {mode}。模式应该是 "activation" 或 "weight"。')

    # Find INT4 op and set the config:
    int4_cfg = extra_qparams.get('int4_op', None)
    if int4_cfg:
        if "module_name" not in qconfig:
            qconfig["module_name"] = {}
        w_fakequantize = 'LearnableFakeQuantize'
        a_fakequantize = 'LearnableFakeQuantize'
        w_observer = 'MinMaxObserver'
        a_observer = 'EMAMinMaxObserver'
        w_qscheme = {
            'bit': 4,
            'symmetry': True,
            'per_channel': False,
            'pot_scale': False
        }
        a_qscheme = {
            'bit': 4,
            'symmetry': True,
            'per_channel': False,
            'pot_scale': False
        }
        int4_qconfig = createQConfig(w_fakequantize=w_fakequantize,
                                        a_fakequantize=a_fakequantize,
                                        w_qscheme=w_qscheme, a_qscheme=a_qscheme)
        for name in int4_cfg:
            print('insert INT4 FakeQuantize::', name)
            qconfig['module_name'][name] = int4_qconfig

    # Find INT8 op and set the config:
    int8_cfg = extra_qparams.get('int8_op', None)
    if int8_cfg:
        if "module_name" not in qconfig:
            qconfig["module_name"] = {}
        w_fakequantize = 'LearnableFakeQuantize'
        a_fakequantize = 'LearnableFakeQuantize'
        w_observer = 'MinMaxObserver'
        a_observer = 'EMAMinMaxObserver'
        w_qscheme = {
            'bit': 8,
            'symmetry': True,
            'per_channel': False,
            'pot_scale': False
        }
        a_qscheme = {
            'bit': 8,
            'symmetry': True,
            'per_channel': False,
            'pot_scale': False
        }
        int8_qconfig = createQConfig(w_fakequantize=w_fakequantize,
                                        a_fakequantize=a_fakequantize,
                                        w_qscheme=w_qscheme, a_qscheme=a_qscheme)
        for name in int8_cfg:
            print('insert INT8 FakeQuantize::', name)
            qconfig['module_name'][name] = int8_qconfig

    # Find F16 op and set the config
    f16_cfg = extra_qparams.get('f16_op', None)
    if f16_cfg:
        if "module_name" not in qconfig:
            qconfig["module_name"] = {}
        w_fakequantize = 'Fp16FakeQuantize'
        a_fakequantize = 'Fp16FakeQuantize'
        w_observer = 'MinMaxObserver'
        a_observer = 'EMAMinMaxObserver'
        w_qscheme = {
            'bit': 16,
            'symmetry': True,
            'per_channel': False,
            'pot_scale': False
        }
        a_qscheme = {
            'bit': 16,
            'symmetry': True,
            'per_channel': False,
            'pot_scale': False
        }    
        f16_qconfig = createQConfig(w_fakequantize=w_fakequantize, a_fakequantize=a_fakequantize, 
                                    w_qscheme=w_qscheme, a_qscheme=a_qscheme)    
        for name in f16_cfg:
            print('insert F16 FakeQuantize::', name)
            qconfig["module_name"][name] = f16_qconfig
    return qconfig

def chipparams(chip,extra_qparams,FakeQuantize):
    w_observer = extra_qparams.get('w_observer', None)
    if w_observer:
        assert w_observer in ObserverDict, \
            'Do not support observer name: {}'.format(w_observer)
        w_observer = ObserverDict[w_observer]
    a_observer = extra_qparams.get('a_observer', None)
    if a_observer:
        assert a_observer in ObserverDict, \
            'Do not support observer name: {}'.format(a_observer)
        a_observer = ObserverDict[a_observer]
    w_fakequantize = extra_qparams.get('w_fakequantize', None)
    if w_fakequantize:
        assert w_fakequantize in FakeQuantize, \
            'Do not support fakequantize name: {}'.format(w_fakequantize)
        w_fakequantize = FakeQuantizeDict[w_fakequantize]
    a_fakequantize = extra_qparams.get('a_fakequantize', None)
    if a_fakequantize:
        assert a_fakequantize in FakeQuantize, \
            'Do not support fakequantize name: {}'.format(a_fakequantize)
        a_fakequantize = FakeQuantize[a_fakequantize]
    chip_params = ParamsTable[chip]
    return chip_params,w_observer,a_observer,w_fakequantize,a_fakequantize
def createQConfigForSophgo_activation(bit_num = 4, a_fakequantize = 'LearnableFakeQuantize', a_observer = 'MinMaxObserver', a_fakeq_params = {}, a_observer_extra_args = {}):
    a_observer = ObserverDict[a_observer]
    a_fakequantize = FakeQuantizeDict[a_fakequantize]
    a_qscheme = QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=bit_num) #Sophgo_TPU use sym per-layer
    a_qscheme.kwargs.update(a_observer_extra_args)
    a_qconfig = a_fakequantize.with_args(observer=a_observer, **a_fakeq_params, **a_qscheme.to_observer_params())
    return QConfig(activation=a_qconfig, weight=None)
def createQConfigForSophgo_weight(chip, bit_num = 4, w_fakequantize = 'FixedFakeQuantize', w_observer = 'MinMaxObserver', w_fakeq_params = {}, w_observer_extra_args = {}):
    w_observer = ObserverDict[w_observer]
    w_fakequantize = FakeQuantizeDict[w_fakequantize]
    sym_range = False
    if chip in ['MARS3', 'CV183X', 'CV182X', 'CV181X', 'CV180X', 'CV186X']:
        sym_range = True
    w_qscheme = QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=bit_num, symmetric_range = sym_range) #Sophgo_TPU use sym per-layer
    w_qscheme.kwargs.update(w_observer_extra_args)
    w_qconfig = w_fakequantize.with_args(observer=w_observer, **w_fakeq_params, **w_qscheme.to_observer_params())
    return QConfig(activation=torch.nn.Identity, weight=w_qconfig) #activation use global quant conifg

def createQConfig(w_fakequantize = 'LearnableFakeQuantize', a_fakequantize = 'LearnableFakeQuantize', 
                w_observer = 'MinMaxObserver', a_observer = 'EMAMinMaxObserver', w_qscheme = {}, a_qscheme = {},
                w_fakeq_params = {}, a_fakeq_params = {}, w_observer_extra_args = {}, a_observer_extra_args = {}):
    w_observer = ObserverDict[w_observer]
    w_fakequantize = FakeQuantizeDict[w_fakequantize]
    if w_qscheme is not None:
        w_qscheme = QuantizeScheme(**w_qscheme)
    else:
        w_qscheme = QuantizeScheme(symmetry=True, per_channel=True, pot_scale=False, bit=8)

    w_qscheme.kwargs.update(w_observer_extra_args)
    w_qconfig = w_fakequantize.with_args(observer=w_observer, **w_fakeq_params, **w_qscheme.to_observer_params())

    a_observer = ObserverDict[a_observer]
    a_fakequantize = FakeQuantizeDict[a_fakequantize]
    if a_qscheme is not None:
        a_qscheme = QuantizeScheme(**a_qscheme)
    else:
        a_qscheme = QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8)
    a_qscheme.kwargs.update(a_observer_extra_args)
    a_qconfig = a_fakequantize.with_args(observer=a_observer, **a_fakeq_params, **a_qscheme.to_observer_params())
    return QConfig(activation=a_qconfig, weight=w_qconfig)

def createQConfigForInt4SophgoLiner(w_fakequantize = 'LearnableFakeQuantize', a_fakequantize = 'LearnableFakeQuantize', 
                w_observer = 'MinMaxObserver', a_observer = 'EMAMinMaxObserver', w_qscheme = {}, a_qscheme = {},
                w_fakeq_params = {}, a_fakeq_params = {}, w_observer_extra_args = {}, a_observer_extra_args = {}):
    w_observer = ObserverDict[w_observer]
    w_fakequantize = FakeQuantizeDict[w_fakequantize]
    w_qscheme = QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=4)

    w_qscheme.kwargs.update(w_observer_extra_args)
    w_qconfig = w_fakequantize.with_args(observer=w_observer, **w_fakeq_params, **w_qscheme.to_observer_params())

    a_observer = ObserverDict[a_observer]
    a_fakequantize = FakeQuantizeDict[a_fakequantize]
    a_qscheme = QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=4)
    a_qscheme.kwargs.update(a_observer_extra_args)
    a_qconfig = a_fakequantize.with_args(observer=a_observer, **a_fakeq_params, **a_qscheme.to_observer_params())
    return QConfig(activation=a_qconfig, weight=w_qconfig)

class CustomedTracer(Tracer):
    """
    ``Tracer`` is the class that implements the symbolic tracing functionality
    of ``torch.fx.symbolic_trace``. A call to ``symbolic_trace(m)`` is equivalent
    to ``Tracer().trace(m)``.
    This Tracer override the ``is_leaf_module`` function to make symbolic trace
    right in some cases.
    """
    def __init__(self, *args, customed_leaf_module=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.customed_leaf_module = customed_leaf_module

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.
        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.
        Args:
            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        """
        if self.customed_leaf_module and isinstance(m, self.customed_leaf_module):
            return True
        return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)

def duplicate_reused_nodes(graph: torch.fx.Graph, modules: Dict[str, Any] = {}):
    _dup_prefix = '_dup'
    target_dict = dict()
    dup_modules = dict()
    for node in graph.nodes:
        if node.op == "call_module":
            if node.target not in target_dict:
                target_dict[node.target] = [node]
            else:
                target_dict[node.target].append(node)
    for key in target_dict:
        if len(target_dict[key]) > 1:
            for idx, node in enumerate(target_dict[key]):
                if idx == 0:
                    continue
                module = deepcopy(modules[node.target])
                node.target += _dup_prefix + str(idx)
                dup_modules[node.target] = module
    graph.lint()
    return graph, dup_modules

def prepare_constant_dict(graph: torch.fx.Graph, model: torch.nn.Module):
    def _get_attrs(target, attrs):
        attrs = attrs.split('.')
        for att in attrs:
            target = getattr(target, att)
        return target
    constant_dict = dict()
    for node in graph.nodes:
        if node.op == 'get_attr':
            constant_dict[node.target] = _get_attrs(model, node.target)
    return constant_dict

def prepare_by_platform(
        model: torch.nn.Module,
        input_shape_dict: list = None,
        prepare_custom_config_dict: Dict[str, Any] = {},
        custom_tracer: Tracer = None):
    """
    Args:
        model (torch.nn.Module):

    >>> prepare_custom_config_dict : {
            quant_dict:Dict,Select quantization strategy,chip,mode
            extra_qconfig_dict : Dict, Find explanations in get_qconfig_by_platform,
            extra_quantizer_dict: Extra params for quantizer.
            preserve_attr: Dict, Specify attribute of model which should be preserved
                after prepare. Since symbolic_trace only store attributes which is
                in forward. If model.func1 and model.backbone.func2 should be preserved,
                {"": ["func1"], "backbone": ["func2"] } should work.
            Attr below is inherited from Pytorch.
            concrete_args: Specify input for model tracing.
            extra_fuse_dict: Specify extra fusing patterns and functions.
        }

    """
    model_mode = 'Training' if model.training else 'Eval'

    # Get Qconfig
    extra_qconfig_dict = prepare_custom_config_dict.get('extra_qconfig_dict', {})
    quant_dict = prepare_custom_config_dict.get('quant_dict')
    chip=quant_dict['chip']
    strategy=quant_dict['strategy']
    logger.info("Quantize model Scheme: {} Mode: {}".format(quant_dict['strategy'], model_mode))
    qconfig = get_qconfig_by_platform(quant_dict, extra_qconfig_dict)

    _swap_ff_with_fxff(model)
    # Preserve attr.
    preserve_attr_dict = dict()
    if 'preserve_attr' in prepare_custom_config_dict:
        for submodule_name in prepare_custom_config_dict['preserve_attr']:
            cur_module = model
            if submodule_name != "":
                cur_module = getattr(model, submodule_name)
            preserve_attr_list = prepare_custom_config_dict['preserve_attr'][submodule_name]
            preserve_attr_dict[submodule_name] = {}
            for attr in preserve_attr_list:
                preserve_attr_dict[submodule_name][attr] = getattr(cur_module, attr)
    # Symbolic trace
    concrete_args = prepare_custom_config_dict.get('concrete_args', None)
    customed_leaf_module = prepare_custom_config_dict.get('leaf_module', [])
    tracer = CustomedTracer(customed_leaf_module=tuple(customed_leaf_module))
    if custom_tracer is not None:
        tracer = custom_tracer
    graph = tracer.trace(model, concrete_args)
    print('>>>>>trace graph:',graph)
    name = model.__class__.__name__ if isinstance(model, torch.nn.Module) else model.__name__
    modules = dict(model.named_modules())
    print('>>>>>named_modules:',modules[''])
    graph, duplicated_modules = duplicate_reused_nodes(graph, modules)
    constant_nodes = prepare_constant_dict(graph, model)
    modules.update(duplicated_modules)
    modules.update(constant_nodes)
    graph_module = GraphModule(modules, graph, name)
    if input_shape_dict is not None:
        try:
            from torch.fx.passes import shape_prop
            dev = next(model.parameters()).device
            dummy_input = [torch.rand(shape).to(dev) for shape in input_shape_dict]
            shape_prop.ShapeProp(graph_module).propagate(*dummy_input)
        except:
            print('waring, shape_prop fail')
    # Model fusion.
    extra_fuse_dict = prepare_custom_config_dict.get('extra_fuse_dict', {})
    extra_fuse_dict.update(fuse_custom_config_dict)
    # Prepare
    import sophgo_mq.custom_quantizer  # noqa: F401
    extra_quantizer_dict = prepare_custom_config_dict.get('extra_quantizer_dict', {})
    quantizer = DEFAULT_MODEL_QUANTIZER[chip](extra_quantizer_dict, extra_fuse_dict,quant_dict)
    if chip == "Academic":
        prepared = quantizer.prepare_swint(graph_module, qconfig)
    else:
        prepared = quantizer.prepare(graph_module, qconfig)
    # Restore attr.
    if 'preserve_attr' in prepare_custom_config_dict:
        for submodule_name in prepare_custom_config_dict['preserve_attr']:
            cur_module = prepared
            _type = type(model)
            if submodule_name != "":
                cur_module = getattr(prepared, submodule_name)
                _type = type(getattr(model, submodule_name))
            preserve_attr_list = prepare_custom_config_dict['preserve_attr'][submodule_name]
            for attr_name in preserve_attr_list:
                logger.info("Preserve attr: {}.{}".format(submodule_name, attr_name))
                _attr = preserve_attr_dict[submodule_name][attr_name]
                if inspect.ismethod(_attr):
                    _attr = types.MethodType(getattr(_type, attr_name), cur_module)
                setattr(cur_module, attr_name, _attr)
    return prepared
