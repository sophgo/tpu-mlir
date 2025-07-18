import torch

from sophgo_mq.utils.logger import logger


def enable_calibration(model):
    logger.info('Enable observer and Disable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            logger.debug('Enable observer and Disable quant: {}'.format(name))
            submodule.enable_observer()
            submodule.disable_fake_quant()


def enable_calibration_woquantization(model, quantizer_type='fake_quant'):
    logger.info('Enable observer and Disable quantize for {}'.format(quantizer_type))
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            if quantizer_type not in name:
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            logger.debug('Enable observer and Disable quant: {}'.format(name))
            submodule.enable_observer()
            submodule.disable_fake_quant()


def enable_calibration_quantization(model, quantizer_type='fake_quant'):
    logger.info('Enable observer and Enable quantize for {}'.format(quantizer_type))
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            if quantizer_type not in name:
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            logger.debug('Enable observer and Enable quant: {}'.format(name))
            submodule.enable_observer()
            submodule.enable_fake_quant()


def enable_quantization(model):
    logger.info('Disable observer and Enable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            logger.debug('Disable observer and Enable quant: {}'.format(name))
            submodule.disable_observer()
            if not submodule.only_enable_observer:
                submodule.enable_fake_quant()


def enable_quantization_except_some_block(model, block_name_list):
    logger.info('Disable observer and Enable quantize except some block.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            submodule.disable_observer()
            submodule.disable_fake_quant()
            if name not in block_name_list:
                logger.debug('Disable observer : {}'.format(name))
                submodule.enable_fake_quant()


def disable_all(model):
    logger.info('Disable observer and Disable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            logger.debug('Disable observer and Disable quantize: {}'.format(name))
            submodule.disable_observer()
            submodule.disable_fake_quant()
