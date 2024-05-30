import numpy as np
import argparse
from data.imagenet import load_data
from models import load_model
from utils import parse_config, seed_all, evaluate
from sophgo_mq.prepare_by_platform import prepare_by_platform, BackendType
from sophgo_mq.advanced_ptq import ptq_reconstruction
from sophgo_mq.convert_deploy import convert_deploy

backend_dict = {
    'Academic': BackendType.Academic,
    'Tensorrt': BackendType.Tensorrt,
    'SNPE': BackendType.SNPE,
    'PPLW8A16': BackendType.PPLW8A16,
    'NNIE': BackendType.NNIE,
    'Vitis': BackendType.Vitis,
    'ONNX_QNN': BackendType.ONNX_QNN,
    'PPLCUDA': BackendType.PPLCUDA,
}


def load_calibrate_data(train_loader, cali_batchsize):
    cali_data = []
    for i, batch in enumerate(train_loader):
        cali_data.append(batch[0])
        if i + 1 == cali_batchsize:
            break
    return cali_data


def get_quantize_model(model, config):
    backend_type = BackendType.Academic if not hasattr(
        config.quantize, 'backend') else backend_dict[config.quantize.backend]
    extra_prepare_dict = {} if not hasattr(
        config, 'extra_prepare_dict') else config.extra_prepare_dict
    return prepare_by_platform(
        model, backend_type, prepare_custom_config_dict=extra_prepare_dict)


def deploy(model, config):
    backend_type = BackendType.Academic if not hasattr(
        config.quantize, 'backend') else backend_dict[config.quantize.backend]
    output_path = './' if not hasattr(
        config.quantize, 'deploy') else config.quantize.deploy.output_path
    model_name = config.quantize.deploy.model_name
    deploy_to_qlinear = False if not hasattr(
        config.quantize.deploy, 'deploy_to_qlinear') else config.quantize.deploy.deploy_to_qlinear

    convert_deploy(model, backend_type, {
                   'input': [1, 3, 224, 224]}, output_path=output_path, model_name=model_name, deploy_to_qlinear=deploy_to_qlinear)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ImageNet Solver')
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()
    config = parse_config(args.config)
    # seed first
    seed_all(config.process.seed)
    # load_model
    model = load_model(config.model)
    if hasattr(config, 'quantize'):
        model = get_quantize_model(model, config)
    model.cuda()
    # load_data
    train_loader, val_loader = load_data(**config.data)
    # evaluate
    if not hasattr(config, 'quantize'):
        evaluate(val_loader, model)
    elif config.quantize.quantize_type == 'advanced_ptq':
        print('begin calibration now!')
        cali_data = load_calibrate_data(train_loader, cali_batchsize=config.quantize.cali_batchsize)
        from sophgo_mq.utils.state import enable_quantization, enable_calibration_woquantization
        # do activation and weight calibration seperately for quick MSE per-channel for weight one
        model.eval()
        import torch
        with torch.no_grad():
            enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
            for batch in cali_data:
                model(batch.cuda())
            enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
            model(cali_data[0].cuda())
        print('begin advanced PTQ now!')
        if hasattr(config.quantize, 'reconstruction'):
            model = ptq_reconstruction(
                model, cali_data, config.quantize.reconstruction)
        enable_quantization(model)
        evaluate(val_loader, model)
        if hasattr(config.quantize, 'deploy'):
            deploy(model, config)
    elif config.quantize.quantize_type == 'naive_ptq':
        print('begin calibration now!')
        cali_data = load_calibrate_data(train_loader, cali_batchsize=config.quantize.cali_batchsize)
        from sophgo_mq.utils.state import enable_quantization, enable_calibration_woquantization
        # do activation and weight calibration seperately for quick MSE per-channel for weight one
        model.eval()
        enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
        for batch in cali_data:
            model(batch.cuda())
        enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
        model(cali_data[0].cuda())
        print('begin quantization now!')
        enable_quantization(model)
        evaluate(val_loader, model)
        if hasattr(config.quantize, 'deploy'):
            deploy(model, config)
    else:
        print("The quantize_type must in 'naive_ptq' or 'advanced_ptq',")
        print("and 'advanced_ptq' need reconstruction configration.")
