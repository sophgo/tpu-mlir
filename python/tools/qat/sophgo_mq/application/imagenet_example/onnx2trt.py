import onnx
import pycuda.autoinit # noqa F401
import tensorrt as trt
import torch
import json
import pycuda.driver as cuda
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import numpy as np
import argparse

from main import AverageMeter, accuracy

def onnx2trt(onnx_model,
             trt_path,
             dataset_path,
             batch_size=1,
             cali_batch=10,
             log_level=trt.Logger.ERROR,
             max_workspace_size=1 << 30,
             device_id=0,
             mode='fp32',
             is_explicit=False,
             dynamic_range_file=None):
    if os.path.exists(trt_path):
        print(f'The "{trt_path}" exists. Remove it and continue.')
        os.remove(trt_path)

    device = torch.device('cuda:{}'.format(device_id))

    # create builder and network
    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)

    if isinstance(onnx_model, str):
        onnx_model = onnx.load(onnx_model)

    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'parse onnx failed:\n{error_msgs}')

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size

    if mode == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        if dynamic_range_file:
            with open(dynamic_range_file, 'r') as f:
                dynamic_range = json.load(f)['tensorrt']['blob_range']

            for input_index in range(network.num_inputs):
                input_tensor = network.get_input(input_index)
                if input_tensor.name in dynamic_range:
                    amax = dynamic_range[input_tensor.name]
                    input_tensor.dynamic_range = (-amax, amax)
                    print(f'Set dynamic range of {input_tensor.name} as [{-amax}, {amax}]')

            for layer_index in range(network.num_layers):
                layer = network[layer_index]
                output_tensor = layer.get_output(0)
                if output_tensor.name in dynamic_range:
                    amax = dynamic_range[output_tensor.name]
                    output_tensor.dynamic_range = (-amax, amax)
                    print(f'Set dynamic range of {output_tensor.name} as [{-amax}, {amax}]')
        elif is_explicit:
            # explicit mode do not need calibrator
            pass
        else:
            from calibrator import ImagenetCalibrator
            calidir = os.path.join(dataset_path, 'cali')
            dataset = datasets.ImageFolder(calidir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]))
            cali_num = min(len(dataset), batch_size * cali_batch)
            cali_dataset = torch.utils.data.Subset(dataset, indices=torch.arange(cali_num))
            cali_loader = torch.utils.data.DataLoader(cali_dataset, batch_size=batch_size, shuffle=False,
                                                      num_workers=1, pin_memory=False)
            calibrator = ImagenetCalibrator(cali_loader, cache_file='imagenet.cache')
            config.int8_calibrator = calibrator
            print(f'Calibration Set!')

    # create engine
    engine = builder.build_engine(network, config)

    with open(trt_path, mode='wb') as f:
        f.write(bytearray(engine.serialize()))
    return engine

def infer(engine, img, batch_size, context):
    h_input = img = img
    h_input_mem = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(0)[1:]), dtype=np.float32)
    h_output_mem = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(1)[1:]), dtype=np.float32)
    # import pdb; pdb.set_trace()
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input_mem.nbytes)
    d_output = cuda.mem_alloc(h_output_mem.nbytes)
    # Transfer input data to the GPU.
    cuda.memcpy_htod(d_input, h_input)
    # Run inference.
    context.execute_v2([d_input, d_output])
    # Return the host output.
    cuda.memcpy_dtoh(h_output_mem, d_output)
    d_input.free()
    d_output.free()

    return h_output_mem.reshape(batch_size, *engine.get_binding_shape(1)[1:])

def validate(trt_file, batch_size=64, dataset_path=None):
    # deserialize engine
    trt_logger = trt.Logger(trt.Logger.INFO)

    with trt.Runtime(trt_logger) as runtime:
        with open(trt_file, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

    # prepare dateset
    valdir = os.path.join(dataset_path, 'val')
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=False)

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # evaluate
    for index, (images, target) in enumerate(val_loader):
        images = images.detach().numpy()
        output = infer(engine, images, len(images), context)
        output = torch.from_numpy(output)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.shape[0])
        top5.update(acc5[0], images.shape[0])

        if index % 100 == 0:
            print(f' {index} ==> * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    print(f' Final ==> * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Onnx to tensorrt')
    parser.add_argument('--onnx-path', type=str, default=None)
    parser.add_argument('--trt-path', type=str, default=None)
    parser.add_argument('--mode', choices=['fp32', 'int8'], default='int8')
    parser.add_argument('--clip-range-file', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--explicit', action='store_true')
    parser.add_argument('--data-path', type=str, required=True)
    args = parser.parse_args()

    if args.onnx_path:
        onnx2trt(args.onnx_path,
                 trt_path=args.trt_path,
                 mode=args.mode,
                 is_explicit=args.explicit,
                 dataset_path=args.data_path,
                 batch_size=args.batch_size,
                 cali_batch=10,
                 log_level=trt.Logger.VERBOSE if args.verbose else trt.Logger.ERROR,
                 dynamic_range_file=args.clip_range_file)
    if args.evaluate:
        validate(args.trt_path, batch_size=args.batch_size, dataset_path=args.data_path)
