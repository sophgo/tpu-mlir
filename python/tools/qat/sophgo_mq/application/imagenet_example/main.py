import sys
import os
import time
sys.path.append(os.path.abspath('.'))
print(sys.path)

import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from sophgo_mq.convert_deploy import convert_deploy, convert_onnx, export_onnx_with_fakequant_node
from sophgo_mq.prepare_by_platform import prepare_by_platform
from sophgo_mq.utils.state import enable_calibration, enable_quantization, disable_all
from tools.model_runner import mlir_inference

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--train_data', metavar='DIR',
                    help='path to dataset', required=True)
parser.add_argument('--val_data', metavar='DIR',
                    help='path to dataset', required=True)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--cuda', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument('--chip', type=str, choices=['BM1688', 'BM1684X', 'BM1690', 'academic', 'MARS3', 'SG2380','CV183X', 'CV182X', 'CV181X', 'CV180X', 'CV186X'], default='BM1690')
parser.add_argument('--quantmode', type=str, choices=['weight_activation', 'weight_only'], default='weight_activation')
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--not-quant', action='store_true')
parser.add_argument('--deploy', action='store_true')
parser.add_argument('--fast_test', action='store_true')
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--pre_eval_and_export', action='store_true')
parser.add_argument('--deploy_batch_size', default=1, type=int, help='deploy_batch_size.')
parser.add_argument('--fp8_e4m3', action='store_true')
parser.add_argument('--fp8_e5m2', action='store_true')
parser.add_argument('--bf16_mix_prec', action='store_true')
parser.add_argument('--export_onnx_before_training', action='store_true')
parser.add_argument('--print_freq', default=5, type=int, help='print_freq')

best_acc1 = 0

def main():
    time_start = time.time()
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = './'
    if args.fp8_e4m3:
        args.output_path=os.path.join(args.output_path, args.arch+'_fp8_e4m3')
    elif args.fp8_e5m2:
        args.output_path=os.path.join(args.output_path, args.arch+'_fp8_e5m2')
    else:
        args.output_path=os.path.join(args.output_path, args.arch)
    os.system('mkdir -p {}'.format(args.output_path))

    args.quant = not args.not_quant

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.cuda is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.cuda, ngpus_per_node, args)

    time_end = time.time()
    print('totally time is ', time_end-time_start)

def generate_random_string(length):
    """
    生成指定长度的随机字符串，包含大小写字母和数字。
    """
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return ''.join(random.choices(chars, k=length))

features_out_hook = {}
def hook(module, fea_in, fea_out):
    name = generate_random_string(15)
    global features_out_hook
    if name in features_out_hook:
        name = generate_random_string(15)
    if isinstance(fea_out, torch.Tensor):
        features_out_hook[f'f_{name}'] = fea_out.cpu().numpy()
    return None

def gen_test_ref_data(cali_loader, model, args):
    model.eval()
    hook_handles = []
    input_data = {}
    data_num = 3
    for name, child in model.named_modules():
        hd = child.register_forward_hook(hook=hook)
        hook_handles.append(hd)
    if args.cpu:
        model = model.cpu()
    with torch.no_grad():
        for i, (images, target) in enumerate(cali_loader):
            if args.cpu:
                images = images.cpu()
            else:
                images = images.cuda(args.cuda, non_blocking=True)
            output = model(images)
            if i < data_num:
                print("gen_test_ref_data ==> ", i)
                input_data['data'] = images.cpu().numpy()
                file_name = f'input_data_{i}.npz'
                os.system(f'rm -f {file_name}')
                np.savez(os.path.join(args.output_path, file_name), **input_data)
                global features_out_hook
                features_out_hook['data'] = input_data['data']
                file_name = f'layer_outputs_{i}.npz'
                os.system(f'rm -f {file_name}')
                np.savez(os.path.join(args.output_path, file_name), **features_out_hook)
                features_out_hook.clear()
                input_data.clear()
            if i == data_num:
                for hd in hook_handles:
                    hd.remove()
                break
    print("End gen_test_ref_data.")
    return

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.cuda = gpu

    if args.cuda is not None:
        print("Use GPU: {} for training".format(args.cuda))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print(args.arch)
        print(f"=> creating model '{args.arch}'")
        model = models.__dict__[args.arch]()

    # for internal cluster
    if args.model_path:
        state_dict = torch.load(args.model_path)
        print(f'load pretrained checkpoint from: {args.model_path}')
        model.load_state_dict(state_dict)

    train_loader, train_sampler, val_loader, cali_loader, bmodel_test_loader = prepare_dataloader(args)
    criterion = nn.CrossEntropyLoss().cuda(args.cuda)
    if args.cuda is not None:
        model = model.cuda(args.cuda)
    else:
        model = model.cpu()

    if args.pre_eval_and_export:
        print('原始onnx模型精度')
        validate(val_loader, model.eval(), criterion, args)  #这里未执行model.cuda()，会报错

        # kwargs = {
        #     'input_shape_dict': {'data': [args.deploy_batch_size, 3, 224, 224]},
        #     'output_path': args.output_path,
        #     'model_name':  args.arch,
        #     'dummy_input': None,
        #     'onnx_model_path':  os.path.join(args.output_path, '{}_ori.onnx'.format(args.arch)),
        # }
        # module_tmp = copy.deepcopy(model)
        # module_tmp = module_tmp.cpu()
        # convert_onnx(module_tmp.eval(), **kwargs)
        # del module_tmp
        model = model.train() #prepare前一定要是train模式!!

    # quantize model
    if args.quant:
        extra_prepare_dict = {
            'quant_dict': {
                        'chip': args.chip,
                        'quantmode': args.quantmode,
                        'strategy': 'CNN',
                        'bf16_mix_prec': args.bf16_mix_prec
                       },
        }
        if args.fp8_e4m3:
            extra_prepare_dict["extra_qconfig_dict"] = {
                                    'w_observer': 'MinMaxObserver',
                                    'a_observer': 'EMAMinMaxObserver',
                                    "w_fakequantize": 'E4M3FakeQuantize',
                                    "a_fakequantize": 'E4M3FakeQuantize',
                                    'w_qscheme': {  'bit': 8,
                                                    'symmetry': True,
                                                    'per_channel': False,
                                                    'pot_scale': False },
                                    'a_qscheme': {  'bit': 8,
                                                    'symmetry': True,
                                                    'per_channel': False,
                                                    'pot_scale': False }
                                }
        if args.fp8_e5m2:
            extra_prepare_dict["extra_qconfig_dict"] = {
                                    'w_observer': 'MinMaxObserver',
                                    'a_observer': 'EMAMinMaxObserver',
                                    "w_fakequantize": "E5M2FakeQuantize",
                                    "a_fakequantize": 'E5M2FakeQuantize',
                                    'w_qscheme': {  'bit': 8,
                                                    'symmetry': True,
                                                    'per_channel': False,
                                                    'pot_scale': False },
                                    'a_qscheme': {  'bit': 8,
                                                    'symmetry': True,
                                                    'per_channel': False,
                                                    'pot_scale': False }
                                }

        if "mobilenet_v3" in args.arch:
            extra_prepare_dict["extra_quantizer_dict"] = {'module_only_enable_observer': [
                                                                    'features.0.0.weight_fake_quant',
                                                                    'features.1.block.0.0.weight_fake_quant',
                                                                    'features.1.block.1.fc1.weight_fake_quant',
                                                                    'features.1.block.1.fc2.weight_fake_quant',
                                                                    'features.1.block.2.0.weight_fake_quant',
                                                                    'features.2.block.0.0.weight_fake_quant',
                                                                    'features.2.block.1.0.weight_fake_quant',
                                                                    'features.2.block.2.0.weight_fake_quant',

                                                                    'x_post_act_fake_quantizer',
                                                                    'features_0_0_post_act_fake_quantizer',
                                                                    'features_0_2_post_act_fake_quantizer',
                                                                    'features_1_block_0_0_post_act_fake_quantizer',
                                                                    'features_1_block_1_avgpool_post_act_fake_quantizer',
                                                                    'features_1_block_1_fc1_post_act_fake_quantizer',
                                                                    'features_1_block_1_fc2_post_act_fake_quantizer',
                                                                    'features_1_block_1_scale_activation_post_act_fake_quantizer',
                                                                    'mul_post_act_fake_quantizer',
                                                                    'features_1_block_2_0_post_act_fake_quantizer',
                                                                    'features_2_block_0_0_post_act_fake_quantizer',
                                                                    'features_2_block_1_0_post_act_fake_quantizer',
                                                                    ]
                                                                }
        if "extra_qconfig_dict" not in extra_prepare_dict:
            extra_prepare_dict["extra_qconfig_dict"] = {}
        extra_prepare_dict["extra_qconfig_dict"]['object_type'] = {
                                    nn.Linear: {
                                        'mode': 'weight',
                                        'bit': 8,
                                        'wfakequantize': 'LearnableFakeQuantize',
                                        'wobserver': 'MinMaxObserver',
                                        }
                                    }
        model = prepare_by_platform(model, input_shape_dict = {'data': [args.deploy_batch_size, 3, 224, 224]}, prepare_custom_config_dict=extra_prepare_dict)
        print('>>>>>prepared module:', model)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.cuda is not None:
            torch.cuda.set_device(args.cuda)
            model.cuda(args.cuda)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.cuda])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.cuda is not None:
        torch.cuda.set_device(args.cuda)
        model = model.cuda(args.cuda)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            if args.cpu:
                model = model.cpu()
            else:
                # model = torch.nn.DataParallel(model).cuda() #会导致gpu训练保存的模型无法resume后用cpu推理
                model = model.cuda()

    # define loss function (criterion) and optimizer
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=args.weight_decay,
                                     amsgrad=False)

    if args.quant:
        enable_calibration(model)
        calibrate(cali_loader, model, args)
        # export graphmodule with fakequant node to onnx, so we can clearly see the positions of each fakequant node.
        if args.export_onnx_before_training:
            os.system('mkdir -p {}'.format(args.output_path+'_export_onnx_before_training'))
            export_onnx_with_fakequant_node(model.eval(), args.chip, input_shape_dict=
                {'data': [args.deploy_batch_size, 3, 224, 224]},
                model_name='{}_with_fakequant_node'.format(args.arch),
                output_path=args.output_path+'_export_onnx_before_training')
    if args.quant:
        enable_quantization(model)

    cudnn.benchmark = True
    # cudnn.deterministic = True #避免计算结果波动

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.cuda is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.cuda)
                if args.cpu:
                    loc = 'cpu'
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.cuda is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.cuda)

            state_dict = checkpoint['state_dict']
            # if args.cpu:
            #     model = torch.nn.DataParallel(model).cpu()
            model_dict = model.state_dict()
            if 'module.' in list(state_dict.keys())[0] and 'module.' not in list(model_dict.keys())[0]:
                for k in list(state_dict.keys()):
                    state_dict[k[7:]] = state_dict.pop(k)

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}), acc = {}"
                  .format(args.resume, checkpoint['epoch'], best_acc1))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit(1)

        if args.evaluate:
            print('resume模型精度')
            from sophgo_mq.convert_deploy import convert_merge_bn
            module_tmp2 = copy.deepcopy(model)
            convert_merge_bn(module_tmp2.eval())
            validate(val_loader, module_tmp2, criterion, args)
            del module_tmp2
            gen_test_ref_data(cali_loader, model, args)
            convert_deploy(model.eval(), args.chip, val_loader, input_shape_dict={'data': [args.deploy_batch_size, 3, 224, 224]},
                model_name='{}'.format(args.arch), output_path=args.output_path)
        exit(0)

    if args.fast_test:
        args.epochs = 1
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        if epoch == args.epochs - 1:
            print('qat训练后的带量化节点的eval精度:')
        else:
            print(f'epoch{epoch}训练后eval精度:')
        validate(val_loader, model, criterion, args)

    gen_test_ref_data(cali_loader, model, args)
    enable_quantization(model)

    net_type = 'CNN'
    mlir_model_path = convert_deploy(model.eval(), args.chip, val_loader, net_type, input_shape_dict=
        {'data': [args.deploy_batch_size, 3, 224, 224]},
        model_name='{}'.format(args.arch),
        output_path=args.output_path, bf16_mix_prec = args.bf16_mix_prec)
    validate_for_chip_model(bmodel_test_loader, mlir_model_path, criterion, args)

def prepare_dataloader(args):
    traindir = os.path.join(args.train_data, 'train')
    valdir = os.path.join(args.val_data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225] )
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    cali_num = 200
    assert cali_num % args.deploy_batch_size == 0
    cali_dataset = torch.utils.data.Subset(train_dataset, indices=torch.arange(cali_num))
    cali_loader = torch.utils.data.DataLoader(cali_dataset, batch_size=args.deploy_batch_size, shuffle=False,
                                                num_workers=args.workers, pin_memory=True)

    bmodel_test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.deploy_batch_size, shuffle=False,
                                                num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return train_loader, train_sampler, val_loader, cali_loader, bmodel_test_loader

def prepare_dataloader_batch(args, batch_size):
    valdir = os.path.join(args.val_data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return val_loader

def calibrate(cali_loader, model, args):
    model.eval()
    print("Start calibration ...")
    print("Calibrate images number = ", len(cali_loader.dataset))
    with torch.no_grad():
        for i, (images, target) in enumerate(cali_loader):
            if args.cuda is not None:
                images = images.cuda(args.cuda, non_blocking=True)
            output = model(images)
            print("Calibration ==> ", i+1)
    print("End calibration.")
    return

def get_node_name_by_module_name(qname, model):
    nodes = list(model.graph.nodes)
    modules = dict(model.named_modules())
    for node in nodes:
        if node.target in modules and qname == node.target:
            return node.name

def get_node_input_by_module_name(qname, model):
    nodes = list(model.graph.nodes)
    modules = dict(model.named_modules())
    post_str = '_post_act_fake_quantizer'
    input_str = '_input_act_fake_quantizer'
    scale_name = None
    for node in nodes:
        if node.target in modules and qname == node.target:
            print(f'{qname} input:', node.args[0].name)
            if post_str in node.args[0].name:
                scale_name = node.args[0].name
                break
            elif input_str in node.args[0].name:
                node2 = node.args[0]
                print(f'{node.args[0].name}.input:', node2.args[0].name)
                if post_str in node2.args[0].name:
                    scale_name = node2.args[0].name
                    break
                elif 'x' == node2.args[0].name:
                    return 'data'
            break
    if scale_name is not None:
        return scale_name[:len(scale_name)-len(post_str)]
    else:
        return ''

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda is not None and torch.cuda.is_available():
            images = images.cuda(args.cuda, non_blocking=True)
            target = target.cuda(args.cuda, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        # # 检查训练过程参数是否异常
        # for param in model.named_parameters():
        #     sum = torch.isnan(param[1]).sum()
        #     if sum > 0:
        #         print(param[0], 'has Nan', param[1].shape, 'sum:', sum)

        if args.fast_test:
            if i % 64 == 0:
                break

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    print('args.print_freq:', args.print_freq)
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if not args.cpu:
                if args.cuda is not None:
                    images = images.cuda(args.cuda, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(args.cuda, non_blocking=True)
            else:
                images = images.cpu()
                target = target.cpu()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

            if args.fast_test:
                if i % 100 == 0:
                    break

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    return top1.avg


def validate_for_chip_model(bmodel_test_loader, mlir_model_path, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(bmodel_test_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    end = time.time()
    inputs = {}
    for i, (images, target) in enumerate(bmodel_test_loader):
        images = images.cpu()
        target = target.cpu()
        # compute output
        inputs['data'] = images.numpy()
        output = mlir_inference(inputs, mlir_model_path, dump_all = False)
        # print('output.names:', list(output.keys()))
        output = torch.from_numpy(list(output.values())[0])
        # print('output:', output.shape, ' target:', target.shape)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        # if args.fast_test:
        #     if i % 100 == 0:
        #         break

    # TODO: this should also be done with the ProgressMeter
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))

    return top1.avg


def validate_onnx(criterion, args):
    import onnxruntime as rt
    val_loader = prepare_dataloader_batch(args, args.deploy_batch_size)
    model_path = os.path.join(args.output_path, '{}_deploy_model.onnx'.format('{}'.format(args.arch)))
    sess = rt.InferenceSession(model_path, providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')


    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if not args.cpu:
                if args.cuda is not None:
                    images = images.cuda(args.cuda, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(args.cuda, non_blocking=True)
            else:
                images = images.cpu()
                target = target.cpu()

            # compute output
            # output = model(images)
            output = sess.run(None, {input_name:images.clone().detach().cpu().numpy()})
            output = torch.from_numpy(output[0]).cuda(args.cuda, non_blocking=True)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

            # if args.fast_test:
            #     if i % 100 == 0:
            #         break

        # TODO: this should also be done with the ProgressMeter
        print('deploy_model.onnx完成所有处理后的onnxruntime测试精度:')
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'_best')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
