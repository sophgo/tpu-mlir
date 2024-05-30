import os
import sys
import copy
import time
import torch
import random
import q_model
import logging
import datasets
import torchvision
import argparse
import transformers
import numpy as np
from PIL import Image
import torch.nn as nn
from tqdm.auto import tqdm
from itertools import chain
from easydict import EasyDict
from datasets import load_metric
from transformers import Trainer
import torchvision.transforms as transforms
import image_classification_utils
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers.onnx.features import FeaturesManager
from transformers.utils.fx import HFTracer, get_concrete_args
from transformers.trainer_utils import get_last_checkpoint, EvalLoopOutput
from sophgo_mq.convert_deploy import convert_deploy
from sophgo_mq.prepare_by_platform import prepare_by_platform
from sophgo_mq.utils.state import enable_quantization, enable_calibration_woquantization,enable_calibration,disable_all

logger = logging.getLogger("transformer")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def pil_loader(path: str):
    with open(path, "rb") as f:
        print('hi')
        im = Image.open(f)
        return im.convert("RGB")


def set_logger(config_progress):

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = config_progress.log_level
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def quantize_model(model, config_quant):
    tracer = HFTracer()
    input_names = ['pixel_values']
    prepare_custom_config_dict = {
        'quant_dict': {
            'chip': 'Academic',
            'strategy': 'Transformer',
            'quantmode': 'weight_activation'
        },
        'extra_qconfig_dict': {
            'w_observer': config_quant.w_qconfig.observer,
            'a_observer': config_quant.a_qconfig.observer,
            'w_fakequantize': config_quant.w_qconfig.quantizer,
            'a_fakequantize': config_quant.a_qconfig.quantizer,
            'w_qscheme': {
                'bit': config_quant.w_qconfig.bit,
                'symmetry': config_quant.w_qconfig.symmetric,
                'per_channel': True if config_quant.w_qconfig.ch_axis == 0 else False,
                'pot_scale': False,
            },
            'a_qscheme': {
                'bit': config_quant.a_qconfig.bit,
                'symmetry': config_quant.a_qconfig.symmetric,
                'per_channel': True if config_quant.a_qconfig.ch_axis == 0 else False,
                'pot_scale': False,
            },
            'int4_op': [
 
            ],
        },
        'concrete_args': get_concrete_args(model, input_names),
        'preserve_attr': {'': ['config', 'num_labels']},
    }
    model = prepare_by_platform(
        model=model,
        # deploy_backend=backend,
        prepare_custom_config_dict=prepare_custom_config_dict,
        custom_tracer=tracer
    )
    model.eval()
    return model

def calibrate(cali_loader, model):
    model.eval()
    print("Start Calibration ...")
    print("Calibrate images number = ", len(cali_loader.dataset))
    with torch.no_grad():
        for i, (images, target) in enumerate(cali_loader):
            images = images.cuda()
            output = model(images)
            print("Calibration ==> ", i+1)
    print("End Calibration.")
    return

def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    
    model.cuda()
    model.train()

    end = time.time()
    for batch, (images, target) in enumerate(dataloader, start=1):
        data_time.update(time.time() - end)
        images, target = images.to(device), target.to(device)
        pred = model(images)
        loss = loss_fn(pred['logits'], target)

        acc1, acc5 = accuracy(pred['logits'], target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        lr_scheduler.step()
        total_loss += loss.item()

        batch_time.update(time.time() - end)
        end = time.time()

        if batch % 1000 == 0:
            progress.display(batch)

    return total_loss

def test_loop(dataloader, model, loss_fn, mode='Test'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')    
    
    model.eval()
    with torch.no_grad():
        end = time.time()
        for batch, (images, target) in enumerate(dataloader, start=1):
            images, target = images.to(device), target.to(device)
            pred = model(images)
            loss = loss_fn(pred['logits'], target)

            acc1, acc5 = accuracy(pred['logits'], target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if batch % 1000 == 0:
                progress.display(batch)
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))
        
    return top1.avg

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
    
def main(config_path):
    config = image_classification_utils.parse_config(config_path)
    set_seed(config.train.seed)
    training_args = image_classification_utils.make_huggingface_training_args(config.train, config.progress)
    set_logger(config.progress)
    raw_datasets = image_classification_utils.load_image_dataset(config.data, config.model)
    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    labels = raw_datasets["validation"].features["labels"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    model, feature_extractor = image_classification_utils.load_model(config.model, len(labels), label2id, id2label)
    traindir = os.path.join(config.data.train_dir)
    valdir = os.path.join(config.data.validation_dir)
    normalize = transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)

    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = config.train.per_device_train_batch_size,
        shuffle=False, num_workers=1, pin_memory=True, sampler=None
    )

    eval_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size = config.train.per_device_eval_batch_size,
        num_workers=1, shuffle=False, pin_memory=True)
    
    # Calibration
    cali_batch = 20
    cali_batch_size = 10
    cali_dataset = torch.utils.data.Subset(train_dataset, indices=torch.arange(cali_batch_size * cali_batch))
    cali_loader = torch.utils.data.DataLoader(cali_dataset, batch_size=cali_batch_size,
                                              shuffle=False, num_workers=1, pin_memory=True)
    
    if hasattr(config, 'quant'):
        model = quantize_model(model, config.quant)
    model.to(device)
    enable_calibration(model)
    calibrate(cali_loader, model)
    # Original Model
    model_copy = copy.deepcopy(model)
    disable_all(model_copy)
    model_copy = model_copy.train()
    optimizer = AdamW(model_copy.parameters(), lr=1e-5)
    epoch_num = 1
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=epoch_num * len(train_loader),
    )
    loss_fn = nn.CrossEntropyLoss().cuda()
    total_loss = 0
    best_acc = 0
    for t in range(epoch_num):
        print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
        total_loss = train_loop(train_loader, model_copy, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
        Test_acc = test_loop(eval_loader, model_copy, loss_fn, mode='Test')
        if Test_acc > best_acc:
            best_acc = Test_acc
            print('saving new weights...\n')
    print("Done!")
    # Quantized Model
    enable_quantization(model)
    model_prepared = model.train()
    total_loss = 0
    best_acc = 0
    optimizer1 = AdamW(model_prepared.parameters(), lr=1e-5)
    lr_scheduler1 = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=epoch_num * len(train_loader),
    )  
    for t in range(epoch_num):
        print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
        total_loss = train_loop(train_loader, model_prepared, loss_fn, optimizer1, lr_scheduler1, t+1, total_loss)
        Test_acc = test_loop(eval_loader,model_prepared, loss_fn, mode='Test')
        if Test_acc > best_acc:
            best_acc = Test_acc
            print('saving new weights...\n')
    print("Done!") 

    # Model Deploy
    # batch_X1, batch_y1 = next(iter(train_loader))
    # export_inputs = {
    #     'pixel_values': batch_X1.cuda()
    # }
    # convert_deploy(model.eval(), net_type='Transformer', 
    #                dummy_input=(export_inputs,),
    #                model_name='qat_swin_transformer')

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
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='configuration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', default='config-qat.yaml', type=str)
    args = parser.parse_args()
    main(args.config)