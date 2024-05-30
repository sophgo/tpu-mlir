import numpy as np
import os
import sys
import torch
import random
import q_model
import logging
import argparse
import datasets
import transformers
from PIL import Image
from tqdm.auto import tqdm
from itertools import chain
from easydict import EasyDict
from datasets import load_metric
import image_classification_utils
from transformers import Trainer
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
                'permute_3_post_act_fake_quantizer', 'transpose_1_post_act_fake_quantizer',
                'permute_11_post_act_fake_quantizer','transpose_2_post_act_fake_quantizer',
                'permute_18_post_act_fake_quantizer','transpose_3_post_act_fake_quantizer',
                'permute_26_post_act_fake_quantizer','transpose_4_post_act_fake_quantizer',
                'permute_33_post_act_fake_quantizer','transpose_5_post_act_fake_quantizer',
                'permute_41_post_act_fake_quantizer','transpose_6_post_act_fake_quantizer',
                'permute_48_post_act_fake_quantizer','transpose_7_post_act_fake_quantizer',
                'permute_56_post_act_fake_quantizer','transpose_8_post_act_fake_quantizer',
                'permute_63_post_act_fake_quantizer','transpose_9_post_act_fake_quantizer',
                'permute_71_post_act_fake_quantizer','transpose_10_post_act_fake_quantizer',
                'permute_78_post_act_fake_quantizer','transpose_11_post_act_fake_quantizer',
                'permute_85_post_act_fake_quantizer','transpose_12_post_act_fake_quantizer',
            ],
        },
        'concrete_args': get_concrete_args(model, input_names),
        'preserve_attr': {'': ['config', 'num_labels']},
    }
    model = prepare_by_platform(
        model=model,
        prepare_custom_config_dict=prepare_custom_config_dict,
        custom_tracer=tracer
    )
    model.eval()
    return model


def calibration(trainer, config_quant):
    trainer.model.cuda()
    trainer.model.eval()

    calibrate_datasets = trainer.train_dataset.shuffle().select(range(config_quant.calibrate))
    calibrate_dataloader = trainer.get_eval_dataloader(calibrate_datasets)
    enable_calibration(trainer.model)
    logger.info("***** Running Calibration *****")
    with torch.no_grad():
        for inputs in tqdm(calibrate_dataloader):
            # import ipdb;ipdb.set_trace()
            for k, v in inputs.items():
                inputs[k] = v.cuda()
            trainer.model(**inputs)
    # enable_calibration_woquantization(trainer.model, quantizer_type='act_fake_quant')
    # logger.info("***** Running Calibration Act *****")
    # with torch.no_grad():
    #     for inputs in tqdm(calibrate_dataloader):
    #         for k, v in inputs.items():
    #             inputs[k] = v.cuda()
    #         trainer.model(**inputs)
    # calibrate_datasets = calibrate_datasets.select(range(2))
    # calibrate_dataloader = trainer.get_eval_dataloader(calibrate_datasets)
    # enable_calibration_woquantization(trainer.model, quantizer_type='weight_fake_quant')
    # logger.info("***** Running Calibration Weight *****")
    # with torch.no_grad():
    #     for inputs in tqdm(calibrate_dataloader):
    #         for k, v in inputs.items():
    #             inputs[k] = v.cuda()
    #         trainer.model(**inputs)


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

    # Load the accuracy metric from the datasets package
    metric = load_metric("./accuracy.py")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    model, feature_extractor = image_classification_utils.load_model(config.model, len(labels), label2id, id2label)

    if hasattr(config, 'quant'):
        model = quantize_model(model, config.quant)
    # Define torchvision transforms to be applied to each image.
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    _train_transforms = Compose(
        [
            RandomResizedCrop((224,224)),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    _val_transforms = Compose(
        [
            Resize((224,224)),
            CenterCrop((224,224)),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]
        ]
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [_val_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]]
        return example_batch

    if training_args.do_train or hasattr(config, 'quant'):
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        if config.data.max_train_samples is not None:
            raw_datasets["train"] = (
                raw_datasets["train"].shuffle(seed=training_args.seed).select(range(config.data.max_train_samples))
            )
        # Set the training transforms
        raw_datasets["train"].set_transform(train_transforms)

    if training_args.do_eval or hasattr(config, 'quant'):
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        if config.data.max_eval_samples is not None:
            raw_datasets["validation"] = (
                raw_datasets["validation"].shuffle(seed=training_args.seed).select(range(config.data.max_eval_samples))
            )
        # Set the validation transforms
        raw_datasets["validation"].set_transform(val_transforms)

    # Initalize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"] if training_args.do_train else None,
        eval_dataset=raw_datasets["validation"] if training_args.do_eval or hasattr(config, 'quant') else None,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
        data_collator=collate_fn,
    )
    model_with_label = q_model.QuantViT(model)
    trainer.model = model_with_label
    # Calibration
    if hasattr(config, 'quant'):
        calibration(trainer, config.quant)
    
    if hasattr(config, 'quant'):
        disable_all(trainer.model.cuda())
    metrics_ori = trainer.evaluate()
    trainer.log_metrics("eval", metrics_ori)
    trainer.save_metrics("eval", metrics_ori)
    # Training
    # if training_args.do_train:
    #     checkpoint = None
    #     if training_args.resume_from_checkpoint is not None:
    #         checkpoint = training_args.resume_from_checkpoint
    #     # elif last_checkpoint is not None:
    #     #     checkpoint = last_checkpoint
    #     train_result = trainer.train(resume_from_checkpoint=checkpoint)
    #     # trainer.save_model()
    #     trainer.log_metrics("train", train_result.metrics)
    #     # trainer.save_metrics("train", train_result.metrics)
    #     # trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        if hasattr(config, 'quant'):
            enable_quantization(trainer.model)
        metrics_quant = trainer.evaluate()
        trainer.log_metrics("eval", metrics_quant)
        trainer.save_metrics("eval", metrics_quant)

    # model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature='default')
    # onnx_config = model_onnx_config(model.config)
    # export_inputs = {
    #     'pixel_values': torch.tensor(raw_datasets["validation"][0]['pixel_values']).unsqueeze(0).cuda()
    # }
    # convert_deploy(model.eval(),
    #                net_type="Transformer",
    #                dummy_input=(export_inputs,),
    #                model_name="ptq_swin_transformer")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='configuration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', default='config.yaml', type=str)
    args = parser.parse_args()
    main(args.config)
