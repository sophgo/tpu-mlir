import sys
import torch
import inspect
import logging
import datasets
import argparse
import transformers
import q_model
import glue_utils
import numpy as np
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
)
from transformers.utils.fx import HFTracer
from transformers.onnx.features import FeaturesManager
from itertools import chain
from sophgo_mq.prepare_by_platform import prepare_by_platform
from sophgo_mq.convert_deploy import convert_deploy
from sophgo_mq.utils.state import enable_quantization, enable_calibration_woquantization


logger = logging.getLogger("transformer")

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

def evaluate(trainer, eval_datasets, num_samples=-1):
    logger.info("*** Evaluate ***")
    if isinstance(eval_datasets, tuple):
        for i in range(len(eval_datasets)):
            if num_samples != -1:
                metrics = trainer.evaluate(eval_dataset=eval_datasets[i].shuffle().select(range(num_samples)))
            else:
                metrics = trainer.evaluate(eval_dataset=eval_datasets[i])
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
    else:
        if num_samples != -1:
            metrics = trainer.evaluate(eval_dataset=eval_datasets.shuffle().select(range(num_samples)))
        else:
            metrics = trainer.evaluate(eval_dataset=eval_datasets)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def quantize_model(model, config_quant):
    if not hasattr(config_quant, 'chip'):
        config_quant.chip = 'BM1690'
    sig = inspect.signature(model.forward)
    input_names = ['input_ids', 'attention_mask', 'token_type_ids']
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}
    prepare_custom_config_dict = {
        'concrete_args': concrete_args,
        'preserve_attr': {'': ['config', 'num_labels']},
        'extra_qconfig_dict':{
                'w_observer': config_quant.w_qconfig.observer,
                'a_observer': config_quant.a_qconfig.observer,
                'w_fakequantize': config_quant.w_qconfig.quantizer,
                'a_fakequantize': config_quant.a_qconfig.quantizer,
                'w_qscheme': {
                    'bit': config_quant.w_qconfig.bit,
                    'symmetry': config_quant.w_qconfig.symmetric,
                    'per_channel': config_quant.w_qconfig.per_channel,
                    'pot_scale': config_quant.pot_scale
                },
                'a_qscheme': {
                    'bit': config_quant.a_qconfig.bit,
                    'symmetry': config_quant.a_qconfig.symmetric,
                    'per_channel': config_quant.a_qconfig.per_channel,
                    'pot_scale': config_quant.pot_scale
                }
            },
        'extra_quantizer_dict':{
            'exclude_module_name':['bert.embeddings.word_embeddings', 'bert.embeddings.token_type_embeddings', 'bert.embeddings.position_embeddings']
        },
        'quant_dict': {
                       'chip': config_quant.chip,
                       'quantmode': config_quant.quantmode,
                       'strategy': 'Transformer',
                       }
    }
    model = prepare_by_platform(model, prepare_custom_config_dict=prepare_custom_config_dict, custom_tracer=HFTracer())
    return model
#'bert.pooler.dense'

def main(config_path):
    config = glue_utils.parse_config(config_path)
    glue_utils.set_seed(config.train.seed)
    training_args = glue_utils.make_huggingface_training_args(config.train, config.progress)
    set_logger(config.progress)
    raw_datasets, num_labels, label_list = glue_utils.load_dataset_labels(config.data)
    tokenizer, model = glue_utils.load_model(config.model, config.data, num_labels)
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and config.data.task_name is not None
        and config.data.task_name != 'stsb'
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in model.config.label2id.items()}
    elif config.data.task_name is not None and config.data.task_name != 'stsb':
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in model.config.label2id.items()}
    config.data.max_seq_length = min(config.data.max_seq_length, tokenizer.model_max_length)
    raw_datasets = glue_utils.preprocess_dataset(config.data, training_args, raw_datasets, label_to_id, tokenizer)

    if config.data.task_name == 'mnli':
        eval_datasets = (
            raw_datasets['validation_matched'], raw_datasets['validation_mismatched']
        )
    else:
        eval_datasets = raw_datasets['validation']
    metric = datasets.load_metric("glue", config.data.task_name)

    if hasattr(config, 'quant'):
        model = quantize_model(model, config.quant)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if config.data.task_name=='stsb' else np.argmax(preds, axis=1)

        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    
    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if config.data.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_datasets if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    model_with_label = q_model.Quant_Bert(model)
    trainer.model = model_with_label
    if hasattr(config, 'quant'):
        # calibrate the model
        calibrate_datasets = raw_datasets['train'].shuffle().select(range(config.quant.calibrate))
        enable_calibration_woquantization(trainer.model, quantizer_type='act_fake_quant')
        evaluate(trainer, calibrate_datasets)
        enable_calibration_woquantization(trainer.model, quantizer_type='weight_fake_quant')
        evaluate(trainer, calibrate_datasets.select(range(2)))
    if training_args.do_eval:
        if hasattr(config, 'quant'):
            enable_quantization(trainer.model)
        evaluate(trainer, eval_datasets) #此步骤进行了fake quant操作，注释以后即可跳过fake quant的计算操作
    
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature='default')
    onnx_config = model_onnx_config(model.config)
    export_inputs = {}
    export_inputs['input_ids'] = torch.tensor(eval_datasets[0]['input_ids']).unsqueeze(0)
    export_inputs['token_type_ids'] = torch.tensor(eval_datasets[0]['token_type_ids']).unsqueeze(0)
    export_inputs['attention_mask'] = torch.tensor(eval_datasets[0]['attention_mask']).unsqueeze(0)

    net_type = 'Transformer'
    convert_deploy(model,
                net_type,
                dummy_input=(export_inputs,),
                output_path=config.train.output_dir,
                model_name='bert-base-uncased',
                input_names=list(onnx_config.inputs.keys()),
                output_names=list(onnx_config.outputs.keys()),
                dynamic_axes={name: axes for name, axes in chain(onnx_config.inputs.items(), onnx_config.outputs.items())}
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='configuration',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default='config.yaml', type=str)
    args = parser.parse_args()
    main(args.config)