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
from sophgo_mq.prepare_by_platform import prepare_by_platform, BackendType
from sophgo_mq.convert_deploy import convert_deploy, convert_onnx
from sophgo_mq.utils.state import enable_quantization, enable_calibration_woquantization, enable_calibration
import re
from sophgo_mq.fake_quantize import global_var
import copy

backends = {
    'academic': BackendType.Academic_NLP,
    'tensorrt': BackendType.Tensorrt_NLP,
}

device = torch.device('cuda')

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
    if not hasattr(config_quant, 'backend'):
        config_quant.backend = 'academic'
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
            }
    }
    backend = backends[config_quant.backend] 
    model = prepare_by_platform(model, backend, prepare_custom_config_dict, custom_tracer=HFTracer())
    return model

inp_out_hooks = []

def insert_model_info(model, valid_layers=(torch.nn.Conv2d, torch.nn.Linear, transformers.Conv1D)):
    print('>'*6, 'Start Insert Info')
    for name, module in model.named_modules():
        try:
            items = module._modules.items()
            assert(len(items))
        except:
            # print(name)
            if('.weight_fake_quant' in name):
                layer_name = name.split('.weight_fake_quant')[0]
                layer_names = layer_name.split('.')
                upper_layer = model
                for l in layer_names:
                    upper_layer = getattr(upper_layer, l)
                
                if isinstance(upper_layer, valid_layers):
                    print('>'*8, 'Insert', layer_name, type(upper_layer))
                    def get_inp_out(layer_name):
                        def tmp(_, inp, out):
                            # print(inp[0].data, out)
                            global_var.set_value(layer_name+'.weight_fake_quant.inp', inp[0].data)
                            global_var.set_value(layer_name+'.weight_fake_quant.out', out.data)
                        return tmp
                    inp_out_hooks.append(upper_layer.register_forward_hook(get_inp_out(layer_name)))
                    layer_module = copy.deepcopy(upper_layer)
                    layer_module.weight_fake_quant = torch.nn.Sequential()
                    layer_module.requires_grad = False
                    setattr(upper_layer.weight_fake_quant, 'layer_module', layer_module)
                    setattr(upper_layer.weight_fake_quant, 'layer_type', type(layer_module))
                    setattr(upper_layer.weight_fake_quant, 'layer_name', layer_name+'.weight_fake_quant')
                    setattr(upper_layer.weight_fake_quant, 'is_gptq_valid', True)
                    setattr(upper_layer.weight_fake_quant, 'is_gptq_done', False)
            else:
                setattr(upper_layer.weight_fake_quant, 'is_gptq_valid', False)
    print('>'*6, 'End Insert Info')

def remove_model_info(model, valid_layers=(torch.nn.Conv2d, torch.nn.Linear, transformers.Conv1D)):
    print('>'*6, 'Start Remove Info')
    print('>'*4, 'Remove hooks')
    for hook in inp_out_hooks:
        hook.remove()
    print('>'*4, 'Set GPTQ Done')
    layer_modules = []
    for name, module in model.named_modules():
        try:
            items = module._modules.items()
            assert(len(items))
        except:
            # print(name)
            if('.weight_fake_quant' in name):
                layer_name = name.split('.weight_fake_quant')[0]
                layer_names = layer_name.split('.')
                upper_layer = model
                for l in layer_names:
                    upper_layer = getattr(upper_layer, l)
                
                if isinstance(upper_layer, valid_layers):
                    if (getattr(upper_layer.weight_fake_quant, 'is_gptq_valid') == True):
                        setattr(upper_layer.weight_fake_quant, 'is_gptq_done', True)
                    if hasattr(upper_layer.weight_fake_quant, 'layer_module'):
                        layer_modules.append(upper_layer.weight_fake_quant)
                
                #     print('>'*8, 'Remove', layer_name, type(upper_layer))
                #     inp_out_hooks[name].remove()
            # else:
            #     setattr(upper_layer.weight_fake_quant, 'is_gptq_valid', False)
    for lm in layer_modules:
        if hasattr(lm, 'layer_module'):
            del lm.layer_module
    print('>'*6, 'End Remove Info')

def main(config_path):
    global_var._init()
    config = glue_utils.parse_config(config_path)
    glue_utils.set_seed(config.train.seed)
    training_args = glue_utils.make_huggingface_training_args(config.train, config.progress)
    set_logger(config.progress)
    raw_datasets, num_labels, label_list = glue_utils.load_dataset_labels(config.data)
    tokenizer, model = glue_utils.load_model(config.model, config.data, num_labels)
    i = 0
    for sts in model.state_dict():
        print(sts, model.state_dict()[sts])
        i = i + 1
        if (i>10):
            break
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
        print("sophgo_mq Model:")
        print(model)

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
    dicts = model.state_dict()
    print(len(dicts))
    for layer in dicts:
        print(layer, '\t', dicts[layer].shape)
    if hasattr(config, 'quant'):
        # calibrate the model
        # calibrate_datasets = raw_datasets['train'].shuffle().select(range(config.quant.calibrate))
        # enable_calibration_woquantization(trainer.model, quantizer_type='act_fake_quant')
        # eval_dataloader = trainer.get_eval_dataloader(calibrate_datasets)
        # for step, inputs in enumerate(eval_dataloader):
        #     print(step, inputs)
        #     break
        # inp = (inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'])
        # # inp.to(torch.device('cpu'))
        # trainer.model = trainer.model.to(torch.device('cpu'))
        # torch.onnx.export(trainer.model, inp, "bert_woquantization.onnx", do_constant_folding=False)
        # trainer.model = trainer.model.to(torch.device('cuda'))
        # evaluate(trainer, calibrate_datasets)
        # enable_calibration_woquantization(trainer.model, quantizer_type='weight_fake_quant')
        # for layer in model.named_modules():
        #     print("============ layer 1 ===============")
        #     print(layer[1])
        #     break
        # evaluate(trainer, calibrate_datasets.select(range(2)))

        # calibrate_datasets = raw_datasets['train'].shuffle().select(range(config.quant.calibrate))
        # enable_calibration_woquantization(trainer.model, quantizer_type='act_fake_quant')
        # evaluate(trainer, calibrate_datasets)
        # enable_calibration_woquantization(trainer.model, quantizer_type='weight_fake_quant')
        # evaluate(trainer, calibrate_datasets.select(range(2)))

        calibrate_datasets = raw_datasets['train'].shuffle().select(range(config.quant.calibrate))
        insert_model_info(trainer.model)
        # export model
        torch.save({'net': trainer.model.state_dict()}, 'torch_model_before.pth')
        eval_dataloader = trainer.get_eval_dataloader(calibrate_datasets)
        step, inputs = next(enumerate(eval_dataloader), 'end')
        # inputs = inputs.to(device)
        input_ids = inputs['input_ids'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        # inp = (inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'])
        enable_calibration(trainer.model)
        evaluate(trainer, calibrate_datasets)
        # trainer.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        enable_quantization(trainer.model)
        evaluate(trainer, calibrate_datasets.select(range(1)))
        # trainer.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        print("GPTQ End.")
        # evaluate(trainer, calibrate_datasets.select(range(2)))
        remove_model_info(trainer.model)
        torch.save({'net': trainer.model.state_dict()}, 'torch_model_after.pth')

    if training_args.do_eval:
        if hasattr(config, 'quant'):
            enable_quantization(trainer.model)
        evaluate(trainer, eval_datasets)
    
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature='default')
    onnx_config = model_onnx_config(model.config)
    export_inputs = {}
    # device = torch.device('cpu')
    export_inputs['input_ids'] = torch.tensor(eval_datasets[0]['input_ids']).unsqueeze(0).to(torch.device('cpu'))
    export_inputs['token_type_ids'] = torch.tensor(eval_datasets[0]['token_type_ids']).unsqueeze(0).to(torch.device('cpu'))
    export_inputs['attention_mask'] = torch.tensor(eval_datasets[0]['attention_mask']).unsqueeze(0).to(torch.device('cpu'))

    model = model.to(torch.device('cpu'))
    for tensor in model.state_dict(): 
        if (model.state_dict()[tensor].device.type != 'cpu'): 
            print('*'*10, 'not same device', tensor)

    # kwargs = {
    #     'input_shape_dict': {'input_ids': [1, 128], 'token_type_ids': [1, 128], 'attention_mask': [1, 128]},
    #     'output_path': './',
    #     'model_name':  'sophgo_mq_model_gptq',
    #     'dummy_input': export_inputs, 
    #     'onnx_model_path':  './sophgo_mq_model_gptq.onnx',
    # }
    # convert_onnx(model, **kwargs)

    convert_deploy(model,
                backends[config.quant.backend],
                dummy_input=(export_inputs,),
                model_name='sophgo_mq_model_gptq',
                input_names=list(onnx_config.inputs.keys()),
                output_names=list(onnx_config.outputs.keys()),
                dynamic_axes={name: axes for name, axes in chain(onnx_config.inputs.items(), onnx_config.outputs.items())}
    )
    print("Done.")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='configuration',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default='/home/zhang/Projects/quantization/sophgo_mq/application/nlp_example/config-gptq.yaml', type=str)
    args = parser.parse_args()
    main(args.config)
