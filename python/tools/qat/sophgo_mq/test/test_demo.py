import torch
import torchvision.models as models
import inspect
from transformers.utils.fx import HFTracer
from transformers.onnx.features import FeaturesManager
from itertools import chain

from sophgo_mq.prepare_by_platform import prepare_by_platform
from sophgo_mq.convert_deploy import convert_deploy
from sophgo_mq.utils.state import enable_quantization, enable_calibration


def resnet18_test():
    ## 1.  define model
    model = models.__dict__['resnet18']()

    ## 2. prepare model
    extra_prepare_dict = {
        'quant_dict': {
            'chip': 'BM1690',
            'quantmode': 'weight_activation',
            'strategy': 'CNN',
        },
    }
    model = prepare_by_platform(model, prepare_custom_config_dict=extra_prepare_dict)

    ## 3. PTQ foward
    model.eval()
    enable_calibration(model)
    rand_input = torch.rand(10, 3, 224, 224)
    output = model(rand_input)
    enable_quantization(model)

    ## 4. deploy
    convert_deploy(model,
                   net_type='CNN',
                   input_shape_dict={'input': [1, 3, 224, 224]},
                   output_path='./',
                   model_name='resnet18')


def bert_test():

    ### 1. define model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    ## 2. prepare model
    sig = inspect.signature(model.forward)
    input_names = ['input_ids', 'attention_mask', 'token_type_ids']
    concrete_args = {
        p.name: p.default
        for p in sig.parameters.values() if p.name not in input_names
    }
    prepare_custom_config_dict = {
        'concrete_args': concrete_args,
        'preserve_attr': {
            '': ['config', 'num_labels']
        },
        'extra_qconfig_dict': {
            'w_observer': 'MinMaxObserver',
            'a_observer': 'EMAQuantileObserver',
            'w_fakequantize': 'FixedFakeQuantize',
            'a_fakequantize': 'FixedFakeQuantize',
            'w_qscheme': {
                'bit': 8,
                'symmetry': True,
                'per_channel': False,
                'pot_scale': False
            },
            'a_qscheme': {
                'bit': 8,
                'symmetry': True,
                'per_channel': False,
                'pot_scale': False
            }
        },
        'extra_quantizer_dict': {
            'exclude_module_name': [
                'bert.embeddings.word_embeddings', 'bert.embeddings.token_type_embeddings',
                'bert.embeddings.position_embeddings'
            ]
        },
        'quant_dict': {
            'chip': 'BM1690',
            'quantmode': 'weight_only',
            'strategy': 'Transformer',
        }
    }
    model = prepare_by_platform(model,
                                prepare_custom_config_dict=prepare_custom_config_dict,
                                custom_tracer=HFTracer())

    ## 3. PTQ forward
    sentence = 'This is a sentence'
    inputs = tokenizer(sentence, return_tensors='pt')
    outputs = model(**inputs)
    print(outputs)

    ### 4. deploy
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
        model, feature='default')
    onnx_config = model_onnx_config(model.config)
    export_inputs = {}
    export_inputs['input_ids'] = inputs['input_ids']
    export_inputs['token_type_ids'] = inputs['token_type_ids']
    export_inputs['attention_mask'] = inputs['attention_mask']

    net_type = 'Transformer'
    convert_deploy(
        model,
        net_type,
        dummy_input=(export_inputs, ),
        output_path='./',
        model_name='bert-base-uncased',
        input_names=list(onnx_config.inputs.keys()),
        output_names=list(onnx_config.outputs.keys()),
        dynamic_axes={
            name: axes
            for name, axes in chain(onnx_config.inputs.items(), onnx_config.outputs.items())
        })


if __name__ == '__main__':

    resnet18_test()
    bert_test()
