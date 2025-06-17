import inspect
import unittest
from itertools import chain

from sophgo_mq.prepare_by_platform import prepare_by_platform, BackendType
from sophgo_mq.convert_deploy import convert_deploy


class TestQuantizeNLPModel(unittest.TestCase):

    def test_bert_base(self):
        try:
            from transformers import BertTokenizer, BertModel
            from transformers.utils.fx import HFTracer
            from transformers.onnx.features import FeaturesManager
        except ModuleNotFoundError:
            return

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained("bert-base-uncased")
        text = "We are testing this project now."
        encoded_input = tokenizer([text, text], return_tensors='pt')
        output = model(**encoded_input)

        sig = inspect.signature(model.forward)
        input_names = encoded_input.keys()
        concrete_args = {
            p.name: p.default
            for p in sig.parameters.values() if p.name not in input_names
        }

        prepare_custom_config_dict = {
            'concrete_args': concrete_args,
            'preserve_attr': {
                '': ['config']
            },
            'extra_qconfig_dict': {
                'w_observer': 'MinMaxObserver',
                'a_observer': 'EMAMinMaxObserver',
                'w_fakequantize': 'FixedFakeQuantize',
                'a_fakequantize': 'LearnableFakeQuantize',
                'w_qscheme': {
                    'bit': 4,
                    'symmetry': True,
                    'per_channel': False,
                    'pot_scale': False
                },
                'a_qscheme': {
                    'bit': 4,
                    'symmetry': True,
                    'per_channel': False,
                    'pot_scale': False
                }
            }
        }

        quantized_model = prepare_by_platform(model,
                                              BackendType.Academic_NLP,
                                              prepare_custom_config_dict,
                                              custom_tracer=HFTracer())
        output = quantized_model(**encoded_input)

        model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
            model, feature='default')
        onnx_config = model_onnx_config(model.config)
        convert_deploy(
            quantized_model,
            BackendType.Academic_NLP,
            dummy_input=(dict(encoded_input), ),
            model_name='bert-base-uncased-sophgo_mq',
            input_names=list(encoded_input.keys()),
            output_names=list(onnx_config.outputs.keys()),
            dynamic_axes={
                name: axes
                for name, axes in chain(onnx_config.inputs.items(), onnx_config.outputs.items())
            })
