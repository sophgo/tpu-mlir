#导入所需的库
import argparse
import transformers
import torch
import torch.nn as nn
import inspect
import unittest
import copy
from itertools import chain
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from transformers import AdamW, get_scheduler
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import default_data_collator
from transformers.onnx.features import FeaturesManager
from datasets import load_dataset,load_metric
import torch.optim as optim
from sophgo_mq.convert_deploy import convert_deploy, convert_onnx
from sophgo_mq.prepare_by_platform import prepare_by_platform, BackendType
from sophgo_mq.utils.state import enable_calibration, enable_quantization, disable_all
from transformers import logging
import matplotlib.pyplot as plt
import torch.onnx 
import pandas as pd
import json
import logging
import os
import collections
import six
from transformers import DistilBertConfig
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import BertTokenizer, BertModel
from transformers.utils.fx import HFTracer
from transformers import Trainer, TrainingArguments, PreTrainedModel

parser = argparse.ArgumentParser(description='sophgo_mq bertbase Training')

parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16), this is the total ')
parser.add_argument('--lr', '--learning-rate', default=2e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='wd')
parser.add_argument('--wbit', default=8, type=int,
                    metavar='wbit', help='weight bit')
parser.add_argument('--abit', default=8, type=int,
                    metavar='abit', help='active bit')
parser.add_argument('--wob', default='LSQObserver', type=str,
                    metavar='wob', help='weight observer')
parser.add_argument('--aob', default='EMAQuantileObserver', type=str,
                    metavar='aob', help='active observer')
parser.add_argument('--wfq', default='FixedFakeQuantize', type=str,
                    metavar='wfq', help='weight fakequantize')
parser.add_argument('--afq', default='E4M3FakeQuantize', type=str,
                    metavar='afq', help='active fakequantize')                                         
parser.add_argument('--backend', type=str, choices=['Academic_NLP', 'Tensorrt_NLP'], default='Academic_NLP')


#前处理数据
def prepare_train_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

def prepare_validation_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples
def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size = 20, max_answer_length = 30):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []
        
        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}
        
        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        if not squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer

    return predictions
def calibrate(cali_loader, model):
    model.eval()
    print("Start calibration ...")
    print("Calibrate data number = ", len(cali_loader))
    with torch.no_grad():
        for i in range(len(cali_loader)):
            X= next(iter(cali_loader))
            batch_input =X['input_ids'].to(device)  
            batch_seg = X['attention_mask'].to(device)
            start_logits, end_logits = model(input_ids=batch_input,
                                                attention_mask=batch_seg)
            print("Calibration ==> ", i+1)
    print("End calibration.")
    return

def prec(datasets,trainer):
    validation_features1 = datasets["validation"].map(
    prepare_validation_features,
    batched=True,
    remove_columns=datasets["validation"].column_names
    )
    raw_predictions1 = trainer.predict(validation_features1)

    validation_features1.set_format(type=validation_features1.format["type"], columns=list(validation_features1.features.keys()))
    examples1 = datasets["validation"]
    features1 = validation_features1
    example_id_to_index1 = {k: i for i, k in enumerate(examples1["id"])}
    features_per_example1 = collections.defaultdict(list)
    for i, feature in enumerate(features1):
        features_per_example1[example_id_to_index1[feature["example_id"]]].append(i)
    
    final_predictions1 = postprocess_qa_predictions(datasets["validation"], validation_features1, raw_predictions1.predictions)
    metric = load_metric("squad_v2" if squad_v2 else "squad")
    if squad_v2:
        formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in final_predictions1.items()]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions1.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
    result=metric.compute(predictions=formatted_predictions, references=references)
    print(result)
    return
###################################################################################################################

#输入参数
args = parser.parse_args()
squad_v2 = False
model_checkpoint = "distilbert-base-uncased"
batch_size = args.b

#导入数据
datasets = load_dataset("squad_v2" if squad_v2 else "squad")
#快速分词
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
#预处理参数导入
max_length = 384 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
pad_on_right = tokenizer.padding_side == "right"
n_best_size = 20
#对训练数据进行处理
tokenized_datasets = datasets.map(prepare_train_features, batched=True, remove_columns=datasets["train"].column_names)
#训练参数导入
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
model_name = model_checkpoint.split("/")[-1]
args1 = TrainingArguments(
    f"{model_name}-finetuned-squad",
    evaluation_strategy = "epoch",
    learning_rate=args.lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=args.epochs,
    weight_decay=args.wd
)
data_collator = default_data_collator

###############################################################################################################

#量化模型参数准备
sig = inspect.signature(model.forward)
input_names =['input_ids','token_type_ids','attention_mask']
#input_names =['input_ids','attention_mask']
concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}
extra_qconfig_dict={
            'w_observer': args.wob,#'MinMaxObserver',
            'a_observer': args.aob,#'EMAMinMaxObserver',
            'w_fakequantize':args.wfq,   #'FixedFakeQuantize',
            'a_fakequantize':args.afq,  # 'LearnableFakeQuantize',
            'w_qscheme': {
                'bit':args.wbit,
                'symmetry':True,
                'per_channel':False,
                'pot_scale': False
            },
            'a_qscheme': {
                'bit':args.abit,
                'symmetry': True,
                'per_channel': False,
                'pot_scale': False
            }
        }
preserve_attr={'': ['config']}
prepare_custom_config_dict = {
    'concrete_args': concrete_args,
    'preserve_attr': preserve_attr,
    #'work_mode':'all_int4_qat',
    'extra_qconfig_dict':extra_qconfig_dict}
#插入量化节点
model_prepared= prepare_by_platform(model, BackendType.Academic_NLP,prepare_custom_config_dict=prepare_custom_config_dict, custom_tracer=HFTracer())

#校准
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cali =[]
for i in range(64):
    text=tokenized_datasets["train"][i]
    cali.append(text)
cali_loader = DataLoader(cali, batch_size=16, shuffle=True, collate_fn= default_data_collator)
enable_calibration(model_prepared)
model_prepared=model_prepared.to(device)
calibrate(cali_loader, model_prepared)

#模型后处理
enable_quantization(model_prepared)
model_prepared.train()
class BertForQuestionAnswering(PreTrainedModel):
    """
    用于建模类似SQuAD这样的问答数据集
    """
    def __init__(self,config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = model_prepared

    def forward(self, input_ids,
                attention_mask=None,
                start_positions=None,
                end_positions=None):
        bert_output= self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask)
        start_logits=bert_output['start_logits']
        end_logits=bert_output['end_logits']
        start_logits = start_logits.squeeze(-1) 
        end_logits = end_logits.squeeze(-1) 
        
        if start_positions is not None and end_positions is not None:
            # 由于部分情况下start/end 位置会超过输入的长度
            # （例如输入序列的可能大于512，并且正确的开始或者结束符就在512之后）
            # 那么此时就要进行特殊处理
            ignored_index = start_logits.size(1)  # 取输入序列的长度
            start_positions.clamp_(0, ignored_index)
            # 如果正确起始位置start_positions中，存在输入样本的开始位置大于输入长度，
            # 那么直接取输入序列的长度作为开始位置
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            # 这里指定ignored_index其实就是为了忽略掉超过输入序列长度的（起始结束）位置
            # 在预测时所带来的损失，因为这些位置并不能算是模型预测错误的（只能看做是没有预测），
            # 同时如果不加ignore_index的话，那么可能会影响模型在正常情况下的语义理解能力
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            return (start_loss + end_loss) / 2, start_logits, end_logits
        else:
            return start_logits, end_logits
config1 = DistilBertConfig.from_pretrained('distilbert-base-uncased')
# 创建自定义配置对象
model_prepared2=BertForQuestionAnswering(config1)
# 原始模型训练
model_prepared22=copy.deepcopy(model_prepared2)
disable_all(model_prepared22)
model_prepared22=model_prepared22.train()
trainer1 = Trainer(
    model_prepared22,
    args1,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer1.train()
print("原始模型精度：")
prec(datasets,trainer1)
print("**************************************************")
# 量化模型训练
trained_model = trainer1.model
enable_quantization(trained_model)
trainer2 = Trainer(
    trained_model,
    args1,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
print("量化模型精度：")
prec(datasets,trainer2)
print("**************************************************")

#模型部署
keys_to_copy = ['input_ids', 'attention_mask']
copied_cali=[]
for i in range(len(cali)):
    text= {key: cali[i][key] for key in keys_to_copy}
    copied_cali.append(text)
cali_loader1 = DataLoader(copied_cali, batch_size=1, shuffle=True, collate_fn= default_data_collator)
X=next(iter(cali_loader1))
model_prepared.eval()
model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model_prepared, feature='default')
onnx_config = model_onnx_config(model_prepared.config)
convert_deploy(model_prepared,
            BackendType.Academic_NLP,
            dummy_input=((dict(X)),),
            model_name='bert-base-uncased-sophgo_mq-squad'
            ) 
