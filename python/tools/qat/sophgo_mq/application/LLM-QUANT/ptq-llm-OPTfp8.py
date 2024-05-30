import torch
import torch.nn as nn
import numpy as np
import random
import inspect
import argparse
import unittest
import transformers
from cleantext import clean
import nltk
import datasets
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import pandas as pd
import numpy as np
import random
import datetime
import time
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import GPT2PreTrainedModel, GPT2Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from itertools import chain
from tqdm.auto import tqdm
from transformers import AdamW, get_scheduler
from transformers import AutoModel
from transformers import AutoTokenizer,AutoConfig,DataCollatorWithPadding
from transformers import default_data_collator
from transformers.onnx.features import FeaturesManager
from datasets import load_dataset,load_metric
import torch.optim as optim
from sophgo_mq.convert_deploy import convert_deploy, convert_onnx
from sophgo_mq.prepare_by_platform import prepare_by_platform, BackendType
from sophgo_mq.utils.state import enable_calibration, enable_quantization, disable_all
from transformers import logging
import torch.onnx 
import logging
import os
import collections
import torch.nn.functional as F
import csv
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DataParallel
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer,AutoModelForCausalLM
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertModel
from transformers.utils.fx import HFTracer
from deepspeed.module_inject import HFBertLayerPolicy
import deepspeed
import copy
import ipdb

parser = argparse.ArgumentParser(description='sophgo_mq LLM')

parser.add_argument('--epochs', default=1, type=int, metavar='N', 
                    help='number of total epochs to run')
parser.add_argument('--b', '--batch-size', default=1, type=int,
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
parser.add_argument('--wob', default='MinMaxObserver', type=str,
                    metavar='wob', help='weight observer')
parser.add_argument('--aob', default='MinMaxObserver', type=str,
                    metavar='aob', help='active observer')
parser.add_argument('--wfq', default='E4M3FakeQuantize', type=str,
                    metavar='wfq', help='weight fakequantize')
parser.add_argument('--afq', default='E4M3FakeQuantize', type=str,
                    metavar='afq', help='active fakequantize')  
#train
def train(model,epochs,optimizer,scheduler,train_dataloader,validation_dataloader,total_steps):
    
    total_t0 = time.time()
    progress_bar = tqdm(range(total_steps))
    training_stats = []
    count=0
    model = model.to(device)
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            b_input_ids = torch.tensor(batch['input_ids']).to(device)
            b_labels = torch.tensor(batch['input_ids']).to(device)
            b_masks = torch.tensor(batch['attention_mask']).to(device)
            shift_attentions = b_masks[:, 1:].contiguous()
            model.zero_grad()        
            outputs = model(input_ids=b_input_ids,
                            attention_mask = b_masks)
            logits = outputs['logits']
            # if shift_attentions.sum(1)==0:
            #     continue
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = b_labels[..., 1:].contiguous()
            loss_fct =nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            batch_loss = loss.item()
            total_train_loss += batch_loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            progress_bar.update(1)
            count+=1
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss /count #len(train_dataloader) 
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
        print("")
        print("Running Validation...")
        ipdb.set_trace()
        t0 = time.time()
        model.eval()
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader: 
            b_input_ids = torch.tensor(batch['input_ids']).to(device)
            b_labels = torch.tensor(batch['input_ids']).to(device)
            b_masks = torch.tensor(batch['attention_mask']).to(device)       
            with torch.no_grad():
                outputs  = model(b_input_ids, 
    #                            token_type_ids=None, 
                                 attention_mask = b_masks)
                logits = outputs['logits']
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = b_labels[..., 1:].contiguous()
                loss_fct =nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))              
            batch_loss = loss.item()
            total_eval_loss += batch_loss        
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = format_time(time.time() - t0)    
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))
        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    return model,training_stats
def cal_ppl(model,test_dataloader):
    total_ppl=0
    count=0
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            b_input_ids = torch.tensor(batch['input_ids']).to(device)
            b_masks = torch.tensor(batch['attention_mask']).to(device)
            b_labels=b_input_ids
            outputs = model(b_input_ids,attention_mask = b_masks)
            # batch = {k: v.to(device) for k, v in batch.items()}
            # outputs = model(**batch)
            bs, sl = b_input_ids.size()
            logits = outputs[1]
            # Shift so that tokens < n predict n
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = b_input_ids[:, 1:].contiguous()
            shift_attentions = b_masks[:, 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=0, reduction="none")          
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).detach().reshape(bs, -1)
            for i in range(loss.shape[0]):
                if shift_attentions.sum(1)[i]==0:
                    total_ppl+=1
                    count+=1
                else:
                    meanloss=loss[i,:shift_attentions.sum(1)[i]].sum() / shift_attentions.sum(1)[i]
                    ppl = torch.exp(meanloss).cpu()
                    ppl=ppl.numpy().tolist()
                    total_ppl+=ppl
                    count+=1
        print(total_ppl)
        avg_ppl=total_ppl/count
        return avg_ppl
def cal_ppl_1(model,test_dataloader):
    total_ppl=0
    count=0
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            b_input_ids = torch.tensor(batch['input_ids']).to(device)
            b_masks = torch.tensor(batch['attention_mask']).to(device)
            b_labels=b_input_ids
            outputs = model(b_input_ids,attention_mask = b_masks)
            # batch = {k: v.to(device) for k, v in batch.items()}
            # outputs = model(**batch)
            bs, sl = b_input_ids.size()
            logits = outputs[1]
            # Shift so that tokens < n predict n
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = b_input_ids[:, 1:].contiguous()
            shift_attentions = b_masks[:, 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=0, reduction="none")
            #ipdb.set_trace()            
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).detach().reshape(bs, -1)
            for i in range(loss.shape[0]):
                if shift_attentions.sum(1)[i]==0:
                    continue
                else:
                    meanloss=loss[i,:shift_attentions.sum(1)[i]].sum() / shift_attentions.sum(1)[i]
                    ppl = torch.exp(meanloss).cpu()
                    ppl=ppl.numpy().tolist()
                    total_ppl+=ppl
                    count+=1
        print(total_ppl)
        avg_ppl=total_ppl/count
        return avg_ppl
def cal_ppl_2(model,test_dataloader):
    total_loss=0
    count=0
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            b_input_ids = torch.tensor(batch['input_ids']).to(device)
            b_masks = torch.tensor(batch['attention_mask']).to(device)
            b_labels=b_input_ids
            outputs = model(b_input_ids,attention_mask = b_masks)
            # batch = {k: v.to(device) for k, v in batch.items()}
            # outputs = model(**batch)
            bs, sl = b_input_ids.size()
            logits = outputs[1]
            # Shift so that tokens < n predict n
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = b_input_ids[:, 1:].contiguous()
            shift_attentions = b_masks[:, 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=0, reduction="none")
            #ipdb.set_trace()            
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).detach().reshape(bs, -1)
            for i in range(loss.shape[0]):
                if shift_attentions.sum(1)[i]==0:
                    continue
                else:
                    batch_loss=loss[i,:shift_attentions.sum(1)[i]].sum() 
                    total_loss+=batch_loss
                    count+=shift_attentions.sum(1)[i]
        print(total_loss)
        avg_loss=total_loss/count
        ppl = torch.exp(avg_loss).cpu()
        return ppl
def calibrate(cali_loader, model):
    model.eval()
    print("Start calibration ...")
    print("Calibrate data number = ", len(cali_loader))
    with torch.no_grad():
        for step, batch in enumerate(cali_loader):
            b_input_ids = torch.tensor(batch['input_ids']).to(device)
            b_masks = torch.tensor(batch['attention_mask']).to(device)
            b_labels=b_input_ids
            outputs = model(b_input_ids,
                            attention_mask = b_masks)
            print("Calibration ==> ", step+1)
    print("End calibration.")
    return
def preprocess_function(examples):
    max_seq_length=256
    padding="longest"
    #result = tokenizer(examples["text"],max_length=max_seq_length,padding=padding,truncation=True,return_tensors="pt")
    result = tokenizer(examples["text"],padding=padding,truncation=True,return_tensors="pt")
    return result
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))
######################################################################################################################

args = parser.parse_args()
checkpoint = "lnair/opt-350m-wikitext2"
#load parameters
batch_size =args.b
epochs = args.epochs
learning_rate = args.lr
warmup_steps = 1e2
epsilon = 1e-8
#load data
wikitext_train = load_dataset("wikitext","wikitext-2-raw-v1",split="train")
wikitext_validation = load_dataset("wikitext","wikitext-2-raw-v1",split="validation")
wikitext_test = load_dataset("wikitext","wikitext-2-raw-v1",split="test")

# wikitext_train = load_dataset("zhengxuanzenwu/wikitext-2-split-128",split="train")
# wikitext_validation = load_dataset("zhengxuanzenwu/wikitext-2-split-128",split="validation")
# wikitext_test = load_dataset("zhengxuanzenwu/wikitext-2-split-128",split="test")

random_indices = random.sample(range(len(wikitext_train)),100)
wikitext_cali= [wikitext_train[i] for i in random_indices]
random_samples_dict = {'text': [sample['text'] for sample in wikitext_cali]}
wikitext_cali = datasets.Dataset.from_dict(random_samples_dict)

#load tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint,pad_token="<|pad|>",padding_side="right")
tokenizer.add_special_tokens({"pad_token": "<|pad|>"}) 
#Building dataset
tokenized_train = wikitext_train.map(preprocess_function, batched=True, remove_columns=wikitext_train.column_names)
tokenized_validation = wikitext_validation.map(preprocess_function, batched=True, remove_columns=wikitext_validation.column_names)
tokenized_test = wikitext_test.map(preprocess_function, batched=True, remove_columns=wikitext_test.column_names)
tokenized_cali = wikitext_cali.map(preprocess_function, batched=True,remove_columns=wikitext_test.column_names)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#dataloader
train_dataloader = DataLoader(
            tokenized_train,  # The training samples.
            shuffle=True, 
            batch_size = batch_size, # Trains with this batch size.
            collate_fn=data_collator
        )

validation_dataloader = DataLoader(
            tokenized_validation, # The validation samples.
            shuffle=False,  
            batch_size = 2, # Evaluate with this batch size.
            collate_fn=data_collator
        )
cali_loader = DataLoader(
            tokenized_cali, 
            shuffle=False, 
            batch_size = 2,
            collate_fn=data_collator
        )
test_dataloader = DataLoader(
            tokenized_test, 
            shuffle=False, 
            batch_size = batch_size,
            collate_fn=data_collator
        )
#load model
configuration = AutoConfig.from_pretrained(checkpoint)
model =AutoModelForCausalLM.from_pretrained(checkpoint, config=configuration)
model.resize_token_embeddings(len(tokenizer))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model=model.to(device)
seed_val = 32
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
#训练参数
# optimizer = optim.AdamW(model.parameters(),
#                   lr = learning_rate,
#                   eps = epsilon
#                 )
# total_steps = len(train_dataloader) * epochs

# scheduler = get_linear_schedule_with_warmup(optimizer, 
#                                             num_warmup_steps = warmup_steps, 
#                                             num_training_steps = total_steps)
# #原始模型训练
# model=model.train()
# model_prepared2,training_stats1=train(model,epochs,optimizer,scheduler,train_dataloader,validation_dataloader,total_steps)

# # Create a DataFrame from our training statistics.
# df_stats1 = pd.DataFrame(data=training_stats1)
# # Use the 'epoch' as the row index.
# df_stats1 = df_stats1.set_index('epoch')
# # Display the table.
# print(df_stats1)

#quantize 
sig = inspect.signature(model.forward)
input_names =['input_ids','attention_mask']
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
#Insert quantization node
model_prepared= prepare_by_platform(model, BackendType.Academic_NLP,prepare_custom_config_dict=prepare_custom_config_dict, custom_tracer=HFTracer())
#
class Quantizemodel(nn.Module):
    """
    用于建模类似SQuAD这样的问答数据集
    """
    def __init__(self,model_prepared):
        super().__init__()
        self.model = model_prepared
        self.config= model_prepared.config
        
    def forward(self, input_ids,attention_mask,labels=None):
        labels=input_ids
        bs, sl = input_ids.size()
        model_output= self.model(input_ids=input_ids,attention_mask=attention_mask)
        lm_logits = model_output['logits']
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct =nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss,lm_logits
model_prepared1=Quantizemodel(model_prepared)


#原始模型PPL
disable_all(model_prepared1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
avg_ppl1=cal_ppl_2(model_prepared1,validation_dataloader)
print("原始模型PPL:{}".format(avg_ppl1))

#校准
model_prepared1.eval()
enable_calibration(model_prepared1)
model_prepared1=model_prepared1.to(device)
calibrate(cali_loader, model_prepared1)

#量化模型PPL
enable_quantization(model_prepared1)
avg_ppl2=cal_ppl_2(model_prepared1,validation_dataloader)
print("量化模型PPL:{}".format(avg_ppl2))
