import torch
import random
import inspect
import argparse
import unittest
import datetime
import time
import copy 
import ipdb
import transformers
import torch.onnx 
import logging
import os
import collections
import csv
import torch.nn as nn
import pandas as pd
import numpy as np
import deepspeed
import torch.nn.functional as F
from cleantext import clean
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import GPT2PreTrainedModel, GPT2Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from itertools import chain
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
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DataParallel
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertModel
from transformers.utils.fx import HFTracer
import ipdb

#clean data
def cleaning(text,punct):
    cleaned_text = clean(text,
        fix_unicode=False,               # fix various unicode errors
        to_ascii=False,                  # transliterate to closest ASCII representation
        lower=True,                     # lowercase text
        no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
        no_urls=False,                  # replace all URLs with a special token
        no_emails=False,                # replace all email addresses with a special token
        no_phone_numbers=False,         # replace all phone numbers with a special token
        no_numbers=False,               # replace all numbers with a special token
        no_digits=False,                # replace all digits with a special token
        no_currency_symbols=False,      # replace all currency symbols with a special token
        no_punct=punct,                 # remove punctuations
        lang="en"                       # set to 'de' for German special handling
    )
    
    tokens = word_tokenize(cleaned_text)
    filtered_sentence = [w for w in tokens if not w in stopwords.words('english')]
    cleaned_text_0 = ' '.join(filtered_sentence)
    return cleaned_text_0
def train(model,epochs,traindataset,validation_dataloader):
    parameters = model.parameters()
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args, model=model, model_parameters=parameters, training_data=traindataset)
    total_t0 = time.time()
    for epoch_i in range(0, epochs):
        running_loss = 0.0
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        for step, batch in enumerate(trainloader):
            b_input_ids = batch[0].to(model_engine.local_rank)
            b_labels = batch[0].to(model_engine.local_rank)
            b_masks = batch[1].to(model_engine.local_rank)       
            outputs = model_engine(b_input_ids,
                            attention_mask = b_masks,
                            labels=b_labels
                            )
            loss = outputs[0]
            model_engine.backward(loss)
            model_engine.step()
            running_loss+=loss.item()
            if (step+1) % args.log_interval==0:
                print("GPUs:%d:epoch:%d,loss:%.3f,device:%d\n"%(torch.cuda.device_count(),epoch_i,running_loss/(step+1),model_engine.local_rank))
        # Measure how long this epoch took.
        print("GPUs:%d:epoch:%d,loss:%.3f,device:%d\n"%(torch.cuda.device_count(),epoch_i,running_loss/len(trainloader),model_engine.local_rank))
        training_time = format_time(time.time() - t0)
        print("")
        print("Running Validation...")

        t0 = time.time()
        # Evaluate data for one epoch
        running_loss1 = 0.0
        with torch.no_grad():
            for batch in validation_dataloader: 
                b_input_ids = batch[0].to(model_engine.local_rank)
                b_labels = batch[0].to(model_engine.local_rank)
                b_masks = batch[1].to(model_engine.local_rank)       
                outputs  = model_engine(b_input_ids, 
    #                            token_type_ids=None, 
                                attention_mask = b_masks,
                                labels=b_labels)
                loss = outputs[0]
                running_loss1+=loss.item()
            print("GPUs:%d:epoch:%d,loss:%.3f,device:%d\n"%(torch.cuda.device_count(),epoch_i,running_loss1/len(validation_dataloader),model_engine.local_rank))              
        validation_time = format_time(time.time() - t0)    
        # Record all statistics from this epoch.
        
    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    return model_engine
#quant train
def train1(model_engine,epochs,trainloader,validation_dataloader):
    total_t0 = time.time()
    for epoch_i in range(0, epochs):
        running_loss = 0.0
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        for step, batch in enumerate(trainloader):
            b_input_ids = batch[0].to(model_engine.local_rank)
            b_labels = batch[0].to(model_engine.local_rank)
            b_masks = batch[1].to(model_engine.local_rank)       
            outputs = model_engine(b_input_ids,
                            attention_mask = b_masks,
                            labels=b_labels
                            )
            loss = outputs[0]
            model_engine.backward(loss)
            model_engine.step()
            running_loss+=loss.item()
            if (step+1) % args.log_interval==0:
                print("GPUs:%d:epoch:%d,loss:%.3f,device:%d\n"%(torch.cuda.device_count(),epoch_i,running_loss/(step+1),model_engine.local_rank))
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)
        print("")
        print("Running Validation...")

        t0 = time.time()
        # Evaluate data for one epoch
        running_loss1 = 0.0
        with torch.no_grad():
            for batch in validation_dataloader: 
                b_input_ids = batch[0].to(model_engine.local_rank)
                b_labels = batch[0].to(model_engine.local_rank)
                b_masks = batch[1].to(model_engine.local_rank)       
                outputs  = model_engine(b_input_ids, 
    #                            token_type_ids=None, 
                                attention_mask = b_masks,
                                labels=b_labels)
                loss = outputs[0] 
                running_loss1+=loss.item() 
            print("GPUs:%d:epoch:%d,loss:%.3f,device:%d\n"%(torch.cuda.device_count(),epoch_i,running_loss1/len(validation_dataloader),model_engine.local_rank))              
        validation_time = format_time(time.time() - t0)    
        # Record all statistics from this epoch.
        
    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    return model_engine
def cal_ppl_bygpt2(model,test_dataloader):
    total_ppl=0
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            b_input_ids = batch[0].to(model.local_rank)
            b_masks = batch[1].to(model.local_rank)
            b_labels=b_input_ids
            outputs = model(b_input_ids,
                            attention_mask = b_masks,
                            labels=b_labels)
            bs, sl = b_input_ids.size()
            logits = outputs[1]
            # Shift so that tokens < n predict n
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = b_input_ids[:, 1:].contiguous()
            shift_attentions = b_masks[:, 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=0, reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).detach().reshape(bs, -1)
            meanloss = loss.sum(1) / shift_attentions.sum(1)
            ppl = torch.exp(meanloss).cpu()
            ppl=ppl.numpy().tolist()
            total_ppl+=ppl[0]
        avg_ppl=total_ppl/(len(test_dataloader))
        print(len(test_dataloader))
    return avg_ppl
def calibrate(cali_loader,model_engine):
    print("Start calibration ...")
    print("Calibrate data number = ", len(cali_loader))
    with torch.no_grad():
        for step, batch in enumerate(cali_loader):
            b_input_ids = batch[0].to(model_engine.local_rank)
            b_masks = batch[1].to(model_engine.local_rank)
            b_labels=b_input_ids
            outputs = model_engine(b_input_ids,
                            attention_mask = b_masks,
                           labels=b_labels)
            print("Calibration ==> ", step+1)
    print("End calibration.")
    return
class GPT2Dataset(Dataset):

    def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=1024):

        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        for txt in txt_list:
            encodings_dict = tokenizer( '<|startoftext|>'+ txt + '<|endoftext|>', truncation=True, max_length=max_length,padding="max_length") #, padding="max_length"
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx] 
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))     
###########################################################################################
def main(args):
    #load data
    squad = load_dataset("squad", split="train[:10000]")
    #processing data
    que = []
    con = []
    ans = []
    for i in squad:
        que.append(cleaning(i['question'],True))
        con.append(cleaning(i['context'],True))
        ans.append(cleaning(i['answers']['text'],True))
    main_data = pd.DataFrame()
    main_data['Question'] = que
    main_data['Context'] = con
    main_data['Answer'] = ans
    main_data['train_data'] = main_data['Question']+'<cont>'+main_data['Context']+'<ans>'+main_data['Answer']
    main_data = main_data.sample(frac=1,random_state=32).reset_index(drop=True)
    train_data = main_data[0:9000].reset_index(drop=True)
    cali_data=train_data['train_data'][0:100].reset_index(drop=True)
    test_data = main_data[9000:10000].reset_index(drop=True)
    validation = train_data['train_data'][8500:9000]
    validation1 = validation.apply( lambda x: x.split('<ans>')[0]+'<ans>')
    validation1.reset_index(drop=True,inplace=True)
    #load GPT tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2',bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium
    traindataset = GPT2Dataset(train_data['train_data'][0:8500], tokenizer)
    validdataset = GPT2Dataset(validation1, tokenizer)
    cali_loader=GPT2Dataset(cali_data, tokenizer)
    testdataset=GPT2Dataset(test_data['train_data'], tokenizer)
    validation_dataloader = DataLoader(
            validdataset, # The validation samples.
            sampler = SequentialSampler(validdataset), # Pull out batches sequentially.
            batch_size = args.b # Evaluate with this batch size.
        )
    test_dataloader = DataLoader(
            testdataset, # The validation samples.
            sampler = SequentialSampler(testdataset), # Pull out batches sequentially.
            batch_size = 2 # Evaluate with this batch size.
        )
    cali_loader = DataLoader(
            cali_loader, 
            sampler = SequentialSampler(cali_loader),
            batch_size = args.b 
        )
    
    #导入模型
    configuration = GPT2Config.from_pretrained('gpt2',resid_pdrop = 0.3 , output_hidden_states=False)
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
    model.resize_token_embeddings(len(tokenizer))
    model=model.train()
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
    # #Post processing
    class Quantizegpt2(GPT2PreTrainedModel):
        """
        用于建模类似SQuAD这样的问答数据集
        """
        def __init__(self,config):
            super(Quantizegpt2, self).__init__(config)
            self.gpt2 = model_prepared
            
        def forward(self, input_ids,attention_mask,labels=None):
            
            gpt2_output= self.gpt2(input_ids=input_ids,attention_mask=attention_mask)
            lm_logits = gpt2_output['logits']
            loss = None
            if labels is not None:
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct =nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                return loss,lm_logits
    configuration = GPT2Config.from_pretrained('gpt2',resid_pdrop = 0.3 , output_hidden_states=False)
    #model_prepared1=Quantizegpt2(config=configuration)
    model=Quantizegpt2(config=configuration)
    model1=copy.deepcopy(model)
    disable_all(model1)

    #deepspeed 原始模型
    epochs=args.epochs
    model1=train(model1,epochs,traindataset,validation_dataloader)
    avg_ppl1=cal_ppl_bygpt2(model1,test_dataloader)
    print("原始模型PPL:{}".format(avg_ppl1))

    #deepspeed量化模型
    torch.cuda.empty_cache()
    enable_calibration(model)
    parameters = model.parameters()
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args, model=model, model_parameters=parameters, training_data=traindataset)
    calibrate(cali_loader, model_engine)
    enable_quantization(model_engine)
    model=train1(model_engine,epochs,trainloader,validation_dataloader)
    avg_ppl2=cal_ppl_bygpt2(model,test_dataloader)
    print("量化模型PPL:{}".format(avg_ppl2))

def add_argument():
    parser = argparse.ArgumentParser(description='sophgo_mq gpt2 Training')

    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--b', '--batch-size', default=2, type=int,
                        metavar='N',
                        help='mini-batch size , this is the total ')
    parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='wd')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--log_interval',
                        type=int,
                        default=200,
                        help="output logging information at a given interval")
    parser.add_argument('--wbit', default=4, type=int,
                        metavar='wbit', help='weight bit')
    parser.add_argument('--abit', default=8, type=int,
                        metavar='abit', help='active bit')
    parser.add_argument('--wob', default='LSQObserver', type=str,
                        metavar='wob', help='weight observer')
    parser.add_argument('--aob', default='EMAQuantileObserver', type=str,
                        metavar='aob', help='active observer')
    parser.add_argument('--wfq', default='LearnableFakeQuantize', type=str,
                        metavar='wfq', help='weight fakequantize')
    parser.add_argument('--afq', default='LearnableFakeQuantize', type=str,
                        metavar='afq', help='active fakequantize') 
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args 

if __name__ == '__main__':
    
    args = add_argument()
    main(args)


