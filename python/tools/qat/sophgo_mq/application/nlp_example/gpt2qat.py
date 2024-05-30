import torch
import torch.nn as nn
import numpy as np
import random
import inspect
import argparse
import unittest
import transformers
from cleantext import clean
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import random
import datetime
import time
import copy
import ipdb 
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
from transformers import AutoTokenizer
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
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertModel
from transformers.utils.fx import HFTracer

parser = argparse.ArgumentParser(description='sophgo_mq gpt2 Training')

parser.add_argument('--epochs', default=3, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16), this is the total ')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='wd')
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
#train
def train(model,epochs,optimizer,scheduler,train_dataloader,validation_dataloader):
    
    total_t0 = time.time()
    training_stats = []
    model = model.to(device)
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)
            model.zero_grad()        
            outputs = model(b_input_ids,
                            attention_mask = b_masks,
                            labels=b_labels
                            )
            loss = outputs[0]  
            batch_loss = loss.item()
            total_train_loss += batch_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader) 
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
        print("")
        print("Running Validation...")

        t0 = time.time()
        model.eval()
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader: 
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)       
            with torch.no_grad():
                outputs  = model(b_input_ids, 
    #                            token_type_ids=None, 
                                 attention_mask = b_masks,
                                labels=b_labels)
                loss = outputs[0]              
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
#quant train
def train1(model,epochs,optimizer,scheduler,train_dataloader,validation_dataloader):
    
    total_t0 = time.time()
    training_stats = []
    model = model.to(device)
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        total_train_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)
            model.zero_grad()        
            outputs = model(b_input_ids,
                            attention_mask = b_masks,
                            labels=b_labels
                            )
            loss = outputs[0]  
            batch_loss = loss.item()
            total_train_loss += batch_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader) 
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
        print("")
        print("Running Validation...")

        t0 = time.time()
        model.eval()
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader: 
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)       
            with torch.no_grad():
                outputs  = model(b_input_ids, 
    #                            token_type_ids=None, 
                                 attention_mask = b_masks,
                                labels=b_labels)
                loss = outputs[0]              
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
def cal_ppl_bygpt2(model,test_dataloader):
    total_ppl=0
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            b_input_ids = batch[0].to(device)
            b_masks = batch[1].to(device)
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
        avg_ppl=total_ppl/len(test_dataloader)
        return avg_ppl
def cal_ppl_bygpt22(model,test_dataloader):
    total_ppl=0
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            b_input_ids = batch[0].to(device)
            b_masks = batch[1].to(device)
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
        avg_ppl=total_ppl/len(test_dataloader)
        return avg_ppl
def calibrate(cali_loader, model):
    model.eval()
    print("Start calibration ...")
    print("Calibrate data number = ", len(cali_loader))
    with torch.no_grad():
        for step, batch in enumerate(cali_loader):
            b_input_ids = batch[0].to(device)
            b_masks = batch[1].to(device)
            b_labels=b_input_ids
            outputs = model(b_input_ids,
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

######################################################################################################################

args = parser.parse_args()
#load parameters
batch_size =args.b
epochs = args.epochs
learning_rate = args.lr
warmup_steps = 1e2
epsilon = 1e-8
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
#save data
train_data.to_csv('training_data.csv',index=False)
test_data.to_csv('testing_data.csv',index=False)
validation1.to_csv('validation_data.csv',index=False)
#load GPT tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2',bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium
#Building GPT dataset
traindataset = GPT2Dataset(train_data['train_data'][0:8500], tokenizer)
validdataset = GPT2Dataset(validation1, tokenizer)
cali_loader=GPT2Dataset(cali_data, tokenizer)
testdataset=GPT2Dataset(test_data['train_data'], tokenizer)
#Generate Text Collection
test_set = pd.DataFrame()
test_set['train_data']=test_data['train_data'][:500]
test_set['True_end_train_data1'] = test_set['train_data'].str.split().str[-20:].apply(' '.join)
test_set['train_data1'] = test_set['train_data'].str.split().str[:-20].apply(' '.join)
#dataloader
train_dataloader = DataLoader(
            traindataset,  # The training samples.
            sampler = RandomSampler(traindataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            validdataset, # The validation samples.
            sampler = SequentialSampler(validdataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )
cali_loader = DataLoader(
            cali_loader, 
            sampler = SequentialSampler(cali_loader),
            batch_size = 2 
        )
test_dataloader = DataLoader(
            testdataset, 
            sampler = RandomSampler(testdataset), 
            batch_size =batch_size 
        )
#load model
configuration = GPT2Config.from_pretrained('gpt2',resid_pdrop = 0.3 , output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
model.resize_token_embeddings(len(tokenizer))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model=model.to(device)
seed_val = 32
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
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

#Post processing
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
model_prepared1=Quantizegpt2(config=configuration)

#train parameters
optimizer = AdamW(model_prepared1.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_steps, 
                                            num_training_steps = total_steps)
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

#Original model training
torch.cuda.empty_cache()
model_prepared11=copy.deepcopy(model_prepared1)
optimizer1 = AdamW(model_prepared11.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )
scheduler1 = get_linear_schedule_with_warmup(optimizer1, 
                                            num_warmup_steps = warmup_steps, 
                                            num_training_steps = total_steps)
disable_all(model_prepared11)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#model_prepared1=DataParallel(model_prepared1)
model_prepared11=model_prepared11.train()
model_prepared2,training_stats1=train(model_prepared11,epochs,optimizer1,scheduler1,train_dataloader,validation_dataloader)

# Display floats with two decimal places.
#pd.set_option('precision', 2)
# Create a DataFrame from our training statistics.
df_stats1 = pd.DataFrame(data=training_stats1)
# Use the 'epoch' as the row index.
df_stats1 = df_stats1.set_index('epoch')
# Display the table.
print(df_stats1)

#Original model PPL
avg_ppl1=cal_ppl_bygpt2(model_prepared2,test_dataloader)
print("原始模型PPL:{}".format(avg_ppl1))

#calibration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
enable_calibration(model_prepared1)
model_prepared1=model_prepared1.to(device)
calibrate(cali_loader, model_prepared1)

#quantize model train
enable_quantization(model_prepared1)
model_prepared1=model_prepared1.train()
model_prepared3,training_stats2=train1(model_prepared1,epochs,optimizer,scheduler,train_dataloader,validation_dataloader)

# Create a DataFrame from our training statistics.
df_stats2 = pd.DataFrame(data=training_stats2)
# Use the 'epoch' as the row index.
df_stats2 = df_stats2.set_index('epoch')
# Display the table.
print(df_stats2)

#quantize model PPL
avg_ppl2=cal_ppl_bygpt22(model_prepared3,test_dataloader)
print("量化模型PPL:{}".format(avg_ppl2))







