import torch
import torch.nn as nn
import inspect
import unittest
import argparse
import copy 
from itertools import chain
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, get_scheduler
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import BertTokenizer, BertModel
from transformers.utils.fx import HFTracer
from transformers.onnx.features import FeaturesManager
from datasets import load_dataset
import torch.optim as optim
from sophgo_mq.convert_deploy import convert_deploy, convert_onnx
from sophgo_mq.prepare_by_platform import prepare_by_platform
from sophgo_mq.utils.state import enable_calibration, enable_quantization, disable_all
from transformers import logging
import matplotlib.pyplot as plt
import torch.onnx 

parser = argparse.ArgumentParser(description='sophgo_mq bertbase Training')

parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--b', '--batch-size', default=4, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16), this is the total ')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
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
parser.add_argument('--wfq', default='AdaRoundFakeQuantize', type=str,
                    metavar='wfq', help='weight fakequantize')
parser.add_argument('--afq', default='LearnableFakeQuantize', type=str,
                    metavar='afq', help='active fakequantize')                                         

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_type):
        self.data = self.load_data(data_type)
    
    def load_data(self, data_type):
        tmp_dataset = load_dataset(path='laugustyniak/abusive-clauses-pl', split = data_type)
        Data = {}
        for idx, line in enumerate(tmp_dataset):
            sample = line
            Data[idx] = sample
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
def collote_fn(batch_samples):
    batch_text= []
    batch_label = []
    for sample in batch_samples:
        batch_text.append(sample['text'])
        batch_label.append(int(sample['label']))
    X = tokenizer(
        batch_text, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    y = torch.tensor(batch_label)
    return X, y
def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = (epoch-1)*len(dataloader)
    
    model.train()
    for batch, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_batch_num + batch):>7f}')
        progress_bar.update(1)
        #losses.append(loss.item())
        #print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")
    return total_loss #losses

def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    print(f"{mode} Accuracy: {(100*correct):>0.1f}%\n")
    return correct
def calibrate(cali_loader, model):
    model.eval()
    print("Start calibration ...")
    print("Calibrate data number = ", len(cali_loader.dataset))
    with torch.no_grad():
        for i, (X, y) in enumerate(cali_loader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            print("Calibration ==> ", i+1)
    print("End calibration.")
    return

##################################################################################################################

args = parser.parse_args()
#load data
train_data = Dataset('train')
test_data = Dataset('test')

#load parameters
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
learning_rate = args.lr
epoch_num = args.epochs

#dataloader
train_dataloader = DataLoader(train_data, batch_size=args.b, shuffle=True, collate_fn=collote_fn)
test_dataloader = DataLoader(test_data, batch_size=args.b, shuffle=True, collate_fn=collote_fn)

#quantize
model1=AutoModel.from_pretrained(checkpoint)
#量化模型参数准备
sig = inspect.signature(model1.forward)
input_names =['input_ids','token_type_ids','attention_mask']
concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}
extra_qconfig_dict={
            'w_observer': args.wob,#'MinMaxObserver',
            'a_observer': args.aob,#'EMAMinMaxObserver',
            'w_fakequantize':args.wfq,   #'FixedFakeQuantize',
            'a_fakequantize':args.afq,  # 'LearnableFakeQuantize',
            'w_qscheme': {
                'bit':args.wbit,
                'symmetry':True,
                'per_channel': False,
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
quant_dict={
            'chip': 'BM1688',
            'quantmode': 'weight_only',
            'strategy': 'Transformer',
            }
prepare_custom_config_dict = {
    'concrete_args': concrete_args,
    'preserve_attr': preserve_attr,
    'extra_qconfig_dict':extra_qconfig_dict,
    'quant_dict': quant_dict
    }
#插入量化节点
model_prepared= prepare_by_platform(model1, prepare_custom_config_dict=prepare_custom_config_dict, custom_tracer=HFTracer())
#后处理
class NeuralNetwork2(nn.Module):
    def __init__(self):
        super(NeuralNetwork2, self).__init__()
        self.bert_encoder = model_prepared
        self.classifier = nn.Linear(768, 2)
    
    def forward(self, x):
        bert_output = self.bert_encoder(**x)
        cls_vectors = bert_output['last_hidden_state'][:, 0]
        logits = self.classifier(cls_vectors)
        return logits
model_prepared1 = NeuralNetwork2().to(device)
#校准
cali =[]
for i in range(20):
    text=train_data[i]
    cali.append(text)
cali_loader = DataLoader(cali, batch_size=args.b, shuffle=True, collate_fn=collote_fn)
enable_calibration(model_prepared1)
model_prepared1=model_prepared1.to(device)
calibrate(cali_loader, model_prepared1)
#原始模型精度
model_prepared11=copy.deepcopy(model_prepared1)
disable_all(model_prepared11)
model_prepared11=model_prepared11.train()
optimizer = AdamW(model_prepared11.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)
loss_fn = nn.CrossEntropyLoss()
total_loss = 0
best_acc = 0
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model_prepared11, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    Test_acc = test_loop(test_dataloader,model_prepared11, mode='Test')
    if Test_acc > best_acc:
        best_acc = Test_acc
        print('saving new weights...\n')
print("Done!")

#量化模型精度
enable_quantization(model_prepared1)
model_prepared1=model_prepared1.train()
total_loss = 0
best_acc = 0
optimizer1 = AdamW(model_prepared1.parameters(), lr=learning_rate)
lr_scheduler1 = get_scheduler(
    "linear",
    optimizer=optimizer1,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader),
)
for t in range(epoch_num):
    print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model_prepared1, loss_fn, optimizer1, lr_scheduler1, t+1, total_loss)
    Test_acc = test_loop(test_dataloader,model_prepared1, mode='Test')
    if Test_acc > best_acc:
        best_acc = Test_acc
        print('saving new weights...\n')
print("Done!")

#量化模型部署
train_dataloader1 = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=collote_fn)
batch_X1, batch_y1 = next(iter(train_dataloader1))
model_prepared.eval()
model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model_prepared, feature='default')
onnx_config = model_onnx_config(model_prepared.config)
net_type = 'Transformer'
convert_deploy(model_prepared,
            net_type,
            dummy_input=(dict(batch_X1),),
            output_path="./",
            model_name='bert-base-uncased'
            )