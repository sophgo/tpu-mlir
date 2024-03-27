import os
import argparse
import copy
import logging
import torch
import time
from tools.train.tpu_mlir_jit import device, aot_backend
import tools.train.tpu_mlir_jit as tpu_mlir_jit
from torch import nn

def enable_dynamo_debug_info():
    import torch._dynamo as td
    td.config.log_level = logging.DEBUG
    td.config.verbose = True
    td.config.output_code = True
    os.environ["TORCHDYNAMO_PRINT_GUARDS"] = "1"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip", default="bm1684x", choices=['bm1684x', 'bm1690','sg2260'],
                        help="chip name")
    parser.add_argument("--debug", default="",
                        help="debug")
    parser.add_argument("--cmp", action='store_true',
                        help="enable cmp")
    parser.add_argument("--skip_tpu_mlir", action='store_true',
                        help="skip_tpu_mlir")
    parser.add_argument("--model", default="",help="model name")
    start_time = time.time()
    args = parser.parse_args()
    tpu_mlir_jit.args = args
    # enable_dynamo_debug_info()
    if args.model == "test":
        ## yolo ###
        # input = torch.randn((1,3,224,224))
        # from nets.yolo import YoloBody
        # anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        # num_classes = 80
        # phi = 's'
        # mod = YoloBody(anchors_mask, num_classes, phi)

        ### resnet ###
        # input = torch.randn((1, 3, 224, 224))
        # import torchvision.models as models
        # mod = models.resnet18()

        # input = torch.randn((1, 3, 224, 224))
        # from resnet50 import resnet50
        # mod = resnet50()

        # from tools.train.test_model import test_model
        # mod = test_model() #.train()
        # input = torch.randn((1, 3, 16, 16))
        # # # input = torch.randn((20, 1024))

        d_model=10 #768  #test ok at 0918
        from tools.train.gpt2 import TransformerBlocks #backward can not use this
        mod = TransformerBlocks(d_model=d_model, nlayers=1) #.train()
        input = torch.randn((1,4,d_model))

        # from tools.train.gpt2 import GPT2
        # mod = GPT2()
        # input = torch.randint(0, 50257,(1, 5))

        net_d = copy.deepcopy(mod)
        input_d = input.to(device)
        net_d.to(device)

        optimizer = torch.optim.SGD(mod.parameters(), lr=0.01)
        # net_d, optimizer = amp.initialize(net_d, optimizer, opt_level='O2')
        model_opt = torch.compile(net_d, backend=aot_backend)

        # optimizer.zero_grad()

        # res_oringal = mod(input)
        # print('res_oringal:', res_oringal)

        # with torch.autocast(tpu_dev):
        # loss_d = model_opt(input_d, labels = input_d)
        loss_d = model_opt(input_d)
        # print('res_oringal_opt:', loss_d)

        # cos_sim = cosine_similarity(res_oringal, loss_d)
        # print(f'cos_sim:{cos_sim}')
        # assert (
        #     cos_sim > COSINE_THRESHOLD,
        #     f"Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}"
        # )
        # print('loss_d:', loss_d.size(), loss_d.device)

        # loss_d.backward() #for TransformerBlocks
        loss_d[0,0].backward() #for resnet18
        # optimizer.step()
        # exit(0)
        # with autocast(): #tpu device
    elif args.model == "resnet":
        input = torch.randn((1, 3, 224, 224))
        import torchvision.models as models
        mod = models.resnet50()
        # from resnet50 import resnet50
        # mod = resnet50()
        net_d = copy.deepcopy(mod)
        net_d.to(device)
        net_d.train()
        input_d = input.to(device)
        optimizer = torch.optim.SGD(net_d.parameters(), lr=0.01)
        optimizer.zero_grad()
        model_opt = torch.compile(net_d, backend=aot_backend)
        loss_d = model_opt(input_d)
        loss_d.sum().backward()
        optimizer.step()
    elif args.model == "yolo":
        input = torch.randn((1,3,224,224))
        from nets.yolo import YoloBody
        anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        num_classes = 80
        phi = 's'
        mod = YoloBody(anchors_mask, num_classes, phi)
        net_d = copy.deepcopy(mod)
        net_d.to(device)
        net_d.train()
        input_d = input.to(device)
        optimizer = torch.optim.SGD(net_d.parameters(), lr=0.01)
        optimizer.zero_grad()
        model_opt = torch.compile(net_d, backend=aot_backend)
        loss_d = model_opt(input_d)
        loss_d[0].sum().backward()
        optimizer.step()
    elif args.model == "llama":
        from transformers import LlamaForCausalLM, LlamaTokenizer,AutoTokenizer
        model_path = "/workspace/tpu_1010/module/llama/llama-2-7b"
        tokenizer_path = "/workspace/tpu_1010/module/llama/tokenizer.model"
        model_name = 'meta-llama/Llama-2-7b-hf'
        model = LlamaForCausalLM.from_pretrained(model_name,torch_dtype=torch.float32)
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        print("llama model is completely loading now\n")
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        input_ = "Redbags is traditional Chinese gifts"
        target_ = "Redbags are traditional Chinese gifts"
        prompt = f"Correct this to standard English: {input_}\n---\nCorrected: "
        prompt_ids = tokenizer.encode(tokenizer.bos_token + prompt, add_special_tokens=False)
        label_ids = tokenizer.encode(target_ + tokenizer.eos_token, add_special_tokens=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        model.train()
        model.to(device)
        model_opt = torch.compile(model, backend=aot_backend)
        input = {
            "input_ids": prompt_ids + label_ids,
            "attention_mask": [1] * len(prompt_ids + label_ids),
            "labels": [-100] * len(prompt_ids) + label_ids
        }
        for key in input.keys():
            input[key] = torch.tensor(input[key])
            input[key] = input[key].unsqueeze(0)
            input[key] = input[key].to(device)
            if key == "labels":
                input[key] = input[key].long()
        loss = model_opt(**input).loss
        loss.sum().backward()
        optimizer.step()
    elif args.model == "sd":
        from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler,DiffusionPipeline
        from transformers import CLIPTokenizer
        cliptokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        model_id = "stabilityai/stable-diffusion-2-1-base"
        mod = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        unet_input = torch.randn(10,4,16,16).to(device) #10,4,16,16
        vae_input = torch.randn(10,3,16,16).to(device)
        timestep = torch.tensor(10).to(device) #10
        encoder_hidden_state = torch.randn(10,49,1024).to(device) #10,49,1024
        net_d = copy.deepcopy(mod.vae)
        example_text = 'I will be on holiday two days later'
        clip_input = cliptokenizer(example_text,padding='max_length',
                            max_length = 10,
                            truncation=True,
                            return_tensors="pt")
        mask = clip_input['attention_mask'].to(device)
        input_id = clip_input['input_ids'].to(device)
        net_d.train()
        optimizer = torch.optim.SGD(net_d.parameters(), lr=0.01)
        model_opt = torch.compile(net_d, backend=aot_backend)
        optimizer.zero_grad()
        # loss_d = model_opt(unet_input,timestep,encoder_hidden_state) #sd unet
        loss_d = model_opt(vae_input) #sd vae
        # out = model_opt(input_id,mask) #sd text_encoder
        # loss_d = out[0][:, 0, :]  # [batch, 768]
        # loss_d.sum().backward()
        loss_d.sample.sum().backward() #sd vae
        optimizer.step()
    elif args.model == "bert":
        from transformers import BertTokenizer, BertModel
        max_len = 256
        mod = BertModel.from_pretrained('bert-base-cased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        example_text = 'I will watch Memento tonight'
        bert_input = tokenizer(example_text,padding='max_length',
                            max_length = 10,
                            truncation=True,
                            return_tensors="pt")
        mask = bert_input['attention_mask'].to(device)
        input_id = bert_input['input_ids'].to(device)
        net_d = copy.deepcopy(mod)
        net_d.to(device)
        net_d.train()
        optimizer = torch.optim.SGD(net_d.parameters(), lr=0.01)
        optimizer.zero_grad()
        model_opt = torch.compile(net_d, backend=aot_backend)
        out = model_opt(input_id,mask)
        loss_d = out[0][:, 0, :]  # [batch, 768]
        loss_d.sum().backward()
        optimizer.step()
    elif args.model == "test_model":
        from tools.train.test_model import test_model
        mod = test_model()
        input = torch.randn((5,10))
        net_d = copy.deepcopy(mod)
        net_d.to(device)
        net_d.train()
        input_d = input.to(device)
        optimizer = torch.optim.SGD(net_d.parameters(), lr=0.01)
        optimizer.zero_grad()
        model_opt = torch.compile(net_d, backend=aot_backend)
        loss_d = model_opt(input_d)
        loss_d.sum().backward()
        optimizer.step()
    elif args.model == "bert_large":
        from transformers import BertTokenizer, BertModel
        max_len = 256
        mod = BertModel.from_pretrained('bert-large-cased')
        tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        example_text = 'I will watch Memento tonight'
        bert_input = tokenizer(example_text,padding='max_length',
                            max_length = 10,
                            truncation=True,
                            return_tensors="pt")
        mask = bert_input['attention_mask'].to(device)
        input_id = bert_input['input_ids'].to(device)
        net_d = copy.deepcopy(mod)
        net_d.to(device)
        net_d.train()
        optimizer = torch.optim.SGD(net_d.parameters(), lr=0.01)
        optimizer.zero_grad()
        model_opt = torch.compile(net_d, backend=aot_backend)
        out = model_opt(input_id,mask)
        loss_d = out[0][:, 0, :]  # [batch, 768]
        loss_d.sum().backward()
        optimizer.step()
    else:
        from transformers import GPT2Config
        # ====== configure ======
        configure = GPT2Config()
        configure.attn_pdrop = 0
        configure.embd_pdrop = 0
        configure.resid_pdrop = 0
        configure.vocab_size= 100
        configure.n_positions= 128
        configure.n_layer= 1
        configure.activation_function= "relu"
        batch = 1
        sequence = 4

        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Model
        net = GPT2LMHeadModel(configure)
        net_d = copy.deepcopy(net)
        optimizer_tpu = torch.optim.AdamW(net.parameters(), lr = 0.01)
        # net_d = net_d.to("privateuseone:0")
        net_d = net_d.to(device)
        model_opt = torch.compile(net_d, backend=aot_backend)
        optimizer_tpu.zero_grad()
        inp = torch.randint(0, configure.vocab_size,(batch, sequence))#, dtype=torch.int64)
        # inp_tpu = inp.to("privateuseone:0")
        inp = inp.to(device)
        out_tpu = model_opt(input_ids = inp, labels = inp)
        out_tpu["logits"].sum().backward()
        optimizer_tpu.step()
