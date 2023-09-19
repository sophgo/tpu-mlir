import os
import argparse
import copy
import logging
import torch
from tools.train.tpu_mlir_jit import device, aot_backend
import tools.train.tpu_mlir_jit as tpu_mlir_jit

def enable_dynamo_debug_info():
    import torch._dynamo as td
    td.config.log_level = logging.DEBUG
    td.config.verbose = True
    td.config.output_code = True
    os.environ["TORCHDYNAMO_PRINT_GUARDS"] = "1"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip", default="bm1684x", choices=['bm1684x', 'sg2260'],
                        help="chip name")
    parser.add_argument("--debug", default="",
                        help="debug")
    parser.add_argument("--cmp", action='store_true',
                        help="enable cmp")
    parser.add_argument("--skip_tpu_mlir", action='store_true',
                        help="skip_tpu_mlir")

    args = parser.parse_args()
    tpu_mlir_jit.args = args
    # enable_dynamo_debug_info()
    if True:
        input = torch.randn((1, 3, 224, 224))
        import torchvision.models as models
        mod = models.resnet18()

        # input = torch.randn((1, 3, 224, 224))
        # from resnet50 import resnet50
        # mod = resnet50()

        # from tools.train.test_model import test_model
        # mod = test_model() #.train()
        # input = torch.randn((1, 3, 16, 16))

        # d_model=10 #768  #test ok at 0918
        # from tools.train.gpt2 import TransformerBlocks #backward can not use this
        # mod = TransformerBlocks(d_model=d_model, nlayers=3) #.train()
        # input = torch.randn((1,4,d_model))

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

        loss_d.backward() #for TransformerBlocks
        # loss_d[0,0].backward() #for resnet18
        # optimizer.step()
        # exit(0)
        # with autocast(): #tpu device
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
