import os
import argparse
import copy
import logging
import torch
import numpy as np
from tools.train.tpu_mlir_jit import device, aot_backend
import tools.train.tpu_mlir_jit as tpu_mlir_jit
from numpy_helper.npz_compare import npz_compare

from torch import autocast
import torch_tpu

def enable_dynamo_debug_info():
    import torch._dynamo as td
    td.config.log_level = logging.DEBUG
    td.config.verbose = True
    td.config.output_code = True
    os.environ["TORCHDYNAMO_PRINT_GUARDS"] = "1"

layer_names = []
features_out_hook = {}
i = 0
def hook(module, fea_in, fea_out):
    global i
    name = layer_names[i]
    i += 1
    global features_out_hook
    features_out_hook[name] = fea_out.detach().numpy()
    return None

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--chip", default="bm1684x", choices=['bm1684x', 'sg2260'],
#                         help="chip name")
#     parser.add_argument("--debug", default="",
#                         help="debug")
#     parser.add_argument("--cmp", action='store_true',
#                         help="enable cmp")

#     args = parser.parse_args()
#     tpu_mlir_jit.args = args
#     os.system('rm -rf ref_data.npz')
#     # enable_dynamo_debug_info()
#     if True:
#         # input = torch.randn((1, 3, 224, 224))
#         # import torchvision.models as models
#         # mod = models.resnet18()

#         # from tools.train.test_model import test_model5
#         # mod = test_model5() #.train()
#         # input = torch.randn((1, 3, 224, 224))
#         # # input0 = torch.randn((2, 3, 224, 224))

#         input = torch.randn((1, 3, 224, 224))
#         from tools.train.resnet import resnet18
#         mod = resnet18()
#         # if args.cmp:
#         if False:
#             hook_handles = []
#             exclude_module = ['torch.fx', 'batchnorm']
#             for name, child in mod.named_modules():
#                 if not any([i in str(type(child)) for i in exclude_module]):
#                     print("add hook on", str(type(child)), name)
#                     layer_names.append(name)
#                     hd = child.register_forward_hook(hook=hook)
#                     hook_handles.append(hd)
#             mod(input)
#             np.savez('ref_data.npz', **features_out_hook)
#             for hd in hook_handles:
#                 hd.remove()

#         # d_model=10 #768  #test ok at 0918
#         # from tools.train.gpt2 import TransformerBlocks #backward can not use this
#         # mod = TransformerBlocks(d_model=d_model, nlayers=2) #.train()
#         # input = torch.randn((1,4,d_model))

#         # from tools.train.gpt2 import GPT2
#         # mod = GPT2()
#         # input = torch.randint(0, 50257,(1, 5))

#         net_d = copy.deepcopy(mod)
#         input_d = input.to(device)
#         # input0_d = input0.to(device)
#         net_d.to(device)

#         optimizer = torch.optim.SGD(mod.parameters(), lr=0.01)
#         # net_d, optimizer = amp.initialize(net_d, optimizer, opt_level='O2')
#         model_opt = torch.compile(net_d, backend=aot_backend)

#         # optimizer.zero_grad()

#         # res_oringal = mod(input)
#         # print('res_oringal:', res_oringal)

#         # with torch.autocast(tpu_dev):
#         # loss_d = model_opt(input_d, labels = input_d)
#         # loss_d = model_opt(input_d, input0_d)
#         for i in range(1):
#             print(f"now run forward{i}")
#             loss_d = model_opt(input_d)
#             # if args.cmp:
#             #     np.savez('model_opt_data.npz', **loss_d)
#             #     npz_compare(['model_opt_data.npz', 'ref_data.npz', "--tolerance", "0.95,0.95", "-v"])
#             # print('res_oringal_opt:', loss_d)

#             # cos_sim = cosine_similarity(res_oringal, loss_d)
#             # print(f'cos_sim:{cos_sim}')
#             # assert (
#             #     cos_sim > COSINE_THRESHOLD,
#             #     f"Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}"
#             # )
#             # print('loss_d:', loss_d.size(), loss_d.device)

#             print(f"now run backward{i}")
#             loss_d.backward() #for TransformerBlocks
#             # loss_d[0,0].backward() #for resnet18
#             # optimizer.step()
#             # exit(0)
#             # with autocast(): #tpu device
#     else:
#         from transformers import GPT2Config
#         # ====== configure ======
#         configure = GPT2Config()
#         configure.attn_pdrop = 0
#         configure.embd_pdrop = 0
#         configure.resid_pdrop = 0
#         configure.vocab_size= 100
#         configure.n_positions= 128
#         configure.n_layer= 1
#         configure.activation_function= "relu"
#         batch = 1
#         sequence = 4

#         from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Model
#         net = GPT2LMHeadModel(configure)
#         net_d = copy.deepcopy(net)
#         optimizer_tpu = torch.optim.AdamW(net.parameters(), lr = 0.01)
#         # net_d = net_d.to("privateuseone:0")
#         net_d = net_d.to(device)
#         model_opt = torch.compile(net_d, backend=aot_backend)
#         optimizer_tpu.zero_grad()
#         inp = torch.randint(0, configure.vocab_size,(batch, sequence))#, dtype=torch.int64)
#         # inp_tpu = inp.to("privateuseone:0")
#         inp = inp.to(device)
#         out_tpu = model_opt(input_ids = inp, labels = inp)
#         out_tpu["logits"].sum().backward()
#         optimizer_tpu.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip", default="bm1690", choices=['bm1684x', 'bm1690'],
                        help="chip name")
    parser.add_argument("--debug", default="",
                        help="debug")
    parser.add_argument("--cmp", action='store_true',
                        help="enable cmp")
    parser.add_argument("--skip_module_num", default=0, type=int,
                        help='skip_module_num')
    parser.add_argument("--num_core", default=1, type=int,
                        help='The numer of TPU cores used for parallel computation')
    args = parser.parse_args()
    if args.chip == 'bm1690':
        os.system('ln -sf $TPUC_ROOT/lib/libcmodel_bm1690.so $TPUC_ROOT/lib/libcmodel.so')
    if args.chip == 'bm1684x':
        os.system('ln -sf $TPUC_ROOT/lib/libcmodel_1684x.so $TPUC_ROOT/lib/libcmodel.so')
    tpu_mlir_jit.args = args
    input = torch.randn((1, 3, 224, 224))
    input_d = input.to(device)

    # print('start test test_model1')
    # from tools.train.test_model import test_model1
    # mod = test_model1(for_train = True)
    # mod.to(device)
    # model_opt = torch.compile(mod, backend=aot_backend)
    # for i in range(1):
    #     print(f"now run forward{i}")
    #     res = model_opt(input_d)
    #     print(f"now run backward{i}")
    #     res.sum().backward()

    # print('start test test_model2')
    # from tools.train.test_model import test_model2
    # mod = test_model2(for_train = True)
    # mod.to(device)
    # model_opt = torch.compile(mod, backend=aot_backend)
    # for i in range(1):
    #     print(f"now run forward{i}")
    #     res = model_opt(input_d)
    #     print(f"now run backward{i}")
    #     res.sum().backward()

    # print('start test resnet50')
    # from tools.train.resnet import resnet50
    # mod = resnet50()
    # mod.to(device)
    # model_opt = torch.compile(mod, backend=aot_backend)
    # for i in range(1):
    #     print(f"now run forward{i}")
    #     res = model_opt(input_d)
    #     print(f"now run backward{i}")
    #     res.sum().backward()

    # # FP16 model
    print('start test FP16 model')
    # define model
    # from tools.train.test_model import test_model1
    # mod = test_model1(for_train = True)
    # from tools.train.test_model import test_model2
    # mod = test_model2(for_train = True)
    from tools.train.resnet import resnet50
    mod = resnet50()
    mod.to(device)
    # mod.half()
    model_opt = torch.compile(mod, backend=aot_backend)
    # forward
    scaler = torch.tpu.amp.GradScaler()
    with autocast(device_type = device, dtype = torch.float16):
        input_d = input.to(device, dtype=torch.float16)
        res = model_opt(input_d)
        loss = res.sum()
    scaler.scale(loss.float()).backward()

    # NEW FP16 model
    # print('start test new FP16 model')
    # input = torch.randn((1, 64, 224, 224))
    # input_d = input.to(device)
    # # define model
    # # from tools.train.test_model import test_model_mine
    # # mod = test_model_mine(for_train = True)
    # # from tools.train.test_model import test_model_mine3
    # # mod = test_model_mine3(for_train = True)
    # from tools.train.resnet import resnet50
    # mod = resnet50()
    # mod.conv1 = torch.nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # # import pdb;pdb.set_trace()
    # mod.to(device)
    # model_opt = torch.compile(mod, backend=aot_backend)
    # # forward
    # scaler = torch.tpu.amp.GradScaler()
    # with autocast(device_type = device, dtype = torch.float16):
    #     input_d = input.to(device, dtype=torch.float16)
    #     res = model_opt(input_d)
    #     loss = res.sum()
    # scaler.scale(loss.float()).backward()
