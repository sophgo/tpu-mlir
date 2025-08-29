import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import random
import json
import subprocess


class MyMatMul(nn.Module):

    def __init__(self, in_f, hidden_f, out_f):
        super().__init__()
        self.weight1 = nn.Parameter(torch.randn(in_f, hidden_f))
        self.weight2 = nn.Parameter(torch.randn(hidden_f, out_f))

    def forward(self, x):
        y = x.matmul(self.weight1)  # First matmul
        z = y.matmul(self.weight2)  # Second matmul
        return z


matmul = MyMatMul(1536, 8960, 1536)
# np.savez("matmul_weight.npz", matmul.weight.detach().numpy())
matmul.eval()

prun_dim = 1  # random.randint(0, 1)
print("prun_dim:", prun_dim)
rand_pchl = random.randint(1, 10) * 64
dummy_input = torch.randn(960, 1536)  # batch_size=1，in_f=128
mask_1 = torch.ones_like(matmul.weight1)
mask_2 = torch.ones_like(matmul.weight2)

while 1:
    mask_channel = [random.randint(0, 8959) for i in range(rand_pchl)]
    mask_channel = sorted(list(set(mask_channel)))
    if len(mask_channel) % 64 == 0:
        break
# dummy_input[:, mask_channel] = 0
mask_1[:, mask_channel] = 0
# dummy_input_1 = torch.tensor(np.delete(dummy_input.detach().numpy(), mask_channel, axis=1))

mask_2[mask_channel, :] = 0

# Call torch.onnx.export to export
torch.onnx.export(
    matmul,  # Model to be exported
    dummy_input,  # Model input (example tensor)
    "mymatmul.onnx",  # Output filename
    export_params=True,  # Whether to export all parameters to an ONNX file?
    opset_version=11,  # ONNX version
    do_constant_folding=True,  # Whether to perform constant folding to optimize the graph
    input_names=['input'],  # Input tensor name
    output_names=['output'],  # Output tensor name
)
print("✅ 导出完成： mymatmul.onnx")

# Construct as above mask
prune.custom_from_mask(matmul, 'weight1', mask_1)
prune.custom_from_mask(matmul, 'weight2', mask_2)
y = matmul(dummy_input)
np.savez("mymatmul_in_f32.npz", input=dummy_input)
data = np.load("mymatmul_in_f32.npz")
print(data.files)
print(data["input"].shape)
print(data.f.input)

# if prun_dim == 1:
#     y_2 = np.delete(y.detach().numpy(), mask_channel, axis=1)
# else:
y_2 = y.detach().numpy()

# np.savez("matmul_output.npz", y.detach().numpy())
np.savez("matmul_output.npz", output_MatMul_f32=y_2, output_MatMul=y_2)
# print("non-zero rows:", (matmul.weight_mask.sum(dim=(1 - prun_dim)) != 0).nonzero().flatten())
data = np.load("matmul_output.npz")
print(data.files)
print(data["output_MatMul_f32"].shape)
print(data.f.output_MatMul_f32)

# rand-gen pruning.json
mm_dic = dict()
mm_dic_1 = dict()
mm_dic["MatMul"] = []
mm_dic_1["idx"] = 0
mm_dic_1["prun_dim"] = 1
mm_dic_1["pruned_channel"] = mask_channel
mm_dic_2 = dict()
mm_dic_2["idx"] = 1
mm_dic_2["prun_dim"] = 0
mm_dic_2["pruned_channel"] = mask_channel
mm_dic["MatMul"].append(mm_dic_1)
mm_dic["MatMul"].append(mm_dic_2)
with open('test_pruning.json', 'w', encoding='utf-8') as f:
    json.dump(mm_dic, f, ensure_ascii=False, indent=4)

# # gen block input_npz
# input_states = torch.rand((1, 960, 1536), dtype=torch.float32)
# position_ids = torch.randn(1, 960).clamp(0, 100)
# attention_mask = torch.rand((1, 1, 960, 960), dtype=torch.float32)
# np.savez("block_model_in_f32.npz", input_states=input_states, position_ids=position_ids, attention_mask=attention_mask)

# subprocess.run('model_runner.py --input block_model_in_f32.npz --model pruned_model.onnx --output block_model_ref_outputs.npz', shell=True)
# data = np.load("block_model_ref_outputs.npz")
# old_keys = ['hidden_states', 'past_k', 'past_v']
# new_keys = ['hidden_states_Add', 'past_k_Add', 'past_v_Reshape']
# new_data = {new: data[old] for old, new in zip(old_keys, new_keys)}
# np.savez('block_model_ref_outputs.npz', **new_data)

# # gen qwen_block_0_input
# pruned_channel = [random.randint(0, 2815) for i in range(rand_pchl)]
# pruned_channel = sorted(list(set(pruned_channel)))
# mm_dic_1 = dict()
# mm_dic_1["idx"] = 4
# mm_dic_1["prun_dim"] = 1
# mm_dic_1["pruned_channel"] = pruned_channel
# mm_dic_2 = dict()
# mm_dic_2["idx"] = 5
# mm_dic_2["prun_dim"] = 1
# mm_dic_2["pruned_channel"] = pruned_channel
# mm_dic_3 = dict()
# mm_dic_3["idx"] = 6
# mm_dic_3["prun_dim"] = 0
# mm_dic_3["pruned_channel"] = pruned_channel
# mm_dic = dict()
# mm_dic["MatMul"] = []
# mm_dic["MatMul"].append(mm_dic_1)
# mm_dic["MatMul"].append(mm_dic_2)
# mm_dic["MatMul"].append(mm_dic_3)
# with open('block_pruning_config.json', 'w', encoding='utf-8') as f:
#     json.dump(mm_dic, f, ensure_ascii=False, indent=4)
