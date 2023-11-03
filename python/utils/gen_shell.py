#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import os
import torch
from torch import nn
import numpy as np
import inspect
from torch import nn
import shutil
from typing import List, Union, Dict


sh_template = r"""

model_transform.py \
   --model_name {model_name} \
   --model_def {model_name}.{suf} \
   --input_shapes {shape_str} \
   --test_input data.npz \
   --keep_aspect_ratio \
   --tolerance 0.85,0.7 \
   --debug \
   --test_result {model_name}_top_output.npz \
   --mlir {model_name}.mlir

model_deploy.py \
  --mlir {model_name}.mlir \
  --quantize F32 \
  --chip bm1684x \
  --debug \
  --test_input {model_name}_in_f32.npz \
  --test_reference {model_name}_top_output.npz \
  --model {model_name}_f32.bmodel

model_deploy.py \
  --mlir {model_name}.mlir \
  --quantize F16 \
  --chip bm1684x \
  --debug \
  --test_input {model_name}_in_f32.npz \
  --test_reference {model_name}_top_output.npz \
  --model {model_name}_f16.bmodel

model_deploy.py \
  --mlir {model_name}.mlir \
  --quantize BF16 \
  --chip bm1684x \
  --debug \
  --test_input {model_name}_in_f32.npz \
  --test_reference {model_name}_top_output.npz \
  --model {model_name}_bf16.bmodel

run_calibration.py {model_name}.mlir \
  --dataset cali_data/ \
  --input_num 100 \
  -o {model_name}_cali_table

model_deploy.py \
  --quantize int8 \
  --mlir {model_name}.mlir \
  --chip bm1684x \
  --tolerance 0.8,0.4 \
  --debug \
  --test_input {model_name}_in_f32.npz \
  --calibration_table {model_name}_cali_table \
  --test_reference {model_name}_top_output.npz \
  --model {model_name}_int8.bmodel

"""

yaml_template = r"""

---
name: {model_name}
pure_name: {model_name}

gops: [1]
shapes:
{shape_list}

model: $(home)/{model_name}.{suf}

cali_input_key: {model_name}/cali_data

time_rountds: 50
time: true
precision: true


mlir_transform: model_transform.py
  --model_name $(name)
  --model_def $(model)
  --tolerance 0.99,0.9
  --test_input $(root)/dataset/{model_name}/test_input.npz
  --input_shapes [$(shape_param)]
  --test_result $(workdir)/$(pure_name)_top_outputs.npz
  --mlir $(workdir)/$(pure_name).mlir

mlir_calibration: run_calibration.py $(workdir)/$(pure_name).mlir
  --dataset $(root)/dataset/$(cali_input_key)
  --input_num 10
  -o $(workdir)/$(pure_name)_cali_table

BM1684:
  deploy:
    - model_deploy.py
      --mlir $(workdir)/$(pure_name).mlir
      --quantize F32
      --chip $(target)
      --test_input $(workdir)/$(pure_name)_in_f32.npz
      --test_reference $(workdir)/$(pure_name)_top_outputs.npz
      --model $(workdir)/$(pure_name)_$(target)_f32.bmodel

    - model_deploy.py
      --mlir $(workdir)/$(pure_name).mlir
      --quantize INT8
      --chip $(target)
      --tolerance 0.8,0.4
      --test_input $(workdir)/$(pure_name)_in_f32.npz
      --test_reference $(workdir)/$(pure_name)_top_outputs.npz
      --model $(workdir)/$(pure_name)_$(target)_int8_sym.bmodel
      --calibration_table $(workdir)/$(pure_name)_cali_table

BM1684X:
  deploy:
    - model_deploy.py
      --mlir $(workdir)/$(pure_name).mlir
      --quantize F32
      --chip $(target)
      --test_input $(workdir)/$(pure_name)_in_f32.npz
      --test_reference $(workdir)/$(pure_name)_top_outputs.npz
      --model $(workdir)/$(pure_name)_$(target)_f32.bmodel

    - model_deploy.py
      --mlir $(workdir)/$(pure_name).mlir
      --quantize F16
      --chip $(target)
      --test_input $(workdir)/$(pure_name)_in_f32.npz
      --test_reference $(workdir)/$(pure_name)_top_outputs.npz
      --model $(workdir)/$(pure_name)_$(target)_f16.bmodel

    - model_deploy.py
      --mlir $(workdir)/$(pure_name).mlir
      --quantize BF16
      --chip $(target)
      --test_input $(workdir)/$(pure_name)_in_f32.npz
      --test_reference $(workdir)/$(pure_name)_top_outputs.npz
      --model $(workdir)/$(pure_name)_$(target)_bf16.bmodel

    - model_deploy.py
      --mlir $(workdir)/$(pure_name).mlir
      --quantize INT8
      --chip $(target)
      --tolerance 0.8,0.4
      --test_input $(workdir)/$(pure_name)_in_f32.npz
      --test_reference $(workdir)/$(pure_name)_top_outputs.npz
      --model $(workdir)/$(pure_name)_$(target)_int8_sym.bmodel
      --calibration_table $(workdir)/$(pure_name)_cali_table

harness:
  type: {model_name}
  args:
    - name: FP32
      bmodel: $(workdir)/$(pure_name)_$(target)_f32.bmodel
    - name: FP16
      bmodel: $(workdir)/$(pure_name)_$(target)_f16.bmodel
    - name: BF16
      bmodel: $(workdir)/$(pure_name)_$(target)_bf16.bmodel
    - name: INT8
      bmodel: $(workdir)/$(pure_name)_$(target)_int8_sym.bmodel

"""


def get_ordered_input_names(callable, without_first=True):
    try:
        tensor_args = inspect.getargspec(callable).args
        if without_first:
            tensor_args = tensor_args[1:]
    except:
        tensor_args = list(inspect.signature(callable).parameters.keys())
    return tensor_args


def shape_list_to_str(shape_list: List[List[int]]) -> List[str]:
    """_summary_

    Args:
        shape_list (List[List[int]]): [[1,3,224,224],[1,3,224,224]]

    Returns:
        List[str]: ["[1,3,224,224]","[1,3,224,224]"]
    """
    res = []
    for shape in shape_list:
        shape_str = ",".join([str(i) for i in shape])
        res.append(f"[{shape_str}]")
    return res


def generate_shell(
    model_name: str, shape_list: List[List[int]], workspace_root: str, suf: str = "pt"
):
    shape_str = ",".join(shape_list_to_str(shape_list))
    sh = sh_template.format(model_name=model_name, shape_str=f"[{shape_str}]", suf=suf)
    with open(os.path.join(workspace_root, f"convert.sh"), "w") as w:
        w.write(sh)


def generate_yaml(
    model_name: str, shape_list: List[List[int]], workspace_root: str, suf: str = "pt"
):
    shape_str = ",".join(shape_list_to_str(shape_list))
    shape_list_str = shape_list_str = f" - {shape_str}"
    yaml = yaml_template.format(
        model_name=model_name, gops=[1, 1], shape_list=shape_list_str, suf=suf
    )
    with open(os.path.join(workspace_root, f"{model_name}.mlir.config.yaml"), "w") as w:
        w.write(yaml)


def generate(
    model_name: str,
    model: nn.Module,
    data: Union[List[torch.Tensor], Dict[str, torch.Tensor]],
    workspace_root: str,
    input_names: list = None,
    use_onnx=True,
):
    """
    generate onnx/jit model, transform/deploy scripts and model-zoo yaml config files by given pytorch model and input tensor list.


    from utils.gen_shell import generate

    def foo():
      model = build_model(config)
      sd = torch.load(
          "/workspace/model_swin/swin_small_patch4_window7_224.pth", map_location="cpu"
      )
      model.load_state_dict(sd['model'])
      input_tensor = torch.rand(1, 3, 224, 224)

      + generate("swin_s", model, [input_tensor], "/workspace/model_swin/swin_s_workspace")


    ls /workspace/model_swin/swin_s_workspace
      ./
      ../
      cali_data/
      convert.sh
      data.npz
      swin_s.mlir.yaml
      swin_s.onnx
      swin_s.pt

    """
    os.makedirs(workspace_root, exist_ok=True)

    if input_names is None:
        if isinstance(model, nn.Module):
            input_names = get_ordered_input_names(model.forward)
        elif callable(model):
            input_names = get_ordered_input_names(model.forward, without_first=False)
        else:
            raise NotImplementedError()

    if isinstance(data, (list, tuple)):
        input_lis = data
    elif isinstance(data, dict):
        input_lis = [data[k] for k in input_names if k in data]

    shape_list = [list(i.shape) for i in input_lis if hasattr(i, "shape")]
    # generate convert.sh + mlir.config.yaml
    generate_shell(
        model_name, shape_list, workspace_root, suf="onnx" if use_onnx else "pt"
    )
    generate_yaml(
        model_name, shape_list, workspace_root, suf="onnx" if use_onnx else "pt"
    )

    input_names = get_ordered_input_names(model.forward)
    def detach(v):
        try:
            return v.cpu().detach().numpy()
        except:
            return v

    data_dic = {f"{k}.1": detach(v) for k, v in zip(input_names, input_lis)}
    print(data_dic.keys())
    # Make npz
    os.makedirs(os.path.join(workspace_root, f"cali_data"), exist_ok=True)
    np.savez(os.path.join(workspace_root, f"cali_data/data.npz"), **data_dic)
    np.savez(os.path.join(workspace_root, "data.npz"), **data_dic)

    # Export model
    traced = torch.jit.trace(model, input_lis)
    torch.jit.save(traced, os.path.join(workspace_root, f"{model_name}.pt"))
    torch.onnx.export(
        model,  # model being run
        tuple(input_lis),  # model input (or a tuple for multiple inputs)
        os.path.join(
            workspace_root, f"{model_name}.onnx"
        ),  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,
        input_names=list(data_dic),
    )  # whether to execute constant folding for optimization)


def generate_onnx(
    model_name: str,
    model_path: str,
    workspace_root: str,
    data_path: str = None,
    fake_npz: bool = False,
):
    import onnx

    model = onnx.load(model_path)
    initializer_names = [x.name for x in model.graph.initializer]
    inputs = [ipt for ipt in model.graph.input if ipt.name not in initializer_names]
    shape_list = []
    for v in inputs:
        shape_list.append([dim.dim_value for dim in v.type.tensor_type.shape.dim])
    shutil.copy(model_path, os.path.join(workspace_root, f"{model_name}.onnx"))
    generate_shell(model_name, shape_list, workspace_root, suf="onnx")
    generate_yaml(model_name, shape_list, workspace_root, suf="onnx")
