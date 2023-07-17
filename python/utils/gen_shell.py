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
import numpy as np
import inspect

mlir_repo_root = os.environ["PROJECT_ROOT"]

sh_template = r"""
source {mlir_repo_root}/envsetup.sh

python py_{model_name}.py

model_transform.py \
   --model_name {model_name} \
   --model_def {model_name}.pt \
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
  --model {model_name}_f32.bmodel

model_deploy.py \
  --mlir {model_name}.mlir \
  --quantize BF16 \
  --chip bm1684x \
  --debug \
  --test_input {model_name}_in_f32.npz \
  --test_reference {model_name}_top_output.npz \
  --model {model_name}_f32.bmodel

run_calibration.py {model_name}.mlir \
  --dataset cali_data/ \
  --input_num 100 \
  -o {model_name}_cali_table \

model_deploy.py \
  --quantize int8 \
  --mlir {model_name}.mlir \
  --chip bm1684x \
  --tolerance 0.85,0.8 \
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

gops: [1, 1]
shapes:
{shape_list}

model: $(home)/{model_name}.pt

cali_input_key: cali_{model_name}

time_rountds: 50
time: true
precision: true

excepts: ""

mlir_transform: model_transform.py
  --model_name $(name)
  --model_def $(model)
  --tolerance 0.99,0.9
  --test_input __test_input__
  --input_shapes [$(shape_param)]
  --excepts $(excepts)
  --test_result $(workdir)/$(pure_name)_top_outputs.npz
  --mlir $(workdir)/$(pure_name).mlir

mlir_calibration: run_calibration.py $(workdir)/$(pure_name).mlir
  --dataset $(root)/dataset/basicvsr/$(cali_input_key)
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
      --quantize_table $(home)/$(name)_qtable
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
      --quantize_table $(home)/$(name)_qtable
      --calibration_table $(workdir)/$(pure_name)_cali_table

harness:
  type: __harness__
  args:
    - name: FP32
      bmodel: $(workdir)/$(pure_name)_$(target)_f32.bmodel
    - name: FP16
      bmodel: $(workdir)/$(pure_name)_$(target)_f16.bmodel
    - name: BF16
      bmodel: $(workdir)/$(pure_name)_$(target)_bf16.bmodel
    - name: INT8
      bmodel: $(workdir)/$(pure_name)_$(target)_int8_sym.bmodel

val_file: __val_file__

"""


def generate(model_name, model, input_lis, workspace_root):
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

    shapes = []
    for i in input_lis:
        ishape = ",".join([f"{x}" for x in i.shape])
        shapes.append(f"[{ishape}]")
    shape_str = ",".join(shapes)

    sh = sh_template.format(
        model_name=model_name, mlir_repo_root=mlir_repo_root, shape_str=shape_str
    )

    shape_list = "\n".join([f" - {shape_str}\n"] * 2)

    yaml = yaml_template.format(
        model_name=model_name, gops=[1, 1], shape_list=shape_list
    )

    tensor_args = inspect.getargspec(model.forward).args[1:]
    data_dic = {f"{k}.1": v for k, v in zip(tensor_args, input_lis)}

    traced = torch.jit.trace(model, input_lis)
    torch.jit.save(traced, os.path.join(workspace_root, f"{model_name}.pt"))

    np.savez(os.path.join(workspace_root, "data.npz"), **data_dic)

    os.makedirs(os.path.join(workspace_root, f"cali_data"), exist_ok=True)
    np.savez(os.path.join(workspace_root, f"cali_data/data.npz"), **data_dic)
    with open(os.path.join(workspace_root, f"{model_name}.mlir.yaml"), "w") as w:
        w.write(yaml)

    with open(os.path.join(workspace_root, f"convert.sh"), "w") as w:
        w.write(sh)

    # Export the model
    torch.onnx.export(
        model,  # model being run
        tuple(input_lis),  # model input (or a tuple for multiple inputs)
        os.path.join(
            workspace_root, f"{model_name}.onnx"
        ),  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,
    )  # whether to execute constant folding for optimization)
