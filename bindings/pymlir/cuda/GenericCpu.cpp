//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaGenericCpuOp(tpu::GenericCpuOp op) {
  auto func_name = op.getCpuOpName();
  if (func_name == "quant") {
    if (!module::isUniformQuantized(op.getOutputs()[0])) {
      UNREACHABLE_OP("Not Implemented", op);
    }
    auto param = op.getParam().value();
    float scale = param.get("scale").cast<FloatAttr>().getValueAsDouble();
    void *input = getCudaData(op.getInputs()[0]);
    void *output = getCudaData(op.getOutputs()[0]);
    int num_elems = module::getNumElements(op.getInputs()[0]);
    cuda::f32ScaleToInt8(input, output, scale, num_elems, true,
                         cuda::RD_HALF_AWAY_FROM_ZERO);
  } else if (func_name == "embedding") {
    auto in = op.getInputs()[0];
    auto embed = op.getInputs()[1];
    auto out = op.getOutputs()[0];
    void *in_ptr = getCudaData(in);
    void *embed_ptr = getCudaData(embed);
    void *out_ptr = getCudaData(out);
    auto in_type = getCudaType(in);
    auto out_type = getCudaType(out);
    int num_in = module::getNumElements(in);
    int num_embed = module::getNumElements(embed);
    auto embed_shape = module::getShape(embed);
    int embed_dim = embed_shape[0];
    int inner_dim = num_embed / embed_dim;
    cuda::gather(in_ptr, embed_ptr, out_ptr, num_in, embed_dim, inner_dim,
                 in_type, out_type);
  } else if (func_name == "argmax_v3") {
    auto param = op.getParam().value();
    int axis = param.get("axis").cast<IntegerAttr>().getInt();
    auto scale = param.get("scale").cast<FloatAttr>().getValueAsDouble();
    auto in_type = module::getStorageType(op.getInputs()[0]);
    if (!in_type.isSignedInteger()) {
      scale = 1.0;
    }
    auto input0 = getCudaData(op.getInputs()[0]);
    auto input1 = getCudaData(op.getInputs()[1]);
    auto output = getCudaData(op.getOutputs()[0]);
    auto input0_shape = module::getShape(op.getInputs()[0]);
    int64_t outer_dim = 1, axis_dim = 1, inner_dim = 1;
    for (size_t i = 0; i < input0_shape.size(); ++i) {
      if (i < axis) {
        outer_dim *= input0_shape[i];
      } else if (i == axis) {
        axis_dim = input0_shape[i];
      } else {
        inner_dim *= input0_shape[i];
      }
    }
    int input_bytes = module::getDtypeSize(op.getInputs()[0]);
    cuda::argIndex(input0, input1, output, outer_dim, axis_dim, inner_dim, input_bytes, scale);
  } else if (func_name == "gatherelements_pt") {
    auto param = op.getParam().value();
    int axis = param.get("axis").cast<IntegerAttr>().getInt();
    void *input = getCudaData(op.getInputs()[0]);
    void *indices = getCudaData(op.getInputs()[1]);
    void *output = getCudaData(op.getOutputs()[0]);
    auto input_shape = module::getShape(op.getInputs()[0]);
    auto indices_shape = module::getShape(op.getInputs()[1]);
    if (axis < 0) {
      axis += input_shape.size();
    }
    int outer_dim = 1, inner_dim = 1;
    for (size_t i = 0; i < input_shape.size(); ++i) {
      if (i < axis) {
        outer_dim *= input_shape[i];
      } else if (i > axis) {
        inner_dim *= input_shape[i];
      }
    }
    auto input_type = getCudaType(op.getInputs()[0]);
    auto index_type = getCudaType(op.getInputs()[1]);
    cuda::gatherElements(indices, input, output, indices_shape[axis], input_shape[axis], outer_dim, inner_dim, index_type, input_type);
  } else if (func_name == "grid_sampler") {
    auto param = op.getParam().value();
    auto mode = param.get("mode").cast<IntegerAttr>().getInt();
    auto padding_mode = param.get("padding_mode").cast<IntegerAttr>().getInt();
    auto align_corners = param.get("align_corners").cast<BoolAttr>().getValue();
    void *input = getCudaData(op.getInputs()[0]);
    void *grid = getCudaData(op.getInputs()[1]);
    void *output = getCudaData(op.getOutputs()[0]);
    auto input_shape = module::getShape(op.getInputs()[0]);
    auto grid_shape = module::getShape(op.getInputs()[1]);
    if (input_shape.size() != 4) {
      llvm_unreachable("Only support 4D input for GridSampler now");
    }
    cuda::grid_sample_interpolation_mode_t interpolation_mode =
        static_cast<cuda::grid_sample_interpolation_mode_t>(mode);
    cuda::grid_sample_padding_mode_t cuda_padding_mode =
        static_cast<cuda::grid_sample_padding_mode_t>(padding_mode);
    cuda::GridSample4D(input, grid, output, input_shape[0], input_shape[1],
             input_shape[2], input_shape[3], grid_shape[1],
             grid_shape[2], align_corners, interpolation_mode,
             cuda_padding_mode);
  } else {
    llvm_unreachable("Generic CPU operation not implemented");
  }
}
