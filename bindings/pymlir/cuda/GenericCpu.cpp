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
    auto input_type = getCudaType(op.getInputs()[0]);
    auto index_type = getCudaType(op.getInputs()[1]);
    cuda::gatherElements(indices, input, output, input_shape.data(),
                         indices_shape.data(), input_shape.size(), axis,
                         index_type, input_type);
  } else if (func_name == "gathernd_tf") {
    auto param = op.getParam().value();
    int batch_dims = param.get("batch_dims").cast<IntegerAttr>().getInt();
    auto input = op.getInputs()[0];
    auto indices = op.getInputs()[1];
    auto output = op.getOutputs()[0];
    auto in_shape = module::getShape(input);
    auto idx_shape = module::getShape(indices);
    int in_rank = in_shape.size();
    int idx_rank = idx_shape.size();
    int coord_dim = idx_shape[idx_rank - 1];

    int in_strides_arr[8] = {};
    int stride = 1;
    for (int d = in_rank - 1; d >= 0; d--) {
      in_strides_arr[d] = stride;
      stride *= in_shape[d];
    }

    int idx_strides_arr[8] = {};
    stride = 1;
    for (int d = idx_rank - 2; d >= 0; d--) {
      idx_strides_arr[d] = stride;
      stride *= idx_shape[d];
    }

    int out_total = 1;
    for (int d = batch_dims; d < idx_rank - 1; d++) {
      out_total *= idx_shape[d];
    }
    for (int d = 0; d < batch_dims; d++) {
      out_total *= idx_shape[d];
    }

    int copy_len = 1;
    for (int d = batch_dims + coord_dim; d < in_rank; d++) {
      copy_len *= in_shape[d];
    }

    cuda_ptr idx_f32;
    void *idx_data = getCudaData(indices);
    if (getCudaType(indices) != cuda::DT_F32) {
      idx_f32 = newCudaData(indices, cuda::DT_F32);
      idx_data = idx_f32.get();
    }
    cuda::bmGatherND(getCudaData(input), idx_data, getCudaData(output),
                     (int *)in_shape.data(), in_strides_arr,
                     (int *)idx_shape.data(), idx_strides_arr, batch_dims,
                     idx_rank, coord_dim, out_total, copy_len);
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
