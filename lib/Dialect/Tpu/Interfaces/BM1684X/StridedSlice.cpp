//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

// int8
void tpu::StridedSliceOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  strideslice_common_spec_t param = {0};
  param.begin_mask = getBeginMask();
  param.end_mask = getEndMask();

  std::vector<int64_t> input_shape = module::getShape(getInput());
  std::vector<int64_t> output_shape = module::getShape(getOutput());

  auto in_dims = input_shape.size();
  auto out_dims = output_shape.size();
  assert(in_dims == out_dims);
  auto start_v =
      cast<top::WeightOp>(getStarts().getDefiningOp()).read<int32_t>();
  auto stride_v =
      cast<top::WeightOp>(getStrides().getDefiningOp()).read<int32_t>();
  auto end_v = cast<top::WeightOp>(getEnds().getDefiningOp()).read<int32_t>();
  for (int i = 0; i < in_dims; i++) {
    param.begin_index[i] = start_v->at(i);
    param.end_index[i] = end_v->at(i);
    param.strides[i] = stride_v->at(i);
  }
  BM168x::call_global_func("backend_api_strideslice_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::StridedSliceOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(strideslice_common_spec_t);
  strideslice_common_spec_t param = {0};
  param.begin_mask = getBeginMask();
  param.end_mask = getEndMask();

  std::vector<int64_t> input_shape = module::getShape(getInput());
  std::vector<int64_t> output_shape = module::getShape(getOutput());

  auto in_dims = input_shape.size();
  auto out_dims = output_shape.size();
  assert(in_dims == out_dims);
  auto start_v =
      cast<top::WeightOp>(getStarts().getDefiningOp()).read<int32_t>();
  auto stride_v =
      cast<top::WeightOp>(getStrides().getDefiningOp()).read<int32_t>();
  auto end_v = cast<top::WeightOp>(getEnds().getDefiningOp()).read<int32_t>();
  for (int i = 0; i < in_dims; i++) {
    param.begin_index[i] = start_v->at(i);
    param.end_index[i] = end_v->at(i);
    param.strides[i] = stride_v->at(i);
  }
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::StridedSliceOp::get_fw_type_bm1684x() {
  return FW_BMNET_STRIDESLICE;
}
