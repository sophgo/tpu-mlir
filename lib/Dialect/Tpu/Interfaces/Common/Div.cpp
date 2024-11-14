//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Interfaces/IndexingMapsInterface.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"

LogicalResult tpu::DivOp::init(InferenceParameter &p) {
  auto binary = new Binary();
  int index0 = 0, index1 = 1;
  if (getIsReverse()) {
    index0 = 1, index1 = 0;
  }
  auto lhs_shape = module::getShape(getInputs()[index0]);
  auto rhs_shape = module::getShape(getInputs()[index1]);

  (*binary)
      .hs(p.inputs[index0], p.inputs[index1], lhs_shape, rhs_shape)
      .dst(p.outputs[0], module::getShape(getOutput()))
      .do_relu(getDoRelu())
      .relu_limit(getReluLimit().convertToDouble())
      .algorithem(algorithm::binary_div)
      .setup();

  p.handle = (void *)binary;

  return success();
}

void tpu::DivOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto binary = (Binary *)p.handle;
    delete binary;
    p.handle = nullptr;
  }
}

LogicalResult tpu::DivOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto binary = (Binary *)p.handle;
  binary->run();
  auto out_type = module::getStorageType(getOutput());
  auto dst = p.outputs[0];
  auto num_elements = module::getNumElements(getOutput());
  if (out_type.isF16()) {
    F16(dst, dst, num_elements);
  } else if (out_type.isBF16()) {
    BF16(dst, dst, num_elements);
  }
  return success();
}

LogicalResult tpu::DivOp::LocalGenSupport() {
  if (!(module::isBM1684XFamily() || module::isBM1690Family())) {
    return failure();
  }
  return BroadCastBinaryLocalGenSupport(getOperation());
}

void tpu::DivOp::assign_sec_info(int64_t n_step, int64_t c_step, int64_t h_step,
                                 int64_t d_step, int64_t w_step,
                                 group_type_t group_type,
                                 local_sec_info_t &sec_info) {
  memset(&sec_info, 0, sizeof(local_sec_info_t));
  sec_info.group_type = group_type;
  int64_t n0, c0, d0, h0, w0, n1, c1, d1, h1, w1, on, oc, od, oh, ow;
  auto input0 = getOperand(0);
  auto input1 = getOperand(1);
  auto output = getResult();
  module::getNCDHW(input0, n0, c0, d0, h0, w0, group_type);
  module::getNCDHW(input1, n1, c1, d1, h1, w1, group_type);
  module::getNCDHW(output, on, oc, od, oh, ow, group_type);
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);
  auto in0_gi = LocalGenInterface::getGroupInfo(input0, n_step, h_step, d_step,
                                                w_step, c_step);
  auto in1_gi = LocalGenInterface::getGroupInfo(input1, n_step, h_step, d_step,
                                                w_step, c_step);
  sec_info.n_slice = std::max(in0_gi.n_slice, in1_gi.n_slice);
  sec_info.c_slice = std::max(in0_gi.c_slice, in1_gi.c_slice);
  sec_info.d_slice = std::max(in0_gi.d_slice, in1_gi.d_slice);
  sec_info.h_slice = std::max(in0_gi.h_slice, in1_gi.h_slice);
  sec_info.w_slice = std::max(in0_gi.w_slice, in1_gi.w_slice);
  sec_info.n_idx = std::max(in0_gi.n_idx, in1_gi.n_idx);
  sec_info.d_idx = std::max(in0_gi.d_idx, in1_gi.d_idx);
  sec_info.c_idx = std::max(in0_gi.c_idx, in1_gi.c_idx);
  sec_info.is_c_split =
      !(std::max(in0_gi.c_idx, in1_gi.c_idx) == 0 &&
        std::max(in0_gi.c_slice, in1_gi.c_slice) == std::max(c0, c1));
  sec_info.h_idx = std::max(in0_gi.h_idx, in1_gi.h_idx);
  sec_info.is_h_split =
      !(std::max(in0_gi.h_idx, in1_gi.h_idx) == 0 &&
        std::max(in0_gi.h_slice, in1_gi.h_slice) == std::max(h0, h1));
  sec_info.w_idx = std::max(in0_gi.w_idx, in1_gi.w_idx);
  sec_info.is_w_split =
      !(std::max(in0_gi.w_idx, in1_gi.w_idx) == 0 &&
        std::max(in0_gi.w_slice, in1_gi.w_slice) == std::max(w0, w1));
  sec_info.out_n_slice = gi.n_slice;
  sec_info.out_h_idx = gi.h_idx;
  sec_info.out_h_slice = gi.h_slice;
  sec_info.out_w_idx = gi.w_idx;
  sec_info.out_w_slice = gi.w_slice;
}

void tpu::DivOp::assign_fw_param(void *param) {
  fw_broadcast_binary_layer_param_t *fw_broadcast_binary_layer_param =
      (fw_broadcast_binary_layer_param_t *)param;
  fw_broadcast_binary_layer_param->binary_op = BINARY_DIV;
  fw_broadcast_binary_layer_param->a_is_coeff =
      module::isWeight(getInputs()[0]);
  fw_broadcast_binary_layer_param->b_is_coeff =
      module::isWeight(getInputs()[1]);
  fw_broadcast_binary_layer_param->if_relu = getDoRelu() ? 1 : 0;
  fw_broadcast_binary_layer_param->relu_upper_limit =
      getReluLimit().convertToDouble();
  fw_broadcast_binary_layer_param->buffer_addr = 0; // no use
  auto input_num = getInputs().size();
  assert(input_num == 2);
  std::vector<int32_t> muls = {getMultiplier(), 1};
  std::vector<int32_t> rs = {getRshift(), 0};
  for (int i = 0; i < input_num; ++i) {
    fw_broadcast_binary_layer_param->scale[i] = muls[i];
    fw_broadcast_binary_layer_param->rshift_num[i] = rs[i];
    auto dtype = BM1684::getDataType(getInputs()[i]);
    if (dtype == DTYPE_INT8)
      fw_broadcast_binary_layer_param->opd_sign[i] = 1;
    else if (dtype == DTYPE_UINT8)
      fw_broadcast_binary_layer_param->opd_sign[i] = 0;
    else if (dtype == DTYPE_INT16)
      fw_broadcast_binary_layer_param->opd_sign[i] = 2;
    else if (dtype == DTYPE_UINT16)
      fw_broadcast_binary_layer_param->opd_sign[i] = 3;
  }
  int a_shape[MAX_SHAPE_DIMS], b_shape[MAX_SHAPE_DIMS];
  module::getGlobalShape(getInputs()[0], a_shape);
  module::getGlobalShape(getInputs()[1], b_shape);
  fw_broadcast_binary_layer_param->a_dims =
      module::getShape(getInputs()[0]).size();
  memcpy(&(fw_broadcast_binary_layer_param->a_shape[0]), &a_shape[0],
         MAX_SHAPE_DIMS * sizeof(int));
  fw_broadcast_binary_layer_param->b_dims =
      module::getShape(getInputs()[1]).size();
  memcpy(&(fw_broadcast_binary_layer_param->b_shape[0]), &b_shape[0],
         MAX_SHAPE_DIMS * sizeof(int));
}

ArrayAttr tpu::DivOp::getIndexingMaps() {
  return getBinaryIndexingMaps(getOperation());
};

bool tpu::DivOp::support_multi_core() { return false; }
