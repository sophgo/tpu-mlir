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
#include "tpu_mlir/Support/Float8.h"

LogicalResult tpu::MulOp::init(InferenceParameter &p) {
  auto binary = new Binary();
  auto lhs_shape = module::getShape(getInputs()[0]);
  auto rhs_shape = module::getShape(getInputs()[1]);

  (*binary)
      .hs(p.inputs[0], p.inputs[1], lhs_shape, rhs_shape)
      .dst(p.outputs[0], module::getShape(getOutput()))
      .do_relu(getDoRelu())
      .relu_limit(getReluLimit().convertToDouble())
      .algorithem(algorithm::binary_mul)
      .setup();
  p.handle = (void *)binary;
  return success();
}

void tpu::MulOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto binary = (Binary *)p.handle;
    delete binary;
    p.handle = nullptr;
  }
}

LogicalResult tpu::MulOp::inference(InferenceParameter &p) {
  auto output_shape = computer_broadcast_shape(getOperation());
  module::setShape(getOutput(), output_shape);
  auto num_elem = module::getNumElements(getOutput());
  auto out_type = module::getStorageType(getOutput());

  auto asym = module::isAsymmetric();
  auto binary = (Binary *)p.handle;
  binary->run();
  if (out_type.isa<FloatType>()) {
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isFloat8E4M3FN()) {
      if (!getOutF8Scales().has_value())
        llvm_unreachable("should have out scale for Mul in f8 mode");
      f64_array_t scales = module::getF64Array(getOutF8Scales().value());
      [[maybe_unused]] auto out_scale = scales->at(0);
      [[maybe_unused]] auto out_scale_reciprocal = 1 / scales->at(0);
      F8E4M3(p.outputs[0], p.outputs[0], num_elem, out_scale_reciprocal, true);
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++)
        p.outputs[0][i] = F16(p.outputs[0][i], false);
      F8E4M3(p.outputs[0], p.outputs[0], num_elem, 1.0, true);
    } else if (out_type.isFloat8E5M2()) {
      F8E5M2(p.outputs[0], p.outputs[0], num_elem, 1.0, true);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else if (out_type.isInteger(32)) {
    return success();
  } else if (asym == false) {
    auto qmode = getQuantMode();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int i = 0; i < num_elem; i++) {
      float sum = p.outputs[0][i];
      if (module::isCV18xx()) {
        sum = applyMultiplierAndRShift(sum, getMultiplier(), getRshift(), qmode,
                                       ROUNDING_HALF_AWAY_FROM_ZERO);
      } else {
        sum =
            applyMultiplierAndRShift(sum, getMultiplier(), getRshift(), qmode);
      }
      p.outputs[0][i] = saturate(sum, out_type);
    }
  } else {
    auto qmode = getQuantMode();
    auto num_elem = module::getNumElements(getOutput());
    auto lhs_num_elem = module::getNumElements(getInputs()[0]);
    auto rhs_num_elem = module::getNumElements(getInputs()[1]);
    std::vector<float> lhs_tmp(lhs_num_elem);
    std::vector<float> rhs_tmp(rhs_num_elem);
    auto o_qtype = module::getUniformQuantizedType(getOutput());
    auto l_qtype = module::getUniformQuantizedType(getInputs()[0]);
    int rzp = 0;
    if (!module::isWeight(getInputs()[1])) {
      auto r_qtype = module::getUniformQuantizedType(getInputs()[1]);
      rzp = r_qtype.getZeroPoint();
    }
#pragma omp parallel for schedule(static, omp_schedule(lhs_num_elem))
    for (int i = 0; i < lhs_num_elem; i++) {
      lhs_tmp[i] = p.inputs[0][i] - l_qtype.getZeroPoint();
    }
#pragma omp parallel for schedule(static, omp_schedule(rhs_num_elem))
    for (int i = 0; i < rhs_num_elem; i++) {
      rhs_tmp[i] = p.inputs[1][i] - rzp;
    }

    auto binary = (Binary *)p.handle;
    (*binary)
        .lhs(lhs_tmp.data(), module::getShape(getInputs()[0]))
        .rhs(rhs_tmp.data(), module::getShape(getInputs()[1]))
        .run();

#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int i = 0; i < num_elem; i++) {
      float sum = p.outputs[0][i];
      sum = applyMultiplierAndRShift(sum, getMultiplier(), getRshift(), qmode) +
            o_qtype.getZeroPoint();
      p.outputs[0][i] = saturate(sum, out_type);
    }
  }
  return success();
}

LogicalResult tpu::MulOp::LocalGenSupport() {
  return BroadCastBinaryLocalGenSupport(getOperation());
}

void tpu::MulOp::assign_sec_info(int64_t n_step, int64_t c_step, int64_t h_step,
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

void tpu::MulOp::DumpQuantAgnosticAttrs(llvm::raw_string_ostream &os) {
  for (auto attr : getOperation()->getAttrs()) {
    auto attr_name = attr.getName().str();
    if (attr_name == "ginfo" || attr_name == "rshift" || attr_name == "multiplier") {
      continue;
    }
    os << attr_name << "=";
    attr.getValue().print(os);
    os << "; ";
  }

  auto rshift_v = getRshift();
  auto multiplier_v = getMultiplier();
  if (rshift_v == 0) {
    // do-nothing.
  } else {
    os << "rshift_len=1; ";
  }
  if (multiplier_v == 1) {
    // do-nothing.
  } else {
    os << "multiplier=1; ";
  }
}

void tpu::MulOp::assign_fw_param(void *param) {
  fw_broadcast_binary_layer_param_t *fw_broadcast_binary_layer_param =
      (fw_broadcast_binary_layer_param_t *)param;
  fw_broadcast_binary_layer_param->binary_op = BINARY_MUL;
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
  fw_broadcast_binary_layer_param->opd_sign[2] = module::isSign(getOutput());
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

ArrayAttr tpu::MulOp::getIndexingMaps() {
  return getBinaryIndexingMaps(getOperation());
};

bool tpu::MulOp::support_multi_core() { return false; }
