//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"

LogicalResult tpu::SubOp::init(InferenceParameter &p) {
  int index0 = 0, index1 = 1;
  if (getIsReverse()) {
    index0 = 1, index1 = 0;
  }
  auto lhs_shape = module::getShape(getInputs()[index0]);
  auto rhs_shape = module::getShape(getInputs()[index1]);

  auto binary = new Binary();
  (*binary)
      .hs(p.inputs[index0], p.inputs[index1], lhs_shape, rhs_shape)
      .dst(p.outputs[0], module::getShape(getOutput()))
      .do_relu(getDoRelu())
      .relu_limit(getReluLimit().convertToDouble())
      .algorithem(algorithm::binary_sub)
      .setup();
  p.handle = (void *)binary;
  return success();
}

void tpu::SubOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto binary = (Binary *)p.handle;
    delete binary;
    p.handle = nullptr;
  }
}

LogicalResult tpu::SubOp::inference(InferenceParameter &p) {
  auto num_elem = module::getNumElements(getOutput());
  auto out_type = module::getStorageType(getOutput());
  memset(p.outputs[0], 0, num_elem * sizeof(float));
  auto asym = module::isAsymmetric();
  bool is_cv18xx = module::isCV18xx();
  if (out_type.isa<FloatType>()) {
    auto binary = (Binary *)p.handle;
    binary->run();
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else if (out_type.isInteger(32)) {
    auto binary = (Binary *)p.handle;
    binary->run();
  } else if (asym == false) {
    if (is_cv18xx) {
      // cv18xx interpreter
      auto multiplier_v = module::getI64Array(getMultipliers(), 2, 1);
      auto rshift_v = module::getI64Array(getRshifts(), 1, 0);
      auto lhs_num_elem = module::getNumElements(getInputs()[0]);
      auto rhs_num_elem = module::getNumElements(getInputs()[1]);
      std::vector<float> lhs_tmp(lhs_num_elem);
      std::vector<float> rhs_tmp(rhs_num_elem);
#pragma omp parallel for schedule(static, omp_schedule(lhs_num_elem))
      for (int i = 0; i < lhs_num_elem; i++) {
        lhs_tmp[i] = p.inputs[0][i] * multiplier_v->at(0);
      }
#pragma omp parallel for schedule(static, omp_schedule(rhs_num_elem))
      for (int i = 0; i < rhs_num_elem; i++) {
        rhs_tmp[i] = p.inputs[1][i] * multiplier_v->at(1);
      }

      auto binary = (Binary *)p.handle;
      (*binary)
          .lhs(lhs_tmp.data(), module::getShape(getInputs()[0]))
          .rhs(rhs_tmp.data(), module::getShape(getInputs()[1]))
          .run();

#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++) {
        auto &out = p.outputs[0][i];
        out = applyMultiplierAndRShift(out, 1, rshift_v->at(0));
        out = saturate(out, out_type);
      }
    } else {
      auto multiplier_v = module::getI64Array(getMultipliers(), 2, 1);
      auto rshift_v = module::getI64Array(getRshifts(), 2, 0);
      auto lhs_num_elem = module::getNumElements(getInputs()[0]);
      auto rhs_num_elem = module::getNumElements(getInputs()[1]);
      std::vector<float> lhs_tmp(lhs_num_elem);
      std::vector<float> rhs_tmp(rhs_num_elem);
#pragma omp parallel for schedule(static, omp_schedule(lhs_num_elem))
      for (int i = 0; i < lhs_num_elem; i++) {
        lhs_tmp[i] = applyMultiplierAndRShift(
            p.inputs[0][i], multiplier_v->at(0), rshift_v->at(0));
      }
#pragma omp parallel for schedule(static, omp_schedule(rhs_num_elem))
      for (int i = 0; i < rhs_num_elem; i++) {
        rhs_tmp[i] = applyMultiplierAndRShift(
            p.inputs[1][i], multiplier_v->at(1), rshift_v->at(1));
      }

      auto binary = (Binary *)p.handle;
      (*binary)
          .lhs(lhs_tmp.data(), module::getShape(getInputs()[0]))
          .rhs(rhs_tmp.data(), module::getShape(getInputs()[1]))
          .run();

#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++) {
        p.outputs[0][i] = saturate(p.outputs[0][i], out_type);
      }
    }
  } else {
    auto lhs_num_elem = module::getNumElements(getInputs()[0]);
    auto rhs_num_elem = module::getNumElements(getInputs()[1]);
    std::vector<float> lhs_tmp(lhs_num_elem);
    std::vector<float> rhs_tmp(rhs_num_elem);
    auto qtype = module::getUniformQuantizedType(getInputs()[0]);
#pragma omp parallel for schedule(static, omp_schedule(lhs_num_elem))
    for (int i = 0; i < lhs_num_elem; i++) {
      lhs_tmp[i] = (p.inputs[0][i] - (float)qtype.getZeroPoint()) *
                   (float)qtype.getScale();
    }
    qtype = module::getUniformQuantizedType(getInputs()[0]);
#pragma omp parallel for schedule(static, omp_schedule(rhs_num_elem))
    for (int i = 0; i < rhs_num_elem; i++) {
      rhs_tmp[i] = (p.inputs[1][i] - (float)qtype.getZeroPoint()) *
                   (float)qtype.getScale();
    }
    auto binary = (Binary *)p.handle;
    (*binary)
        .lhs(lhs_tmp.data(), module::getShape(getInputs()[0]))
        .rhs(rhs_tmp.data(), module::getShape(getInputs()[1]))
        .run();

    auto o_qtype = module::getUniformQuantizedType(getOutput());
    auto zp = o_qtype.getZeroPoint();
    auto scale = o_qtype.getScale();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int i = 0; i < num_elem; i++) {
      p.outputs[0][i] = p.outputs[0][i] * (float)(1.0 / scale) + zp;
      p.outputs[0][i] = saturate(p.outputs[0][i], out_type);
    }
    return success();
  }

  return success();
}

LogicalResult tpu::SubOp::LocalGenSupport() {
  if (module::isCV18xx()) {
    return failure();
  }
  return BroadCastBinaryLocalGenSupport(getOperation());
}

void tpu::SubOp::assign_sec_info(int64_t n_step, int64_t c_step, int64_t h_step,
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

void tpu::SubOp::assign_fw_param(void *param) {
  IR_PARAM_BROADCAST_BINARY(BINARY_SUB);
}
