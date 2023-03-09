//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::AddOp::init(InferenceParameter &p) {
  if (module::isCV18xx() && getInputs().size() != 2) {
    p.handle = nullptr;
  } else {
    auto in0_shape = module::getShape(getInputs()[0]);
    auto in1_shape = module::getShape(getInputs()[1]);
    auto binary = new Binary();
    // fix me. naive impletment.
    // It should be o = alpha * i0 + beta * i1
    auto coeff_ = module::getF64Array(getCoeff(), 2, 1);
    bool is_add = true;
    if (module::getStorageType(getOutput()).isa<FloatType>()) {
      if (coeff_->at(0) == 1 && coeff_->at(1) == -1) {
        is_add = false;
      }
    }

    (*binary)
        .hs(p.inputs[0], p.inputs[1], in0_shape, in1_shape)
        .dst(p.outputs[0], module::getShape(getOutput()))
        .do_relu(getDoRelu())
        .relu_limit(getReluLimit().convertToDouble())
        .algorithem(is_add ? algorithm::binary_add : algorithm::binary_sub)
        .setup();
    p.handle = (void *)binary;
  }
  return success();
}

void tpu::AddOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto binary = (Binary *)p.handle;
    delete binary;
    p.handle = nullptr;
  }
}

LogicalResult tpu::AddOp::inference(InferenceParameter &p) {
  auto num_elem = module::getNumElements(getOutput());
  auto out_type = module::getStorageType(getOutput());
  memset(p.outputs[0], 0, num_elem * sizeof(float));
  auto asym = module::isAsymmetric();
  if (out_type.isa<FloatType>()) {
    auto binary = (Binary *)p.handle;
    binary->run();
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else if (out_type.isInteger(32)) {
    // auto in0 = reinterpret_cast<int32_t*>(p.inputs[0]);
    // auto in1 = reinterpret_cast<int32_t*>(p.inputs[1]);
    // auto out = reinterpret_cast<int32_t*>(p.outputs[0]);
    auto binary = (Binary *)p.handle;
    binary->run();
  } else if (asym == false) {
    if (module::isCV18xx()) {
      // cv18xx interpreter
      auto multiplier_v =
          module::getI64Array(getMultipliers(), getInputs().size(), 1);
      auto rshift_v = module::getI64Array(getRshifts(), 1, 0);
      auto ninputs = getInputs().size();
      if (getInputs().size() == 2) {
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
      } else {
#pragma omp parallel for schedule(static, omp_schedule(ninputs))
        for (int i = 0; i < ninputs; ++i) {
          for (int j = 0; j < num_elem; ++j) {
            p.outputs[0][j] += p.inputs[i][j] * multiplier_v->at(i);
          }
        }
      }
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

LogicalResult tpu::AddOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                    int64_t out_idx, int64_t out_slice) {

  in_idx = out_idx;
  in_slice = out_slice;
  auto do_early_stride = getDoEarlyStride();
  if(do_early_stride.has_value() && do_early_stride.value()) {
    auto h_stride = getEarlyStrideH().value();
    in_idx = out_idx * h_stride;
    in_slice = out_slice * h_stride;
  }
  return success();
}

LogicalResult tpu::AddOp::LocalGenSupport() {
  return BroadCastBinaryLocalGenSupport(getOperation());
}

void tpu::AddOp::assign_sec_info(int64_t n_step, int64_t h_step,
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
  auto gi = getGroupInfo(n_step, h_step);
  auto in0_gi = LocalGenInterface::getGroupInfo(input0, n_step, h_step);
  auto in1_gi = LocalGenInterface::getGroupInfo(input1, n_step, h_step);
  sec_info.n_slice = std::max(in0_gi.n_slice, in1_gi.n_slice);
  sec_info.d_slice = std::max(d0, d1);
  sec_info.h_slice = std::max(in0_gi.h_slice, in1_gi.h_slice);
  sec_info.h_idx = std::max(in0_gi.h_idx, in1_gi.h_idx);
  sec_info.is_h_split = !(std::max(in0_gi.h_idx, in1_gi.h_idx) == 0 && std::max(in0_gi.h_slice, in1_gi.h_slice) == std::max(h0, h1));
  sec_info.w_slice = std::max(w0, w1);
  sec_info.out_n_slice = gi.n_slice;
  sec_info.out_h_idx = gi.h_idx;
  sec_info.out_h_slice = gi.h_slice;
  sec_info.out_w_slice = ow;
}
