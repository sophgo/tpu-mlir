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
#include "tpu_mlir/Support/Float8.h"

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Interfaces/IndexingMapsInterface.h"

LogicalResult tpu::AddOp::init(InferenceParameter &p) {
  if (module::isCV18xx() && getInputs().size() != 2) {
    p.handle = nullptr;
  } else {
    p.handle = nullptr;
    // auto in0_shape = module::getShape(getInputs()[0]);
    // auto in1_shape = module::getShape(getInputs()[1]);
    // auto binary = new Binary();
    // // fix me. naive impletment.
    // // It should be o = alpha * i0 + beta * i1
    // auto coeff_ = module::getF64Array(getCoeff(), 2, 1);
    // bool is_add = true;
    // if (module::getStorageType(getOutput()).isa<FloatType>()) {
    //   if (coeff_->at(0) == 1 && coeff_->at(1) == -1) {
    //     is_add = false;
    //   }
    // }

    // (*binary)
    //     .hs(p.inputs[0], p.inputs[1], in0_shape, in1_shape)
    //     .dst(p.outputs[0], module::getShape(getOutput()))
    //     .do_relu(getDoRelu())
    //     .relu_limit(getReluLimit().convertToDouble())
    //     .algorithem(is_add ? algorithm::binary_add : algorithm::binary_sub)
    //     .setup();
    // p.handle = (void *)binary;
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
  auto output_shape = computer_broadcast_shape(getOperation());
  module::setShape(getOutput(), output_shape);
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

  auto num_elem = module::getNumElements(getOutput());
  auto out_type = module::getStorageType(getOutput());
  memset(p.outputs[0], 0, num_elem * sizeof(float));
  auto asym = module::isAsymmetric();
  if (out_type.isa<FloatType>()) {
    if (out_type.isa<Float8E4M3FNType>()) {
      auto scales = module::getF64Array(getF8Scales(), 2, 1.);
      scales->at(0) =
          F16(scales->at(0), false); // should be true ? align to kernel
      scales->at(1) = F16(scales->at(1), false);
      auto lhs_num_elem = module::getNumElements(getInputs()[0]);
      auto rhs_num_elem = module::getNumElements(getInputs()[1]);
      std::vector<float> lhs_tmp(lhs_num_elem);
      std::vector<float> rhs_tmp(rhs_num_elem);
#pragma omp parallel for schedule(static, omp_schedule(lhs_num_elem))
      for (int i = 0; i < lhs_num_elem; i++)
        lhs_tmp[i] = p.inputs[0][i] * scales->at(0);
#pragma omp parallel for schedule(static, omp_schedule(rhs_num_elem))
      for (int i = 0; i < rhs_num_elem; i++)
        rhs_tmp[i] = p.inputs[1][i] * scales->at(1);

#pragma omp parallel for schedule(static, omp_schedule(lhs_num_elem))
      for (int i = 0; i < lhs_num_elem; i++)
        lhs_tmp[i] = F16(lhs_tmp[i], false);
#pragma omp parallel for schedule(static, omp_schedule(rhs_num_elem))
      for (int i = 0; i < rhs_num_elem; i++)
        rhs_tmp[i] = F16(rhs_tmp[i], false);

      auto binary = (Binary *)p.handle;
      (*binary)
          .lhs(lhs_tmp.data(), module::getShape(getInputs()[0]))
          .rhs(rhs_tmp.data(), module::getShape(getInputs()[1]))
          .run();

#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++)
        p.outputs[0][i] = F16(p.outputs[0][i], false);
      F8E4M3(p.outputs[0], p.outputs[0], num_elem, 1.0, true);
    } else {
      auto binary = (Binary *)p.handle;
      binary->run();
      if (out_type.isBF16()) {
        BF16(p.outputs[0], p.outputs[0], num_elem);
      } else if (out_type.isF16()) {
        F16(p.outputs[0], p.outputs[0], num_elem);
      } else if (out_type.isFloat8E5M2()) {
        F8E5M2(p.outputs[0], p.outputs[0], num_elem, 1.0, true);
      }
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
    auto multiplier_v = module::getI64Array(getMultipliers(), 2, 1);
    auto rshift_v = module::getI64Array(getRshifts(), 2, 0);
    auto o_qtype = module::getUniformQuantizedType(getOutput());
    auto lhs_num_elem = module::getNumElements(getInputs()[0]);
    auto rhs_num_elem = module::getNumElements(getInputs()[1]);
    std::vector<float> lhs_tmp(lhs_num_elem);
    std::vector<float> rhs_tmp(rhs_num_elem);
    if (isa<top::WeightOp>(getInputs()[0].getDefiningOp())) {
#pragma omp parallel for schedule(static, omp_schedule(lhs_num_elem))
      for (int i = 0; i < lhs_num_elem; i++) {
        lhs_tmp[i] = applyMultiplierAndRShift(
            p.inputs[0][i], multiplier_v->at(0), rshift_v->at(0));
      }
    } else {
      auto l_qtype = module::getUniformQuantizedType(getInputs()[0]);
#pragma omp parallel for schedule(static, omp_schedule(lhs_num_elem))
      for (int i = 0; i < lhs_num_elem; i++) {
        lhs_tmp[i] =
            applyMultiplierAndRShift(p.inputs[0][i] - l_qtype.getZeroPoint(),
                                     multiplier_v->at(0), rshift_v->at(0));
      }
    }
    if (isa<top::WeightOp>(getInputs()[1].getDefiningOp())) {
#pragma omp parallel for schedule(static, omp_schedule(rhs_num_elem))
      for (int i = 0; i < rhs_num_elem; i++) {
        rhs_tmp[i] = applyMultiplierAndRShift(
            p.inputs[1][i], multiplier_v->at(1), rshift_v->at(1));
      }
    } else {
      auto r_qtype = module::getUniformQuantizedType(getInputs()[1]);
#pragma omp parallel for schedule(static, omp_schedule(rhs_num_elem))
      for (int i = 0; i < rhs_num_elem; i++) {
        rhs_tmp[i] =
            applyMultiplierAndRShift(p.inputs[1][i] - r_qtype.getZeroPoint(),
                                     multiplier_v->at(1), rshift_v->at(1));
      }
    }
    auto binary = (Binary *)p.handle;
    (*binary)
        .lhs(lhs_tmp.data(), module::getShape(getInputs()[0]))
        .rhs(rhs_tmp.data(), module::getShape(getInputs()[1]))
        .run();

#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int i = 0; i < num_elem; i++) {
      p.outputs[0][i] =
          saturate(p.outputs[0][i] + o_qtype.getZeroPoint(), out_type);
    }
  }

  return success();
}

LogicalResult tpu::AddOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                    int64_t out_idx, int64_t out_slice) {

  in_idx = out_idx;
  in_slice = out_slice;
  auto do_early_stride = getDoEarlyStride();
  if (do_early_stride.has_value() && do_early_stride.value()) {
    auto h_stride = getEarlyStrideH().value();
    in_idx = out_idx * h_stride;
    in_slice = out_slice * h_stride;
  }
  return success();
}

LogicalResult tpu::AddOp::BackwardW(int64_t &in_idx, int64_t &in_slice,
                                    int64_t out_idx, int64_t out_slice) {

  in_idx = out_idx;
  in_slice = out_slice;
  auto do_early_stride = getDoEarlyStride();
  if (do_early_stride.has_value() && do_early_stride.value()) {
    auto w_stride = getEarlyStrideW().value();
    in_idx = out_idx * w_stride;
    in_slice = out_slice * w_stride;
  }
  return success();
}

LogicalResult tpu::AddOp::LocalGenSupport() {
  // for onnx AddConst
  if (getNumOperands() == 1)
    return success();
  return BroadCastBinaryLocalGenSupport(getOperation());
}

void tpu::AddOp::DumpQuantAgnosticAttrs(llvm::raw_string_ostream &os) {
  for (auto attr : getOperation()->getAttrs()) {
    auto attr_name = attr.getName().str();
    if (attr_name == "ginfo" || attr_name == "rshifts" ||
        attr_name == "multipliers") {
      continue;
    }
    os << attr_name << "=";
    attr.getValue().print(os);
    os << "; ";
  }

  if (getRshifts().has_value()) {
    auto rshift_v = module::getI64Array(getRshifts().value());
    if (std::all_of(rshift_v->begin(), rshift_v->end(),
                    [](int64_t x) { return x == 0; })) {
      // do-nothing.
    } else {
      os << "rshifts_len=" << rshift_v->size() << "; ";
    }
  }
  if (getMultipliers().has_value()) {
    auto multiplier_v = module::getI64Array(getMultipliers().value());
    if (std::all_of(multiplier_v->begin(), multiplier_v->end(),
                    [](int64_t x) { return x == 1; })) {
      // do-nothing.
    } else {
      os << "multipliers_len=" << multiplier_v->size() << "; ";
    }
  }
}

void tpu::AddOp::assign_sec_info(int64_t n_step, int64_t c_step, int64_t h_step,
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
  sec_info.h_slice = gi.h_slice;
  // set "w_slice" for 5d broadcast (i.e., broadcast across both h and w dims)
  sec_info.w_slice = std::max(in0_gi.w_slice, in1_gi.w_slice) >= gi.w_slice
                         ? gi.w_slice
                         : std::max(in0_gi.w_slice, in1_gi.w_slice);
  setHWMargins(sec_info.hw_margins_opdA, in0_gi, gi);
  setHWMargins(sec_info.hw_margins_opdB, in1_gi, gi);
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

void tpu::AddOp::assign_fw_param(void *param) {
  IR_PARAM_BROADCAST_BINARY(BINARY_ADD);
}

ArrayAttr tpu::AddOp::getIndexingMaps() {
  return getBinaryIndexingMaps(getOperation());
};

bool tpu::AddOp::support_multi_core() { return false; }
