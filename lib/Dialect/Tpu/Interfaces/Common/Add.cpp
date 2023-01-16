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
  auto in0_shape = module::getShape(getInputs()[0]);
  auto in1_shape = module::getShape(getInputs()[1]);
  int dims = std::max(in0_shape.size(), in1_shape.size());
  auto input0_shape = shape_expand_dim(in0_shape, dims);
  auto input1_shape = shape_expand_dim(in1_shape, dims);
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
      .lhs(p.inputs[0], input0_shape)
      .rhs(p.inputs[1], input1_shape)
      .dst(p.outputs[0], module::getShape(getOutput()))
      .do_relu(getDoRelu())
      .relu_limit(getReluLimit().convertToDouble())
      .algorithem(is_add ? algorithm::binary_add : algorithm::binary_sub)
      .setup();
  p.handle = (void *)binary;
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
        out = applyMultiplierAndRShift(out, 1, rshift_v->at(0), CVI_QUANT_NORMAL);
        out = out_type.isUnsignedInteger(8) ? to_uint8(out)
                                            : to_int8(out);
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
        auto &out = p.outputs[0][i];
        out = out_type.isUnsignedInteger(8) ? to_uint8(out)
                                            : to_int8(out);
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
      p.outputs[0][i] = out_type.isUnsignedInteger(8)
                            ? to_uint8(p.outputs[0][i])
                            : to_int8(p.outputs[0][i]);
    }
    return success();
  }

  return success();
}

LogicalResult tpu::AddOp::LocalGenSupport() {
  // BackwardH and BackwardN can not handle more than one input right now.
  // The same n_slice and h_slice value will propagate to each inputs.
  // Thus, the local layer is only safe when we do not need to slice n and h
  // dimensions.
  auto out_shape = module::getShape(getOutput());
  auto lhs_shape = module::getShape(getInputs()[0]);
  auto rhs_shape = module::getShape(getInputs()[1]);
  if (getOperand(1).getDefiningOp() &&
      isa<top::WeightOp>(getOperand(1).getDefiningOp()))
    return failure();
  // left align
  switch (out_shape.size()) {
  case 2:
    if (lhs_shape[0] != rhs_shape[0])
      return failure();
  case 3:
  case 4:
    if (lhs_shape[0] != rhs_shape[0])
      return failure();
    if (lhs_shape[2] != rhs_shape[2])
      return failure();
  default:
    success();
  }
  return success();
}
