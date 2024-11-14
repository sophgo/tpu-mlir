//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Interfaces/IndexingMapsInterface.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"

LogicalResult tpu::PReluOp::init(InferenceParameter &p) {
  auto w_shape = module::getShape(getSlope());
  auto weight_shape =
      channel_expand_dim(w_shape, module::getShape(getInput()).size());
  auto prelu = new PRelu();
  (*prelu)
      .src(p.inputs[0], module::getShape(getInput()))
      .weights(p.inputs[1], weight_shape)
      .dst(p.outputs[0], module::getShape(getOutput()))
      .setup();
  p.handle = (void *)prelu;
  return success();
}
void tpu::PReluOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto prelu = (PRelu *)p.handle;
    delete prelu;
    p.handle = nullptr;
  }
}

LogicalResult tpu::PReluOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }

  auto num_elem = module::getNumElements(getOutput());
  auto out_type = module::getStorageType(getOutput());
  bool asym = module::isAsymmetric();
  bool is_cv18xx = module::isCV18xx();
  if (out_type.isa<FloatType>()) {
    auto prelu = (PRelu *)p.handle;
    prelu->run();
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else if (asym == false) {
    auto shift = getRshift();
    int64_t shift_pos, multiplier_pos;
    if (is_cv18xx) {
      shift_pos = getRshiftPos().value();
      multiplier_pos = getMultiplierPos().value();
    }
    auto num_slope = module::getNumElements(getSlope());
    auto in_shape = module::getShape(getInput());
    int64_t num_inner = 1;
    int64_t num_outer = 1;
    if (in_shape.size() > 1) {
      num_outer = std::accumulate(in_shape.begin(), in_shape.begin() + 2, 1,
                                  std::multiplies<int64_t>());
      num_inner = std::accumulate(in_shape.begin() + 2, in_shape.end(), 1,
                                  std::multiplies<int64_t>());
    } else {
      num_outer = in_shape[0];
      num_inner = 1;
    }
#pragma omp parallel for schedule(static, omp_schedule(num_outer))
    for (int64_t i = 0; i < num_outer; i++) {
      int idx_slope = i % num_slope;
      int8_t slopei = p.inputs[1][idx_slope];
      for (int64_t j = 0; j < num_inner; j++) {
        int64_t idx = i * num_inner + j;
        if (is_cv18xx) {
          int64_t v;
          if (p.inputs[0][idx] < 0) {
            v = applyMultiplierAndRShift(p.inputs[0][idx], slopei, shift);
          } else {
            v = applyMultiplierAndRShift(p.inputs[0][idx], multiplier_pos,
                                         shift_pos);
          }
          p.outputs[0][idx] = saturate(v, out_type);
        } else {
          if (p.inputs[0][idx] < 0) {
            auto v = applyMultiplierAndRShift(p.inputs[0][idx], slopei, shift);
            p.outputs[0][idx] = saturate(v, out_type);
          } else {
            p.outputs[0][idx] = p.inputs[0][idx];
          }
        }
      }
    }
  } else {
    llvm_unreachable("Unknown type");
  }

  return success();
}

ArrayAttr tpu::PReluOp::getIndexingMaps() {
  return getBinaryIndexingMaps(getOperation());
};

bool tpu::PReluOp::support_multi_core() { return false; }
