//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::ScatterNDOp::codegen_global_cv18xx(int64_t layer_id) {
  // parse param
  std::shared_ptr<std::vector<float>> indices;
  auto inputShape = module::getShape(getInputData());
  auto indicesShape = module::getShape(getIndices());
  if (auto indicesOp = cast<top::WeightOp>(getIndices().getDefiningOp())) {
    indices = indicesOp.read<float>();
  } else {
    llvm_unreachable("Not support activation indices.");
  }
  auto dims = indicesShape.back();
  if (dims > 4) {
    llvm_unreachable("ScatterND not support k > 4.");
  }
  std::vector<float> diffs;
  for (int i = 0; i < dims; i++) {
    auto _stride = std::accumulate(indicesShape.begin() + i, indicesShape.end(),
                                   1, std::multiplies<int64_t>()) /
                   indicesShape[i];
    float preValues;
    for (int j = 0; j < indicesShape[i]; j++) {
      auto curVaule = indices->at(j * _stride + i);
      if (j == 0) {
        if (indicesShape[i] == 1) {
          diffs.push_back(1);
        }
      } else if (j == 1) {
        diffs.push_back(curVaule - preValues);
      } else if (curVaule - preValues != diffs[i]) {
        llvm_unreachable("Not fully support ScatterNDOp now.");
      }
      preValues = curVaule;
    }
  }
  int64_t in, ic, ih, iw, un, uc, uh, uw;
  std::vector<int> o_stride(4, 0);
  module::getNCHW(getInputData(), in, ic, ih, iw);
  std::vector<int> i_shape = {(int)in, (int)ic, (int)ih, (int)iw};
  module::getNCHW(getUpdates(), un, uc, uh, uw);
  std::vector<int> u_shape = {(int)un, (int)uc, (int)uh, (int)uw};

  for (int i = diffs.size(); i < 4; i++) {
    diffs.push_back(1);
  }
  o_stride[3] = diffs[3] * 1;
  o_stride[2] = diffs[2] * iw;
  o_stride[1] = diffs[1] * ih * iw;
  o_stride[0] = diffs[0] * ic * ih * iw;

  std::vector<int> start_idxs;
  for (int i = 0; i < dims; i++) {
    start_idxs.push_back(indices->at(i));
  }
  for (int i = start_idxs.size(); i < 4; i++) {
    diffs.push_back(0);
  }

  uint32_t offset = 0;
  for (int i = 0; i < dims; i++) {
    uint32_t _stride = 1;
    for (int j = i + 1; j < inputShape.size(); j++) {
      _stride *= inputShape[j];
    }
    offset += indices->at(i) * _stride;
  }

  // codegen
  gaddr_t input_gaddr = module::getAddress(getInputData());
  gaddr_t updates_gaddr = module::getAddress(getUpdates());
  gaddr_t output_gaddr = module::getAddress(getOutput());
  auto fmt =
      module::isUniformQuantized(getOutput()) ? CVK_FMT_I8 : CVK_FMT_BF16;
  cvi_backend_tg_scatterND_kernel(input_gaddr, updates_gaddr, output_gaddr,
                                  i_shape, u_shape, o_stride, offset, fmt);
}
