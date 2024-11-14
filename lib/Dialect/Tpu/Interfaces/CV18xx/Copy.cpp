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

void tpu::CopyOp::codegen_global_cv18xx(int64_t layer_id) {

  gaddr_t input_gaddr = module::getAddress(this->getInput());
  gaddr_t output_gaddr = module::getAddress(this->getOutput());
  // parseparam
  auto shape = module::getI64Array(this->getShape());
  auto i_stride = module::getI64Array(this->getInputStride());
  auto o_stride = module::getI64Array(this->getOutputStride());
  std::vector<int32_t> shape_4;
  std::vector<int32_t> i_stride_4;
  std::vector<int32_t> o_stride_4;
  shape_4 = {1, 1, 1, 1};
  i_stride_4 = {0, 0, 0, 0};
  o_stride_4 = {0, 0, 0, 0};
  int num_dims = shape->size();
  assert(num_dims <= 4);
  assert(i_stride->size() == shape->size());
  assert(o_stride->size() == shape->size());
  for (int end = num_dims - 1, idx = 3; end >= 0 && idx >= 0; end--, idx--) {
    shape_4[idx] = shape->at(end);
    i_stride_4[idx] = i_stride->at(end);
    o_stride_4[idx] = o_stride->at(end);
  }
  if (module::isUniformQuantized(getOutput())) {
    cvi_backend_tg_copy_kernel(input_gaddr, output_gaddr, shape_4, i_stride_4,
                               o_stride_4, CVK_FMT_I8);
  } else {
    cvi_backend_tg_copy_kernel(input_gaddr, output_gaddr, shape_4, i_stride_4,
                               o_stride_4, CVK_FMT_BF16);
  }
}
