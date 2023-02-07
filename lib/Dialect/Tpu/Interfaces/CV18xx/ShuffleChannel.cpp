//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

// #include "tpu_mlir/Backend/BM168x/cv18xx.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Support/Module.h"




using namespace tpu_mlir::backend;

void tpu::ShuffleChannelOp::codegen_global_cv18xx( int64_t layer_id) {
  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  std::vector<int64_t> input_shape;
  module::getShapeVec(getInput(), input_shape);
  int64_t n, c, h, w;
  module::getNCHW(input_shape, n, c, h, w, false);
  int64_t group = this->getGroup();
  if (module::isUniformQuantized(getOutput())) {
    cvi_backend_tg_permute_kernel( layer_id, ga_input, ga_output,
          n, group, c / group, h * w,
          0, 2, 1, 3, CVK_FMT_I8);
  } else {
    cvi_backend_tg_permute_kernel( layer_id, ga_input, ga_output,
          n, group, c / group, h * w,
          0, 2, 1, 3, CVK_FMT_BF16);
  }
}
