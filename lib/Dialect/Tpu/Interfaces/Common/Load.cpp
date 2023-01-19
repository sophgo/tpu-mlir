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

#include "tpu_mlir/Support/Module.h"



LogicalResult tpu::LoadOp::init(InferenceParameter &p) { return success(); }
void tpu::LoadOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::LoadOp::inference(InferenceParameter &p) {
  llvm_unreachable("Not Implemented");
  return success();
}

void tpu::LoadOp::assign_sec_info(int64_t n_step, int64_t h_step,
                                  local_sec_info_t &sec_info) {
  memset(&sec_info, 0, sizeof(local_sec_info_t));

  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  auto gi = getGroupInfo(n_step, h_step);
  sec_info.n_slice = gi.n_slice;
  sec_info.d_slice = 1;
  sec_info.h_slice = gi.h_slice;
  sec_info.h_idx = gi.h_idx;
  sec_info.is_h_split = !(gi.h_idx == 0 && gi.h_slice == h);
  sec_info.w_slice = w;
  sec_info.out_n_slice = gi.n_slice;
  sec_info.out_h_idx = gi.h_idx;
  sec_info.out_h_slice = gi.h_slice;
  sec_info.out_w_slice = w;
}
