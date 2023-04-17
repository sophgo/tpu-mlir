//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"
#include "mlir/IR/AsmState.h"

using namespace llvm;
using namespace mlir;
namespace tpu_mlir {
namespace tpu {

class ProfileCtx {
public:
  ProfileCtx() {}
  ProfileCtx(Operation *moduleOp, bool enable_profile);
  void log_str(const char *fmt, ...);
  void log_tensor(Value value, bool is_in, int64_t n_step=0, int64_t h_step=0);
  void log_local_layer(Operation *op, int64_t n_step, int64_t h_step);
  void log_global_layer(Operation *op);

  bool get_enable_profile() { return enable_profile_; };
  void set_enable_profile(bool enable_profile) {
    enable_profile_ = enable_profile;
  };

  int32_t get_cur_net_idx() { return cur_net_idx; }

  void set_profile_start();
  void set_profile_end();
  FILE *get_fp_profile() { return fp_profile[cur_net_idx]; }

private:
  int32_t net_num;
  int32_t cur_net_idx;
  bool enable_profile_;
  std::vector<FILE *> fp_profile;
  AsmState::LocationMap opToLineCol;
};

} // namespace tpu
} // namespace tpu_mlir
