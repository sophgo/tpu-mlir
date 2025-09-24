//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Support/LLVM.h"
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"
#include <list>
#include <map>
#include <set>

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/CycleCalculator.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupDefs.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LgPass.h"

namespace tpu_mlir {
namespace tpu {

class LayerGroupProfile {
public:
  LayerGroupProfile(const LgOptions &options);
  ~LayerGroupProfile() { module::setCoreNum(old_num_core_); }

  bool process(LgPassIR *pass_ir);

  bool process(std::vector<LgInfo> &lg_infos,
               std::vector<shape_secs_t> &shape_secs,
               std::vector<BasicTimeStepPtr> &time_steps);

  void analyzeGlobalOp(Operation *op);
  void analyzeGroupOp(BasicTimeStepPtr &time_step, shape_secs_t &shape_secs,
                      group_type_t group_type);

  LayerGroupPerf getLayerGroupPerf() const { return perf_; }

private:
  LgOptions options_;
  LayerGroupPerf perf_;
  std::unique_ptr<CycleCalculator> cycle_calculator_;
  int64_t old_num_core_;
};

std::unique_ptr<LgPass> CreateLayerGroupProfilePass(const LgOptions &options);

} // namespace tpu
} // namespace tpu_mlir
