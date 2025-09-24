//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupProfile.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"

namespace tpu_mlir {
namespace tpu {

LayerGroupProfile::LayerGroupProfile(const LgOptions &options) {
  options_ = options;
  cycle_calculator_ = CycleCalculatorFactory::create(options.num_core);

  old_num_core_ = module::getCoreNum();
  module::setCoreNum(options_.num_core);
}

bool LayerGroupProfile::process(std::vector<LgInfo> &lg_infos,
                                std::vector<shape_secs_t> &shape_secs,
                                std::vector<BasicTimeStepPtr> &time_steps) {
  for (size_t i = 0; i < lg_infos.size(); ++i) {
    auto runmode = getRunMode(lg_infos[i].group_ops[0]);
    if (runmode != RunMode::TPU_STATIC) {
      continue;
    }
    if (lg_infos[i].group_ops.size() == 1) {
      analyzeGlobalOp(lg_infos[i].group_ops[0]);
    } else {
      analyzeGroupOp(time_steps[i], shape_secs[i], lg_infos[i].type);
    }
  }
  return true;
}

void LayerGroupProfile::analyzeGlobalOp(Operation *op) {
  auto time_info = cycle_calculator_->getGlobalLayerCycle(op);
  perf_.time_info += time_info;
}

void LayerGroupProfile::analyzeGroupOp(BasicTimeStepPtr &time_step,
                                       shape_secs_t &shape_secs,
                                       group_type_t group_type) {
  auto time_info =
      cycle_calculator_->getGroupCycle(time_step, shape_secs, group_type);
  perf_.time_info += time_info;
}

class LayerGroupProfilePass : public LgPass {
public:
  LayerGroupProfilePass(const LgOptions &options) { options_ = options; }
  virtual bool run(LgPassIR *pass_ir) override {
    auto profile = LayerGroupProfile(options_);
    profile.process(pass_ir->lg_infos, pass_ir->shape_secs,
                    pass_ir->time_steps);
    pass_ir->perf = profile.getLayerGroupPerf();
    return true;
  }

  virtual std::string name() override { return "LayerGroupProfilePass"; }
  virtual std::string brief() override {
    return "Get the Performance Info for each group.";
  }
};

std::unique_ptr<LgPass> CreateLayerGroupProfilePass(const LgOptions &options) {
  return std::unique_ptr<LgPass>(new LayerGroupProfilePass(options));
}

} // namespace tpu
} // namespace tpu_mlir
