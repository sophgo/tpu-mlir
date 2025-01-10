//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LgPass.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"

namespace tpu_mlir {
namespace tpu {

// LgOptions LgPass::OPTIONS = {
//     .dyn_compile = false,
//     .opt = 0,
//     .group_by_cores = false,
//     .nnvlc_mode = NnvlcMode::NONE,
//     .lgcache = false,
//     .num_core = 0,
// };

void LgPassIR::clear() {
  lg_infos.clear();
  time_steps.clear();
  shape_secs.clear();
  ILP_time_steps.clear();
}

void LgPassManager::add_pass(std::unique_ptr<LgPass> pass) {
  passes.emplace_back(std::move(pass));
}

#define PASS_RUN(pass)                                                         \
  LAYER_GROUP_LOG_DEBUG_BLOCK({                                                \
    llvm::outs() << "==---------------------------==\n";                       \
    llvm::outs() << "Run " << pass->name() << " : \n";                         \
    llvm::outs() << "    " << pass->brief() << "\n";                           \
    llvm::outs() << "==---------------------------==\n";                       \
  });                                                                          \
  if (!pass->run(pass_ir)) {                                                   \
    LAYER_GROUP_LOG_DEBUG_BLOCK(llvm::outs()                                   \
                                    << pass->name().c_str() << " pass failed." \
                                    << "\n";);                                 \
  }

void LgPassManager::run(LgPassIR *pass_ir) {
  for (auto op : pass_ir->subnet_ops) {
    set_weight_allow_split_attr(op);
    generate_fake_global_addr(op);
    set_fake_local_layer_param(op, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
  }

  for (size_t i = 0; i < this->passes.size(); i++) {
    PASS_RUN(this->passes[i]);
  }

  for (auto op : pass_ir->subnet_ops) {
    delete_weight_allow_split_attr(op);
    delete_fake_global_addr(op);
    delete_fake_local_layer_param(op);
  }
}
#undef PASS_RUN

static inline LgOptimizerMap &get_lg_optimizers() {
  static LgOptimizerMap map;
  return map;
}

const LgOptimizerMap &get_registered_optimizers() {
  return get_lg_optimizers();
}

LgOptimizerReg::LgOptimizerReg(const std::string &name,
                               std::shared_ptr<LgOptimizer> optimizer) {
  if (get_lg_optimizers().find(name) != get_lg_optimizers().end()) {
    llvm::outs() << "Lg optimizer name \"" << name
                 << "\" had beed already regitered."
                 << "So we replace it by the newest one.\n";
  }
  get_lg_optimizers().emplace(name, optimizer.get());
}

} // namespace tpu
} // namespace tpu_mlir
