#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LgPass.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"

namespace tpu_mlir {
namespace tpu {

void LgPassIR::clear() {
  lg_infos.clear();
  time_steps.clear();
  shape_secs.clear();
}

void LgPassManager::add_pass(std::unique_ptr<LgPass> pass) {
  passes.emplace_back(std::move(pass));
}

#define PASS_RUN(pass)                                                         \
  llvm::errs() << "==---------------------------==\n";                         \
  llvm::errs() << "Run " << pass->name() << " : \n";                           \
  llvm::errs() << "    " << pass->brief() << "\n";                             \
  llvm::errs() << "==---------------------------==\n";                         \
  if (!pass->run(pass_ir)) {                                                   \
    llvm::errs() << pass->name().c_str() << " pass failed."                    \
                 << "\n";                                                      \
  }

void LgPassManager::run(LgPassIR *pass_ir) {
  for (auto op : pass_ir->subnet_ops) {
    generate_fake_global_addr(op);
    set_fake_local_layer_param(op, 0, 1, 0, 1);
  }

  for (size_t i = 0; i < this->passes.size(); i++) {
    PASS_RUN(this->passes[i]);
  }

  for (auto op : pass_ir->subnet_ops) {
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
    llvm::errs() << "Lg optimizer name \"" << name
                 << "\" had beed already regitered."
                 << "So we replace it by the newest one.\n";
  }
  get_lg_optimizers().emplace(name, optimizer.get());
}

} // namespace tpu
} // namespace tpu_mlir
