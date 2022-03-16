#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Support/Helper/Module.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "float.h"
#include <map>
using namespace llvm;
using namespace mlir;
namespace sophgo {
namespace helper {
constexpr llvm::StringRef Module::Attr::NAME;
constexpr llvm::StringRef Module::Attr::STATE;
constexpr llvm::StringRef Module::Attr::CHIP;
constexpr llvm::StringRef Module::Attr::WEIGHT_FILE;

constexpr llvm::StringRef Module::State::TOP_F32;
constexpr llvm::StringRef Module::State::TOP_CALIBRATED;
constexpr llvm::StringRef Module::State::TOP_QUANTIZED;
constexpr llvm::StringRef Module::State::TPU_QUANTIZED;

ModuleOp Module::getModuleOp(Operation *op) {
  auto moduleOp = op->getParentOp();
  while (moduleOp && !isa<mlir::ModuleOp>(moduleOp)) {
    moduleOp = moduleOp->getParentOp();
  }
  if (!moduleOp) {
    op->dump();
    llvm_unreachable("can't get module op");
  }
  auto mOp = llvm::cast<ModuleOp>(moduleOp);
  return mOp;
}

StringRef Module::getWeightFile(ModuleOp module) {
  return module->getAttrOfType<StringAttr>(Attr::WEIGHT_FILE);
}

void Module::setWeightFile(ModuleOp module, StringRef weight_file) {
  module->setAttr(Attr::WEIGHT_FILE,
                  StringAttr::get(module.getContext(), weight_file));
  auto dialect = module->getContext()->getLoadedDialect("top");
  auto top_dialect = llvm::cast<top::TopDialect>(dialect);
  top_dialect->wFile->save(weight_file.str());
}

StringRef Module::getState(ModuleOp module) {
  return module->getAttrOfType<StringAttr>(Attr::STATE);
}

void Module::setState(ModuleOp module, StringRef state) {
  module->setAttr(Attr::STATE, StringAttr::get(module.getContext(), state));
}

bool Module::isState(ModuleOp module, llvm::StringRef state) {
  auto _state = getState(module);
  return _state == state;
}

StringRef Module::getChip(ModuleOp module) {
  return module->getAttrOfType<StringAttr>(Attr::CHIP);
}

void Module::setChip(ModuleOp module, StringRef chip) {
  module->setAttr(Attr::CHIP, StringAttr::get(module.getContext(), chip));
}
} // namespace help
} // namespace sophgo
