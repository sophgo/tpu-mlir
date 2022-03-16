#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Support/ModuleHelper.h"
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

ModuleOp getModuleOp(Operation *op) {
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

StringRef getMlirWeightFile(ModuleOp module) {
  return module->getAttrOfType<StringAttr>(kWeightFileAttrName);
}

void setMlirWeightFile(ModuleOp module, StringRef weight_file) {
  module->setAttr(kWeightFileAttrName,
                  StringAttr::get(module.getContext(), weight_file));
  auto dialect = module->getContext()->getLoadedDialect("top");
  auto top_dialect = llvm::cast<top::TopDialect>(dialect);
  top_dialect->wFile->save(weight_file.str());
}

StringRef getMlirState(ModuleOp module) {
  return module->getAttrOfType<StringAttr>(kStateAttrName);
}

void setMlirState(ModuleOp module, StringRef state) {
  module->setAttr(kStateAttrName, StringAttr::get(module.getContext(), state));
}

StringRef getMlirChip(ModuleOp module) {
  return module->getAttrOfType<StringAttr>(kChipAttrName);
}

void setMlirChip(ModuleOp module, StringRef chip) {
  module->setAttr(kChipAttrName, StringAttr::get(module.getContext(), chip));
}
} // namespace sophgo
