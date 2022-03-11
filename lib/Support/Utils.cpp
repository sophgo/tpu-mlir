#include "sophgo/Support/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include <map>
using namespace llvm;
namespace mlir {

StringRef getMlirWeightFile(ModuleOp module) {
  return module->getAttrOfType<StringAttr>("mlir.weight_file");
}

StringRef getMlirState(ModuleOp module) {
  return module->getAttrOfType<StringAttr>("mlir.state");
}

void setMlirState(mlir::ModuleOp module, StringRef state) {
  module->setAttr("mlir.state", StringAttr::get(module.getContext(), state));
}

llvm::StringRef getMlirChip(mlir::ModuleOp module) {
  return module->getAttrOfType<StringAttr>("mlir.chip");
}
void setMlirChip(mlir::ModuleOp module, StringRef chip) {
  module->setAttr("mlir.chip", StringAttr::get(module.getContext(), chip));
}

}
