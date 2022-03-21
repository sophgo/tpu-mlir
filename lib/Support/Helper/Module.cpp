#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Support/Helper/Module.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
constexpr llvm::StringRef Module::State::TPU_WEIGHT_REORDERD;

constexpr llvm::StringRef Module::Chip::ALL;
constexpr llvm::StringRef Module::Chip::BM1684;
constexpr llvm::StringRef Module::Chip::BM1686;

top::NoneOp Module::getNoneOp(Operation *op) {
  assert(op != nullptr);
  if (auto noneOp = dyn_cast<top::NoneOp>(op)) {
    return noneOp;
  }
  auto funcOp = cast<mlir::FuncOp>(op->getParentOp());
  auto &block = funcOp.front();
  auto &topOp = block.front();
  if (auto noneOp = dyn_cast<top::NoneOp>(topOp)) {
    return noneOp;
  }
  auto ctx = op->getContext();
  auto builder = OpBuilder(ctx);
  builder.setInsertionPointToStart(&block);
  auto NoneOp =
      builder.create<top::NoneOp>(op->getLoc(), builder.getNoneType());
  return NoneOp;
}

ModuleOp Module::getModuleOp(Operation *op) {
  assert(op != nullptr);
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

void Module::updateModuleTypes(ModuleOp module) {
  auto ctx = module.getContext();
  Builder builder(ctx);
  // sync ReturnOp type to function type
  for (auto func : module.getOps<FuncOp>()) {
    // alter the function type to match the real type
    // of InputOp and ReturnOp
    std::vector<mlir::Type> arguments;
    std::vector<mlir::Type> returns;
    Block &entryBlock = func.front();
    auto returnOp = dyn_cast<func::ReturnOp>(entryBlock.back()).getOperation();
    for (uint32_t i = 0; i < entryBlock.getNumArguments(); ++i) {
      arguments.push_back(entryBlock.getArgument(i).getType());
    }
    for (uint32_t i = 0; i < returnOp->getNumOperands(); ++i) {
      returns.push_back(returnOp->getOperand(i).getType());
    }
    auto fnType = builder.getFunctionType(llvm::ArrayRef<mlir::Type>{arguments},
                                          llvm::ArrayRef<mlir::Type>{returns});
    func.setType(fnType);
  }
}

std::string Module::genWeightFileName(ModuleOp module) {
  auto name = getName(module);
  auto state = getState(module);
  auto chip = getChip(module);
  std::string weight_file_name = name.lower() + std::string("_") +
                                 state.lower() + std::string("_") +
                                 chip.lower() + "_weight.npz";
  return weight_file_name;
}

} // namespace helper
} // namespace sophgo
