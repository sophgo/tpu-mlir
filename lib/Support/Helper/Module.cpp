#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Support/Helper/Module.h"
#include "sophgo/Support/Helper/Quant.h"
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

int64_t Module::getAddress(Value v) {
  auto op = v.getDefiningOp();
  if (!op->hasAttr("addr")) {
    v.dump();
    llvm_unreachable("Value has no addr attribute");
  }
  return op->getAttr("addr").cast<IntegerAttr>().getSInt();
}

size_t Module::getBytes(Value v) {
  auto type = v.getType().cast<RankedTensorType>();
  auto elm_count = type.getNumElements();
  auto etype = type.getElementType();
  int elm_bytes = 0;
  if (auto qType = etype.dyn_cast<quant::CalibratedQuantizedType>()) {
    elm_bytes = qType.getExpressedType().getIntOrFloatBitWidth() / 8;
  } else if (auto qType = etype.dyn_cast<quant::UniformQuantizedType>()) {
    elm_bytes = qType.getStorageType().getIntOrFloatBitWidth() / 8;
  } else {
    elm_bytes = etype.getIntOrFloatBitWidth() / 8;
  }
  return elm_count * elm_bytes;
}

static void getNCHW_align_right(llvm::ArrayRef<int64_t> &shape, int64_t &n,
                                int64_t &c, int64_t &h, int64_t &w) {
  int num_dims = shape.size();
  n = 1, c = 1, h = 1, w = 1;
  if (num_dims > 0) {
    w = shape[num_dims - 1];
  }
  if (num_dims > 1) {
    h = shape[num_dims - 2];
  }
  if (num_dims > 2) {
    c = shape[num_dims - 3];
  }
  if (num_dims > 3) {
    n = shape[num_dims - 4];
  }
  for (int i = 4; i < num_dims; i++) {
    n *= shape[num_dims - i - 1];
  }
}

static void getNCHW_align_left(llvm::ArrayRef<int64_t> shape, int64_t &n,
                               int64_t &c, int64_t &h, int64_t &w) {
  int num_dims = shape.size();
  n = 1, c = 1, h = 1, w = 1;
  if (num_dims >= 4) {
    n = std::accumulate(shape.begin(), shape.begin() + num_dims - 3, 1,
                        std::multiplies<int64_t>());
    c = shape[num_dims - 3];
    h = shape[num_dims - 2];
    w = shape[num_dims - 1];
  } else if (num_dims == 3) {
    n = shape[num_dims - 3];
    c = shape[num_dims - 2];
    h = shape[num_dims - 1];
  } else if (num_dims == 2) {
    n = shape[num_dims - 2];
    c = shape[num_dims - 1];
  } else if (num_dims == 1) {
    n = shape[num_dims - 1];
  } else if (num_dims == 0) {
    // scalar
  } else {
    llvm_unreachable("unsupported shape size");
  }
}

void Module::getNCHW(Value v, int64_t &n, int64_t &c, int64_t &h, int64_t &w,
                     bool align_left) {
  auto shape = v.getType().cast<RankedTensorType>().getShape();
  if (align_left) {
    getNCHW_align_left(shape, n, c, h, w);
  } else {
    getNCHW_align_right(shape, n, c, h, w);
  }
}

} // namespace helper
} // namespace sophgo
