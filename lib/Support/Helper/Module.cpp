//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Helper/Module.h"
#include "float.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include <map>
using namespace llvm;
using namespace mlir;
namespace tpu_mlir {
namespace helper {
constexpr llvm::StringRef Module::Attr::NAME;
constexpr llvm::StringRef Module::Attr::STATE;
constexpr llvm::StringRef Module::Attr::CHIP;
constexpr llvm::StringRef Module::Attr::FLOPS;
constexpr llvm::StringRef Module::Attr::WEIGHT_FILE;
constexpr llvm::StringRef Module::Attr::COEFF_ADDR;
constexpr llvm::StringRef Module::Attr::COEFF_SIZE;
constexpr llvm::StringRef Module::Attr::NEURON_ADDR;
constexpr llvm::StringRef Module::Attr::NEURON_SIZE;
constexpr llvm::StringRef Module::Attr::GMEM_PRIVATE_SIZE;
constexpr llvm::StringRef Module::Attr::ASYMMETRIC;
constexpr llvm::StringRef Module::Attr::MODE;

constexpr llvm::StringRef Module::State::TOP_F32;
constexpr llvm::StringRef Module::State::TOP_CALIBRATED;
constexpr llvm::StringRef Module::State::TOP_QUANTIZED;
constexpr llvm::StringRef Module::State::TPU_LOWERED;
constexpr llvm::StringRef Module::State::TPU_REORDERED;
constexpr llvm::StringRef Module::State::TPU_DIVIDED;
constexpr llvm::StringRef Module::State::TPU_ADDRESSED;

constexpr llvm::StringRef Module::Chip::ALL;
constexpr llvm::StringRef Module::Chip::BM1684;
constexpr llvm::StringRef Module::Chip::BM1684x;
constexpr llvm::StringRef Module::Chip::CV182x;
constexpr llvm::StringRef Module::Chip::CV183x;

top::NoneOp Module::getNoneOp(Operation *op) {
  assert(op != nullptr);
  if (auto noneOp = dyn_cast<top::NoneOp>(op)) {
    return noneOp;
  }
  FuncOp funcOp;
  if (isa<FuncOp>(op)) {
    funcOp = cast<FuncOp>(op);
  } else {
    funcOp = cast<FuncOp>(op->getParentOp());
  }
  auto &block = funcOp.front();
  auto &topOp = block.front();
  if (auto noneOp = dyn_cast<top::NoneOp>(topOp)) {
    return noneOp;
  }
  auto ctx = op->getContext();
  auto builder = OpBuilder(ctx);
  builder.setInsertionPointToStart(&block);
  auto NoneOp = builder.create<top::NoneOp>(builder.getUnknownLoc(),
                                            builder.getNoneType());
  return NoneOp;
}

Value Module::getOperand(Operation* op, int i) {
  auto v = op->getOperand(i);
  if (auto block_arg = v.dyn_cast_or_null<mlir::BlockArgument>()) {
    int idx = block_arg.getArgNumber();
    auto parent_op = v.getParentBlock()->getParentOp();
    if (auto func_op = dyn_cast_or_null<FuncOp>(parent_op)) {
      auto module = getModuleOp(parent_op);
      // cur call op
      auto call_op = getCallOp(module, func_op);
      // pre call op
      auto operand = call_op.getOperand(idx);
      auto result = operand.cast<OpResult>();
      auto opd = result.getDefiningOp();
      if (isa<top::InputOp>(opd)) {
        return operand;
      }
      auto pre_call_op = dyn_cast<func::CallOp>(opd);
      auto pre_func_op = getFuncOp(module, pre_call_op.getCallee());
      auto return_op = dyn_cast<func::ReturnOp>(pre_func_op.front().back());
      return return_op.getOperand(result.getResultNumber());
    }
  } else if (v.getDefiningOp() != 0x0){
    return v;
  }
  llvm_unreachable("Failed to get preOperation.FIx me");
}

ModuleOp Module::getModuleOp(Operation *op) {
  auto moduleOp = op;
  while (moduleOp && !isa<ModuleOp>(moduleOp)) {
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
  // update callee func's return types
  for (auto func : module.getOps<FuncOp>()) {
    if (func.getName() == "main") {
      continue;
    }
    std::vector<Type> returns;
    Block &entryBlock = func.front();
    auto returnOp = dyn_cast<func::ReturnOp>(entryBlock.back()).getOperation();
    for (uint32_t i = 0; i < returnOp->getNumOperands(); ++i) {
      returns.push_back(returnOp->getOperand(i).getType());
    }
    auto fnType = builder.getFunctionType(func.getArgumentTypes(),
                                          llvm::ArrayRef<Type>{returns});
    func.setType(fnType);
    auto callee = getCallOp(module, func);
    if (callee) {
      for (auto it : llvm::zip(callee.getResults(), returns)) {
        std::get<0>(it).setType(std::get<1>(it));
      }
    }
  }
  // update callee arg types
  for (auto func : module.getOps<FuncOp>()) {
    if (func.getName() == "main") {
      continue;
    }
    auto callee = getCallOp(module, func);
    if (!callee) {
      continue;
    }
    std::vector<Type> arguments;
    for (auto it :
         llvm::zip(callee.getOperandTypes(), func.front().getArguments())) {
      arguments.push_back(std::get<0>(it));
      std::get<1>(it).setType(std::get<0>(it));
    }
    auto fnType = builder.getFunctionType(llvm::ArrayRef<Type>(arguments),
                                          func.getResultTypes());
    func.setType(fnType);
  }
  // update main op return types
  auto mainFunc = getMainFuncOp(module);
  Block &entryBlock = mainFunc.front();
  auto returnOp = dyn_cast<func::ReturnOp>(entryBlock.back()).getOperation();
  std::vector<Type> returns;
  for (uint32_t i = 0; i < returnOp->getNumOperands(); ++i) {
    returns.push_back(returnOp->getOperand(i).getType());
  }
  auto fnType = builder.getFunctionType(mainFunc.getArgumentTypes(),
                                        llvm::ArrayRef<Type>{returns});
  mainFunc.setType(fnType);
}

void Module::removeUnusedOp(ModuleOp module) {
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      if (isa<func::ReturnOp, FuncOp, tpu::YieldOp>(op)) {
      } else {
        if (op->getUsers().empty()) {
          op->erase();
        }
      }
    });
  }
}

std::string Module::genWeightFileName(ModuleOp module, bool &same_name) {
  auto name = getName(module);
  auto state = getState(module);
  auto chip = getChip(module);
  auto old_name = getWeightFile(module);
  std::string file_name = name.lower() + std::string("_") + state.lower() +
                          std::string("_") + chip.lower();
  if (std::string(chip) != "ALL") {
    auto mode = getMode(module);
    std::string sym = "";
    if (mode == Quant::Type::INT8) {
      sym = getAsymmetric(module) ? "_asym" : "_sym";
    }
    file_name += std::string("_") + mode.lower() + sym;
  }
  auto new_name = file_name + "_weight.npz";
  same_name = (old_name == new_name);
  if (same_name) {
    new_name = file_name + "_weight_fix.npz";
  }
  return new_name;
}

int64_t Module::getAddress(Value v) {
  auto attr = v.getType().cast<RankedTensorType>().getEncoding();
  if (!attr) {
    if (auto block_arg = v.dyn_cast_or_null<mlir::BlockArgument>()) {
      int index = block_arg.getArgNumber();
      auto parent_op = v.getParentBlock()->getParentOp();
      auto funcOp = dyn_cast_or_null<FuncOp>(parent_op);
      if (funcOp) {
        mlir::func::CallOp callee = getCallOp(getModuleOp(parent_op), funcOp);
        return Module::getAddress(callee.getOperand(index));
      }
    }
  } else {
    assert(attr.isa<IntegerAttr>());
    return attr.cast<IntegerAttr>().getInt();
  }
  llvm_unreachable("can't get address");
  return 0;
}

void Module::setAddress(Value v, int64_t addr) {
  auto type = v.getType().cast<RankedTensorType>();
  Builder builder(v.getContext());
  auto addrAttr = builder.getI64IntegerAttr(addr);
  auto new_type =
      RankedTensorType::get(type.getShape(), type.getElementType(), addrAttr);
  v.setType(new_type);
}

size_t Module::getBytes(Value v) {
  auto type = v.getType().cast<RankedTensorType>();
  auto elm_count = type.getNumElements();
  auto etype = getStorageType(v);
  int elm_bytes = etype.getIntOrFloatBitWidth() / 8;
  return elm_count * elm_bytes;
}

int Module::getDtypeSize(Value v) {
  auto type = v.getType().cast<RankedTensorType>();
  auto etype = getStorageType(v);
  int elm_bytes = etype.getIntOrFloatBitWidth() / 8;
  return elm_bytes;
}

int64_t Module::getNumElements(Value v) {
  auto type = v.getType().cast<RankedTensorType>();
  return type.getNumElements();
}

llvm::ArrayRef<int64_t> Module::getShape(Value v) {
  auto type = v.getType().cast<RankedTensorType>();
  return type.getShape();
}

std::shared_ptr<std::vector<int64_t>> Module::getI64Array(ArrayAttr arrayAttr) {
  auto data = std::make_shared<std::vector<int64_t>>();
  for (auto en : llvm::enumerate(arrayAttr)) {
    auto attr = en.value().dyn_cast<IntegerAttr>();
    if (attr) {
      data->push_back(attr.getInt());
    } else {
      arrayAttr.dump();
      llvm_unreachable("not int64_t type");
    }
  }
  return std::move(data);
}

std::shared_ptr<std::vector<int64_t>>
Module::getI64Array(Optional<ArrayAttr> arrayAttr, int64_t num_elem,
                    int64_t default_value) {
  if (arrayAttr.has_value()) {
    auto arr = getI64Array(arrayAttr.value());
    assert(arr->size() == num_elem);
    return std::move(arr);
  }
  return std::make_shared<std::vector<int64_t>>(num_elem, default_value);
}

std::shared_ptr<std::vector<double>> Module::getF64Array(ArrayAttr arrayAttr) {
  auto data = std::make_shared<std::vector<double>>();
  for (auto en : llvm::enumerate(arrayAttr)) {
    auto attr = en.value().dyn_cast<FloatAttr>();
    data->push_back(attr.getValueAsDouble());
  }
  return std::move(data);
}

std::shared_ptr<std::vector<double>>
Module::getF64Array(Optional<ArrayAttr> arrayAttr, int64_t num_elem,
                    double default_value) {
  if (arrayAttr.has_value()) {
    auto arr = getF64Array(arrayAttr.value());
    assert(arr->size() == num_elem);
    return std::move(arr);
  }
  return std::make_shared<std::vector<double>>(num_elem, default_value);
}

Type Module::getStorageType(Type type) {
  if (type.isa<RankedTensorType>()) {
    type = type.cast<RankedTensorType>().getElementType();
  }
  if (auto qType = type.dyn_cast<quant::CalibratedQuantizedType>()) {
    return qType.getExpressedType();
  } else if (auto qType = type.dyn_cast<quant::UniformQuantizedType>()) {
    auto stype = qType.getStorageType();
    bool isSign = qType.isSigned();
    if (stype.isSignlessInteger()) {
      auto bits = stype.getIntOrFloatBitWidth();
      auto sign = isSign ? IntegerType::Signed : IntegerType::Unsigned;
      return IntegerType::get(type.getContext(), bits, sign);
    }
    return stype;
  } else if (auto qType = type.dyn_cast<quant::UniformQuantizedPerAxisType>()) {
    return qType.getStorageType();
  }
  return type;
}

Type Module::getStorageType(Value v) { return getStorageType(v.getType()); }

Type Module::getElementType(Value v) {
  auto type = v.getType();
  if (type.isa<RankedTensorType>()) {
    auto rtype = v.getType().cast<RankedTensorType>();
    return rtype.getElementType();
  }
  return type;
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
  if (num_dims > 0) {
    n = shape[0];
  }
  if (num_dims > 1) {
    c = shape[1];
  }
  if (num_dims > 2) {
    h = shape[2];
  }
  for (size_t i = 3; i < num_dims; ++i) {
    w *= shape[i];
  }
}

void Module::getNCHW(llvm::ArrayRef<int64_t> shape, int64_t &n, int64_t &c,
                     int64_t &h, int64_t &w, bool left_align) {
  if (left_align) {
    getNCHW_align_left(shape, n, c, h, w);
  } else {
    getNCHW_align_right(shape, n, c, h, w);
  }
}

void Module::getNCHW(Value v, int64_t &n, int64_t &c, int64_t &h, int64_t &w,
                     bool left_align) {
  auto shape = v.getType().cast<RankedTensorType>().getShape();
  getNCHW(shape, n, c, h, w, left_align);
}

void Module::getShapeVec(Value v, std::vector<int64_t> &vec_shape) {
  auto shape = v.getType().cast<RankedTensorType>().getShape();
  int num_dims = shape.size();
  for (int i = 0; i < num_dims; i++) {
    vec_shape.push_back(shape[i]);
  }
}

bool Module::isOpInGroup(Operation *Op) {
  if (Op == nullptr) {
    return false;
  }
  auto parent = Op->getParentOp();
  if (parent != nullptr && isa<tpu::GroupOp>(parent)) {
    return true;
  }
  return false;
}

FuncOp Module::getFuncOp(ModuleOp module, StringRef func_name) {
  for (auto func : module.getOps<FuncOp>()) {
    if (func.getName() == func_name) {
      return func;
    }
  }
  llvm::errs() << "Can't find FuncOp:" << func_name << "\n";
  llvm_unreachable("Error getFuncOp !!\n");
  return nullptr;
}

func::CallOp Module::getCallOp(ModuleOp module, FuncOp func) {
  auto mainFunc = getMainFuncOp(module);
  func::CallOp call = nullptr;
  mainFunc.walk([&](func::CallOp op) {
    if (!call && op.getCallee() == func.getName()) {
      call = op;
    }
  });
  return call;
}

StringRef Module::getName(Operation *op, int index) {
  if (auto module = dyn_cast<ModuleOp>(op)) {
    return getName(module);
  }
  if (auto loc = op->getLoc().dyn_cast<mlir::NameLoc>()) {
    return loc.getName();
  }
  if (auto loc = op->getLoc().dyn_cast<mlir::FusedLoc>()) {
    auto locs = loc.getLocations();
    assert(index < locs.size());
    if (auto name_loc = locs[index].dyn_cast<mlir::NameLoc>()) {
      return name_loc.getName();
    }
  }
  op->print(llvm::errs(), OpPrintingFlags().useLocalScope().enableDebugInfo());
  llvm::errs() << "\n";
  llvm_unreachable("op has no name location!!!");
  return "";
}

StringRef Module::getName(Value v) {
  if (auto loc = v.getLoc().dyn_cast<mlir::NameLoc>()) {
    return loc.getName();
  } else if (auto op = v.getDefiningOp()) {
    if (op->getNumResults() == 1) {
      return Module::getName(op);
    } else {
      auto r = v.cast<OpResult>();
      return Module::getName(op, r.getResultNumber());
    }
  }
  v.dump();
  llvm_unreachable("No name info");
  return "";
}

StringRef Module::getChip(Operation *op) {
  auto module = getModuleOp(op);
  return getChip(module);
}

void Module::getInputsOutputs(ModuleOp module, std::vector<Value> &inputs,
                              std::vector<Value> &outputs) {
  auto main_func = Module::getMainFuncOp(module);
  main_func.walk([&](top::InputOp op) { inputs.push_back(op.output()); });
  main_func.walk([&](func::ReturnOp op) {
    for (auto out : op.getOperands()) {
      auto result = out.cast<OpResult>();
      auto call_op = result.getDefiningOp<func::CallOp>();
      auto func_op = getFuncOp(module, call_op.getCallee());
      auto return_op = dyn_cast<func::ReturnOp>(func_op.front().back());
      assert(return_op);
      outputs.push_back(return_op.getOperand(result.getResultNumber()));
    }
  });
}

void Module::getInputsOutputs(func::CallOp call, std::vector<Value> &inputs,
                              std::vector<Value> &outputs) {
  auto module = getModuleOp(call);
  for (auto opd : call.getOperands()) {
    auto result = opd.cast<OpResult>();
    auto op = result.getDefiningOp();
    if (isa<top::InputOp>(op)) {
      inputs.push_back(opd);
    } else if (auto call_ = dyn_cast<func::CallOp>(op)) {
      auto func_op = getFuncOp(module, call_.getCallee());
      auto return_op = dyn_cast<func::ReturnOp>(func_op.front().back());
      assert(return_op);
      inputs.push_back(return_op.getOperand(result.getResultNumber()));
    } else {
      llvm_unreachable("input is illegal");
    }
  }
  auto func = getFuncOp(module, call.getCallee());
  func.walk([&](func::ReturnOp op) {
    for (auto output : op.getOperands()) {
      outputs.push_back(output);
    }
  });
}

} // namespace helper
} // namespace tpu_mlir
