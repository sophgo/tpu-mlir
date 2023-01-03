//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"
#include "float.h"
#include "mlir/IR/PatternMatch.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "mlir/Dialect/Quant/FakeQuantSupport.h"

#include <map>

namespace tpu_mlir {
namespace module {
constexpr llvm::StringRef Attr::NAME;
constexpr llvm::StringRef Attr::STATE;
constexpr llvm::StringRef Attr::CHIP;
constexpr llvm::StringRef Attr::FLOPS;
constexpr llvm::StringRef Attr::WEIGHT_FILE;
constexpr llvm::StringRef Attr::COEFF_ADDR;
constexpr llvm::StringRef Attr::COEFF_SIZE;
constexpr llvm::StringRef Attr::NEURON_ADDR;
constexpr llvm::StringRef Attr::NEURON_SIZE;
constexpr llvm::StringRef Attr::GMEM_PRIVATE_SIZE;
constexpr llvm::StringRef Attr::ASYMMETRIC;
constexpr llvm::StringRef Attr::MODE;

constexpr llvm::StringRef State::TOP_F32;
constexpr llvm::StringRef State::TOP_CALIBRATED;
constexpr llvm::StringRef State::TOP_QUANTIZED;
constexpr llvm::StringRef State::TPU_LOWERED;
constexpr llvm::StringRef State::TPU_REORDERED;
constexpr llvm::StringRef State::TPU_DIVIDED;
constexpr llvm::StringRef State::TPU_ADDRESSED;

constexpr llvm::StringRef Chip::ALL;
constexpr llvm::StringRef Chip::BM1684;
constexpr llvm::StringRef Chip::BM1684X;
constexpr llvm::StringRef Chip::CV182x;
constexpr llvm::StringRef Chip::CV183x;
constexpr llvm::StringRef Chip::BM1686;

static ModuleOp m = nullptr;
static MLIRContext *ctx = nullptr;
static llvm::StringRef chip = "";

void init(ModuleOp module) {
  m = module;
  ctx = m.getContext();
  chip = m->getAttrOfType<StringAttr>(Attr::CHIP).getValue();
}

top::NoneOp getNoneOp(Operation *op) {
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

Value getOriValue(Value &v) {
  if (auto block_arg = v.dyn_cast_or_null<BlockArgument>()) {
    int idx = block_arg.getArgNumber();
    auto parent_op = v.getParentBlock()->getParentOp();
    if (auto func_op = dyn_cast_or_null<FuncOp>(parent_op)) {
      // cur call op
      auto call_op = getCallOp(func_op);
      // pre call op
      auto operand = call_op.getOperand(idx);
      auto result = operand.cast<OpResult>();
      auto opd = result.getDefiningOp();
      if (isa<top::InputOp>(opd)) {
        return operand;
      }
      auto pre_call_op = dyn_cast<func::CallOp>(opd);
      auto pre_func_op = getFuncOp(pre_call_op.getCallee());
      auto return_op = dyn_cast<ReturnOp>(pre_func_op.front().back());
      return return_op.getOperand(result.getResultNumber());
    }
  } else if (auto pre_op = v.getDefiningOp()) {
    if (isa<func::CallOp>(pre_op)) {
      auto call_op = dyn_cast<func::CallOp>(pre_op);
      int index = v.cast<OpResult>().getResultNumber();
      for (auto func : m.getOps<FuncOp>()) {
        if (call_op.getCallee() == func.getName()) {
          Block &entryBlock = func.front();
          auto returnOp = dyn_cast<ReturnOp>(entryBlock.back()).getOperation();
          return returnOp->getOperand(index);
        }
      }
    } else {
      return v;
    }
  }
  llvm_unreachable("Failed to get preOperation.FIx me");
}

Value getOperand(Operation *op, int i) {
  auto v = op->getOperand(i);
  return getOriValue(v);
}

void updateModuleTypes() {
  Builder builder(ctx);
  // update callee func's return types
  for (auto func : m.getOps<FuncOp>()) {
    if (func.getName() == "main") {
      continue;
    }
    std::vector<Type> returns;
    Block &entryBlock = func.front();
    auto returnOp = dyn_cast<ReturnOp>(entryBlock.back()).getOperation();
    for (uint32_t i = 0; i < returnOp->getNumOperands(); ++i) {
      returns.push_back(returnOp->getOperand(i).getType());
    }
    auto fnType = builder.getFunctionType(func.getArgumentTypes(),
                                          llvm::ArrayRef<Type>{returns});
    func.setType(fnType);
    auto callee = getCallOp(func);
    if (callee) {
      for (auto it : llvm::zip(callee.getResults(), returns)) {
        std::get<0>(it).setType(std::get<1>(it));
      }
    }
  }
  // update callee arg types
  for (auto func : m.getOps<FuncOp>()) {
    if (func.getName() == "main") {
      continue;
    }
    auto callee = getCallOp(func);
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
  auto mainFunc = getMainFuncOp();
  Block &entryBlock = mainFunc.front();
  auto returnOp = dyn_cast<ReturnOp>(entryBlock.back()).getOperation();
  std::vector<Type> returns;
  for (uint32_t i = 0; i < returnOp->getNumOperands(); ++i) {
    returns.push_back(returnOp->getOperand(i).getType());
  }
  auto fnType = builder.getFunctionType(mainFunc.getArgumentTypes(),
                                        llvm::ArrayRef<Type>{returns});
  mainFunc.setType(fnType);
}

void removeUnusedOp() {
  std::vector<Operation *> all_ops;
  for (auto func : m.getOps<FuncOp>()) {
    for (auto &op : func.getOps()) {
      if (false == isa<ReturnOp, FuncOp, tpu::YieldOp>(op)) {
        all_ops.push_back(&op);
      }
    }
  }
  for (auto iter = all_ops.rbegin(); iter != all_ops.rend(); iter++) {
    if ((*iter)->use_empty()) {
      (*iter)->erase();
    }
  }
}

std::string genWeightFileName(bool &same_name) {
  auto name = getModuleName();
  auto state = getState();
  auto chip = getChip();
  auto old_name = getWeightFile();
  std::string file_name = name.lower() + std::string("_") + state.lower() +
                          std::string("_") + chip.lower();
  if (std::string(chip) != "ALL") {
    auto mode = getMode();
    std::string sym = "";
    if (mode == Mode::INT8) {
      sym = isAsymmetric() ? "_asym" : "_sym";
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

int64_t getAddress(Value v) {
  if (v.getType().isa<NoneType>()) {
    return 0;
  }
  auto attr = v.getType().cast<RankedTensorType>().getEncoding();
  if (!attr) {
    if (auto block_arg = v.dyn_cast_or_null<BlockArgument>()) {
      int index = block_arg.getArgNumber();
      auto parent_op = v.getParentBlock()->getParentOp();
      auto funcOp = dyn_cast_or_null<FuncOp>(parent_op);
      if (funcOp) {
        func::CallOp callee = getCallOp(funcOp);
        return getAddress(callee.getOperand(index));
      }
    }
  } else {
    assert(attr.isa<IntegerAttr>());
    return attr.cast<IntegerAttr>().getInt();
  }
  llvm_unreachable("can't get address");
  return 0;
}

void setAddress(Value v, int64_t addr) {
  auto type = v.getType().cast<RankedTensorType>();
  Builder builder(v.getContext());
  auto addrAttr = builder.getI64IntegerAttr(addr);
  auto new_type =
      RankedTensorType::get(type.getShape(), type.getElementType(), addrAttr);
  v.setType(new_type);
}

size_t getBytes(Value v) {
  if (v.getType().isa<NoneType>()) {
    return 0;
  }
  auto type = v.getType().cast<RankedTensorType>();
  auto elm_count = type.getNumElements();
  auto etype = getStorageType(v);
  int elm_bytes = etype.getIntOrFloatBitWidth() / 8;
  return elm_count * elm_bytes;
}

int getDtypeSize(Value v) {
  auto type = v.getType().cast<RankedTensorType>();
  auto etype = getStorageType(v);
  int elm_bytes = etype.getIntOrFloatBitWidth() / 8;
  return elm_bytes;
}

int64_t getNumElements(Value v) {
  auto type = v.getType().cast<RankedTensorType>();
  return type.getNumElements();
}

llvm::ArrayRef<int64_t> getShape(Value v) {
  if (v.getType().isa<NoneType>()) {
    v.dump();
    llvm_unreachable("v is none type");
  }
  auto type = v.getType().cast<RankedTensorType>();
  return type.getShape();
}

i32_array_t getI32Array(ArrayAttr arrayAttr) {
  auto data = std::make_shared<std::vector<int32_t>>();
  for (auto en : llvm::enumerate(arrayAttr)) {
    auto attr = en.value().dyn_cast<IntegerAttr>();
    if (attr) {
      data->push_back(attr.getInt());
    } else {
      arrayAttr.dump();
      llvm_unreachable("not int32_t type");
    }
  }
  return std::move(data);
}

i32_array_t getI32Array(Optional<ArrayAttr> arrayAttr, int64_t num_elem,
                        int32_t default_value) {
  if (arrayAttr.has_value()) {
    auto arr = getI32Array(arrayAttr.value());
    assert(arr->size() == num_elem);
    return std::move(arr);
  }
  return std::make_shared<std::vector<int32_t>>(num_elem, default_value);
}

i64_array_t getI64Array(ArrayAttr arrayAttr) {
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

i64_array_t getI64Array(Optional<ArrayAttr> arrayAttr, int64_t num_elem,
                        int64_t default_value) {
  if (arrayAttr.has_value()) {
    auto arr = getI64Array(arrayAttr.value());
    assert(arr->size() == num_elem);
    return std::move(arr);
  }
  return std::make_shared<std::vector<int64_t>>(num_elem, default_value);
}

f64_array_t getF64Array(ArrayAttr arrayAttr) {
  auto data = std::make_shared<std::vector<double>>();
  for (auto en : llvm::enumerate(arrayAttr)) {
    auto attr = en.value().dyn_cast<FloatAttr>();
    data->push_back(attr.getValueAsDouble());
  }
  return std::move(data);
}

f64_array_t getF64Array(Optional<ArrayAttr> arrayAttr, int64_t num_elem,
                        double default_value) {
  if (arrayAttr.has_value()) {
    auto arr = getF64Array(arrayAttr.value());
    assert(arr->size() == num_elem);
    return std::move(arr);
  }
  return std::make_shared<std::vector<double>>(num_elem, default_value);
}

Type getStorageType(Type type) {
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

Type getStorageType(Value v) { return getStorageType(v.getType()); }

Type getElementType(Value v) {
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

void getNCHW(llvm::ArrayRef<int64_t> shape, int64_t &n, int64_t &c, int64_t &h,
             int64_t &w, bool left_align) {
  if (left_align) {
    getNCHW_align_left(shape, n, c, h, w);
  } else {
    getNCHW_align_right(shape, n, c, h, w);
  }
}

void getNCHW(Value v, int64_t &n, int64_t &c, int64_t &h, int64_t &w,
             bool left_align) {
  auto shape = v.getType().cast<RankedTensorType>().getShape();
  getNCHW(shape, n, c, h, w, left_align);
}

void getShapeVec(Value v, std::vector<int64_t> &vec_shape) {
  auto shape = v.getType().cast<RankedTensorType>().getShape();
  int num_dims = shape.size();
  for (int i = 0; i < num_dims; i++) {
    vec_shape.push_back(shape[i]);
  }
}

bool isOpInGroup(Operation *Op) {
  if (Op == nullptr) {
    return false;
  }
  auto parent = Op->getParentOp();
  if (parent != nullptr && isa<tpu::GroupOp>(parent)) {
    return true;
  }
  return false;
}

FuncOp getFuncOp(StringRef func_name) {
  for (auto func : m.getOps<FuncOp>()) {
    if (func.getName() == func_name) {
      return func;
    }
  }
  llvm::errs() << "Can't find FuncOp:" << func_name << "\n";
  llvm_unreachable("Error getFuncOp !!\n");
  return nullptr;
}

func::CallOp getCallOp(FuncOp func) {
  auto mainFunc = getMainFuncOp();
  func::CallOp call = nullptr;
  mainFunc.walk([&](func::CallOp op) {
    if (!call && op.getCallee() == func.getName()) {
      call = op;
    }
  });
  return call;
}

StringRef getName(Operation *op, int index) {
  if (auto module = dyn_cast<ModuleOp>(op)) {
    return getName(module);
  }
  if (auto loc = op->getLoc().dyn_cast<NameLoc>()) {
    return loc.getName();
  }
  if (auto loc = op->getLoc().dyn_cast<FusedLoc>()) {
    auto locs = loc.getLocations();
    assert(index < locs.size());
    if (auto name_loc = locs[index].dyn_cast<NameLoc>()) {
      return name_loc.getName();
    }
  }
  op->print(llvm::errs(), OpPrintingFlags().useLocalScope().enableDebugInfo());
  llvm::errs() << "\n";
  llvm_unreachable("op has no name location!!!");
  return "";
}

FuncOp getMainFuncOp() { return getFuncOp("main"); }

llvm::StringRef getModuleName() {
  return m->getAttrOfType<StringAttr>(Attr::NAME).getValue();
}

int64_t getCoeffSize() {
  return m->getAttrOfType<IntegerAttr>(Attr::COEFF_SIZE).getInt();
}
void setCoeffSize(int64_t size) {
  m->setAttr(Attr::COEFF_SIZE, Builder(ctx).getI64IntegerAttr(size));
}
int64_t getGmemPrivateSize() {
  return m->getAttrOfType<IntegerAttr>(Attr::GMEM_PRIVATE_SIZE).getInt();
}
void setGmemPrivateSize(int64_t size) {
  m->setAttr(Attr::GMEM_PRIVATE_SIZE, Builder(ctx).getI64IntegerAttr(size));
}
int64_t getCoeffAddr() {
  return m->getAttrOfType<IntegerAttr>(Attr::COEFF_ADDR).getInt();
}

void setCoeffAddr(int64_t addr) {
  m->setAttr(Attr::COEFF_ADDR, Builder(ctx).getI64IntegerAttr(addr));
}
int64_t getNeuronSize() {
  return m->getAttrOfType<IntegerAttr>(Attr::NEURON_SIZE).getInt();
}
void setNeuronSize(int64_t size) {
  m->setAttr(Attr::NEURON_SIZE, Builder(ctx).getI64IntegerAttr(size));
}
int64_t getNeuronAddr() {
  return m->getAttrOfType<IntegerAttr>(Attr::NEURON_ADDR).getInt();
}
void setNeuronAddr(int64_t addr) {
  m->setAttr(Attr::NEURON_ADDR, Builder(ctx).getI64IntegerAttr(addr));
}

llvm::StringRef getChip() { return chip; }
llvm::StringRef getMode() {
  return m->getAttrOfType<StringAttr>(Attr::MODE).getValue();
}
llvm::StringRef getFuncMode(FuncOp func) {
  return func->getAttr("mode").cast<StringAttr>().getValue();
}
void setChip(StringRef chip_) {
  m->setAttr(Attr::CHIP, StringAttr::get(m.getContext(), chip_.upper()));
  chip = m->getAttrOfType<StringAttr>(Attr::CHIP).getValue();
}

void setMode(StringRef mode) {
  m->setAttr(Attr::MODE, StringAttr::get(ctx, mode.upper()));
}

StringRef getWeightFile() {
  return m->getAttrOfType<StringAttr>(Attr::WEIGHT_FILE).getValue();
}
void setWeightFile(StringRef weight_file) {
  m->setAttr(Attr::WEIGHT_FILE, StringAttr::get(ctx, weight_file));
}
int64_t getFLOPs() {
  return m->getAttrOfType<IntegerAttr>(Attr::FLOPS).getInt();
}
void setFLOPs(int64_t flops) {
  auto intType = IntegerType::get(ctx, 64);
  m->setAttr(Attr::FLOPS, IntegerAttr::get(intType, flops));
}

bool isAsymmetric() {
  if (m->hasAttrOfType<BoolAttr>(Attr::ASYMMETRIC)) {
    return m->getAttrOfType<BoolAttr>(Attr::ASYMMETRIC).getValue();
  }
  return false;
}
void setAsymmetric(bool is_asymmetric) {
  m->setAttr(Attr::ASYMMETRIC, BoolAttr::get(ctx, is_asymmetric));
}
StringRef getState() {
  return m->getAttrOfType<StringAttr>(Attr::STATE).getValue();
}
void setState(StringRef state) {
  m->setAttr(Attr::STATE, StringAttr::get(ctx, state));
}
bool isState(llvm::StringRef state) { return state == getState(); }
bool isTpuOp(Operation *op) {
  return (op->getDialect()->getNamespace() == "tpu");
}

bool isCV18xx() { return (chip == Chip::CV183x || chip == Chip::CV182x); }
bool isBM1684Family() { return (chip == Chip::BM1684); }
bool isBM1684XFamily() {
  return (chip == Chip::BM1684X || chip == Chip::BM1686);
}
bool isBM1686() { return (chip == Chip::BM1686); }

ModuleOp getModuleOp() { return m; }

Location getLoc() { return m.getLoc(); }

MLIRContext *getCtx() { return ctx; }

void push_back(FuncOp funcOp) { m.push_back(funcOp); }

double getThreshold(Value v) {
  auto type = getCalibratedType(v);
  assert(type.getMax() == -type.getMin());
  return type.getMax();
}

StringRef getName(Value v) {
  if (auto loc = v.getLoc().dyn_cast<NameLoc>()) {
    return loc.getName();
  } else if (auto op = v.getDefiningOp()) {
    if (op->getNumResults() == 1) {
      return getName(op);
    } else {
      auto r = v.cast<OpResult>();
      return getName(op, r.getResultNumber());
    }
  }
  v.dump();
  llvm_unreachable("No name info");
  return "";
}

void getInputsOutputs(std::vector<Value> &inputs, std::vector<Value> &outputs) {
  auto main_func = getMainFuncOp();
  main_func.walk([&](top::InputOp op) { inputs.push_back(op.getOutput()); });
  main_func.walk([&](ReturnOp op) {
    for (auto out : op.getOperands()) {
      auto result = out.cast<OpResult>();
      auto call_op = result.getDefiningOp<func::CallOp>();
      auto func_op = getFuncOp(call_op.getCallee());
      auto return_op = dyn_cast<ReturnOp>(func_op.front().back());
      assert(return_op);
      outputs.push_back(return_op.getOperand(result.getResultNumber()));
    }
  });
}

void getInputsOutputs(func::CallOp call, std::vector<Value> &inputs,
                      std::vector<Value> &outputs) {
  for (auto opd : call.getOperands()) {
    auto result = opd.cast<OpResult>();
    auto op = result.getDefiningOp();
    if (isa<top::InputOp>(op)) {
      inputs.push_back(opd);
    } else if (auto call_ = dyn_cast<func::CallOp>(op)) {
      auto func_op = getFuncOp(call_.getCallee());
      auto return_op = dyn_cast<ReturnOp>(func_op.front().back());
      assert(return_op);
      inputs.push_back(return_op.getOperand(result.getResultNumber()));
    } else {
      llvm_unreachable("input is illegal");
    }
  }
  auto func = getFuncOp(call.getCallee());
  func.walk([&](ReturnOp op) {
    for (auto output : op.getOperands()) {
      outputs.push_back(output);
    }
  });
}

constexpr llvm::StringRef Mode::INT8;
constexpr llvm::StringRef Mode::INT4;
constexpr llvm::StringRef Mode::BF16;
constexpr llvm::StringRef Mode::F16;
constexpr llvm::StringRef Mode::F32;

void getScaleAndZeroPoint(double rmin, double rmax, double &scale,
                          int64_t &zeroPoint, int bitwidth) {
  int qmin = rmin < 0 ? -128 : 0;
  int qmax = rmin < 0 ? 127 : 255;
  if (bitwidth == 4) {
    qmin = rmin < 0 ? -8 : 0;
    qmax = rmin < 0 ? 7 : 15;
  }
  // Determine the scale.
  double qminDouble = qmin;
  double qmaxDouble = qmax;
  scale = (rmax - rmin) / (qmaxDouble - qminDouble);
  double zeroPointFromMin = qminDouble - rmin / scale;

  // Now nudge the zero point to be an integer.
  zeroPoint = round(zeroPointFromMin);
  if (zeroPointFromMin < qminDouble) {
    zeroPoint = qmin;
    scale = rmax / (qmaxDouble - zeroPoint);
  } else if (zeroPointFromMin > qmaxDouble) {
    zeroPoint = qmax;
    scale = rmin / (qminDouble - zeroPoint);
  }
}

double getScale(double threshold, bool sign, int bitwidth) {
  if (bitwidth == 8) {
    if (sign) {
      return threshold / 127.0;
    } else {
      return threshold / 255.0;
    }
  } else {
    if (sign) {
      return threshold / 7.0;
    } else {
      return threshold / 15.0;
    }
  }
}

void getScaleAndZeroPoint(Value v, double &scale, int64_t &zeropoint,
                          bool asymmetric, int bitwidth) {
  bool sign;
  getScaleAndZeroPoint(v, scale, zeropoint, sign, asymmetric, bitwidth);
}

void getScaleAndZeroPoint(Value v, double &scale, int64_t &zeropoint,
                          bool &sign, bool asymmetric, int bitwidth) {
  if (isCalibratedType(v)) {
    auto qtype = getCalibratedType(v);
    auto max = qtype.getMax();
    auto min = qtype.getMin();
    sign = min < 0;
    if (asymmetric) {
      getScaleAndZeroPoint(min, max, scale, zeropoint, bitwidth);
    } else {
      zeropoint = 0;
      scale = getScale(max, sign, bitwidth);
    }
  } else if (isUniformQuantized(v)) {
    auto qtype = getUniformQuantizedType(v);
    scale = qtype.getScale();
    zeropoint = qtype.getZeroPoint();
    sign = qtype.isSigned();
  } else {
    v.dump();
    llvm_unreachable("can't get scale and zeropoint");
  }
}

bool isCalibratedType(Type type) {
  return type.cast<RankedTensorType>()
      .getElementType()
      .isa<quant::CalibratedQuantizedType>();
}

bool isCalibratedType(Value v) { return isCalibratedType(v.getType()); }

bool isUniformQuantized(Type type) {
  return type.cast<RankedTensorType>()
      .getElementType()
      .isa<quant::UniformQuantizedType>();
}

bool isUniformQuantized(Value v) { return isUniformQuantized(v.getType()); }

quant::CalibratedQuantizedType getCalibratedType(Value v) {
  return v.getType()
      .cast<RankedTensorType>()
      .getElementType()
      .cast<quant::CalibratedQuantizedType>();
}

quant::CalibratedQuantizedType getCalibratedType(Type t) {
  return t.cast<RankedTensorType>()
      .getElementType()
      .cast<quant::CalibratedQuantizedType>();
}

quant::UniformQuantizedType getUniformQuantizedType(Value v) {
  return v.getType()
      .cast<RankedTensorType>()
      .getElementType()
      .cast<quant::UniformQuantizedType>();
}

quant::UniformQuantizedType getUniformQuantizedType(Type t) {
  return t.cast<RankedTensorType>()
      .getElementType()
      .cast<quant::UniformQuantizedType>();
}

} // namespace module
} // namespace tpu_mlir
