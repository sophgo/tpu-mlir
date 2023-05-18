//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Backend/Arch.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Interfaces/LocalGenInterface.h"

#include "float.h"
#include "mlir/Dialect/Quant/FakeQuantSupport.h"
#include "mlir/IR/PatternMatch.h"
#include <map>

#include "tpu_mlir/Support/ModuleEnum.cpp.inc"

namespace tpu_mlir {
namespace module {
struct Attr {
  static constexpr llvm::StringRef NAME = "module.name";
  static constexpr llvm::StringRef STATE = "module.state";
  static constexpr llvm::StringRef CHIP = "module.chip";
  static constexpr llvm::StringRef WEIGHT_FILE = "module.weight_file";
  static constexpr llvm::StringRef FLOPS = "module.FLOPs";
  static constexpr llvm::StringRef COEFF_ADDR = "module.coeff_addr";
  static constexpr llvm::StringRef COEFF_SIZE = "module.coeff_size";
  static constexpr llvm::StringRef NEURON_ADDR = "module.neuron_addr";
  static constexpr llvm::StringRef NEURON_SIZE = "module.neuron_size";
  static constexpr llvm::StringRef GMEM_PRIVATE_SIZE = "module.private_size";
  static constexpr llvm::StringRef ASYMMETRIC = "module.asymmetric";
  static constexpr llvm::StringRef MODE = "module.mode";
  static constexpr llvm::StringRef PLATFORM = "module.platform";
};

static ModuleOp m = nullptr;
static MLIRContext *ctx = nullptr;
static Chip chip = Chip::ALL;
static Platform platform = Platform::ONNX;
static std::unique_ptr<mlir::TensorFile> wFile = nullptr;
static std::string weightFileName = "";

void init(ModuleOp module) {
  m = module;
  ctx = m.getContext();
  auto chip_ = m->getAttrOfType<StringAttr>(Attr::CHIP);
  chip = symbolizeChip(chip_).value_or(Chip::ALL);
  wFile = nullptr;
  if (m->hasAttrOfType<StringAttr>(Attr::PLATFORM)) {
    auto p = m->getAttrOfType<StringAttr>(Attr::PLATFORM);
    platform = symbolizePlatform(p).value_or(Platform::ONNX);
  } else {
    platform = Platform::ONNX;
  }
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

Value getOriValue(Value v) {
  if (auto block_arg = v.dyn_cast_or_null<BlockArgument>()) {
    int idx = block_arg.getArgNumber();
    auto parent_op = v.getParentBlock()->getParentOp();
    if (auto func_op = dyn_cast_or_null<FuncOp>(parent_op)) {
      // cur call op
      auto call_op = getCallOp(func_op);
      // pre call op
      auto operand = call_op.getOperand(idx);
      if (operand.isa<BlockArgument>()) {
        auto find_root = [](auto&&Me, Value v) ->Value {
          if (v.isa<BlockArgument>()) {
              int index = dyn_cast<BlockArgument>(v).getArgNumber();
              auto p_op = v.getParentBlock()->getParentOp();
              auto func_op = dyn_cast<FuncOp>(p_op);
              auto call_op = getCallOp(func_op);
              return Me(Me, call_op.getOperand(index));
          } else {
            return v;
          }
        };

        Value src_v = find_root(find_root, operand);
        return src_v;
      }
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

Operation *getNextOp(Operation *op, int i) {
  Operation *nextOp = nullptr;
  if (op->getResult(i).hasOneUse()) {
    for (auto &use : op->getResult(i).getUses()) {
      nextOp = use.getOwner();
      break;
    }
    assert(nextOp && "nextOp is nullptr");
  } else {
    auto users = op->getUsers();
    if (1 == std::distance(users.begin(), users.end())) {
      nextOp = *users.begin();
    }
  }
  // if not found, will return NULL
  return nextOp;
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
    //for to support nested region's op
    func.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (!isa<ReturnOp, FuncOp, tpu::YieldOp, top::YieldOp>(op))
        all_ops.push_back(op);
    });
  }
  for (auto iter = all_ops.rbegin(); iter != all_ops.rend(); iter++) {
    if ((*iter)->use_empty()) {
      (*iter)->erase();
    }
  }
}

int64_t getAddress(Value v) {
  if (v.getType().isa<NoneType>()) {
    return 0;
  }
  auto attr = v.getType().cast<RankedTensorType>().getEncoding();
  if (attr) {
    assert(attr.isa<IntegerAttr>());
    return attr.cast<IntegerAttr>().getInt();
  }
  if (auto block_arg = v.dyn_cast_or_null<BlockArgument>()) {
    int index = block_arg.getArgNumber();
    auto parent_op = v.getParentBlock()->getParentOp();
    auto funcOp = dyn_cast_or_null<FuncOp>(parent_op);
    if (funcOp) {
      func::CallOp callee = getCallOp(funcOp);
      return getAddress(callee.getOperand(index));
    }
  }
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
  int elm_bits = etype.getIntOrFloatBitWidth();
  return align_up(elm_count * elm_bits, (int64_t)8) / 8;
}

double getDtypeSize(Value v) {
  auto type = v.getType().cast<RankedTensorType>();
  auto etype = getStorageType(v);
  double elm_bytes = (double)etype.getIntOrFloatBitWidth() / 8;
  return elm_bytes;
}

int64_t getNumElements(Value v) {
  if (v.getType().isa<RankedTensorType>() == false) {
    return 0;
  }
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

void getGlobalShape(Value v, int *shape, int dim) {
  for (auto v : llvm::enumerate(getShape(v)))
    shape[v.index()] = (int)v.value();
  for (int i = getShape(v).size(); i < dim; ++i)
    shape[i] = 1;
}

void getLocalShape(Value v, int64_t n_step, int64_t h_step, int *shape) {
  int64_t n, c, h, w;
  module::getNCHW(v, n, c, h, w);
  group_info_t gi = LocalGenInterface::getGroupInfo(v, n_step, h_step);
  shape[0] = (int)gi.n_slice;
  shape[1] = (int)c;
  shape[2] = (int)gi.h_slice;
  shape[3] = (int)w;
}

void get128BtyeAlignedStrideForNBit(int *stride, int *shape, int npu_num,
                                    int bit) {
  assert(bit == 8 || bit == 16 || bit == 32);
  int aligned_bit;
  switch (bit) {
  case 8:
    aligned_bit = 128;
    break;
  case 16:
    aligned_bit = 64;
    break;
  case 32:
    aligned_bit = 32;
    break;
  }
  const int cstride = align_up(shape[3] * shape[2], aligned_bit);
  stride[0] = (int)std::ceil((double)shape[1] / npu_num) * cstride;
  stride[1] = cstride;
  stride[2] = shape[3];
  stride[3] = 1;
}

void getCompactStride(int *stride, int *shape, int npu_num) {
  const int cstride = shape[2] * shape[3];
  stride[0] = (int)std::ceil((double)shape[1] / npu_num) * cstride;
  stride[1] = cstride;
  stride[2] = shape[3];
  stride[3] = 1;
}

void getContinousStride(int *stride, int *shape) {
  stride[3] = 1;
  stride[2] = shape[3];
  stride[1] = shape[3] * shape[2];
  stride[0] = stride[1] * shape[1];
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
    auto rtype = type.cast<RankedTensorType>();
    return rtype.getElementType();
  } else if (type.isa<UnrankedTensorType>()) {
    auto rtype = type.cast<UnrankedTensorType>();
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

void getNCHW(llvm::ArrayRef<int64_t> shape, int64_t &n, int64_t &c, int64_t &h,
             int64_t &w, group_type_t group_type) {
  if (group_type == GROUP_NORMAL) {
    module::getNCHW(shape, n, c, h, w, true);
  } else if (group_type == GROUP_SMALL_C) {
    int64_t npu_num = backend::Arch::NPU_NUM;
    auto shape_vec = shape.vec();
    shape_vec.resize(4);
    // shape.size() == 2/1 is for MatMul weight and bias
    if (shape.size() == 2) {
      shape_vec[3] = 1;
      shape_vec[2] = shape[1];
      shape_vec[1] = shape[0];
      shape_vec[0] = 1;
    } else if (shape.size() == 1) {
      shape_vec[3] = 1;
      shape_vec[2] = shape[0];
      shape_vec[1] = 1;
      shape_vec[0] = 1;
    } else if (shape.size() == 4) {
      shape_vec[3] = 1;
      shape_vec[2] = shape[3];
      shape_vec[1] = shape[2];
      shape_vec[0] = shape[1] * shape[0];
      if (shape[2] * shape[1] * shape[0] % npu_num == 0) {
        shape_vec[1] = npu_num;
        shape_vec[0] = shape[2] * shape[1] * shape[0] / npu_num;
      }
    } else if (shape.size() == 5) {
      shape_vec[3] = 1;
      shape_vec[2] = shape[4];
      shape_vec[1] = shape[3];
      shape_vec[0] = shape[2] * shape[1] * shape[0];
      if (shape[3] * shape[2] * shape[1] * shape[0] % npu_num == 0) {
        shape_vec[1] = npu_num;
        shape_vec[0] = shape[3] * shape[2] * shape[1] * shape[0] / npu_num;
      }
    }
    module::getNCHW(shape_vec, n, c, h, w, false);
  } else if (GROUP_MM_INT4 == group_type) {
    assert(shape.size() == 2);
    n = shape[0];
    c = 1;
    h = shape[1];
    w = 1;
  } else {
    module::getNCHW(shape, n, c, h, w, true);
  }
}

void getNCHW(Value v, int64_t &n, int64_t &c, int64_t &h, int64_t &w,
             group_type_t group_type) {
  auto shape = v.getType().cast<RankedTensorType>().getShape();
  getNCHW(shape, n, c, h, w, group_type);
}

void getNCDHW(Value v, int64_t &n, int64_t &c, int64_t &d, int64_t &h,
              int64_t &w, group_type_t group_type) {
  auto shape = v.getType().cast<RankedTensorType>().getShape();
  int num_dims = shape.size();
  if (GROUP_3D == group_type) {
    n = num_dims > 0 ? shape[0] : 1;
    c = num_dims > 1 ? shape[1] : 1;
    d = num_dims > 2 ? shape[2] : 1;
    h = num_dims > 3 ? shape[3] : 1;
    w = 1;
    for (size_t i = 4; i < num_dims; ++i) {
      w *= shape[i];
    }
    return;
  } else if (GROUP_MM_INT4 == group_type) {
    assert(num_dims == 2);
    n = shape[0];
    c = 1;
    d = 1;
    h = shape[1];
    w = 1;
  } else {
    d = 1;
    getNCHW(shape, n, c, h, w, group_type);
  }
}

bool isOpInGroup(Operation *Op, int64_t *group_type) {
  if (Op == nullptr) {
    return false;
  }
  auto parent = Op->getParentOp();
  if (parent != nullptr && isa<tpu::GroupOp>(parent)) {
    if (group_type) {
      if (auto groupop = dyn_cast<tpu::GroupOp>(Op)) {
        *group_type = groupop.getGroupType();
      }
    }
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
  func::CallOp call = nullptr;
  for (auto each_func : m.getOps<FuncOp>()) {
    WalkResult result = each_func.walk<WalkOrder::PreOrder>([&](func::CallOp op) {
      if (!call && op.getCallee() == func.getName()) {
        call = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      break;
  }
  return call;
}

FuncOp getMainFuncOp() { return getFuncOp("main"); }

bool isSign(Value v) {
  auto stype = getStorageType(v);
  if (stype.isUnsignedInteger()) {
    return false;
  }
  return true;
}

bool isWeight(Value v) {
  auto op = v.getDefiningOp();
  if (op == nullptr) {
    return false;
  }
  if (isa<top::WeightOp>(op)) {
    return true;
  }
  return false;
}

bool isAllWeight(Operation *op) {
  for (auto in : op->getOperands()) {
    if (isNone(in) || isWeight(in)) {
      continue;
    }
    return false;
  }
  return true;
}

bool isNone(Value v) { return v.getType().isa<mlir::NoneType>(); }

bool isUnranked(Value v) { return v.getType().isa<mlir::UnrankedTensorType>(); }

void setShapeOrVerify(Value v, llvm::ArrayRef<int64_t> shape) {
  if (isUnranked(v)) {
    auto newType = RankedTensorType::get(shape, getElementType(v));
    v.setType(newType);
  } else {
    auto s = getShape(v);
    if (s != shape) {
      v.dump();
      llvm_unreachable("Shape Verify failed");
    }
  }
}

bool isGlobalBuffer(Value v) {
  const auto op = v.getDefiningOp();
  if (!op)
    return false;
  return isa<tpu::BufferOp>(op);
}

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

Chip getChip() { return chip; }

Mode getMode() {
  auto s = m->getAttrOfType<StringAttr>(Attr::MODE);
  return symbolizeMode(s).value_or(Mode::F32);
}

void setChip(Chip chip_) {
  chip = chip_;
  auto s = stringifyChip(chip_);
  m->setAttr(Attr::CHIP, StringAttr::get(m.getContext(), s));
}

bool isChip(Chip chip_) { return chip == chip_; }

void setMode(Mode mode) {
  auto s = stringifyMode(mode);
  m->setAttr(Attr::MODE, StringAttr::get(ctx, s));
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

State getState() {
  auto s = m->getAttrOfType<StringAttr>(Attr::STATE);
  return symbolizeState(s).value_or(State::TOP_F32);
}

Platform getPlatform() { return platform; }

bool isPlatform(Platform plt) { return platform == plt; }

void setState(State state) {
  auto s = stringifyState(state);
  m->setAttr(Attr::STATE, StringAttr::get(ctx, s));
}

bool isState(State state) { return state == getState(); }

bool isTpuOp(Operation *op) {
  return (op->getDialect()->getNamespace() == "tpu");
}

bool isInt4Op(Operation *op) {
  // if (isa<top::ConvOp, tpu::Conv2DOp>(op)) {
  if (isa<top::ConvOp, top::MatMulOp, tpu::Conv2DOp, tpu::MatMulOp>(op)) {
    if (auto convOp = dyn_cast<top::ConvOp>(op)) {
      if (convOp.parseParam().is_dw)
        return false;
    }
    if (auto convOp = dyn_cast<tpu::Conv2DOp>(op)) {
      if (convOp.parseParam().is_dw)
        return false;
    }
    return true;
  }

  return false;
}

bool isCV18xx() {
  return (chip == Chip::CV183x || chip == Chip::CV182x ||
          chip == Chip::CV181x || chip == Chip::CV180x);
}
bool isBM1684Family() { return (chip == Chip::BM1684); }
bool isBM1684XFamily() {
  return (chip == Chip::BM1684X || chip == Chip::BM1686 || chip == Chip::CV186X);
}
bool isBM1686() { return (chip == Chip::BM1686 || chip == Chip::CV186X); }
bool isBM1684X() { return (chip == Chip::BM1684X); }

ModuleOp getModuleOp() { return m; }

Location getLoc() { return m.getLoc(); }

MLIRContext *getCtx() { return ctx; }

void push_back(FuncOp funcOp) { m.push_back(funcOp); }

double getThreshold(Value v) {
  auto type = getCalibratedType(v);
  return type.getMax();
}

uint32_t getIdx(Value v) {
  uint32_t idx = 0;
  if (auto r = v.dyn_cast<OpResult>()) {
    idx = r.getResultNumber();
  } else if (auto r = v.dyn_cast<BlockArgument>()) {
    idx = r.getArgNumber();
  } else {
    v.dump();
    llvm_unreachable("Not Implemented");
  }
  return idx;
}

void setLoc(Value v, NameLoc loc) {
  if (v.getLoc().isa<NameLoc>()) {
    v.setLoc(loc);
    return;
  }
  if (auto fuse_loc = v.getLoc().dyn_cast<FusedLoc>()) {
    std::vector<mlir::Location> locs = fuse_loc.getLocations();
    uint32_t idx = getIdx(v);
    locs[idx] = loc;
    auto new_loc = FusedLoc::get(v.getContext(), locs);
    v.setLoc(new_loc);
    return;
  }
  if (auto op = v.getDefiningOp()) {
    auto op_loc = op->getLoc();
    if (op_loc.isa<NameLoc>()) {
      op->setLoc(loc);
      return;
    }
    if (auto fuse_loc = op->getLoc().dyn_cast<FusedLoc>()) {
      std::vector<mlir::Location> locs = fuse_loc.getLocations();
      auto idx = getIdx(v);
      locs[idx] = loc;
      auto new_loc = FusedLoc::get(v.getContext(), locs);
      op->setLoc(new_loc);
      return;
    }
  }
  v.dump();
  llvm_unreachable("Not Implemented");
}

NameLoc getLoc(Value v) {
  if (auto loc = v.getLoc().dyn_cast<NameLoc>()) {
    return loc;
  } else if (auto fuse_loc = v.getLoc().dyn_cast<FusedLoc>()) {
    auto locs = fuse_loc.getLocations();
    uint32_t idx = getIdx(v);
    if (auto name_loc = locs[idx].dyn_cast<NameLoc>()) {
      return name_loc;
    }
  } else if (auto op = v.getDefiningOp()) {
    auto loc = op->getLoc();
    if (auto name_loc = loc.dyn_cast<NameLoc>()) {
      return name_loc;
    }
    if (auto fuse_loc = loc.dyn_cast<FusedLoc>()) {
      uint32_t idx = getIdx(v);
      auto locs = fuse_loc.getLocations();
      if (auto name_loc = locs[idx].dyn_cast<NameLoc>()) {
        return name_loc;
      }
    }
  }
  v.dump();
  llvm_unreachable("Not Implemented");
  return nullptr;
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

StringRef getName(Value v) { return getLoc(v).getName().strref(); }

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
    inputs.emplace_back(module::getOriValue(opd));
  }
  auto func = getFuncOp(call.getCallee());
  func.walk([&](ReturnOp op) {
    for (auto output : op.getOperands()) {
      outputs.push_back(output);
    }
  });
}

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
    if (bitwidth == 8) {
      auto pre_op = v.getDefiningOp();
      if (module::isInt4Op(pre_op)) {
        if (auto convOp = dyn_cast<top::ConvOp>(pre_op)) {
          if (convOp.getOutInt8Scale().has_value()) {
            scale = convOp.getOutInt8Scale()
                        .value_or(APFloat(1.0))
                        .convertToDouble();
            zeropoint = int64_t(
                convOp.getOutInt8Zp().value_or(APFloat(0.0)).convertToDouble());
            return;
          }
        } else {
          if (auto matmulOp = dyn_cast<top::MatMulOp>(pre_op)) {
            // break; //todo matmul need support int4
            if (matmulOp.getOutInt8Scale().has_value()) {
              scale = matmulOp.getOutInt8Scale()
                          .value_or(APFloat(1.0))
                          .convertToDouble();
              zeropoint = int64_t(matmulOp.getOutInt8Zp()
                                      .value_or(APFloat(0.0))
                                      .convertToDouble());
              return;
            }
          }
        }
      }
    }

    auto qtype = getCalibratedType(v);
    auto max = qtype.getMax();
    auto min = qtype.getMin();
    sign = min < 0;
    if (asymmetric) {
      getScaleAndZeroPoint(min, max, scale, zeropoint, bitwidth);
    } else {
      zeropoint = 0;
      auto th = std::max(std::abs(max), std::abs(min));
      scale = getScale(th, sign, bitwidth);
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
  if (type.isa<RankedTensorType>() == false) {
    return false;
  }
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

//-----------------------------------------------------------------
// Helper Functions for weight
//-----------------------------------------------------------------
static std::string genWeightFileName(bool &same_name) {
  auto name = getModuleName();
  auto state = getState();
  auto chip_ = getChip();
  auto chip = stringifyChip(chip_);
  auto old_name = m->getAttrOfType<StringAttr>(Attr::WEIGHT_FILE).getValue();
  std::string file_name = name.lower() + std::string("_") +
                          stringifyState(state).lower() + std::string("_") +
                          chip.lower();
  if (!isChip(Chip::ALL)) {
    auto mode = getMode();
    std::string sym = "";
    if (mode == Mode::INT8) {
      sym = isAsymmetric() ? "_asym" : "_sym";
    }
    auto mode_ = stringifyMode(mode);
    file_name += std::string("_") + mode_.lower() + sym;
  }
  auto new_name = file_name + "_weight.npz";
  same_name = (old_name == new_name);
  if (same_name) {
    new_name = file_name + "_weight_fix.npz";
  }
  return new_name;
}

void saveWeight() {
  // check name conflict
  std::set<StringRef> all_names;
  for (auto func : m.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      if (op->getLoc().dyn_cast<NameLoc>() && !module::isOpInGroup(op)) {
        auto name = module::getName(op);
        //if op have more than two regions, it can have the same op Name
        if (all_names.find(name) != all_names.end() && !isa<tpu::YieldOp, tpu::IfOp>(op)) {
          op->dump();
          llvm_unreachable("op name conflict");
        }
        all_names.insert(name);
      }
    });
  }
  bool same_name = true;
  std::string filename_;
  if (weightFileName == "") {
    filename_ = module::genWeightFileName(same_name);
  } else {
    same_name = false;
    filename_ = weightFileName;
  }
  // weight remove unused in npz
  if (wFile == nullptr) {
    if (!same_name) {
      weightFile().save(filename_);
      m->setAttr(Attr::WEIGHT_FILE, StringAttr::get(ctx, filename_));
    }
    return;
  }
  if (wFile->changed() == false && same_name) {
    return;
  }
  std::set<StringRef> weight_names;
  for (auto func : m.getOps<FuncOp>()) {
    func.walk([&](top::WeightOp op) {
      weight_names.insert(module::getName(op.getOperation()));
    });
  }
  std::set<StringRef> npz_names;
  wFile->getAllNames(npz_names);
  std::set<StringRef> dif_names;
  for (auto name : npz_names) {
    if (weight_names.find(name) == weight_names.end()) {
      dif_names.insert(name);
    }
  }
  for (auto &name : dif_names) {
    wFile->deleteTensor(name);
  }
  if (wFile->changed() == false && same_name) {
    return;
  }
  wFile->save(filename_);
  m->setAttr(Attr::WEIGHT_FILE, StringAttr::get(ctx, filename_));
}

void setWeightFileName(const std::string &name) { weightFileName = name; }
void detachWeightFile() { wFile = nullptr; }

mlir::TensorFile &weightFile() {
  if (wFile == nullptr) {
    auto name = m->getAttrOfType<StringAttr>(Attr::WEIGHT_FILE).getValue();
    wFile = std::make_unique<mlir::TensorFile>(name, false);
  }
  return *wFile;
}

} // namespace module
} // namespace tpu_mlir
