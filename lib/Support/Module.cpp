//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <fstream>
#include "tpu_mlir/Backend/Arch.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/ModuleEnum.cpp.inc"

static uint64_t core_addr[8] = {
    // according to TPU1686/sgdnn_tmp/src/sgdnn_api_common.cpp
    0x800000000 | (3UL << 40),  // region0 core0
    0x900000000 | (3UL << 40),  // region0 core1
    0x1000000000 | (3UL << 40), // region1 core2
    0x1100000000 | (3UL << 40), // region1 core3
    0x1200000000 | (3UL << 40), // region2 core4
    0x1300000000 | (3UL << 40), // region2 core5
    0x1900000000 | (3UL << 40), // region3 core6
    0x1A00000000 | (3UL << 40)  // region3 core7
};

namespace tpu_mlir {
namespace module {
struct Attr {
  static constexpr llvm::StringRef STATE = "module.state";
  static constexpr llvm::StringRef CHIP = "module.chip";
  static constexpr llvm::StringRef WEIGHT_FILE = "module.weight_file";
  static constexpr llvm::StringRef FLOPS = "module.FLOPs";
  static constexpr llvm::StringRef CORES = "module.cores";
  static constexpr llvm::StringRef DEVICES = "module.devices";
  static constexpr llvm::StringRef COEFF_ADDR = "module.coeff_addr";
  static constexpr llvm::StringRef COEFF_SIZE = "module.coeff_size";
  static constexpr llvm::StringRef NEURON_ADDR = "module.neuron_addr";
  static constexpr llvm::StringRef NEURON_SIZE = "module.neuron_size";
  static constexpr llvm::StringRef IO_ADDR = "module.io_addr";
  static constexpr llvm::StringRef IO_SIZE = "module.io_size";
  static constexpr llvm::StringRef GMEM_PRIVATE_SIZE = "module.private_size";
  static constexpr llvm::StringRef ASYMMETRIC = "module.asymmetric";
  static constexpr llvm::StringRef MODE = "module.mode";
  static constexpr llvm::StringRef PLATFORM = "module.platform";
  static constexpr llvm::StringRef POSTPROCESS = "module.postprocess";
  static constexpr llvm::StringRef DEVICE_ID = "module.device_id";
  static constexpr llvm::StringRef STEP = "module.step";
  static constexpr llvm::StringRef INPUTS = "module.inputs";
  static constexpr llvm::StringRef OUTPUTS = "module.outputs";
  static constexpr llvm::StringRef TRAIN = "module.train";
  static constexpr llvm::StringRef ADDR_MODE = "module.addr_mode";
  static constexpr llvm::StringRef QUANT_GROUP_SIZE = "module.q_group_size";
  static constexpr llvm::StringRef TOP_RUN_MODE = "module.top_run_mode";
  static constexpr llvm::StringRef DYNAMIC_COEFF_OFFSET = "module.dynamic_coeff_offset";
};

static ModuleOp m = nullptr;
static MLIRContext *ctx = nullptr;
static Chip chip = Chip::ALL;
static Platform platform = Platform::ONNX;
static std::unique_ptr<mlir::TensorFile> wFile = nullptr;
static std::string weightFileName = "";
static bool b_weight_in_mem = false;
static std::string debug_cmd = "";
std::unordered_map<std::string, int> patternMatchCounts;
std::mutex patternMatchCountsMutex;

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

  std::ifstream file("/tmp/debug_cmd");
  if (file.is_open()) {
    std::getline(file, debug_cmd);
    file.close();
  }
}

// int32_t cur_log_level = 0;
void init_loglevel(int32_t log_level) {
    SetLogFlag(log_level);
}

void setWeightInMemFlag(bool enable) {
  b_weight_in_mem = enable;
}

bool getWeightInMemFlag() {
  return b_weight_in_mem;
}

top::NoneOp getNoneOp(Operation *op) {
  assert(op != nullptr);
  if (auto noneOp = dyn_cast<top::NoneOp>(op)) {
    return noneOp;
  }
  FuncOp funcOp;
  if (isa<FuncOp>(op)) {
    funcOp = cast<FuncOp>(op);
  } else if (isOpInGroup(op)) {
    funcOp = cast<FuncOp>(op->getParentOp()->getParentOp());
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

static ModuleOp getModuleOp(Value v) {
  auto parent_op = v.getParentBlock()->getParentOp();
  while (parent_op != nullptr && !isa<ModuleOp>(parent_op)) {
    parent_op = parent_op->getParentOp();
  }
  if (parent_op == nullptr) {
    return nullptr;
  }
  return cast<ModuleOp>(parent_op);
}

ModuleOp getModuleOp(Operation *op) {
  while (op != nullptr && !isa<ModuleOp>(op)) {
    op = op->getParentOp();
  }
  if (op == nullptr) {
    return nullptr;
  }
  return cast<ModuleOp>(op);
}

Value getOriValue(Value v) {
  auto s = getModuleOp(v);
  if (!s) {
    return v;
  }
  if (auto block_arg = v.dyn_cast_or_null<BlockArgument>()) {
    int idx = block_arg.getArgNumber();
    // blockargument have multi-layers nest.
    FuncOp func_op;
    if (isa<FuncOp>(v.getParentBlock()->getParentOp()))
      func_op = cast<FuncOp>(v.getParentBlock()->getParentOp());
    else if (isa<tpu::LoopOp, tpu::IfOp, top::LoopOp, top::IfOp>(
                 v.getParentBlock()->getParentOp())) {
      return getOriValue(v.getParentBlock()->getParentOp()->getOperand(idx));
    } else
      func_op = v.getParentBlock()->getParentOp()->getParentOfType<FuncOp>();

    if (func_op) {
      // cur call op
      auto call_op = getCallOp(func_op);
      // pre call op
      auto operand = call_op.getOperand(idx);
      if (operand.isa<BlockArgument>()) {
        auto find_root = [](auto &&Me, Value v) -> Value {
          if (v.isa<BlockArgument>()) {
            int index = dyn_cast<BlockArgument>(v).getArgNumber();
            FuncOp func_op;
            if (isa<FuncOp>(v.getParentBlock()->getParentOp()))
              func_op = cast<FuncOp>(v.getParentBlock()->getParentOp());
            else
              func_op =
                  v.getParentBlock()->getParentOp()->getParentOfType<FuncOp>();
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
      auto pre_func_op = getFuncOp(s, pre_call_op.getCallee());
      auto return_op = dyn_cast<ReturnOp>(pre_func_op.front().back());
      return return_op.getOperand(result.getResultNumber());
    }
  } else if (auto pre_op = v.getDefiningOp()) {
    if (isa<func::CallOp>(pre_op)) {
      auto call_op = dyn_cast<func::CallOp>(pre_op);
      int index = v.cast<OpResult>().getResultNumber();
      for (auto func : s.getOps<FuncOp>()) {
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

static void updateModuleTypes(ModuleOp s) {
  Builder builder(ctx);
  // update callee func's return types
  for (auto func : s.getOps<FuncOp>()) {
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
  for (auto func : s.getOps<FuncOp>()) {
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
  auto mainFunc = getMainFuncOp(s);
  Block &entryBlock = mainFunc.front();
  auto returnOp = dyn_cast<ReturnOp>(entryBlock.back()).getOperation();
  std::vector<Type> returns;
  for (uint32_t i = 0; i < returnOp->getNumOperands(); ++i) {
    returns.push_back(returnOp->getOperand(i).getType());
  }
  std::vector<Type> inputs;
  auto args = mainFunc.getArguments();
  for (auto arg : args) {
    inputs.push_back(arg.getType());
  }
  auto fnType = builder.getFunctionType(llvm::ArrayRef<Type>{inputs},
                                        llvm::ArrayRef<Type>{returns});
  mainFunc.setType(fnType);
}

void updateModuleTypes() {
  auto modules = getAllModules();
  for (auto s : *modules) {
    updateModuleTypes(s);
  }
}

static void removeUnusedOp(ModuleOp submodule) {
  std::vector<Operation *> all_ops;
  for (auto func : submodule.getOps<FuncOp>()) {
    // for to support nested region's op
    func.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (!isa<ReturnOp, FuncOp, tpu::YieldOp, top::YieldOp, top::InputOp>(op))
        all_ops.push_back(op);
    });
  }
  for (auto iter = all_ops.rbegin(); iter != all_ops.rend(); iter++) {
    if ((*iter)->use_empty()) {
      (*iter)->erase();
    }
  }
}

void removeUnusedOp() {
  auto modules = getAllModules();
  for (auto s : *modules) {
    removeUnusedOp(s);
  }
}

int64_t getAddress(Value v) {
  if (v.getType().isa<NoneType>()) {
    return 0;
  }
  auto attr = v.getType().cast<RankedTensorType>().getEncoding();
  if (attr) {
    if (isa<IntegerAttr>(attr)) {
      return attr.cast<IntegerAttr>().getInt();
    } else if (isa<tpu::CPInterleaveAttr>(attr)) {
      return attr.cast<tpu::CPInterleaveAttr>().getAddress();
    }
  }
  if (auto block_arg = v.dyn_cast_or_null<BlockArgument>()) {
    int index = block_arg.getArgNumber();
    auto parent_op = v.getParentBlock()->getParentOp();
    FuncOp funcOp;

    if (isa<FuncOp>(parent_op))
      funcOp = cast<FuncOp>(parent_op);
    else
      funcOp = parent_op->getParentOfType<FuncOp>();

    if (funcOp) {
      func::CallOp callee = getCallOp(funcOp);
      return getAddress(callee.getOperand(index));
    }
  }
  return 0;
}

void setAddress(Value v, int64_t addr) {
  auto type = v.getType().cast<RankedTensorType>();
  auto _8chAttr = dyn_cast_or_null<tpu::CPInterleaveAttr>(type.getEncoding());
  if (!_8chAttr) {
    Builder builder(v.getContext());
    auto addrAttr = builder.getI64IntegerAttr(addr);
    auto new_type =
        RankedTensorType::get(type.getShape(), type.getElementType(), addrAttr);
    v.setType(new_type);
  } else {
    auto index = _8chAttr.getRegionId();
    if (index == -1)
      addr = _8chAttr.getAddress();
    set8chAddress(v, index, _8chAttr.getOffset(), addr);
  }
}

void set8chAddress(Value v, size_t index, int64_t offset, int64_t addr) {
  auto type = v.getType().cast<RankedTensorType>();
  if (index != -1)
    addr = offset + core_addr[index];
  auto ddrAttr =
      tpu::CPInterleaveAttr::get(v.getContext(), index, offset, addr);
  auto ddrType = mlir::RankedTensorType::get(type.getShape(),
                                             type.getElementType(), ddrAttr);
  v.setType(ddrType);
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
  if (!isUnranked(v)) {
    auto type = v.getType().cast<RankedTensorType>();
    return type.getShape();
  } else {
    return v.getType().cast<UnrankedTensorType>().getShape();
  }
}

std::vector<int64_t> getShapeVec(Value v) {
  llvm::ArrayRef<int64_t> shape = getShape(v);
  std::vector<int64_t> shapeV(shape.begin(), shape.end());
  return shapeV;
}

void setShape(Value v, llvm::ArrayRef<int64_t> shape) {
  auto newType = RankedTensorType::get(shape, getElementType(v));
  v.setType(newType);
}

void getGlobalShape(Value v, int *shape, int dim) {
  for (auto v : llvm::enumerate(getShape(v)))
    shape[v.index()] = (int)v.value();
  for (int i = getShape(v).size(); i < dim; ++i)
    shape[i] = 1;
}

void getLocalShape(Value v, int64_t n_step, int64_t h_step, int *shape) {
  group_info_t gi = LocalGenInterface::getGroupInfo(v, n_step, h_step);
  int64_t n, c, d, h, w;
  module::getNCDHW(v, n, c, d, h, w, (group_type_t)gi.type);
  shape[0] = (int)gi.n_slice;
  shape[1] = (int)c;
  shape[2] = (int)gi.h_slice;
  shape[3] = (int)w;
  if (module::isBM1684Family() && module::isUniformQuantized(v)) {
    shape[1] *= d;
  } else {
    shape[0] *= d;
  }
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

i32_array_t getI32Array(std::optional<ArrayAttr> arrayAttr, int64_t num_elem,
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

i64_array_t getI64Array(std::optional<ArrayAttr> arrayAttr, int64_t num_elem,
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

f64_array_t getF64Array(std::optional<ArrayAttr> arrayAttr, int64_t num_elem,
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

RankedTensorType getTypeLike(Value v, llvm::ArrayRef<int64_t> shape) {
  return RankedTensorType::get(shape, getElementType(v));
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
    } else if (shape.size() == 3) {
      shape_vec[3] = 1;
      shape_vec[2] = shape[2];
      shape_vec[1] = shape[1] * shape[0];
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
    d = num_dims > 4 ? shape[2] : 1;
    h = num_dims > 4 ? shape[3] : (num_dims > 2 ? shape[2] : 1);
    w = 1;
    for (int i = (num_dims > 4) ? 4 : 3; i < num_dims; i++)
      w *= shape[i];
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
  if (isa_and_nonnull<tpu::GroupOp>(parent)) {
    if (group_type) {
      if (auto groupop = dyn_cast<tpu::GroupOp>(Op)) {
        *group_type = groupop.getGroupType();
      }
    }
    return true;
  }
  return false;
}

bool isOpInCoreParallel(Operation *Op) {
  if (Op == nullptr) {
    return false;
  }
  auto parent = Op->getParentOp();
  if (isa_and_nonnull<tpu::CoreParallelOp>(parent)) {
    return true;
  }
  return false;
}

// op in [CoreBegin, CoreEnd]
bool isOpInCoreMatch(Operation *op) {
  while (!op->use_empty()) {
    op = *op->user_begin();
    if (isa<func::ReturnOp, tpu::CoreBeginOp>(op)) {
      return false;
    }
    if (isa<tpu::CoreEndOp>(op)) {
      return true;
    }
  }
  return false;
}

bool isOpInDevParallel(Operation *op) {
  while (!op->use_empty()) {
    op = *op->user_begin();
    if (isa<func::ReturnOp, tpu::DevBeginOp>(op)) {
      return false;
    }
    if (isa<tpu::DevEndOp>(op)) {
      return true;
    }
  }
  return false;
}

bool isOpInBlock(Operation *op) {
  if (op == nullptr) {
    return false;
  }
  auto parent = op->getParentOp();
  if (parent == nullptr) {
    return false;
  }
  if (isa<func::FuncOp>(parent)) {
    return false;
  }
  return true;
}

FuncOp getFuncOp(ModuleOp mod, StringRef func_name) {
  for (auto func : mod.getOps<FuncOp>()) {
    if (func.getName() == func_name) {
      return func;
    }
  }
  llvm::errs() << "Can't find FuncOp:" << func_name << "\n";
  llvm_unreachable("Error getFuncOp !!\n");
  return nullptr;
}

func::CallOp getCallOp(FuncOp func) {
  auto parent = func->getParentOp();
  auto s = cast<ModuleOp>(parent);
  func::CallOp call = nullptr;
  for (auto each_func : s.getOps<FuncOp>()) {
    WalkResult result =
        each_func.walk<WalkOrder::PreOrder>([&](func::CallOp op) {
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

FuncOp getMainFuncOp(ModuleOp module) { return getFuncOp(module, "main"); }

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

bool isActive(Value v) {
  if (module::isNone(v) || module::isWeight(v)) {
    return false;
  }
  return true;
}

bool isDynWeight(Value v) {
  auto op = v.getDefiningOp();
  if (op == nullptr) {
    return false;
  }
  if (op->hasAttr("dynamic_weight")) {
    // use code below to tag dynamic weight op
    // op->setAttr("dynamic_weight", , rewriter.getBoolAttr(true));
    return true;
  }
  return false;
}

bool isShapeRelatedOp(Value v) {
  auto op = v.getDefiningOp();
  if (op == nullptr) {
    return false;
  }
  if (isa<top::ShapeOp, tpu::ShapeOp, tpu::ShapeSliceOp, tpu::ShapeCastOp>(
          op)) {
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

bool isDynamicShape(Value v) {
  int ret = false;
  auto tensorTy = v.getType().dyn_cast<RankedTensorType>();
  if (tensorTy) {
    for (int64_t dim : tensorTy.getShape()) {
      if (ShapedType::isDynamic(dim) || dim == 0)
        ret = true;
    }
  }
  return ret;
}

void setShapeOrVerify(Value v, llvm::ArrayRef<int64_t> shape) {
  if (isUnranked(v) || isDynamicShape(v)) {
    auto newType = RankedTensorType::get(shape, getElementType(v));
    v.setType(newType);
  } else {
    auto s = getShape(v);
    /* unranked tensor is okay, for example:
       tensor<*xf32>->tensor<1xf32> */
    if ((std::max(s.size(), shape.size()) > 1) && s != shape) {
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

int64_t getCoeffSize(ModuleOp s) {
  return s->getAttrOfType<IntegerAttr>(Attr::COEFF_SIZE).getInt();
}

void setCoeffSize(ModuleOp s, int64_t size) {
  s->setAttr(Attr::COEFF_SIZE, Builder(ctx).getI64IntegerAttr(size));
}

int64_t getDynamicOffset(ModuleOp s) {
  return s->getAttrOfType<IntegerAttr>(Attr::DYNAMIC_COEFF_OFFSET).getInt();
}

void setDynamicOffset(ModuleOp s, int64_t size) {
  s->setAttr(Attr::DYNAMIC_COEFF_OFFSET, Builder(ctx).getI64IntegerAttr(size));
}

int64_t getGmemPrivateSize(ModuleOp s) {
  return s->getAttrOfType<IntegerAttr>(Attr::GMEM_PRIVATE_SIZE).getInt();
}

void setGmemPrivateSize(ModuleOp s, int64_t size) {
  s->setAttr(Attr::GMEM_PRIVATE_SIZE, Builder(ctx).getI64IntegerAttr(size));
}

int64_t getCoreNum() {
  if (auto cores = m->getAttrOfType<IntegerAttr>(Attr::CORES))
    return cores.getInt();
  return 1;
}

void setCoreNum(int64_t core_num) {
  m->setAttr(Attr::CORES, Builder(ctx).getI64IntegerAttr(core_num));
}

int64_t getDeviceNum() {
  if (auto devices = m->getAttrOfType<IntegerAttr>(Attr::DEVICES)) {
    return devices.getInt();
  }
  return 1;
}

void setDeviceNum(int64_t device_num) {
  m->setAttr(Attr::DEVICES, Builder(ctx).getI64IntegerAttr(device_num));
}

int64_t getCoeffAddr(ModuleOp s) {
  return s->getAttrOfType<IntegerAttr>(Attr::COEFF_ADDR).getInt();
}

void setCoeffAddr(ModuleOp s, int64_t addr) {
  s->setAttr(Attr::COEFF_ADDR, Builder(ctx).getI64IntegerAttr(addr));
}

int64_t getNeuronSize(ModuleOp s) {
  return s->getAttrOfType<IntegerAttr>(Attr::NEURON_SIZE).getInt();
}

void setNeuronSize(ModuleOp s, int64_t size) {
  s->setAttr(Attr::NEURON_SIZE, Builder(ctx).getI64IntegerAttr(size));
}

int64_t getNeuronAddr(ModuleOp s) {
  return s->getAttrOfType<IntegerAttr>(Attr::NEURON_ADDR).getInt();
}

void setNeuronAddr(ModuleOp s, int64_t addr) {
  s->setAttr(Attr::NEURON_ADDR, Builder(ctx).getI64IntegerAttr(addr));
}

int64_t getIOSize(ModuleOp s) {
  if (s->hasAttrOfType<IntegerAttr>(Attr::IO_SIZE)) {
    return s->getAttrOfType<IntegerAttr>(Attr::IO_SIZE).getInt();
  }
  return 0;
}

void setIOSize(ModuleOp s, int64_t size) {
  s->setAttr(Attr::IO_SIZE, Builder(ctx).getI64IntegerAttr(size));
}

int64_t getIOAddr(ModuleOp s) {
  if (s->hasAttrOfType<IntegerAttr>(Attr::IO_ADDR)) {
    return s->getAttrOfType<IntegerAttr>(Attr::IO_ADDR).getInt();
  }
  return 0;
}

void setIOAddr(ModuleOp s, int64_t addr) {
  s->setAttr(Attr::IO_ADDR, Builder(ctx).getI64IntegerAttr(addr));
}

llvm::StringRef getPostprocess() {
  if (m->hasAttrOfType<StringAttr>(Attr::POSTPROCESS)) {
    return m->getAttrOfType<StringAttr>(Attr::POSTPROCESS).strref();
  }
  return llvm::StringRef("");
}

void setPostprocess(StringRef post) {
  m->setAttr(Attr::POSTPROCESS, Builder(ctx).getStringAttr(post));
}

Chip getChip() { return chip; }

Mode getMode() {
  if (false == m->hasAttrOfType<StringAttr>(Attr::MODE)) {
    return Mode::F32;
  }
  auto s = m->getAttrOfType<StringAttr>(Attr::MODE);
  return symbolizeMode(s).value_or(Mode::F32);
}

bool isBF16Modes() {
  auto s = m->getAttrOfType<StringAttr>(Attr::MODE);
  auto mode = symbolizeMode(s).value_or(Mode::F32);
  return mode == Mode::BF16 || mode == Mode::W8BF16 || mode == Mode::W4BF16;
}

bool isF16Modes() {
  auto s = m->getAttrOfType<StringAttr>(Attr::MODE);
  auto mode = symbolizeMode(s).value_or(Mode::F32);
  return mode == Mode::F16 || mode == Mode::W8F16 || mode == Mode::W4F16;
}

bool isF8Modes() {
  auto s = m->getAttrOfType<StringAttr>(Attr::MODE);
  auto mode = symbolizeMode(s).value_or(Mode::F32);
  return mode == Mode::F8 || mode == Mode::F8E4M3 || mode == Mode::F8E5M2;
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

std::shared_ptr<std::vector<ModuleOp>> getAllModules() {
  auto modules = std::make_shared<std::vector<ModuleOp>>();
  auto sub = m.getOps<ModuleOp>();
  if (sub.empty()) {
    modules->push_back(m);
  } else {
    modules->assign(sub.begin(), sub.end());
  }
  return std::move(modules);
}

int getNumSubModule() {
  auto sub = m.getOps<ModuleOp>();
  return std::distance(sub.begin(), sub.end());
}

void setSubModuleId(ModuleOp sub, int64_t device_id, int64_t step) {
  sub->setAttr(Attr::DEVICE_ID,
               Builder(sub.getContext()).getI64IntegerAttr(device_id));
  sub->setAttr(Attr::STEP, Builder(sub.getContext()).getI64IntegerAttr(step));
}

void getSubModuleId(ModuleOp sub, int64_t &device_id, int64_t &step) {
  device_id = sub->getAttrOfType<IntegerAttr>(Attr::DEVICE_ID).getInt();
  step = sub->getAttrOfType<IntegerAttr>(Attr::STEP).getInt();
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

int getQuantGroupSize() {
  if (m->hasAttrOfType<IntegerAttr>(Attr::QUANT_GROUP_SIZE)) {
    return m->getAttrOfType<IntegerAttr>(Attr::QUANT_GROUP_SIZE)
        .getValue()
        .getSExtValue();
  }
  return 0;
}

void setQuantGroupSize(int q_group_size) {
  auto intType = IntegerType::get(ctx, 64);
  m->setAttr(Attr::QUANT_GROUP_SIZE, IntegerAttr::get(intType, q_group_size));
}

bool isTrain() {
  if (m->hasAttrOfType<BoolAttr>(Attr::TRAIN)) {
    return m->getAttrOfType<BoolAttr>(Attr::TRAIN).getValue();
  }
  return false;
}

void setTrain(bool is_train) {
  m->setAttr(Attr::TRAIN, BoolAttr::get(ctx, is_train));
}

void setAddrMode(AddrMode mode) {
  auto s = stringifyAddrMode(mode);
  m->setAttr(Attr::ADDR_MODE, StringAttr::get(ctx, s));
}

AddrMode getAddrMode() {
  if (m->hasAttrOfType<StringAttr>(Attr::ADDR_MODE)) {
    auto s = m->getAttrOfType<StringAttr>(Attr::ADDR_MODE);
    return symbolizeAddrMode(s).value_or(AddrMode::BASIC);
  }
  return AddrMode::BASIC;
}

bool isAddrMode(AddrMode mode) { return mode == getAddrMode(); }

void setTopRunMode(TopRunMode mode) {
  auto s = stringifyTopRunMode(mode);
  m->setAttr(Attr::TOP_RUN_MODE, StringAttr::get(ctx, s));
}

TopRunMode getTopRunMode() {
  if (m->hasAttrOfType<StringAttr>(Attr::TOP_RUN_MODE)) {
    auto s = m->getAttrOfType<StringAttr>(Attr::TOP_RUN_MODE);
    return symbolizeTopRunMode(s).value_or(TopRunMode::STATIC);
  }
  return TopRunMode::STATIC;
}

bool isDynamic() { return getTopRunMode() == TopRunMode::DYNAMIC; }

bool isDebugCmdEnable(std::string cmd_str) {
  if (debug_cmd.find(cmd_str) != std::string::npos) {
    return true;
  }
  return false;
}

State getState() {
  auto s = m->getAttrOfType<StringAttr>(Attr::STATE);
  return symbolizeState(s).value_or(State::TOP_F32);
}

void setState(State state) {
  auto s = stringifyState(state);
  m->setAttr(Attr::STATE, StringAttr::get(ctx, s));
}

Platform getPlatform() { return platform; }

bool isPlatform(Platform plt) { return platform == plt; }

void setInputs(ArrayRef<StringRef> inputs) {
  m->setAttr(Attr::INPUTS, Builder(ctx).getStrArrayAttr(inputs));
}

std::shared_ptr<std::vector<StringRef>> getInputs() {
  auto inputs = m->getAttrOfType<ArrayAttr>(Attr::INPUTS);
  auto data = std::make_shared<std::vector<StringRef>>();
  for (auto en : llvm::enumerate(inputs)) {
    auto attr = en.value().dyn_cast<StringAttr>();
    data->push_back(attr.strref());
  }
  return std::move(data);
}

void setOutputs(ArrayRef<StringRef> outputs) {
  m->setAttr(Attr::OUTPUTS, Builder(ctx).getStrArrayAttr(outputs));
}

std::shared_ptr<std::vector<StringRef>> getOutputs() {
  auto outputs = m->getAttrOfType<ArrayAttr>(Attr::OUTPUTS);
  auto data = std::make_shared<std::vector<StringRef>>();
  for (auto en : llvm::enumerate(outputs)) {
    auto attr = en.value().dyn_cast<StringAttr>();
    data->push_back(attr.strref());
  }
  return std::move(data);
}

void removeAttr(mlir::Operation *op, std::string attr_name) {
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    if (attr.getName() != attr_name) {
      attrs.push_back(attr);
    }
  }
  op->setAttrs(attrs);
}

bool isState(State state) { return state == getState(); }

bool isSubnetDividedState() {
  return isState(module::State::TPU_DIVIDED) || isState(module::State::TPU_ADDRESSED);
}

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
  return (chip == Chip::BM1684X || chip == Chip::BM1688 ||
          chip == Chip::CV186X || chip == Chip::MARS3 || chip == Chip::SG2380);
}
bool isBM1690Family() { return (chip == Chip::BM1690); }
bool isSG2380() { return (chip == Chip::SG2380); }
bool isBM1688() {
  return (chip == Chip::BM1688 || chip == Chip::CV186X);
}
bool isBM1684X() { return (chip == Chip::BM1684X); }
bool isMARS3() { return (chip == Chip::MARS3); }

ModuleOp getModuleOp() { return m; }

Location getLoc() { return m.getLoc(); }

MLIRContext *getCtx() { return ctx; }

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

NameLoc getLocLike(Operation *op, llvm::StringRef suffix) {
  return getLocLike(op->getResult(0), suffix);
}

NameLoc getLocLike(Value v, llvm::StringRef suffix) {
  auto name = getName(v);
  auto new_name = name.str() + "_" + suffix.str();
  Builder builder(v.getContext());
  return NameLoc::get(builder.getStringAttr(new_name));
}

void setLocSuffix(Operation *op, llvm::StringRef suffix) {
  if (op->getNumResults() > 1) {
    std::vector<Location> locs;
    for (auto r : op->getResults()) {
      auto loc = getLocLike(r, suffix);
      locs.push_back(loc);
    }
    auto new_loc = FusedLoc::get(op->getContext(), locs);
    op->setLoc(new_loc);
  } else {
    auto loc = getLocLike(op->getResult(0), suffix);
    op->setLoc(loc);
  }
}

StringRef getName(Operation *op, int index) {
  if (auto module = dyn_cast<ModuleOp>(op)) {
    return module.getName().value_or("Unknown");
  }

  if (auto func = dyn_cast<FuncOp>(op)) {
    return func.getName();
  }
  if (isa<mlir::func::CallOp>(op)) {
    return "func.call";
  }
  if (isa<ReturnOp>(op)) {
    return "func.return";
  }
  if (isa<top::NoneOp>(op)) {
    return "NoneOp";
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
  llvm::errs() << "op has no name location!!!\n";
  op->dump();
  llvm_unreachable("op has no name location!!!");
  return "";
}

StringRef getName(Value v) { return getLoc(v).getName().strref(); }

void getInputsOutputs(ModuleOp s, std::vector<Value> &inputs,
                      std::vector<Value> &outputs) {
  auto main_func = getMainFuncOp(s);
  auto args = main_func.front().getArguments();
  for (auto &arg : args) {
    for (auto user : arg.getUsers()) {
      if (auto op = dyn_cast<top::InputOp>(user)) {
        inputs.push_back(op.getOutput());
      } else {
        llvm_unreachable("arg should only be used for InputOp.");
        user->dump();
      }
    }
  }
  // main_func.walk([&](top::InputOp op) { inputs.push_back(op.getOutput()); });
  main_func.walk([&](ReturnOp op) {
    for (auto out : op.getOperands()) {
      auto result = out.cast<OpResult>();
      auto call_op = result.getDefiningOp<func::CallOp>();
      if (call_op) {
        auto func_op = getFuncOp(s, call_op.getCallee());
        auto return_op = dyn_cast<ReturnOp>(func_op.front().back());
        assert(return_op);
        outputs.push_back(return_op.getOperand(result.getResultNumber()));

        func_op.walk([&](tpu::OutBufferOp op) {
          // llvm::errs() <<"ModuleOp dump
          // OutBufferOp:"<<getName(op->getResult(0)).str()<<" as outputs\n";
          bool need_dump = op.getNeedDump();
          if (need_dump) {
            outputs.push_back(op->getResult(0));
          }
        });
      } else {
        outputs.push_back(out);
      }
    }
  });
}

bool isSameOp(Operation *op0, Operation *op1) {
  if (op0 == nullptr || op1 == nullptr) {
    return false;
  }
  if (op0->getName() != op1->getName()) {
    return false;
  }
  if (op0->getNumOperands() != op1->getNumOperands()) {
    return false;
  }
  for (auto it : llvm::zip(op0->getOperands(), op1->getOperands())) {
    if (std::get<0>(it) != std::get<1>(it)) {
      return false;
    }
  }
  if (op0->getNumResults() != op1->getNumResults()) {
    return false;
  }
  for (auto it : llvm::zip(op0->getResultTypes(), op1->getResultTypes())) {
    if (std::get<0>(it) != std::get<1>(it)) {
      return false;
    }
  }
  if (false == op0->getAttrs().equals(op1->getAttrs())) {
    return false;
  }
  return true;
}

void getInputsOutputs(func::CallOp call, std::vector<Value> &inputs,
                      std::vector<Value> &outputs) {
  for (auto opd : call.getOperands()) {
    inputs.emplace_back(module::getOriValue(opd));
  }
  auto md = getModuleOp(call);
  auto func = getFuncOp(md, call.getCallee());
  func.walk([&](ReturnOp op) {
    for (auto output : op.getOperands()) {
      outputs.push_back(output);
    }
  });
  func.walk([&](tpu::OutBufferOp op) {
    llvm::errs() << "func dump OutBufferOp:" << getName(op->getResult(0)).str()
                 << " as outputs\n";
    bool need_dump = op.getNeedDump();
    if (need_dump) {
      outputs.push_back(op->getResult(0));
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
  } else if (bitwidth == 16) {
    qmin = rmin < 0 ? -32768 : 0;
    qmax = rmin < 0 ? 32767 : 65535;
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
  } else if (bitwidth == 4) {
    if (sign) {
      return threshold / 7.0;
    } else {
      return threshold / 15.0;
    }
  } else if (bitwidth == 16) {
    if (sign) {
      return threshold / 32767.0;
    } else {
      return threshold / 65535.0;
    }
  } else {
    llvm_unreachable("not support");
    return 0;
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
  } else if (auto ccop = dyn_cast<top::CompareConstOp>(v.getDefiningOp())) {
    if (ccop.getMode().str() == "And")
      llvm_unreachable("calibration info not set for compareconst And");
    scale = 1.0;
    zeropoint = 0;
    sign = 0;
  } else {
    v.dump();
    llvm_unreachable("can't get scale and zeropoint");
  }
}

bool isScalar(mlir::Operation *op) {
  if (op->hasTrait<trait::ScalarProducer>()) {
    auto is_scalar = op->getAttr("is_scalar").cast<BoolAttr>().getValue();
    return is_scalar;
  }
  return false;
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
// Helper Functions for op translate
//-----------------------------------------------------------------
mlir::Value opSliceAxis(PatternRewriter &rewriter, mlir::Value v, int64_t axis,
                        int64_t offset, int64_t length, std::string mode) {
  auto stype = module::getStorageType(v);
  auto shape = module::getShape(v);
  if (axis < 0) {
    axis += shape.size();
  }
  std::vector<int64_t> new_shape(shape);
  new_shape[axis] = length;
  assert(offset + length <= shape[axis]);
  auto new_type = RankedTensorType::get(new_shape, module::getElementType(v));
  auto suffix = std::to_string(axis) + "_" + std::to_string(offset);
  if (isWeight(v)) {
    auto op = v.getDefiningOp<top::WeightOp>();
    if (stype.isBF16() || stype.isF16()) {
      auto data = op.read<uint16_t>();
      auto new_data =
          tensor_slice(data->data(), shape, axis, offset, length, mode);
      return top::WeightOp::create<uint16_t>(op, suffix, *new_data, new_type);
    } else if (stype.isSignedInteger(8) || stype.isSignlessInteger(8)) {
      auto data = op.read<int8_t>();
      auto new_data =
          tensor_slice(data->data(), shape, axis, offset, length, mode);
      return top::WeightOp::create<int8_t>(op, suffix, *new_data, new_type);
    } else if (stype.isUnsignedInteger(8)) {
      auto data = op.read<uint8_t>();
      auto new_data =
          tensor_slice(data->data(), shape, axis, offset, length, mode);
      return top::WeightOp::create<uint8_t>(op, suffix, *new_data, new_type);
    } else if (stype.isF32()) {
      auto data = op.read<float>();
      auto new_data =
          tensor_slice(data->data(), shape, axis, offset, length, mode);
      return top::WeightOp::create<float>(op, suffix, *new_data, new_type);
    } else if (stype.isInteger(32)) {
      auto data = op.read<int32_t>();
      auto new_data =
          tensor_slice(data->data(), shape, axis, offset, length, mode);
      return top::WeightOp::create<int32_t>(op, suffix, *new_data, new_type);
    }
    op.dump();
    llvm_unreachable("Not Implemented");
  } else {
    std::string name = getName(v).str() + "_" + suffix;
    auto loc = NameLoc::get(rewriter.getStringAttr(name));
    std::vector<NamedAttribute> attrs;
    std::vector<int64_t> offsets(shape.size(), 0);
    offsets[axis] = offset;
    attrs.emplace_back(
        rewriter.getNamedAttr("offset", rewriter.getI64ArrayAttr(offsets)));
    std::vector<int64_t> steps(shape.size(), 1);
    attrs.emplace_back(
        rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr(steps)));
    std::vector<int64_t> ends(shape.size(), -1);
    attrs.emplace_back(
        rewriter.getNamedAttr("ends", rewriter.getI64ArrayAttr(ends)));
    rewriter.setInsertionPointAfterValue(v);
    std::vector<Value> operands;
    operands.emplace_back(v);
    auto none = module::getNoneOp(v.getDefiningOp());
    operands.emplace_back(none);
    operands.emplace_back(none);
    operands.emplace_back(none);
    operands.emplace_back(none);
    auto sliceOp =
        rewriter.create<tpu::SliceOp>(loc, new_type, operands, attrs);
    return sliceOp.getOutput();
  }
}

//-----------------------------------------------------------------
// Helper Functions for weight
//-----------------------------------------------------------------
static std::string genWeightFileName(bool &same_name) {
  auto name = getName(m);
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
  auto modules = getAllModules();
  for (auto s : *modules) {
    for (auto func : s.getOps<FuncOp>()) {
      func.walk([&](Operation *op) {
        if (op->getLoc().dyn_cast<NameLoc>() && !module::isOpInBlock(op) &&
            !isa<func::ReturnOp, func::CallOp, func::FuncOp, tpu::YieldOp,
                 tpu::IfOp, top::InputOp>(op)) {
          auto name = module::getName(op);
          // if op have more than two regions, it can have the same op Name
          ASSERT_OP(all_names.find(name) == all_names.end(), op);
          all_names.insert(name);
        }
      });
    }
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
      setWeightFileAttr(filename_);
    }
    return;
  }
  if (wFile->changed() == false && same_name) {
    return;
  }
  std::set<StringRef> weight_names;
  for (auto s : *modules) {
    for (auto func : s.getOps<FuncOp>()) {
      func.walk([&](top::WeightOp op) {
        weight_names.insert(module::getName(op.getOperation()));
      });
    }
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
  setWeightFileAttr(filename_);
}

void setWeightFileName(const std::string &name) { weightFileName = name; }
void detachWeightFile() { wFile = nullptr; }
void setWeightFileAttr(const std::string &name) {
  m->setAttr(Attr::WEIGHT_FILE, StringAttr::get(ctx, name));
}
llvm::StringRef getWeightFileAttr() {
  return m->getAttrOfType<StringAttr>(Attr::WEIGHT_FILE).getValue();
}

mlir::TensorFile &weightFile() {
  if (wFile == nullptr) {
    auto name = getWeightFileAttr();
    wFile = std::make_unique<mlir::TensorFile>(name, false);
  }
  return *wFile;
}

//-----------------------------------------------------------------
// Helper for shape op inference
//-----------------------------------------------------------------
void ShapeHelper::bindShapeInfo(const Value &v,
                                const std::vector<int64_t> &shape) {
  _shape_info[v] = shape;
}

std::vector<int64_t> ShapeHelper::getShapeInfo(const Value &v) {
  return _shape_info.at(v);
}

bool ShapeHelper::isShape(const Value &v) {
  return _shape_info.find(v) != _shape_info.end();
}

void bindShapeTensorValue(const Value &v, const std::vector<int64_t> &shape) {
  ShapeHelper::getInstance().bindShapeInfo(v, shape);
}

std::vector<int64_t> getShapeTensorValue(const Value &v) {
  return ShapeHelper::getInstance().getShapeInfo(v);
}

bool isShape(const Value &v) { return ShapeHelper::getInstance().isShape(v); }

std::vector<int64_t>
commonShapeValInfer(mlir::Operation *op,
                    const std::vector<std::vector<int64_t>> &in_shapes_v,
                    const std::vector<int64_t> &out_shape) {
  // support scalar
  // assert(out_shape.size() == 1 || out_shape.size() == 0);
  // auto real_out_size = out_shape.size() == 0 ? 1 : out_shape[0];
  int64_t real_out_size = 1;
  for (auto dim : out_shape) {
    real_out_size *= dim;
  }
  InferenceParameter p;
  std::vector<std::vector<float_t>> input_datas;
  for (auto &in_shape_v : in_shapes_v) {
    std::vector<float_t> input_data(in_shape_v.size());
    std::transform(in_shape_v.begin(), in_shape_v.end(), input_data.begin(),
                   [](auto &i) { return static_cast<float_t>(i); });
    input_datas.push_back(input_data);
  }
  std::transform(input_datas.begin(), input_datas.end(),
                 std::back_inserter(p.inputs),
                 [](auto &i) { return i.data(); });
  std::vector<float_t> output_data(real_out_size);
  p.outputs.push_back(output_data.data());
  auto inf_op = dyn_cast<InferenceInterface>(op);
  assert(inf_op);
  inf_op.init(p);
  auto ret = inf_op.inference(p);
  assert(mlir::succeeded(ret));
  inf_op.deinit(p);
  std::vector<int64_t> output_shape_v(real_out_size);
  std::transform(output_data.begin(), output_data.end(), output_shape_v.begin(),
                 [](float_t i) { return static_cast<int64_t>(i); });
  return output_shape_v;
}

void assert_with_dump(bool cond, Operation *op, const char *info,
                      const char *file, unsigned line) {
  if (cond) {
    return;
  }
  unreachable(info, op, file, line);
}

void unreachable(const char *info, Operation *op, const char *file,
                 unsigned line) {
  std::cerr << "ASSERT executed at" << file << ":" << line << std::endl;
  std::cerr << "ASSERT INFO:" << info << std::endl << "Operation:" << std::endl;
  if (op != nullptr) {
    auto inputs = op->getOperands();
    if (!inputs.empty()) {
        for (auto input : inputs)
          input.dump();
    }
    std::cerr << "-> ";
    op->dump();
    for (auto out : op->getResults()) {
      for (auto user : out.getUsers())
        user->dump();
    }
  }
  exit(-1);
}

bool startsWith(const std::string& fullString, const std::string& startingSubstring) {
    if (fullString.length() >= startingSubstring.length()) {
        return (0 == fullString.compare(0, startingSubstring.length(), startingSubstring));
    } else {
        return false;
    }
}

bool endsWith(const std::string& fullString, const std::string& suffix) {
    return fullString.rfind(suffix) == fullString.length() - suffix.length();
}
} // namespace module
} // namespace tpu_mlir
