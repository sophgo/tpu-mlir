//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/AttrStruct.h"

using namespace mlir;
using namespace mlir::func;
using namespace tpu_mlir;

namespace tpu_mlir {

//-----------------------------------------------------------------
// Types
//-----------------------------------------------------------------
typedef std::shared_ptr<std::vector<int32_t>> i32_array_t;
typedef std::shared_ptr<std::vector<int64_t>> i64_array_t;
typedef std::shared_ptr<std::vector<double>> f64_array_t;
namespace module {

// init module by ModuleOp in init pass
void init(ModuleOp module);

//-----------------------------------------------------------------
// Attributes for ModuleOp
//-----------------------------------------------------------------
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
};

struct State {
  static constexpr llvm::StringRef TOP_F32 = "TOP_F32";
  static constexpr llvm::StringRef TOP_CALIBRATED = "TOP_CALIBRATED";
  static constexpr llvm::StringRef TOP_QUANTIZED = "TOP_QUANTIZED";
  static constexpr llvm::StringRef TPU_LOWERED = "TPU_LOWERED";
  static constexpr llvm::StringRef TPU_REORDERED = "TPU_REORDERED";
  static constexpr llvm::StringRef TPU_DIVIDED = "TPU_DIVIDED";
  static constexpr llvm::StringRef TPU_ADDRESSED = "TPU_ADDRESSED";
};

struct Chip {
  static constexpr llvm::StringRef ALL = "ALL";
  static constexpr llvm::StringRef BM1684 = "BM1684";
  static constexpr llvm::StringRef BM1684X = "BM1684X";
  static constexpr llvm::StringRef CV182x = "CV182X";
  static constexpr llvm::StringRef CV183x = "CV183X";
  static constexpr llvm::StringRef BM1686 = "BM1686";
};

struct Mode {
  static constexpr llvm::StringRef INT8 = "INT8";
  static constexpr llvm::StringRef UINT8 = "UINT8";
  static constexpr llvm::StringRef INT4 = "INT4";
  static constexpr llvm::StringRef BF16 = "BF16";
  static constexpr llvm::StringRef F16 = "F16";
  static constexpr llvm::StringRef F32 = "F32";
};

// get/set Attributes
int64_t getCoeffSize();
void setCoeffSize(int64_t size);
int64_t getGmemPrivateSize();
void setGmemPrivateSize(int64_t size);
int64_t getCoeffAddr();
void setCoeffAddr(int64_t addr);
int64_t getNeuronSize();
void setNeuronSize(int64_t size);
int64_t getNeuronAddr();
void setNeuronAddr(int64_t addr);

llvm::StringRef getChip();
llvm::StringRef getMode();
llvm::StringRef getFuncMode(FuncOp func);
void setChip(StringRef chip);
void setMode(StringRef mode);
StringRef getWeightFile();
void setWeightFile(StringRef weight_file);
int64_t getFLOPs();
void setFLOPs(int64_t flops);
bool isAsymmetric();
void setAsymmetric(bool is_asymmetric);
StringRef getState();
void setState(StringRef state);
bool isState(llvm::StringRef state);

//-----------------------------------------------------------------
// Helper Functions for ModuleOp
//-----------------------------------------------------------------

ModuleOp getModuleOp();
Location getLoc();
MLIRContext *getCtx();

void push_back(FuncOp funcOp);

top::NoneOp getNoneOp(Operation *op);
Value getOriValue(Value v);
Value getOperand(Operation *op, int i);
void updateModuleTypes();
void removeUnusedOp();
std::string genWeightFileName(bool &same_name);
int64_t getAddress(Value v);
void setAddress(Value v, int64_t addr);
void getNCHW(Value v, int64_t &n, int64_t &c, int64_t &h, int64_t &w,
             bool left_align = true);
void getNCHW(llvm::ArrayRef<int64_t> shape, int64_t &n, int64_t &c, int64_t &h,
             int64_t &w, bool left_align = true);
void getShapeVec(Value v, std::vector<int64_t> &vec_shape);
int getDtypeSize(Value v);
size_t getBytes(Value v);
int64_t getNumElements(Value v);
Type getStorageType(Value v); // storage type
Type getStorageType(Type type);
Type getElementType(Value v);
llvm::ArrayRef<int64_t> getShape(Value v);
bool isSign(Value v);
bool isWeight(Value v);
FuncOp getMainFuncOp();
i32_array_t getI32Array(ArrayAttr arrayAttr);
i32_array_t getI32Array(llvm::Optional<ArrayAttr> arrayAttr, int64_t num_elem,
                        int32_t default_value);
i64_array_t getI64Array(ArrayAttr arrayAttr);
i64_array_t getI64Array(llvm::Optional<ArrayAttr> arrayAttr, int64_t num_elem,
                        int64_t default_value);
f64_array_t getF64Array(ArrayAttr arrayAttr);
f64_array_t getF64Array(llvm::Optional<ArrayAttr> arrayAttr, int64_t num_elem,
                        double default_value);
bool isOpInGroup(Operation *Op);
FuncOp getFuncOp(StringRef func_name);
func::CallOp getCallOp(FuncOp func);
llvm::StringRef getModuleName();
llvm::StringRef getName(Operation *op, int index = 0);
llvm::StringRef getName(Value v);
void getInputsOutputs(std::vector<Value> &inputs, std::vector<Value> &outputs);
void getInputsOutputs(func::CallOp call, std::vector<Value> &inputs,
                      std::vector<Value> &outputs);

bool isTpuOp(Operation *op);
bool isCV18xx();
bool isBM1684Family();
bool isBM1684XFamily();
bool isBM1686();

//-----------------------------------------------------------------
// Helper Functions for quantization
//-----------------------------------------------------------------
bool isCalibratedType(Type type);
bool isCalibratedType(Value v);
bool isUniformQuantized(Type type);
bool isUniformQuantized(Value v);
template <typename... Args> bool isCalibratedType(Value v, Args... args) {
  return isCalibratedType(v) && isCalibratedType(args...);
}
template <typename... Args> bool isUniformQuantized(Value v, Args... args) {
  return isUniformQuantized(v) && isUniformQuantized(args...);
}

quant::CalibratedQuantizedType getCalibratedType(Value v);
quant::CalibratedQuantizedType getCalibratedType(Type t);
quant::UniformQuantizedType getUniformQuantizedType(Value v);
quant::UniformQuantizedType getUniformQuantizedType(Type t);
double getThreshold(Value v);

// for symmetric
double getScale(double threshold, bool sign, int bitwidth = 8);
// for asymmetric
void getScaleAndZeroPoint(double rmin, double rmax, double &scale,
                          int64_t &zeroPoint, int bitwidth = 8);
void getScaleAndZeroPoint(Value v, double &scale, int64_t &zeropoint,
                          bool asymmetric, int bitwidth = 8);
void getScaleAndZeroPoint(Value v, double &scale, int64_t &zeropoint,
                          bool &sign, bool asymmetric, int bitwidth = 8);

} // namespace module
} // namespace tpu_mlir
