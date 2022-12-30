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
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/AttrStruct.h"

using namespace mlir;
using namespace mlir::func;

namespace tpu_mlir {
namespace helper {
struct Module {
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

  static top::NoneOp getNoneOp(Operation *op);
  static Value getOriValue(Value &v);
  static Value getOperand(Operation *op, int i);
  static void updateModuleTypes();
  static void removeUnusedOp();
  static std::string genWeightFileName(bool &same_name);
  static int64_t getAddress(Value v);
  static void setAddress(Value v, int64_t addr);
  static void getNCHW(Value v, int64_t &n, int64_t &c, int64_t &h, int64_t &w,
                      bool left_align = true);
  static void getNCHW(llvm::ArrayRef<int64_t> shape, int64_t &n, int64_t &c,
                      int64_t &h, int64_t &w, bool left_align = true);
  static void getShapeVec(Value v, std::vector<int64_t> &vec_shape);
  static int getDtypeSize(Value v);
  static size_t getBytes(Value v);
  static int64_t getNumElements(Value v);
  static Type getStorageType(Value v); // storage type
  static Type getStorageType(Type type);
  static Type getElementType(Value v);
  static llvm::ArrayRef<int64_t> getShape(Value v);
  static inline FuncOp getMainFuncOp() { return getFuncOp("main"); }
  static std::shared_ptr<std::vector<int32_t>> getI32Array(ArrayAttr arrayAttr);
  static std::shared_ptr<std::vector<int32_t>>
  getI32Array(llvm::Optional<ArrayAttr> arrayAttr, int64_t num_elem,
              int32_t default_value);
  static std::shared_ptr<std::vector<int64_t>> getI64Array(ArrayAttr arrayAttr);
  static std::shared_ptr<std::vector<int64_t>>
  getI64Array(llvm::Optional<ArrayAttr> arrayAttr, int64_t num_elem,
              int64_t default_value);
  static std::shared_ptr<std::vector<double>> getF64Array(ArrayAttr arrayAttr);
  static std::shared_ptr<std::vector<double>>
  getF64Array(llvm::Optional<ArrayAttr> arrayAttr, int64_t num_elem,
              double default_value);
  static bool isOpInGroup(Operation *Op);
  static FuncOp getFuncOp(StringRef func_name);
  static func::CallOp getCallOp(FuncOp func);
  static inline llvm::StringRef getModuleName() {
    return m->getAttrOfType<StringAttr>(Attr::NAME).getValue();
  }
  static llvm::StringRef getName(Operation *op, int index = 0);
  static llvm::StringRef getName(Value v);
  static void getInputsOutputs(std::vector<Value> &inputs,
                               std::vector<Value> &outputs);
  static void getInputsOutputs(func::CallOp call, std::vector<Value> &inputs,
                               std::vector<Value> &outputs);
  static inline int64_t getCoeffSize() {
    return m->getAttrOfType<IntegerAttr>(Attr::COEFF_SIZE).getInt();
  }
  static inline void setCoeffSize(int64_t size) {
    m->setAttr(Attr::COEFF_SIZE, Builder(ctx).getI64IntegerAttr(size));
  }
  static inline int64_t getGmemPrivateSize() {
    return m->getAttrOfType<IntegerAttr>(Attr::GMEM_PRIVATE_SIZE).getInt();
  }
  static inline void setGmemPrivateSize(int64_t size) {
    m->setAttr(Attr::GMEM_PRIVATE_SIZE, Builder(ctx).getI64IntegerAttr(size));
  }
  static inline int64_t getCoeffAddr() {
    return m->getAttrOfType<IntegerAttr>(Attr::COEFF_ADDR).getInt();
  }
  static inline void setCoeffAddr(int64_t addr) {
    m->setAttr(Attr::COEFF_ADDR, Builder(ctx).getI64IntegerAttr(addr));
  }
  static inline int64_t getNeuronSize() {
    return m->getAttrOfType<IntegerAttr>(Attr::NEURON_SIZE).getInt();
  }
  static inline void setNeuronSize(int64_t size) {
    m->setAttr(Attr::NEURON_SIZE, Builder(ctx).getI64IntegerAttr(size));
  }
  static inline int64_t getNeuronAddr() {
    return m->getAttrOfType<IntegerAttr>(Attr::NEURON_ADDR).getInt();
  }
  static inline void setNeuronAddr(int64_t addr) {
    m->setAttr(Attr::NEURON_ADDR, Builder(ctx).getI64IntegerAttr(addr));
  }
  static inline llvm::StringRef getChip() { return chip; }
  static inline llvm::StringRef getMode() {
    return m->getAttrOfType<StringAttr>(Attr::MODE).getValue();
  }
  static inline llvm::StringRef getFuncMode(FuncOp func) {
    return func->getAttr("mode").cast<StringAttr>().getValue();
  }
  static inline void setChip(StringRef chip_) {
    m->setAttr(Attr::CHIP, StringAttr::get(m.getContext(), chip_.upper()));
    chip = m->getAttrOfType<StringAttr>(Attr::CHIP).getValue();
  }

  static inline void setMode(StringRef mode) {
    m->setAttr(Attr::MODE, StringAttr::get(ctx, mode.upper()));
  }
  static inline StringRef getWeightFile() {
    return m->getAttrOfType<StringAttr>(Attr::WEIGHT_FILE).getValue();
  }
  static inline void setWeightFile(StringRef weight_file) {
    m->setAttr(Attr::WEIGHT_FILE, StringAttr::get(ctx, weight_file));
  }
  static inline int64_t getFLOPs() {
    return m->getAttrOfType<IntegerAttr>(Attr::FLOPS).getInt();
  }
  static inline void setFLOPs(int64_t flops) {
    auto intType = IntegerType::get(ctx, 64);
    m->setAttr(Attr::FLOPS, IntegerAttr::get(intType, flops));
  }
  static inline bool isAsymmetric() {
    if (m->hasAttrOfType<BoolAttr>(Attr::ASYMMETRIC)) {
      return m->getAttrOfType<BoolAttr>(Attr::ASYMMETRIC).getValue();
    }
    return false;
  }
  static inline void setAsymmetric(bool is_asymmetric) {
    m->setAttr(Attr::ASYMMETRIC, BoolAttr::get(ctx, is_asymmetric));
  }
  static inline StringRef getState() {
    return m->getAttrOfType<StringAttr>(Attr::STATE).getValue();
  }
  static inline void setState(StringRef state) {
    m->setAttr(Attr::STATE, StringAttr::get(ctx, state));
  }
  static inline bool isState(llvm::StringRef state) {
    return state == getState();
  }
  static inline bool isTpuOp(Operation *op) {
    return (op->getDialect()->getNamespace() == "tpu");
  }
  static inline bool isCV18xx() {
    return (chip == Module::Chip::CV183x || chip == Module::Chip::CV182x);
  }
  static inline bool isBM1684Family() { return (chip == Module::Chip::BM1684); }
  static inline bool isBM1684XFamily() {
    return (chip == Module::Chip::BM1684X || chip == Module::Chip::BM1686);
  }
  static inline bool isBM1686() { return (chip == Module::Chip::BM1686); }

  static inline ModuleOp getModuleOp() { return m; }

  static inline mlir::Location getLoc() { return m.getLoc(); }

  static inline mlir::MLIRContext *getCtx() { return ctx; }

  static inline void push_back(mlir::func::FuncOp funcOp) {
    m.push_back(funcOp);
  }

  static void init(ModuleOp module) {
    m = module;
    ctx = m.getContext();
    chip = m->getAttrOfType<StringAttr>(Attr::CHIP).getValue();
  }

private:
  static ModuleOp m;
  static llvm::StringRef chip;
  static mlir::MLIRContext *ctx;
};
} // namespace helper
} // namespace tpu_mlir
