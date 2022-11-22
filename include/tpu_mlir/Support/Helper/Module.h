//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

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
    static constexpr llvm::StringRef ATHENA2 = "ATHENA2";
  };

  static top::NoneOp getNoneOp(Operation *op);
  static Value getOperand(Operation* op, int i);
  static ModuleOp getModuleOp(Operation *op);
  static void updateModuleTypes(ModuleOp module);
  static void removeUnusedOp(ModuleOp module);
  static std::string genWeightFileName(ModuleOp module, bool &same_name);
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
  static inline FuncOp getMainFuncOp(ModuleOp module) {
    return getFuncOp(module, "main");
  }
  static std::shared_ptr<std::vector<int64_t>> getI64Array(ArrayAttr arrayAttr);
  static std::shared_ptr<std::vector<int64_t>>
  getI64Array(llvm::Optional<ArrayAttr> arrayAttr, int64_t num_elem,
              int64_t default_value);
  static std::shared_ptr<std::vector<double>> getF64Array(ArrayAttr arrayAttr);
  static std::shared_ptr<std::vector<double>>
  getF64Array(llvm::Optional<ArrayAttr> arrayAttr, int64_t num_elem,
              double default_value);
  static bool isOpInGroup(Operation *Op);
  static FuncOp getFuncOp(ModuleOp module, StringRef func_name);
  static func::CallOp getCallOp(ModuleOp module, FuncOp func);
  static inline llvm::StringRef getName(ModuleOp module) {
    return module->getAttrOfType<StringAttr>(Attr::NAME).getValue();
  }
  static llvm::StringRef getName(Operation *op, int index = 0);
  static llvm::StringRef getName(Value v);
  static void getInputsOutputs(ModuleOp module, std::vector<Value> &inputs,
                               std::vector<Value> &outputs);
  static void getInputsOutputs(func::CallOp call, std::vector<Value> &inputs,
                               std::vector<Value> &outputs);
  static inline int64_t getCoeffSize(ModuleOp module) {
    return module->getAttrOfType<IntegerAttr>(Attr::COEFF_SIZE).getInt();
  }
  static inline void setCoeffSize(ModuleOp module, int64_t size) {
    module->setAttr(Attr::COEFF_SIZE,
                    Builder(module.getContext()).getI64IntegerAttr(size));
  }
  static inline int64_t getGmemPrivateSize(ModuleOp module) {
    return module->getAttrOfType<IntegerAttr>(Attr::GMEM_PRIVATE_SIZE).getInt();
  }
  static inline void setGmemPrivateSize(ModuleOp module, int64_t size) {
    module->setAttr(Attr::GMEM_PRIVATE_SIZE,
                    Builder(module.getContext()).getI64IntegerAttr(size));
  }
  static inline int64_t getCoeffAddr(ModuleOp module) {
    return module->getAttrOfType<IntegerAttr>(Attr::COEFF_ADDR).getInt();
  }
  static inline void setCoeffAddr(ModuleOp module, int64_t addr) {
    module->setAttr(Attr::COEFF_ADDR,
                    Builder(module.getContext()).getI64IntegerAttr(addr));
  }
  static inline int64_t getNeuronSize(ModuleOp module) {
    return module->getAttrOfType<IntegerAttr>(Attr::NEURON_SIZE).getInt();
  }
  static inline void setNeuronSize(ModuleOp module, int64_t size) {
    module->setAttr(Attr::NEURON_SIZE,
                    Builder(module.getContext()).getI64IntegerAttr(size));
  }
  static inline int64_t getNeuronAddr(ModuleOp module) {
    return module->getAttrOfType<IntegerAttr>(Attr::NEURON_ADDR).getInt();
  }
  static inline void setNeuronAddr(ModuleOp module, int64_t addr) {
    module->setAttr(Attr::NEURON_ADDR,
                    Builder(module.getContext()).getI64IntegerAttr(addr));
  }
  static inline llvm::StringRef getChip(ModuleOp module) {
    return module->getAttrOfType<StringAttr>(Attr::CHIP).getValue();
  }
  static StringRef getChip(Operation *op);
  static inline llvm::StringRef getMode(ModuleOp module) {
    return module->getAttrOfType<StringAttr>(Attr::MODE).getValue();
  }
  static inline llvm::StringRef getMode(FuncOp func) {
    return func->getAttr("mode").cast<StringAttr>().getValue();
  }
  static inline void setChip(ModuleOp module, StringRef chip) {
    module->setAttr(Attr::CHIP,
                    StringAttr::get(module.getContext(), chip.upper()));
  }
  static inline void setMode(ModuleOp module, StringRef mode) {
    module->setAttr(Attr::MODE,
                    StringAttr::get(module.getContext(), mode.upper()));
  }
  static inline StringRef getWeightFile(ModuleOp module) {
    return module->getAttrOfType<StringAttr>(Attr::WEIGHT_FILE).getValue();
  }
  static inline void setWeightFile(ModuleOp module, StringRef weight_file) {
    module->setAttr(Attr::WEIGHT_FILE,
                    StringAttr::get(module.getContext(), weight_file));
  }
  static inline int64_t getFLOPs(ModuleOp module) {
    return module->getAttrOfType<IntegerAttr>(Attr::FLOPS).getInt();
  }
  static inline void setFLOPs(ModuleOp module, int64_t flops) {
    auto intType = IntegerType::get(module.getContext(), 64);
    module->setAttr(Attr::FLOPS, IntegerAttr::get(intType, flops));
  }
  static inline bool getAsymmetric(ModuleOp module) {
    if (module->hasAttrOfType<BoolAttr>(Attr::ASYMMETRIC)) {
      return module->getAttrOfType<BoolAttr>(Attr::ASYMMETRIC).getValue();
    }
    return false;
  }
  static inline void setAsymmetric(ModuleOp module, bool is_asymmetric) {
    module->setAttr(Attr::ASYMMETRIC,
                    BoolAttr::get(module.getContext(), is_asymmetric));
  }
  static inline StringRef getState(ModuleOp module) {
    return module->getAttrOfType<StringAttr>(Attr::STATE).getValue();
  }
  static inline void setState(ModuleOp module, StringRef state) {
    module->setAttr(Attr::STATE, StringAttr::get(module.getContext(), state));
  }
  static inline bool isState(ModuleOp module, llvm::StringRef state) {
    return state == getState(module);
  }
  static inline bool isTpuOp(Operation *op) {
    return (op->getDialect()->getNamespace() == "tpu");
  }
  static inline bool isCV18xx(StringRef chip) {
    return (chip == Module::Chip::CV183x
            || chip == Module::Chip::CV182x);
  }
  static inline bool isBM1684Family(StringRef chip){
    return (chip == Module::Chip::BM1684);
  }
  static inline bool isBM1684XFamily(StringRef chip){
    return   (chip == Module::Chip::BM1684X
            || chip == Module::Chip::ATHENA2);
  }
  static inline bool isAthena2(StringRef chip){
    return (chip == Module::Chip::ATHENA2);
  }
};
} // namespace helper
} // namespace tpu_mlir
