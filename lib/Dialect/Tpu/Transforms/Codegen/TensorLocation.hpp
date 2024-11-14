//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#pragma once
#include "mlir/IR/AsmState.h"
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include <llvm/Support/JSON.h>

namespace mlir {
using namespace llvm;
using namespace tpu_mlir::backend;

class LocalGenInterfaceDecorator;
class GlobalGenInterfaceDecorator;
class ScopeVar;

template <typename T>
class Table {
public:
  Table(){};
  bool contains(T key) { return prefix_record.contains(key); }
  int64_t get_index(T key) {
    if (contains(key))
      return prefix_record[key];
    int64_t index = prefix_record.size();
    prefix_record.insert({key, index});
    return index;
  }

  auto take_values() { return prefix_record.takeVector(); }

private:
  MapVector<T, int64_t> prefix_record;
};

class TensorLocationImpl {
public:
  TensorLocationImpl(AsmState::LocationMap *location, std::string filename)
      : OS(filename, EC), J(OS, 2), opToLineCol(*location) {
    J.arrayBegin();
  }
  ~TensorLocationImpl() { J.arrayEnd(); }

  template <typename... Args>
  void before_codegen_local(Operation *op, Args...) {
    const bool is_ld_st = isa<tpu::LoadOp, tpu::StoreOp>(op);
    cmd_before[0] = is_ld_st ? ((int *)((*BM168x::instance())->gdma_node))[0]
                             : (*BM168x::instance()).get_total_id("tiu:0:0");
    cmd_before[1] = is_ld_st ? (*BM168x::instance()).get_total_id("gdma:0:0")
                             : ((int *)((*BM168x::instance())->bdc_node))[1];
  };

  void after_codegen_local(Operation *op, int64_t n_step, int64_t c_step,
                           int64_t h_step, int64_t d_step, int64_t w_step,
                           group_type_t group_type, local_sec_info_t &sec_info);

  void before_codegen_global(Operation *op) {
    cmd_before[0] = (*BM168x::instance()).get_total_id("tiu:0:0");
    cmd_before[1] = (*BM168x::instance()).get_total_id("gdma:0:0");
  };
  void after_codegen_global(Operation *op);

  TensorLocationImpl &operator=(const TensorLocationImpl &T) = delete;

protected:
  void record_loc(Operation *op, const json::Array &operands,
                  const json::Array &results, const json::Array &buffers);

private:
  uint64_t cmd_before[2];
  std::error_code EC;
  llvm::raw_fd_ostream OS;
  json::OStream J;
  const AsmState::LocationMap &opToLineCol;
  friend ScopeVar;
};

class TensorLocation {

public:
  TensorLocation() { getImpl().reset(); }
  template <typename... Args>
  TensorLocation(bool enable, Args... args) {
    if (enable) {
      getImpl() = std::make_unique<TensorLocationImpl>(args...);
    }
  };

  TensorLocationImpl *operator->() const { return getImpl().get(); }

private:
  static std::shared_ptr<TensorLocationImpl> &getImpl() {
    static std::shared_ptr<TensorLocationImpl> impl;
    return impl;
  }
  friend class LocalGenInterfaceDecorator;
  friend class GlobalGenInterfaceDecorator;
  friend class ScopeVar;
};

class ScopeVar {
  json::OStream &J;
  const AsmState::LocationMap &opToLineCol;

public:
  ScopeVar(TensorLocation &tensor_loc, Operation *op)
      : J(tensor_loc->J), opToLineCol(tensor_loc->opToLineCol) {
    J.objectBegin();
    if (auto func = dyn_cast<FuncOp>(op)) {
      J.attribute("function", func.getName());
    } else {
      J.attribute("function", op->getName().getStringRef());
    }
    J.attribute("line", opToLineCol.at(op).first);
    J.attributeBegin("body");
    J.arrayBegin();
  };

  ScopeVar(TensorLocation &tensor_loc, StringRef name)
      : J(tensor_loc->J), opToLineCol(tensor_loc->opToLineCol) {
    J.objectBegin();
    J.attribute("function", name);
    J.attribute("line", nullptr);
    J.attributeBegin("body");
    J.arrayBegin();
  };
  ~ScopeVar() {
    J.arrayEnd();
    J.attributeEnd();
    J.objectEnd();
  };
};

class LocalGenInterfaceDecorator : public LocalGenInterface {
public:
  LocalGenInterfaceDecorator(Operation *op) : LocalGenInterface(op){};
  template <typename... Args>
  void codegen_local_bm168x(Args... args) {
    const auto tl_impl = TensorLocation::getImpl();
    if (tl_impl) {
      tl_impl->before_codegen_local(getOperation(), args...);
    }
    LocalGenInterface::codegen_local_bm168x(args...);
    if (tl_impl) {
      tl_impl->after_codegen_local(getOperation(), args...);
    }
  }
};

class GlobalGenInterfaceDecorator : public GlobalGenInterface {
public:
  GlobalGenInterfaceDecorator(Operation *op) : GlobalGenInterface(op){};
  void codegen_global_bm168x() {
    const auto tl_impl = TensorLocation::getImpl();
    if (tl_impl) {
      tl_impl->before_codegen_global(getOperation());
    }
    GlobalGenInterface::codegen_global_bm168x();
    if (tl_impl) {
      tl_impl->after_codegen_global(getOperation());
    }
  }
};

} // namespace mlir
