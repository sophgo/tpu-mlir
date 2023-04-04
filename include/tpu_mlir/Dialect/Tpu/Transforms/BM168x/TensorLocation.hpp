//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AsmState.h"
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include <llvm/Support/JSON.h>

namespace mlir {
using namespace llvm;
using namespace tpu_mlir::backend;

class LocalGenInterfaceDecorator;
class GlobalGenInterfaceDecorator;
class CodeGeninterface;

class TensorLocationImpl {
public:
  TensorLocationImpl(Operation *moduleOp, std::string filename)
      : OS(filename, EC), J(OS, 2) {
    J.arrayBegin();
    llvm::raw_null_ostream os;
    AsmState state(moduleOp, OpPrintingFlags(), &opToLineCol);
    moduleOp->print(os, state);
  }
  ~TensorLocationImpl() { J.arrayEnd(); }

  template <typename... Args>
  void before_codegen_local(Args...) {
    cmd_before[0] = BM168x::instance()->bdc_total_id;
    cmd_before[1] = BM168x::instance()->gdma_total_id;
  };

  void after_codegen_local(Operation *op, int64_t n_step, int64_t h_step,
                           int64_t d_step, int64_t w_step,
                           group_type_t group_type, local_sec_info_t &sec_info);

  void before_codegen_global(Operation *op) {
    cmd_before[0] = BM168x::instance()->bdc_total_id;
    cmd_before[1] = BM168x::instance()->gdma_total_id;
  };
  void after_codegen_global(Operation *op);

  TensorLocationImpl &operator=(const TensorLocationImpl &T) = delete;

protected:
  void record_loc(Operation *op, const json::Array &operands,
                  const json::Array &results);

private:
  uint64_t cmd_before[2];
  std::error_code EC;
  llvm::raw_fd_ostream OS;
  json::OStream J;
  AsmState::LocationMap opToLineCol;
};

class TensorLocation {
public:
  TensorLocation() { impl.reset(); }
  template <typename... Args>
  TensorLocation(Args... args) {
    impl = std::make_unique<TensorLocationImpl>(args...);
  };

private:
  static std::shared_ptr<TensorLocationImpl> impl;
  friend class LocalGenInterfaceDecorator;
  friend class GlobalGenInterfaceDecorator;
};

class LocalGenInterfaceDecorator : public LocalGenInterface {
public:
  LocalGenInterfaceDecorator(Operation *op) : LocalGenInterface(op){};
  template <typename... Args>
  void codegen_local_bm168x(Args... args) {
    TensorLocation::impl->before_codegen_local(getOperation(), args...);
    LocalGenInterface::codegen_local_bm168x(args...);
    TensorLocation::impl->after_codegen_local(getOperation(), args...);
  }
};

class GlobalGenInterfaceDecorator : public GlobalGenInterface {
public:
  GlobalGenInterfaceDecorator(Operation *op) : GlobalGenInterface(op){};
  void codegen_global_bm168x() {
    TensorLocation::impl->before_codegen_global(getOperation());
    GlobalGenInterface::codegen_global_bm168x();
    TensorLocation::impl->after_codegen_global(getOperation());
  }
};

} // namespace mlir
