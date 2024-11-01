//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "tpu_mlir/Conversion/Conversion.h"
namespace mlir {
#define GEN_PASS_DEF_CONVERTTOPTOTPU
#include "tpu_mlir/Conversion/Passes.h.inc"
} // namespace mlir

namespace tpu_mlir {
struct ConvertTopToTpu : public ::impl::ConvertTopToTpuBase<ConvertTopToTpu> {
  // some implementations are in TopToTpuPass
  // others about quantize are in MixPrecision
public:
  void runOnOperation() override;

protected:
  void calibration_process();
  void device2host_process();
  void host2device_convert_process();
  void relu_process();
  void cast_process();
  template <typename opType>
  bool set_block_fp16(
      Operation *from,
      Operation *to); // must be a block with single input and single output
  void set_add_before_softmax_fp32();
  template <typename opType>
  bool find_in_block(Operation *from, Operation *to,
                     std::vector<Operation *> &ops);
  void float_till_output(Operation *start);
  //kv cache 2024.05
  void kv_cache_process();
  bool kv_cache_mix_precision();
  void match_kv_cache(std::vector<Operation *> &kv_cache);
  //2024.05
  Value do_cast(Value v, Type to, TypeCastMode mode,
                Operation *user_op = nullptr);
  Value insert_18xx_cpu_cast(OpBuilder &builder, Value &v, NameLoc &loc,
                             Type &newType);
  static module::Mode qmode(const std::string &mode);
  void init_qtable();

  ModuleOp module_;
  FuncOp mainFunc_;
  MLIRContext *ctx_;
};
} // namespace tpu_mlir
