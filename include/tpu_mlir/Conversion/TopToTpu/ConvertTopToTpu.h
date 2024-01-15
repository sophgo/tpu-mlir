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
  void host2device_convert_process();
  void relu_process();
  void cast_process();
  bool isSISO(Operation *op);
  void match_bert_ffn(std::vector<Operation *> &ffn);
  void match_bert_mha(std::vector<Operation *> &mha);
  void match_attention(std::vector<Operation *> &attention);
  bool bert_mix_precision();
  void match_vit_mlp(std::vector<Operation *> &mlp);
  void match_vit_mha(std::vector<Operation *> &mha);
  void match_vit_mha1(std::vector<Operation *> &mha);
  void match_deit_mha(std::vector<Operation *> &mha);
  bool deit_mix_precision();
  void match_swint_mlp(std::vector<Operation *> &mlp);
  void match_swint_wmsa(std::vector<Operation *> &wmsa);
  bool convergence(Operation* from, Operation *to);
  bool vit_mix_precision();
  void set_add_before_softmax_fp32();
  bool swin_t_mix_precision();
  void qtable_process();
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
