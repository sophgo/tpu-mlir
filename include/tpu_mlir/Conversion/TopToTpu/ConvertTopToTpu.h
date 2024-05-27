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
  void match_swin_mlp(std::vector<Operation *> &mlp);
  void match_swin_wmsa(std::vector<Operation *> &wmsa,
                       std::vector<Operation *> &sm_ops,
                       std::vector<Operation *> &qkmmops);
  void match_cswin_cswsa(std::vector<Operation *> &cswsa);
  bool convergence(Operation *from, Operation *to);
  bool convergence_with_sm_matmul_slice(Operation *from, Operation *to,
                                        int &sm_cnt, int &matmul_cnt,
                                        int &triple_matmul, int &triple_slice,
                                        int &six_slice);
  template <typename opType>
  bool set_block_fp16(
      Operation *from,
      Operation *to); // must be a block with single input and single output
  bool vit_mix_precision();
  void match_eva2_mlp(std::vector<Operation *> &mlp);
  void match_eva2_mhsa(std::vector<Operation *> &mhsa);
  bool eva2_mix_precision();
  void set_add_before_softmax_fp32();
  void spread_q_config();
  bool swin_mix_precision();
  bool cswin_mix_precision();
  void match_detr_ffn(std::vector<Operation *> &ffn);
  void match_detr_mha(std::vector<Operation *> &mha);
  bool match_detr_decoder(std::vector<Operation *> &dec,
                          std::vector<Operation *> &addops,
                          std::vector<Operation *> &smops);
  void match_detr_encoder_mha(std::vector<Operation *> &mha,
                              std::vector<Operation *> &smops);
  template <typename opType>
  bool find_in_block(Operation *from, Operation *to,
                     std::vector<Operation *> &ops);
  void float_till_output(Operation *start);
  bool detr_mix_precision();
  void qtable_process();
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
