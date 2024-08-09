//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Traits/Traits.h"
#include <llvm/Support/Debug.h>

namespace tpu_mlir {
namespace tpu {
// ===================================
// match helper functions
// ===================================
bool isAttentionPattern(Operation *op, std::vector<Operation *> &begin_ops,
                        std::vector<Operation *> &end_ops, bool GQA,
                        int *num_head);

bool isFAttentionPattern(Operation *op, std::vector<Operation *> &begin_ops,
                        std::vector<Operation *> &end_ops, bool GQA,
                        int *num_head);

bool isChatGLMAttentionPattern(Operation *op,
                               std::vector<Operation *> &begin_ops,
                               std::vector<Operation *> &end_ops, int *num_head);

// ===================================
// distribute helper functions
// ===================================

int64_t get_splited_size(int64_t size, int64_t num_devices, int64_t cur_device,
                         int64_t num_head, int64_t q_group_size);
int64_t get_splited_offset(int64_t size, int64_t num_devices,
                           int64_t cur_device, int64_t num_head,
                           int64_t q_group_size);
std::vector<Operation *> cloneOpWithWeight(PatternRewriter &rewriter,
                                           Operation *next_op, Value &cur_out,
                                           std::string suffix);

std::vector<Operation *> cloneOpWithWeight(PatternRewriter &rewriter,
                                           Operation *next_op, Value &cur_out,
                                           int axis, int num_devices,
                                           int cur_device);

std::vector<Operation *> cloneCommonOp(PatternRewriter &rewriter,
                                       Operation *next_op, Value &cur_out,
                                       int axis, int num_devices,
                                       int cur_device);

Operation *cloneCommonOp(PatternRewriter &rewriter, Operation *next_op,
                         Value &cur_out, std::string &suffix);

std::vector<Operation *> cloneCommonAxisOp(PatternRewriter &rewriter,
                                           Operation *next_op, Value &cur_out,
                                           int axis, int num_devices,
                                           int cur_device, int num_head);

Operation *cloneMultiInsOp(PatternRewriter &rewriter, Operation *next_op,
                           Value &cur_out, std::vector<Value> &operands,
                           int axis, int num_devices, int cur_device);

std::vector<Operation *> cloneMultiInsOps(PatternRewriter &rewriter,
                                          Operation *next_op, Value &cur_out,
                                          std::vector<Value> &operands,
                                          int axis, int num_devices,
                                          int cur_device);

std::vector<Operation *>
cloneRotaryEmbedOp(PatternRewriter &rewriter, Operation *next_op,
                   Value &cur_out, int axis, std::vector<Value> pos_ids,
                   int num_devices, int cur_device, int num_head);

Operation *cloneColParallelMatMul(PatternRewriter &rewriter, Operation *next_op,
                                  Value &cur_out, int num_devices,
                                  int cur_device, int num_head, std::string mode = "default");

Operation *cloneRowParallelMatMul(PatternRewriter &rewriter, Operation *next_op,
                                  Value &cur_out, int num_devices,
                                  int cur_device, int num_head, std::string mode = "default");

void createMulConstOp(PatternRewriter &rewriter, Value &cur_out,
                      int cur_device, float const_val);

void createSubConstOp(PatternRewriter &rewriter, Value &cur_out,
                      int cur_device, float const_val);

Operation *createSliceOp(PatternRewriter &rewriter, Operation *next_op,
                         Value &cur_out, int axis, int num_devices,
                         int cur_device, int num_head);

Operation *cloneSliceAxisOp(PatternRewriter &rewriter, Operation *next_op,
                            Value &cur_out, int axis, int num_devices,
                            int cur_device);

// ===================================
// distribute helper functions for llama2/falcon/qwen
// ===================================

std::vector<Operation *> cloneAttentionInput(PatternRewriter &rewriter,
                                             Operation *next_op, Value &cur_out,
                                             int num_devices, int cur_device,
                                             int num_head);

std::vector<Operation *> cloneAttentionQuery(PatternRewriter &rewriter,
                                             Operation *next_op, Value &cur_out,
                                             std::vector<Value> &pos_ids,
                                             int num_devices, int cur_device,
                                             bool GQA, int num_head);

std::vector<Operation *> cloneAttentionKey(PatternRewriter &rewriter,
                                           Operation *next_op, Value &cur_out,
                                           std::vector<Value> &pos_ids,
                                           std::vector<Value> &outs,
                                           int num_devices, int cur_device,
                                           bool GQA, int num_head);

std::vector<Operation *> cloneAttentionMatrix(PatternRewriter &rewriter,
                                              Operation *next_op,
                                              Value &cur_out, int axis,
                                              std::vector<Value> &other_opds,
                                              int num_devices, int cur_device);

std::vector<Operation *> cloneAttentionValue(PatternRewriter &rewriter,
                                             Operation *next_op, Value &cur_out,
                                             std::vector<Value> &other_opds,
                                             std::vector<Value> &outs, int axis,
                                             int num_devices, int cur_device,
                                             bool GQA, int num_head);

std::vector<Operation *> cloneAttentionOutput(PatternRewriter &rewriter,
                                              Operation *next_op,
                                              Value &cur_out, int num_devices,
                                              int cur_device, int num_head);

std::vector<Operation *> cloneFlashAttention(PatternRewriter &rewriter,
                                             Operation *next_op, Value &cur_out,
                                             std::vector<Value> &pos_ids,
                                             std::vector<Value> &past_kv,
                                             int num_devices, int cur_device,
                                             int q_head, int kv_head);

// ===================================
// distribute helper functions for chatglm2
// ===================================
std::vector<Value> cloneChatGLMPosInput(PatternRewriter &rewriter,
                                        Operation *next_op, Value &cur_out,
                                        int axis, int num_devices,
                                        int cur_device, std::string suffix);

std::vector<Operation *> cloneChatGLMRotaryEmbedOp(PatternRewriter &rewriter,
                                                   Operation *next_op,
                                                   Value &cur_out, int axis,
                                                   std::vector<Value> pos_operands,
                                                   int num_devices, int cur_device);

std::vector<Operation *> cloneChatGLMAttentionQK(PatternRewriter &rewriter,
                                                 Operation *next_op,
                                                 Value &cur_out, int axis,
                                                 std::vector<Value> &pos_operands,
                                                 int num_devices, int cur_device, int num_head);

std::vector<Operation *> cloneChatGLMAttentionValue(PatternRewriter &rewriter,
                                                    Operation *next_op, Value &cur_out,
                                                    std::vector<Value> &other_opds,
                                                    std::vector<Value> &outs, int axis,
                                                    int num_devices, int cur_device,
                                                    int num_head);

std::vector<Operation *> cloneChatGLMAttentionQxK(PatternRewriter &rewriter,
                                                  std::vector<Operation *> query_next_ops,
                                                  std::vector<Operation *> key_next_ops,
                                                  Operation *next_op, Value &cur_out,
                                                  std::vector<Value> &operands, int num_devices, int cur_device);

Operation *cloneChatGLMAttentionOutput(PatternRewriter &rewriter,
                                       Operation *qk_next_op,
                                       Operation *value_next_op,
                                       Operation *next_op, Value &value_branch,
                                       Value &qk_out, Value &cur_out,
                                       int num_devices, int cur_device, int num_head);

void createReshapeOp(PatternRewriter &rewriter, Operation *next_op,
                    Value &cur_out, int cur_device);
} // namespace tpu
} // namespace tpu_mlir
