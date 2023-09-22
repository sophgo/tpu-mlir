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
                        std::vector<Operation *> &end_ops, bool GQA);

// ===================================
// distribute helper functions
// ===================================
std::vector<Operation *> cloneOpWithWeight(PatternRewriter &rewriter,
                                           Operation *next_op, Value &cur_out,
                                           std::string suffix);

std::vector<Operation *> cloneCommonOp(PatternRewriter &rewriter,
                                       Operation *next_op, Value &cur_out,
                                       int axis, int num_devices,
                                       int cur_device);

Operation *cloneCommonOp(PatternRewriter &rewriter, Operation *next_op,
                         Value &cur_out, std::string &suffix);

std::vector<Operation *> cloneCommonAxisOp(PatternRewriter &rewriter,
                                           Operation *next_op, Value &cur_out,
                                           int axis, int num_devices,
                                           int cur_device);

Operation *cloneMultiInsOp(PatternRewriter &rewriter, Operation *next_op,
                           Value &cur_out, std::vector<Value> &operands,
                           int axis, int num_devices, int cur_device);

std::vector<Operation *> cloneRotaryEmbedOp(PatternRewriter &rewriter,
                                            Operation *next_op, Value &cur_out,
                                            int axis,
                                            std::vector<Value> pos_ids,
                                            int num_devices, int cur_device);

Operation *cloneColParallelMatMul(PatternRewriter &rewriter, Operation *next_op,
                                  Value &cur_out, int num_devices,
                                  int cur_device);

Operation *cloneRowParallelMatMul(PatternRewriter &rewriter, Operation *next_op,
                                  Value &cur_out, int num_devices,
                                  int cur_device);

void createMulConstOp(PatternRewriter &rewriter, Value &cur_out,
                      int num_devices, int cur_device);

Operation *createSliceOp(PatternRewriter &rewriter, Operation *next_op,
                         Value &cur_out, int axis, int num_devices,
                         int cur_device);

std::vector<Operation *> cloneAttentionInput(PatternRewriter &rewriter,
                                             Operation *next_op, Value &cur_out,
                                             int num_devices, int cur_device);

std::vector<Operation *> cloneAttentionQuery(PatternRewriter &rewriter,
                                             Operation *next_op, Value &cur_out,
                                             std::vector<Value> &pos_ids,
                                             int num_devices, int cur_device, bool GQA);

std::vector<Operation *> cloneAttentionKey(PatternRewriter &rewriter,
                                           Operation *next_op, Value &cur_out,
                                           std::vector<Value> &pos_ids,
                                           std::vector<Value> &outs,
                                           int num_devices, int cur_device, bool GQA);

std::vector<Operation *> cloneAttentionMatrix(PatternRewriter &rewriter,
                                              Operation *next_op,
                                              Value &cur_out,
                                              std::vector<Value> &other_opds,
                                              int num_devices, int cur_device);

std::vector<Operation *> cloneAttentionValue(PatternRewriter &rewriter,
                                             Operation *next_op, Value &cur_out,
                                             std::vector<Value> &other_opds,
                                             std::vector<Value> &outs,
                                             int num_devices, int cur_device, bool GQA);

std::vector<Operation *> cloneAttentionOutput(PatternRewriter &rewriter,
                                              Operation *next_op,
                                              Value &cur_out, int num_devices,
                                              int cur_device);
} // namespace tpu
} // namespace tpu_mlir
