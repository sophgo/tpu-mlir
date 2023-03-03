//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/BMAddressAssign.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;
using namespace tpu_mlir::backend;

namespace tpu_mlir {

namespace bm168x {

class GRUGlobalBuffer : public OpRewritePattern<tpu::GRUOp> {
public:
  using OpRewritePattern<tpu::GRUOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::GRUOp GRUOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(GRUOp.getBuffer())) {
      return failure();
    }
    auto attr = GRUOp.parseParam();
    auto type = module::getStorageType(GRUOp.getInput());
    // add buffer
    std::vector<int64_t> buffer_shape = {5, attr.batch_size, attr.hidden_size};
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(GRUOp, buffer_type);
    GRUOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
      return operand.get() == GRUOp.getBuffer();
    });
    return success();
  }
};

class LSTMGlobalBuffer : public OpRewritePattern<tpu::LSTMOp> {
public:
  using OpRewritePattern<tpu::LSTMOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::LSTMOp lstmOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(lstmOp.getBuffer())) {
      return failure();
    }
    auto attr = lstmOp.parseParam();
    auto type = module::getStorageType(lstmOp.getInput());
    // add buffer
    std::vector<int64_t> buffer_shape = {5, attr.batch_size, attr.hidden_size};
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(lstmOp, buffer_type);
    lstmOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
      return operand.get() == lstmOp.getBuffer();
    });
    return success();
  }
};

class ReduceGlobalBuffer : public OpRewritePattern<tpu::ReduceOp> {
public:
  using OpRewritePattern<tpu::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(reduceOp.getBuffer())) {
      return failure();
    }
    auto attr = reduceOp.parseParam();
    if (attr.simplified == false) {
      llvm_unreachable("Not Implemented");
    }
    auto type = module::getStorageType(reduceOp.getInput());
    auto input_spec = BM168x::get_input_spec(reduceOp);
    auto output_spec = BM168x::get_output_spec(reduceOp);
    BM168x::fix_shape(input_spec->at(0), {attr.outer_n, attr.outer_c,
                                          attr.axis_dims, attr.inner_dims});
    BM168x::fix_shape(output_spec->at(0),
                      {attr.outer_n, attr.outer_c, 1, attr.inner_dims});
    reduce_full_global_param_t param = {0};
    param.spec.common.axis_num = 1;
    param.spec.common.axis[0] = 2;
    param.spec.common.method = BM168x::get_reduce_type(reduceOp.getMode());
    param.if_getting_buffer_size = true;
    uint64_t buffer_size = 0;
    param.buffer_size_ptr = &buffer_size;
    BM168x::call_global_func("backend_api_reduce_full_global", &param,
                             sizeof(param), input_spec->data(),
                             output_spec->data());
    if (buffer_size > 0) {
      std::vector<int64_t> buffer_shape = {(int64_t)buffer_size};
      auto buffer_type = RankedTensorType::get(buffer_shape, type);
      auto buffer = tpu::BufferOp::create(reduceOp, buffer_type);
      reduceOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
        return operand.get() == reduceOp.getBuffer();
      });
    }

    return success();
  }
};

class PermuteGlobalBuffer : public OpRewritePattern<tpu::PermuteOp> {
public:
  using OpRewritePattern<tpu::PermuteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::PermuteOp permuteOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(permuteOp.getBuffer())) {
      return failure();
    }
    transpose_param_t param = {0};
    param.if_getting_buffer_size = 1;
    auto attr = permuteOp.parseParam();
    for (int i = 0, n = attr.order_fix.size(); i < n; ++i) {
      param.spec.order[i] = attr.order_fix[i];
    }
    uint64_t buffer_size = 0;
    param.buffer_size_ptr = &buffer_size;
    auto input_spec = BM168x::get_input_spec(permuteOp);
    auto output_spec = BM168x::get_output_spec(permuteOp);
    BM168x::fix_shape(input_spec->at(0), attr.in_shape_fix);
    BM168x::fix_shape(output_spec->at(0), attr.out_shape_fix);
    BM168x::call_global_func("backend_api_transpose_global", &param,
                             sizeof(param), input_spec->data(),
                             output_spec->data());
    auto type = module::getStorageType(permuteOp.getInput());
    // add buffer
    if (buffer_size > 0) {
      auto buffer_type = RankedTensorType::get({(int64_t)buffer_size}, type);
      auto buffer = tpu::BufferOp::create(permuteOp, buffer_type);
      permuteOp.setOperand(1, buffer);
      return success();
    }
    return failure();
  }
};

class NonZeroGlobalBuffer : public OpRewritePattern<tpu::NonZeroOp> {
public:
  using OpRewritePattern<tpu::NonZeroOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::NonZeroOp op,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(op.getBuffer())) {
      return failure();
    }
    auto type = module::getStorageType(op.getInput());
    // add buffer
    auto buffer_type = RankedTensorType::get(module::getShape(op.getInput()), type);
    auto buffer = tpu::BufferOp::create(op, buffer_type);
    op.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
      return operand.get() == op.getBuffer();
    });
    return success();
  }
};

void populateGlobalBufferPatterns(RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
      GRUGlobalBuffer,
      LSTMGlobalBuffer,
      ReduceGlobalBuffer,
      PermuteGlobalBuffer,
      NonZeroGlobalBuffer
  >(patterns->getContext());
  // clang-format on
}

} // namespace bm168x
} // namespace tpu_mlir
