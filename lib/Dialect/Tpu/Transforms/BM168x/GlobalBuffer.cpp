//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/BMAddressAssign.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;
using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;

namespace tpu_mlir {

namespace bm168x {

class LSTMGlobalBuffer : public OpRewritePattern<tpu::LSTMOp> {
public:
  using OpRewritePattern<tpu::LSTMOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::LSTMOp lstmOp,
                                PatternRewriter &rewriter) const override {
    if (!lstmOp.buffer().getType().isa<mlir::NoneType>()) {
      return failure();
    }
    auto ctx = rewriter.getContext();
    auto attr = lstmOp.parseParam();
    auto type = Module::getStorageType(lstmOp.input());
    // add buffer
    int64_t buffer_size = attr.batch_size * attr.hidden_size * 5;
    std::vector<int64_t> buffer_shape = {5, attr.batch_size, attr.hidden_size};
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(lstmOp, buffer_type);
    lstmOp.buffer().replaceAllUsesWith(buffer);
    return success();
  }
};

class ReduceGlobalBuffer : public OpRewritePattern<tpu::ReduceOp> {
public:
  using OpRewritePattern<tpu::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    if (!reduceOp.buffer().getType().isa<mlir::NoneType>()) {
      return failure();
    }
    auto ctx = rewriter.getContext();
    auto type = Module::getStorageType(reduceOp.input());
    // add buffer
    /* if reduce n or c, need imm buffer. if reduce h/w, don't need imm buffer
       if reduce c/h, c/w, n/w, will split it to 2 step at fronted, it will not go here.
       if reduce c/h/w, n/h/w, need imm buffer .
       Note: reduce max/mean and n/h, don't support it now (transpose need imm buffer) */
    auto axes_val = Module::getI64Array(reduceOp.axes());
    auto axis_num = axes_val->size();
    auto in_tensor = Module::getShape(reduceOp.input());
    int is_reduce[MAX_SHAPE_DIMS] = {0};
    for (int i = 0; i < axis_num; i++) {
      is_reduce[axes_val->at(i)] = 1;
    }

    /* reducemax/mean and reduce n/w,c/h,c/w, will use max/avg pool to implement it.*/
    if ((axis_num == 1 && (is_reduce[0] || is_reduce[1]))
        || (axis_num == 3 && (is_reduce[0] || is_reduce[1]))
        || (axis_num >= 4 && axis_num == in_tensor.size())) {
      //calculate the actual imm buffer size according to nodechip_reduce
      int imm_buffer_size = 0;
      if (axis_num >=4 && axis_num == in_tensor.size()){
        for (int i = in_tensor.size() - 3; i >=0; i--){
          imm_buffer_size *= in_tensor[i];
        }
        imm_buffer_size *= sizeof(float_t);
      } else if (axis_num == 1 && (is_reduce[0] || is_reduce[1])) {
        for (int i = in_tensor.size() - 1; i >=0; i--){
          imm_buffer_size *= in_tensor[i];
        }
        imm_buffer_size *= sizeof(float_t) *2;
      } else if (axis_num == 3 && (is_reduce[0] || is_reduce[1])) {
        int tmp_val  = sizeof(float_t);
        for (int i = in_tensor.size() - 3; i >=0; i--){
          tmp_val *= in_tensor[i];
        }
        imm_buffer_size += tmp_val;
        tmp_val = sizeof(float_t) * 2;
        for (int i = in_tensor.size() - 1; i >= 0; i--){
          tmp_val *= in_tensor[i];
        }
        imm_buffer_size += tmp_val;
      }

      std::vector<int64_t> buffer_shape = {1, imm_buffer_size};
      auto buffer_type = RankedTensorType::get(buffer_shape, type);
      auto buffer = tpu::BufferOp::create(reduceOp, buffer_type);
      reduceOp.buffer().replaceAllUsesWith(buffer);
    }

    return success();
  }
};

void populateGlobalBufferPatterns(RewritePatternSet *patterns) {
  patterns->add<LSTMGlobalBuffer>(patterns->getContext());
  patterns->add<ReduceGlobalBuffer>(patterns->getContext());
}

} // namespace bm168x
} // namespace tpu_mlir
