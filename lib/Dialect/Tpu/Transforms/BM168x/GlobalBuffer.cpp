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
    // add buffer
    /* if reduce n or c, need imm buffer. if reduce h/w, don't need imm buffer
       if reduce c/h, c/w, n/w, will split it to 2 step at fronted, it will not
       go here. if reduce c/h/w, n/h/w, need imm buffer . Note: reduce max/mean
       and n/h, don't support it now (transpose need imm buffer) */
    auto axis_num = 1;
    bool is_reduce[4] = {false, false, true, false};
    std::vector<int64_t> in_tensor = {attr.outer_n, attr.outer_c,
                                      attr.axis_dims, attr.inner_dims};

    /* reducemax/mean and reduce n/w,c/h,c/w, will use max/avg pool to implement
     * it.*/
    if ((axis_num == 1 && (is_reduce[0] || is_reduce[1])) ||
        (axis_num == 3 && (is_reduce[0] || is_reduce[1])) ||
        (axis_num >= 4 && axis_num == in_tensor.size())) {
      // calculate the actual imm buffer size according to nodechip_reduce
      int imm_buffer_size = 0;
      if (axis_num >= 4 && axis_num == in_tensor.size()) {
        for (int i = in_tensor.size() - 3; i >= 0; i--) {
          imm_buffer_size *= in_tensor[i];
        }
        imm_buffer_size *= sizeof(float_t);
      } else if (axis_num == 1 && (is_reduce[0] || is_reduce[1])) {
        for (int i = in_tensor.size() - 1; i >= 0; i--) {
          imm_buffer_size *= in_tensor[i];
        }
        imm_buffer_size *= sizeof(float_t) * 2;
      } else if (axis_num == 3 && (is_reduce[0] || is_reduce[1])) {
        int tmp_val = sizeof(float_t);
        for (int i = in_tensor.size() - 3; i >= 0; i--) {
          tmp_val *= in_tensor[i];
        }
        imm_buffer_size += tmp_val;
        tmp_val = sizeof(float_t) * 2;
        for (int i = in_tensor.size() - 1; i >= 0; i--) {
          tmp_val *= in_tensor[i];
        }
        imm_buffer_size += tmp_val;
      }

      std::vector<int64_t> buffer_shape = {1, imm_buffer_size};
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

  typedef struct tranpose_spec {
    uint64_t buffer_global_addr;
    uint32_t order[MAX_SHAPE_DIMS];
    uint32_t is_dynamic;
  } transpose_spec_t;

  typedef struct transpose_param {
    transpose_spec_t spec;

    int32_t if_getting_buffer_size;
    uint64_t buffer_size_ptr;
  } transpose_param_t;

  LogicalResult matchAndRewrite(tpu::PermuteOp permuteOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(permuteOp.getBuffer())) {
      return failure();
    }
    transpose_param_t param = {0};
    param.if_getting_buffer_size = 0;
    param.spec.buffer_global_addr = 0;

    auto order = module::getI64Array(permuteOp.getOrder());
    for (int i = 0, n = order->size(); i < n; ++i) {
      param.spec.order[i] = order->at(i);
    }
    int64_t buffer_size = 0;
    param.buffer_size_ptr = (uint64_t)&buffer_size;

    auto value_to_spec = [](mlir::Value v) -> tensor_spec_t {
      tensor_spec_t spec;
      memset(&spec, 0, sizeof(spec));
      spec.dtype = backend::BM168x::getDataType(v);
      auto shape = module::getShape(v);
      spec.dims = shape.size();
      for (int i = 0; i < spec.dims; i++) {
        spec.shape[i] = shape[i];
      }
      spec.elem_num = 0;
      return spec;
    };

    auto input_sepc = value_to_spec(permuteOp.getInput());
    auto output_sepc = value_to_spec(permuteOp.getOutput());
    backend::BM168x::call_global_func("backend_api_transpose_global", &param,
                                      sizeof(param), &input_sepc, &output_sepc);
    auto type = module::getStorageType(permuteOp.getInput());
    // add buffer
    if (buffer_size > 0) {
      auto buffer_type = RankedTensorType::get({buffer_size}, type);
      auto buffer = tpu::BufferOp::create(permuteOp, buffer_type);
      permuteOp.setOperand(1, buffer);
      return success();
    }
    return failure();
  }
};

void populateGlobalBufferPatterns(RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
      GRUGlobalBuffer,
      LSTMGlobalBuffer,
      ReduceGlobalBuffer,
      PermuteGlobalBuffer
  >(patterns->getContext());
  // clang-format on
}

} // namespace bm168x
} // namespace tpu_mlir
