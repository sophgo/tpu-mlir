//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

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
    if (module::isBM1684XFamily()) {
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
    } else if (module::isBM1684Family()) {
      auto type = module::getStorageType(reduceOp.getInput());
      auto in_addr = module::getAddress(reduceOp.getInput());
      auto out_addr = module::getAddress(reduceOp.getOutput());
      int input_dims = module::getShape(reduceOp.getInput()).size();
      uint64_t buffer_size = 0;
      uint64_t *buffer_size_ptr = &buffer_size;
      uint32_t *input_shape = new uint32_t[MAX_SHAPE_DIMS];
      for (auto v : llvm::enumerate(module::getShape(reduceOp.getInput())))
        input_shape[v.index()] = (uint32_t)v.value();
      auto &&axes = reduceOp.getAxes();
      int axis_num = axes.size();
      int axis_list[axis_num];
      for (int i = 0; i < axes.size(); i++)
        axis_list[i] = (axes[i].cast<IntegerAttr>().getInt());
      int method = BM1684::get_reduce_type(reduceOp.getMode());
      uint64_t buffer_addr = module::getAddress(reduceOp.getBuffer());
      if (BM1684::getDataType(reduceOp.getInput()) == DTYPE_FP32 ||
          BM1684::getDataType(reduceOp.getInput()) == DTYPE_INT32 ||
          BM1684::getDataType(reduceOp.getInput()) == DTYPE_UINT32) {
        BM1684::instance().dl_nodechip_reduce_full_v3(
            in_addr, out_addr, (const uint32_t *)input_shape, input_dims,
            axis_list, axis_num, method, buffer_addr, buffer_size_ptr,
            (CMD_ID_NODE *)BM1684::instance().cmdid_node);
      } else {
        int keep_dims = reduceOp.getKeepdims() ? 1 : 0;
        buffer_size = BM1684::instance().dl_nodechip_reduce_get_buffer_size_fix8b(
            input_shape, input_dims, axis_list, axis_num, method, keep_dims);
      }
      if (buffer_size > 0) {
        std::vector<int64_t> buffer_shape = {(int64_t)buffer_size};
        auto buffer_type = RankedTensorType::get(buffer_shape, type);
        auto buffer = tpu::BufferOp::create(reduceOp, buffer_type);
        reduceOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
          return operand.get() == reduceOp.getBuffer();
        });
      }
    }

    return success();
  }
};

class SliceGlobalBuffer : public OpRewritePattern<tpu::SliceOp> {
public:
  using OpRewritePattern<tpu::SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::SliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(sliceOp.getBuffer())) {
      return failure();
    }
    if (!module::isBM1684Family()) {
      return failure();
    }
    if (!module::isUniformQuantized(sliceOp.getInput())) {
      return failure();
    }
    if (!module::isUniformQuantized(sliceOp.getOutput())) {
      return failure();
    }
    auto p = sliceOp.parseParam();
    uint64_t buffer_size = 0;
    int shape_dim = module::getShape(sliceOp.getInput()).size();
    // melloc
    int *input_shape = new int[MAX_SHAPE_DIMS];
    int *begin_index = new int[MAX_SHAPE_DIMS];
    int *end_index = new int[MAX_SHAPE_DIMS];
    int *stride = new int[MAX_SHAPE_DIMS];
    // assign param and call func to get buffer size
    module::getGlobalShape(sliceOp.getInput(), input_shape);
    for (int i = 0; i < shape_dim; ++i) {
      begin_index[i] = p.offset_4[i];
      end_index[i] = p.os_4[i] * p.step_4[i] + p.offset_4[i];
      stride[i] = p.step_4[i];
    }
    BM1684::instance().dl_nodechip_stride_slice_fix8b(
        0, 0, 0, 0, &buffer_size, input_shape, shape_dim, STORE_MODE_4N,
        STORE_MODE_4N, 0, 0, begin_index, end_index, stride, 0, NULL);
    // release
    delete[] input_shape;
    delete[] begin_index;
    delete[] end_index;
    delete[] stride;
    // create bufferOp
    std::vector<int64_t> buffer_shape = {(int64_t)buffer_size};
    auto type = module::getStorageType(sliceOp.getOutput());
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(sliceOp, buffer_type);
    sliceOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
      return operand.get() == sliceOp.getBuffer() && operand.getOwner() == sliceOp;
    });
    return success();
  }
};

class SoftmaxGlobalBuffer : public OpRewritePattern<tpu::SoftmaxOp> {
public:
  using OpRewritePattern<tpu::SoftmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::SoftmaxOp softmaxOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(softmaxOp.getBuffer())) {
      return failure();
    }
    if (!module::isBM1684Family()) {
      return failure();
    }
    if (!module::isUniformQuantized(softmaxOp.getInput())) {
      return failure();
    }
    int64_t n, c, h, w;
    module::getNCHW(softmaxOp.getInput(), n, c, h, w);
    std::vector<int64_t> buffer_shape = {ceiling_func(n, (int64_t)4), c, h, w};
    auto type = module::getStorageType(softmaxOp.getOutput());
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(softmaxOp, buffer_type);
    softmaxOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
      return operand.get() == softmaxOp.getBuffer();
    });
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
    uint64_t buffer_size = 0;
    auto attr = permuteOp.parseParam();
    if (module::isBM1684XFamily()) {
      transpose_param_t param = {0};
      param.if_getting_buffer_size = 1;
      for (int i = 0, n = attr.order_fix.size(); i < n; ++i) {
        param.spec.order[i] = attr.order_fix[i];
      }
      param.buffer_size_ptr = &buffer_size;
      auto input_spec = BM168x::get_input_spec(permuteOp);
      auto output_spec = BM168x::get_output_spec(permuteOp);
      BM168x::fix_shape(input_spec->at(0), attr.in_shape_fix);
      BM168x::fix_shape(output_spec->at(0), attr.out_shape_fix);
      BM168x::call_global_func("backend_api_transpose_global", &param,
                               sizeof(param), input_spec->data(),
                               output_spec->data());
    } else if (module::isBM1684Family()) {
      // melloc
      uint32_t *input_shape = new uint32_t[MAX_SHAPE_DIMS];
      int *order = new int[MAX_SHAPE_DIMS];
      uint64_t *buffer_size_ptr = &buffer_size;

      auto input = permuteOp.getInput();
      auto output = permuteOp.getOutput();
      i32_array_t in_order = module::getI32Array(permuteOp.getOrder());
      auto input_addr = module::getAddress(input);
      auto output_addr = module::getAddress(output);
      int input_dims = module::getShape(input).size();
      auto input_dtype = BM1684::getDataType(input);
      auto output_dtype = BM1684::getDataType(output);
      int type_len = BM1684::getFmtBytes(input_dtype);
      int store_mode;
      for (auto v : llvm::enumerate(module::getShape(input)))
        input_shape[v.index()] = (uint32_t)v.value();
      memcpy(order, in_order->data(), (*in_order).size() * sizeof(int));
      if (input_dtype == DTYPE_FP32 || input_dtype == DTYPE_INT32 ||
          input_dtype == DTYPE_UINT32) {
        store_mode = STORE_MODE_1N;
        BM1684::instance().dl_nodechip_transpose(
            input_addr, output_addr, input_shape, order, input_dims, type_len,
            store_mode, 0, buffer_size_ptr,
            (CMD_ID_NODE *)BM1684::instance().cmdid_node);
      } else if (input_dtype == DTYPE_INT8 || input_dtype == DTYPE_UINT8) {
        store_mode = STORE_MODE_4N;
        assert(output_dtype == DTYPE_INT8 || output_dtype == DTYPE_UINT8);
        BM1684::instance().dl_nodechip_transpose_fix8b(
            input_addr, output_addr, input_shape, order, input_dims, store_mode,
            store_mode, 0, buffer_size_ptr,
            (CMD_ID_NODE *)BM1684::instance().cmdid_node);
      } else {
        llvm_unreachable("Not Implemented");
        return failure();
      }
      // release
      delete[] input_shape;
      delete[] order;
      buffer_size_ptr = NULL;
      ;
    } else {
      return failure();
    }
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
    auto buffer_type =
        RankedTensorType::get(module::getShape(op.getInput()), type);
    auto buffer = tpu::BufferOp::create(op, buffer_type);
    op.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
      return operand.get() == op.getBuffer();
    });
    return success();
  }
};

class InterpGlobalBuffer : public OpRewritePattern<tpu::InterpOp> {
public:
  using OpRewritePattern<tpu::InterpOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::InterpOp interpOp,
                                PatternRewriter &rewriter) const override {
    if(!module::isBM1686()) return failure();

    if (!module::isNone(interpOp.getBuffer())) {
      return failure();
    }

    interp_global_param_t param = {0};
    param.if_getting_buffer_size = true;
    uint64_t buffer_size = 0;
    param.buffer_size_ptr = &buffer_size;
    auto input_spec = BM168x::get_input_spec(interpOp);
    auto output_spec = BM168x::get_output_spec(interpOp);
    BM168x::call_global_func("backend_api_interp_global", &param,
                               sizeof(param), input_spec->data(),
                               output_spec->data());
    // add buffer
    if (buffer_size > 0) {
      auto type = ::mlir::Builder(getContext()).getIntegerType(8);
      auto buffer_type = RankedTensorType::get({(int64_t)buffer_size}, type);
      auto buffer = tpu::BufferOp::create(interpOp, buffer_type);
      interpOp.setOperand(1, buffer);
      return success();
    }

    return failure();
  }
};

class DeformGatherGlobalBuffer : public OpRewritePattern<tpu::DeformGatherOp> {
public:
  using OpRewritePattern<tpu::DeformGatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::DeformGatherOp Op,
                                PatternRewriter &rewriter) const override {
    if (!module::isNone(Op.getBuffer())) {
      return failure();
    }
    if (!module::isBM1684XFamily()) {
      return failure();
    }
    std::vector<int64_t> buffer_shape = {};
    auto type = module::getStorageType(Op.getOutput());
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(Op, buffer_type);
    Op.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
      return operand.get() == Op.getBuffer();
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
      SliceGlobalBuffer,
      SoftmaxGlobalBuffer,
      PermuteGlobalBuffer,
      InterpGlobalBuffer,
      NonZeroGlobalBuffer,
      DeformGatherGlobalBuffer
  >(patterns->getContext());
  // clang-format on
}

} // namespace bm168x
} // namespace tpu_mlir
