//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Backend/BM168x/BM1688.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/CustomLayer.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"
#include "tpu_mlir/Support/TPUNnvlcUtil.h"

using namespace llvm;
using namespace tpu_mlir::backend;

namespace tpu_mlir {

namespace bm168x {

class GRUGlobalBuffer : public OpRewriterPatternEx<tpu::GRUOp> {
public:
  GRUGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::GRUOp>(context,"GRUGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::GRUOp GRUOp,
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
      return operand.get() == GRUOp.getBuffer() && operand.getOwner() == GRUOp;
    });
    return success();
  }
  bool shouldPrint(tpu::GRUOp GRUOp) const override { return false;}
};


class FAttentionGlobalBuffer : public OpRewriterPatternEx<tpu::FAttentionOp> {
public:
  FAttentionGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::FAttentionOp>(context,"FAttentionGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::FAttentionOp op,
                                    PatternRewriter &rewriter) const override {
    // if (!module::isNone(FaOp.getBuffer())) {
    //   return failure();
    // }
    // auto type = module::getStorageType(FaOp.getOutput());
    // int64_t batch = FaOp.getBatch();
    // int64_t head = FaOp.getHead();
    // int64_t mq = FaOp.getMq();
    // int64_t mk = FaOp.getMk();
    // // add buffer
    // std::vector<int64_t> buffer_shape = {batch, mq, head, mk};
    // auto buffer_type = RankedTensorType::get(buffer_shape, type);
    // auto buffer = tpu::BufferOp::create(FaOp, buffer_type);
    // FaOp.getBufferMutable().assign(buffer);
    return success();
  }
  bool shouldPrint(tpu::FAttentionOp op) const override { return false;}
};

class GatherElementsGlobalBuffer : public OpRewriterPatternEx<tpu::GatherElementsOp> {
public:
  GatherElementsGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::GatherElementsOp>(context,"GatherElementsGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::GatherElementsOp GatherElementsOp,
                                    PatternRewriter &rewriter) const override {
    if (!module::isNone(GatherElementsOp.getBuffer())) {
      return failure();
    }
    if (!module::isBM1684XFamily() && !module::isBM1690Family()) {
      return failure();
    }
    auto buffer_type =
        GatherElementsOp.getIndices().getType().cast<RankedTensorType>();
    auto buffer = tpu::BufferOp::create(GatherElementsOp, buffer_type);
    GatherElementsOp.setOperand(3, buffer);
    return success();
  }
  bool shouldPrint(tpu::GatherElementsOp GatherElementsOp) const override { return false;}
};

class LSTMGlobalBuffer : public OpRewriterPatternEx<tpu::LSTMOp> {
public:
  LSTMGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::LSTMOp>(context,"LSTMGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::LSTMOp lstmOp,
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
      return operand.get() == lstmOp.getBuffer() &&
             operand.getOwner() == lstmOp;
    });
    return success();
  }
  bool shouldPrint(tpu::LSTMOp lstmOp) const override { return false;}
};

class ReduceGlobalBuffer : public OpRewriterPatternEx<tpu::ReduceOp> {
public:
  ReduceGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::ReduceOp>(context,"ReduceGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::ReduceOp reduceOp,
                                    PatternRewriter &rewriter) const override {
    if (!module::isNone(reduceOp.getBuffer())) {
      return failure();
    }
    auto attr = reduceOp.parseParam();
    if (attr.simplified == false) {
      UNREACHABLE_OP("Not Implemented", reduceOp);
    }
    if (module::isBM1684XFamily() || module::isBM1690Family()) {
      auto type = module::getStorageType(reduceOp.getInput());
      auto input_spec = BM168x::get_input_spec(reduceOp);
      auto output_spec = BM168x::get_output_spec(reduceOp);
      reduce_full_global_param_t param = {0};
      // need to check if it is dynamic mode
      auto run_mode = tpu::getRunMode(reduceOp.getOperation());
      if (run_mode == tpu::RunMode::TPU_STATIC) {
        BM168x::fix_shape(input_spec->at(0), {attr.outer_n, attr.outer_c,
                                              attr.axis_dims, attr.inner_dims});
        BM168x::fix_shape(output_spec->at(0),
                          {attr.outer_n, attr.outer_c, 1, attr.inner_dims});
        param.spec.common.axis_num = 1;
        param.spec.common.axis[0] = 2;
      } else if (run_mode == tpu::RunMode::TPU_DYNAMIC) {
        auto &&axes = reduceOp.getAxes();
        param.spec.common.axis_num = axes.size();
        for (int i = 0; i < axes.size(); i++) {
          param.spec.common.axis[i] = (axes[i].cast<IntegerAttr>().getInt());
        }
        param.spec.common.keep_dims = reduceOp.getKeepdims() ? 1 : 0;
      }

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
          return operand.get() == reduceOp.getBuffer() &&
                 operand.getOwner() == reduceOp;
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
            (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
      } else {
        int keep_dims = reduceOp.getKeepdims() ? 1 : 0;
        buffer_size =
            BM1684::instance().dl_nodechip_reduce_get_buffer_size_fix8b(
                input_shape, input_dims, axis_list, axis_num, method,
                keep_dims);
      }
      if (buffer_size > 0) {
        std::vector<int64_t> buffer_shape = {(int64_t)buffer_size};
        auto buffer_type = RankedTensorType::get(buffer_shape, type);
        auto buffer = tpu::BufferOp::create(reduceOp, buffer_type);
        reduceOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
          return operand.get() == reduceOp.getBuffer() &&
                 operand.getOwner() == reduceOp;
        });
      }
    }

    return success();
  }
  bool shouldPrint(tpu::ReduceOp reduceOp) const override { return false;}
};


class SliceGlobalBuffer : public OpRewriterPatternEx<tpu::SliceOp> {
public:
  SliceGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::SliceOp>(context,"SliceGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::SliceOp sliceOp,
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
    int shape_dim = p.is_4.size();
    // melloc
    int *input_shape = new int[MAX_SHAPE_DIMS];
    int *begin_index = new int[MAX_SHAPE_DIMS];
    int *end_index = new int[MAX_SHAPE_DIMS];
    int *stride = new int[MAX_SHAPE_DIMS];
    // assign param and call func to get buffer size
    for (int i = 0; i < shape_dim; ++i) {
      input_shape[i] = p.is_4[i];
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
    if (!buffer_size)
      return failure();
    // create bufferOp
    std::vector<int64_t> buffer_shape = {(int64_t)buffer_size};
    auto type = module::getStorageType(sliceOp.getOutput());
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(sliceOp, buffer_type);
    sliceOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
      return operand.get() == sliceOp.getBuffer() &&
             operand.getOwner() == sliceOp;
    });
    return success();
  }
  bool shouldPrint(tpu::SliceOp sliceOp) const override { return false;}
};


class ReshapeGlobalBuffer : public OpRewriterPatternEx<tpu::ReshapeOp> {
public:
  ReshapeGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::ReshapeOp>(context,"ReshapeGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::ReshapeOp reshapeOp,
                                    PatternRewriter &rewriter) const override {
    // only used for 4N ndim reshape!
    if (!module::isBM1684Family()) {
      return failure();
    }
    if (!module::isNone(reshapeOp.getBuffer())) {
      return failure();
    }
    if (!module::isUniformQuantized(reshapeOp.getInput())) {
      return failure();
    }
    if (!module::isUniformQuantized(reshapeOp.getOutput())) {
      return failure();
    }
    int64_t in, ic, ih, iw, on, oc, oh, ow;
    module::getNCHW(reshapeOp.getInput(), in, ic, ih, iw);
    module::getNCHW(reshapeOp.getOutput(), on, oc, oh, ow);
    if (on == in) {
      return failure();
    }

    uint64_t buffer_size = on * oc * oh * ow;
    ;

    // create bufferOp
    std::vector<int64_t> buffer_shape = {(int64_t)buffer_size};
    auto type = module::getStorageType(reshapeOp.getOutput());
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(reshapeOp, buffer_type);
    reshapeOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
      return operand.get() == reshapeOp.getBuffer() &&
             operand.getOwner() == reshapeOp;
    });
    return success();
  }
  bool shouldPrint(tpu::ReshapeOp reshapeOp) const override { return false;}
};

class SoftmaxGlobalBuffer : public OpRewriterPatternEx<tpu::SoftmaxOp> {
public:
  SoftmaxGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::SoftmaxOp>(context,"SoftmaxGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::SoftmaxOp softmaxOp,
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
      return operand.get() == softmaxOp.getBuffer() &&
             operand.getOwner() == softmaxOp;
    });
    return success();
  }
  bool shouldPrint(tpu::SoftmaxOp softmaxOp) const override { return false;}
};

class PermuteGlobalBuffer : public OpRewriterPatternEx<tpu::PermuteOp> {
public:
 PermuteGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::PermuteOp>(context,"PermuteGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::PermuteOp permuteOp,
                                    PatternRewriter &rewriter) const override {
    if (!module::isNone(permuteOp.getBuffer())) {
      return failure();
    }
    uint64_t buffer_size = 0;
    auto attr = permuteOp.parseParam();
    if (module::isBM1684XFamily() || module::isBM1690Family()) {
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
            (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
      } else if (input_dtype == DTYPE_INT8 || input_dtype == DTYPE_UINT8) {
        store_mode = STORE_MODE_4N;
        assert(output_dtype == DTYPE_INT8 || output_dtype == DTYPE_UINT8);
        BM1684::instance().dl_nodechip_transpose_fix8b(
            input_addr, output_addr, input_shape, order, input_dims, store_mode,
            store_mode, 0, buffer_size_ptr,
            (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
      } else {
        UNREACHABLE_OP("Not Implemented", permuteOp);
        return failure();
      }
      // release
      delete[] input_shape;
      delete[] order;
      buffer_size_ptr = NULL;
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
  bool shouldPrint(tpu::PermuteOp permuteOp) const override { return false;}
};

class NonZeroGlobalBuffer : public OpRewriterPatternEx<tpu::NonZeroOp> {
public:
 NonZeroGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::NonZeroOp>(context,"NonZeroGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::NonZeroOp op,
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
      return operand.get() == op.getBuffer() && operand.getOwner() == op;
    });
    return success();
  }
  bool shouldPrint(tpu::NonZeroOp op) const override { return false;}
};

class InterpGlobalBuffer : public OpRewriterPatternEx<tpu::InterpOp> {
public:
 InterpGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::InterpOp>(context,"InterpGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::InterpOp interpOp,
                                    PatternRewriter &rewriter) const override {
    if (!module::isBM1684XFamily() && !module::isBM1690Family())
      return failure();

    if (!module::isNone(interpOp.getBuffer())) {
      return failure();
    }

    interp_global_param_t param = {0};
    param.if_getting_buffer_size = true;
    uint64_t buffer_size = 0;
    param.buffer_size_ptr = &buffer_size;
    auto input_spec = BM168x::get_input_spec(interpOp);
    auto output_spec = BM168x::get_output_spec(interpOp);
    BM168x::call_global_func("backend_api_interp_global", &param, sizeof(param),
                             input_spec->data(), output_spec->data());
    // add buffer
    if (buffer_size > 0) {
      auto type = ::mlir::Builder(getContext()).getIntegerType(8);
      auto buffer_type = RankedTensorType::get({(int64_t)buffer_size}, type);
      auto buffer = tpu::BufferOp::create(interpOp, buffer_type);
      interpOp.setOperand(2, buffer);
      return success();
    }

    return failure();
  }
  bool shouldPrint(tpu::InterpOp interpOp) const override { return false;}
};


class MatMulGlobalBuffer : public OpRewriterPatternEx<tpu::MatMulOp> {
public:
 MatMulGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::MatMulOp>(context,"MatMulGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::MatMulOp matmulOp,
                                    PatternRewriter &rewriter) const override {

    if (!module::isNone(matmulOp.getBuffer())) {
      return failure();
    }

    if(!supportMultiCore(matmulOp)) {
      return failure();
    }

    auto p = matmulOp.parseParam();
    fc_global_spec_t spec = {0};
    memset(&spec, 0, sizeof(spec));
    spec.if_getting_buffer_size = true;
    uint64_t buffer_size = 0;
    spec.buffer_size_ptr = &buffer_size;
    spec.if_relu = p.do_relu;
    spec.relu_limit = p.relu_limit;
    spec.have_bias = p.with_bias;
    spec.requant_mode = -1;
    spec.R_transpose = p.right_transpose;
    if (module::isUniformQuantized(matmulOp.getInput())) {
      spec.rshift = 0;
      spec.is_asymmetric = 1;
      spec.rzp_is_const = 1;
      spec.rzp_const_val = p.right_zp;
      spec.izp_const_val = p.input_zp;
      if (module::isUniformQuantized(matmulOp.getOutput())) {
        auto rshift_v = module::getI64Array(matmulOp.getRshifts(), 1, 0);
        auto multiplier_v = module::getI64Array(matmulOp.getMultipliers(), 1, 1);
        assert(rshift_v->size() == 1);
        assert(multiplier_v->size() == 1);
        spec.requant_mode = static_cast<int>(matmulOp.getQuantMode());
        spec.mul_val = multiplier_v->at(0);
        spec.shift_val = -rshift_v->at(0);
        auto output_type = module::getUniformQuantizedType(matmulOp.getOutput());
        spec.offset_val = output_type.getZeroPoint();
        spec.round_mode = ROUNDING_HALF_AWAY_FROM_ZERO;
      }
    }
    auto input_spec = BM168x::get_input_spec(matmulOp);
    auto output_spec = BM168x::get_output_spec(matmulOp);
    BM168x::call_global_func("backend_api_fc_multi_core_global", &spec, sizeof(spec),
                             input_spec->data(), output_spec->data());
    if (buffer_size > 0) {
      auto type = ::mlir::Builder(getContext()).getIntegerType(8);
      auto buffer_type = RankedTensorType::get({(int64_t)buffer_size}, type);
      auto buffer = tpu::BufferOp::create(matmulOp, buffer_type, tpu::BufferType::L2);
      matmulOp.setOperand(4, buffer);
      return success();
    }
    return failure();
  }
  bool shouldPrint(tpu::MatMulOp matmulOp) const override { return false;}
};

class GridSamplerBuffer : public OpRewriterPatternEx<tpu::GridSamplerOp> {
public:
 GridSamplerBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::GridSamplerOp>(context,"GridSamplerBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::GridSamplerOp gridSamplerOp,
                                    PatternRewriter &rewriter) const override {
    if (!module::isBM1684XFamily()) {
      return failure();
    }
    if (!module::isNone(gridSamplerOp.getBuffer())) {
      return failure();
    }
    auto type = ::mlir::Builder(getContext()).getIntegerType(8);
    auto output_shape = module::getShape(gridSamplerOp.getOutput());
    auto coeff_dtype_size = module::getDtypeSize(gridSamplerOp.getOutput());
    auto dim = output_shape.size();
    int f32_size = 4;
    int64_t buffer_size;
    if (dim == 4) {
      buffer_size = output_shape[0] * align_up(output_shape[2] * output_shape[3], Arch::NPU_NUM) * (2 * f32_size + 4 * coeff_dtype_size);
    } else if (dim == 5) {
      buffer_size = output_shape[0] * align_up(output_shape[2] * output_shape[3] * output_shape[4], Arch::NPU_NUM) * (3 * f32_size + 8 * coeff_dtype_size);
    } else {
      return failure();
    }
    auto buffer_type = RankedTensorType::get({(int64_t)buffer_size}, type);
    auto buffer = tpu::BufferOp::create(gridSamplerOp, buffer_type);
    gridSamplerOp.setOperand(gridSamplerOp.getNumOperands() - 1, buffer);
    return success();
  }
  bool shouldPrint(tpu::GridSamplerOp gridSamplerOp) const override { return false;}
};

class DeformGatherGlobalBuffer : public OpRewriterPatternEx<tpu::DeformGatherOp> {
public:
 DeformGatherGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::DeformGatherOp>(context,"DeformGatherGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::DeformGatherOp Op,
                                    PatternRewriter &rewriter) const override {
    if (!module::isNone(Op.getBuffer())) {
      return failure();
    }
    if (!(module::isBM1684XFamily() || module::isBM1690Family())) {
      return failure();
    }
    deform_gather_attr_t p = Op.parseParam();
    uint64_t buffer_size = 0;
    auto conved_H =
        ((p.ih - (p.dh * (p.kh - 1) + 1) + p.pht + p.phb) / p.sh + 1);
    auto conved_W =
        ((p.iw - (p.dw * (p.kw - 1) + 1) + p.pwl + p.pwr) / p.sw + 1);
    auto full_size = p.kh * p.kw * conved_H * conved_W * sizeof(float);
    buffer_size = 2 * p.n * p.deform_groups * full_size;
    if (p.use_mask)
      buffer_size = std::max(buffer_size, (uint64_t)p.n * p.ic *
                                              p.deform_groups * full_size);
    auto type = module::getStorageType(Op.getInput());
    auto buffer_type = RankedTensorType::get({(int64_t)buffer_size}, type);
    auto buffer = tpu::BufferOp::create(Op, buffer_type);
    Op.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
      return operand.get() == Op.getBuffer() && operand.getOwner() == Op;
    });
    return success();
  }
  bool shouldPrint(tpu::DeformGatherOp Op) const override { return false;}
};

class Space2BatchGlobalBuffer : public OpRewriterPatternEx<tpu::Space2BatchOp> {
public:
 Space2BatchGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::Space2BatchOp>(context,"Space2BatchGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::Space2BatchOp space2batchOp,
                                    PatternRewriter &rewriter) const override {
    if (!module::isNone(space2batchOp.getBuffer())) {
      return failure();
    }
    uint64_t buffer_size = 0;
    if (module::isBM1684XFamily() || module::isBM1690Family()) {
      llvm_unreachable("Not supported now");
      return failure();
    } else if (module::isBM1684Family()) {
      // melloc
      uint32_t *input_shape = new uint32_t[MAX_SHAPE_DIMS];

      auto input = space2batchOp.getInput();
      auto output = space2batchOp.getOutput();
      auto pads_v = module::getI64Array(space2batchOp.getPads());
      int64_t pad_top = pads_v->at(0);
      int64_t pad_bottom = pads_v->at(1);
      int64_t pad_left = pads_v->at(2);
      int64_t pad_right = pads_v->at(3);

      int input_dims = module::getShape(input).size();
      auto input_dtype = BM1684::getDataType(input);
      auto output_dtype = BM1684::getDataType(output);
      int type_len = BM1684::getFmtBytes(input_dtype);
      int store_mode;
      for (auto v : llvm::enumerate(module::getShape(input)))
        input_shape[v.index()] = (uint32_t)v.value();
      if (input_dtype == DTYPE_FP32 || input_dtype == DTYPE_INT32 ||
          input_dtype == DTYPE_UINT32) {
        int height_pad = input_shape[2] + pad_top + pad_bottom;
        int width_pad = input_shape[3] + pad_left + pad_right;
        buffer_size =
            input_shape[0] * input_shape[1] * height_pad * width_pad * type_len;
      } else if (input_dtype == DTYPE_INT8 || input_dtype == DTYPE_UINT8) {
        store_mode = STORE_MODE_4N;
        assert(output_dtype == DTYPE_INT8 || output_dtype == DTYPE_UINT8);
        int block_h = (int)space2batchOp.getBlockH();
        int block_w = (int)space2batchOp.getBlockW();
        int block_size[2] = {block_h, block_w};
        int pad_sizes[4] = {(int)pad_top, (int)pad_bottom, (int)pad_left,
                            (int)pad_right};
        BM1684::instance().dl_nodechip_space2batch_fix8b(
            0, 0, 0, &buffer_size, (int *)input_shape, input_dims, store_mode,
            store_mode, block_size, pad_sizes, NULL, NULL);
      } else {
        UNREACHABLE_OP("Not Implemented", space2batchOp);
        return failure();
      }
      // release
      delete[] input_shape;
    } else {
      UNREACHABLE_OP("Not Implemented", space2batchOp);
      return failure();
    }
    auto type = module::getStorageType(space2batchOp.getInput());
    // add buffer
    if (buffer_size > 0) {
      auto buffer_type = RankedTensorType::get({(int64_t)buffer_size}, type);
      auto buffer = tpu::BufferOp::create(space2batchOp, buffer_type);
      space2batchOp.setOperand(1, buffer);
      return success();
    }
    return failure();
  }
  bool shouldPrint(tpu::Space2BatchOp space2batchOp) const override { return false;}
};

class Batch2SpaceGlobalBuffer : public OpRewriterPatternEx<tpu::Batch2SpaceOp> {
public:
 Batch2SpaceGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::Batch2SpaceOp>(context,"Batch2SpaceGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::Batch2SpaceOp batch2spaceOp,
                                    PatternRewriter &rewriter) const override {
    if (!module::isNone(batch2spaceOp.getBuffer())) {
      return failure();
    }
    uint64_t buffer_size = 0;
    if (module::isBM1684XFamily() || module::isBM1690Family()) {
      llvm_unreachable("Not supported now");
      return failure();
    } else if (module::isBM1684Family()) {
      // melloc
      uint32_t *input_shape = new uint32_t[MAX_SHAPE_DIMS];

      auto input = batch2spaceOp.getInput();
      auto output = batch2spaceOp.getOutput();
      auto crops_v = module::getI64Array(batch2spaceOp.getCrops());
      int64_t crop_top = crops_v->at(0);
      int64_t crop_bottom = crops_v->at(1);
      int64_t crop_left = crops_v->at(2);
      int64_t crop_right = crops_v->at(3);

      int input_dims = module::getShape(input).size();
      auto input_dtype = BM1684::getDataType(input);
      auto output_dtype = BM1684::getDataType(output);
      int type_len = BM1684::getFmtBytes(input_dtype);
      int store_mode;
      for (auto v : llvm::enumerate(module::getShape(input)))
        input_shape[v.index()] = (uint32_t)v.value();
      if (input_dtype == DTYPE_FP32 || input_dtype == DTYPE_INT32 ||
          input_dtype == DTYPE_UINT32) {
        buffer_size = input_shape[0] * input_shape[1] * input_shape[2] *
                      input_shape[3] * type_len;
      } else if (input_dtype == DTYPE_INT8 || input_dtype == DTYPE_UINT8) {
        store_mode = STORE_MODE_4N;
        assert(output_dtype == DTYPE_INT8 || output_dtype == DTYPE_UINT8);
        int block_h = (int)batch2spaceOp.getBlockH();
        int block_w = (int)batch2spaceOp.getBlockW();
        int block_size[2] = {block_h, block_w};
        int pad_sizes[4] = {(int)crop_top, (int)crop_bottom, (int)crop_left,
                            (int)crop_right};
        BM1684::instance().dl_nodechip_batch2space_fix8b(
            0, 0, 0, 0, &buffer_size, (int *)input_shape, input_dims,
            store_mode, store_mode, block_size, pad_sizes, NULL, NULL);
      } else {
        UNREACHABLE_OP("Not Implemented", batch2spaceOp);
        return failure();
      }
      // release
      delete[] input_shape;
    } else {
      UNREACHABLE_OP("Not Implemented", batch2spaceOp);
      return failure();
    }
    auto type = module::getStorageType(batch2spaceOp.getInput());
    // add buffer
    if (buffer_size > 0) {
      auto buffer_type = RankedTensorType::get({(int64_t)buffer_size}, type);
      auto buffer = tpu::BufferOp::create(batch2spaceOp, buffer_type);
      batch2spaceOp.setOperand(1, buffer);
      return success();
    }
    return failure();
  }
  bool shouldPrint(tpu::Batch2SpaceOp batch2spaceOp) const override { return false;}
};


class TileGlobalBuffer : public OpRewriterPatternEx<tpu::TileOp> {
public:
 TileGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::TileOp>(context,"TileGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::TileOp TileOp,
                                    PatternRewriter &rewriter) const override {
    if (!module::isNone(TileOp.getBuffer())) {
      return failure();
    }
    if (!module::isBM1684Family() && (!module::isBM1684XFamily()) &&
        (!module::isBM1690Family())) {
      UNREACHABLE_OP("Not Implemented", TileOp);
      return failure();
    }
    auto input_shape = module::getShape(TileOp.getInput());
    int input_dim = input_shape.size();
    auto output_shape = module::getShape(TileOp.getOutput());
    int tile_coeff[8];
    auto type = module::getStorageType(TileOp.getInput());
    int64_t type_len = module::getDtypeSize(TileOp.getInput());
    uint64_t buffer_size = 0;
    int tile_count = 0;
    int min_tile = output_shape[0] / input_shape[0];
    uint64_t total_size = 1;
    for (int i = 0; i < input_dim; ++i) {
      tile_coeff[i] =
          output_shape[i] < 0 ? 1 : output_shape[i] / input_shape[i];
      if (tile_coeff[i] > 1)
        tile_count++;
      if (tile_coeff[i] < min_tile)
        min_tile = tile_coeff[i];
      total_size *= output_shape[i];
    }
    if (type_len > 0) {
      if (tile_count > 1) {
        buffer_size = total_size / min_tile * type_len;
      }
      if (buffer_size > 0) {
        std::vector<int64_t> buffer_shape = {(int64_t)buffer_size};
        auto buffer_type = RankedTensorType::get(buffer_shape, type);
        auto buffer = tpu::BufferOp::create(TileOp, buffer_type);
        TileOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
          return operand.get() == TileOp.getBuffer() &&
                 operand.getOwner() == TileOp;
        });
        return success();
      }
    }
    return failure();
  }
  bool shouldPrint(tpu::TileOp TileOp) const override { return false;}
};


class IndexPutGlobalBuffer : public OpRewriterPatternEx<tpu::IndexPutOp> {
public:
 IndexPutGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::IndexPutOp>(context,"IndexPutGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::IndexPutOp IndexPutOp,
                                    PatternRewriter &rewriter) const override {
    if (!module::isNone(IndexPutOp.getBuffer()) ||
        !IndexPutOp.getAccumulate()) {
      return failure();
    }
    if (!module::isBM1684XFamily()) {
      return failure();
    }
    auto elment_num = module::getNumElements(IndexPutOp.getValues());
    auto type = module::getStorageType(IndexPutOp.getValues());
    // add buffer
    std::vector<int64_t> buffer_shape = {elment_num}; // double buffer
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(IndexPutOp, buffer_type);
    IndexPutOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
      return operand.get() == IndexPutOp.getBuffer() &&
             operand.getOwner() == IndexPutOp;
    });
    return success();
  }
  bool shouldPrint(tpu::IndexPutOp IndexPutOp) const override { return false;}
};


class PadGlobalBuffer : public OpRewriterPatternEx<tpu::PadOp> {
public:
 PadGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::PadOp>(context,"PadGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::PadOp padOp,
                                    PatternRewriter &rewriter) const override {
    if (!module::isNone(padOp.getBuffer())) {
      return failure();
    }
    if (!module::isBM1684Family()) {
      return failure();
    }
    std::vector<int64_t> shape = module::getShape(padOp.getInput());
    if (shape.size() < 5 && !module::isUniformQuantized(padOp.getInput())) {
      return failure();
    }
    if (shape.size() <= 5) {
      std::vector<int64_t> buffer_shape;
      buffer_shape.push_back(ceiling_func(shape[0], (int64_t)4));
      for (int i = 1; i < shape.size(); ++i) {
        buffer_shape.push_back(shape[i]);
      }
      auto type = module::getStorageType(padOp.getOutput());
      auto buffer_type = RankedTensorType::get(buffer_shape, type);
      auto buffer = tpu::BufferOp::create(padOp, buffer_type);
      padOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
        return operand.get() == padOp.getBuffer() &&
               operand.getOwner() == padOp;
      });
      return success();
    } else {
      return failure();
    }
  }
  bool shouldPrint(tpu::PadOp padOp) const override { return false;}
};

class GatherGlobalBuffer : public OpRewriterPatternEx<tpu::GatherOp> {
public:
  GatherGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::GatherOp>(context, "GatherGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::GatherOp GatherOp,
                                    PatternRewriter &rewriter) const override {
    if (!module::isNone(GatherOp.getBuffer())) {
      return failure();
    }
    if (!module::isBM1684Family()) {
      return failure();
    }
    auto elment_num = module::getNumElements(GatherOp.getInput());
    auto type = module::getStorageType(GatherOp.getInput());
    // add buffer
    std::vector<int64_t> buffer_shape = {2, elment_num}; // double buffer
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(GatherOp, buffer_type);
    GatherOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
      return operand.get() == GatherOp.getBuffer() &&
             operand.getOwner() == GatherOp;
    });
    return success();
  }
  bool shouldPrint(tpu::GatherOp GatherOp) const override { return false;}
};

class Pool3DGlobalBuffer : public OpRewriterPatternEx<tpu::Pool3DOp> {
public:
  Pool3DGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::Pool3DOp>(context, "Pool3DGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::Pool3DOp Pool3DOp,
                                    PatternRewriter &rewriter) const override {
    if (!module::isNone(Pool3DOp.getBuffer())) {
      return failure();
    }
    if (!module::isBM1684Family()) {
      return failure();
    }

    auto elment_num = module::getNumElements(Pool3DOp.getInput());
    auto type = module::getStorageType(Pool3DOp.getInput());
    // add buffer
    std::vector<int64_t> buffer_shape = {2, elment_num}; // double buffer
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(Pool3DOp, buffer_type);
    Pool3DOp.getBuffer().replaceUsesWithIf(buffer, [&](OpOperand &operand) {
      return operand.get() == Pool3DOp.getBuffer() &&
             operand.getOwner() == Pool3DOp;
    });
    return success();
  }
  bool shouldPrint(tpu::Pool3DOp Pool3DOp) const override { return false;}
};


class ScatterElementsGlobalBuffer : public OpRewriterPatternEx<tpu::ScatterElementsOp> {
public:
  ScatterElementsGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::ScatterElementsOp>(context, "ScatterElementsGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::ScatterElementsOp ScatterElementsOp,
                                    PatternRewriter &rewriter) const override {
    if (!module::isNone(ScatterElementsOp.getBuffer())) {
      return failure();
    }
    if (!module::isBM1684XFamily() && !module::isBM1690Family()) {
      return failure();
    }
    auto buffer_type =
        ScatterElementsOp.getIndices().getType().cast<RankedTensorType>();
    auto buffer = tpu::BufferOp::create(ScatterElementsOp, buffer_type);
    ScatterElementsOp.setOperand(4, buffer);
    return success();
  }
  bool shouldPrint(tpu::ScatterElementsOp ScatterElementsOp) const override { return false;}
};


class ScatterNDGlobalBuffer : public OpRewriterPatternEx<tpu::ScatterNDOp> {
public:
  ScatterNDGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::ScatterNDOp>(context, "ScatterNDGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::ScatterNDOp ScatterNDOp,
                                    PatternRewriter &rewriter) const override {
    if (!module::isNone(ScatterNDOp.getBuffer())) {
      return failure();
    }
    if (!module::isBM1684XFamily() && !module::isBM1690Family()) {
      return failure();
    }
    auto buffer_type =
        // ScatterNDOp.getInputData().getType().cast<RankedTensorType>();
        RankedTensorType::get(module::getShape(ScatterNDOp.getInputData()),
                              rewriter.getI32Type());
    auto buffer = tpu::BufferOp::create(ScatterNDOp, buffer_type);
    ScatterNDOp.setOperand(3, buffer);
    return success();
  }
  bool shouldPrint(tpu::ScatterNDOp ScatterNDOp) const override { return false;}
};


class NmsGlobalBuffer : public OpRewriterPatternEx<tpu::NmsOp> {
public:
  NmsGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::NmsOp>(context, "NmsGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::NmsOp NmsOp,
                                    PatternRewriter &rewriter) const override {
    if (!module::isNone(NmsOp.getBuffer())) {
      return failure();
    }
    if (!module::isBM1684XFamily() ||
        (module::isBM1684XFamily() && !module::isBM1688())) {
      return failure();
    }

    int64_t buffer_size = BUFFER_SIZE;
    auto type = module::getStorageType(NmsOp.getInputs()[0]);
    std::vector<int64_t> buffer_shape = {(int64_t)buffer_size};
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(NmsOp, buffer_type);
    NmsOp.setOperand(NmsOp.getNumOperands() - 1, buffer);
    return success();
  }
  bool shouldPrint(tpu::NmsOp NmsOp) const override { return false;}
};


class YoloDetectionGlobalBuffer : public OpRewriterPatternEx<tpu::YoloDetectionOp> {
public:
  YoloDetectionGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::YoloDetectionOp>(context, "YoloDetectionGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::YoloDetectionOp yoloDetectionOp,
                                    PatternRewriter &rewriter) const override {
    if (!module::isNone(yoloDetectionOp.getBuffer())) {
      return failure();
    }
    if (!module::isBM1684XFamily() ||
        (module::isBM1684XFamily() && !module::isBM1688())) {
      return failure();
    }
    auto process = module::getPostprocess();
    if (process.starts_with("yolov5")) {
      return failure();
    }
    int64_t buffer_size = BUFFER_SIZE;
    auto type = module::getStorageType(yoloDetectionOp.getInputs()[0]);
    std::vector<int64_t> buffer_shape = {(int64_t)buffer_size};
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(yoloDetectionOp, buffer_type);
    yoloDetectionOp.setOperand(yoloDetectionOp.getNumOperands() - 1, buffer);
    return success();
  }
  bool shouldPrint(tpu::YoloDetectionOp yoloDetectionOp) const override { return false;}
};


class  DetectionOutputGlobalBuffer : public OpRewriterPatternEx<tpu::DetectionOutputOp> {
public:
   DetectionOutputGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::DetectionOutputOp>(context, "DetectionOutputGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::DetectionOutputOp detectionOutputOp,
                                    PatternRewriter &rewriter) const override {
    if (!module::isNone(detectionOutputOp.getBuffer())) {
      return failure();
    }

    if (!module::isBM1684XFamily() ||
        (module::isBM1684XFamily() && !module::isBM1688())) {
      return failure();
    }

    int64_t buffer_size = BUFFER_SIZE;
    auto type = module::getStorageType(detectionOutputOp.getInputs()[0]);
    std::vector<int64_t> buffer_shape = {(int64_t)buffer_size};
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(detectionOutputOp, buffer_type);
    detectionOutputOp.setOperand(detectionOutputOp.getNumOperands() - 1,
                                 buffer);
    return success();
  }
  bool shouldPrint(tpu::DetectionOutputOp detectionOutputOp) const override { return false;}
};


class  TopKGlobalBuffer : public OpRewriterPatternEx<tpu::TopKOp> {
public:
   TopKGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::TopKOp>(context, "TopKGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::TopKOp TopKOp,
                                    PatternRewriter &rewriter) const override {
    if (!module::isNone(TopKOp.getBufferVal()) ||
        !module::isNone(TopKOp.getBufferIdx())) {
      return failure();
    }

    if (!module::isBM1684XFamily()) {
      return failure();
    }

    int64_t num_elements = module::getNumElements(TopKOp.getInput());
    auto val_type = module::getStorageType(TopKOp.getValues());
    auto idx_type = module::getStorageType(TopKOp.getIndices());
    std::vector<int64_t> buffer_shape = {num_elements};
    auto val_buffer_type = RankedTensorType::get(buffer_shape, val_type);
    auto idx_buffer_type = RankedTensorType::get(buffer_shape, idx_type);

    OpBuilder builder(TopKOp->getContext());
    builder.setInsertionPoint(TopKOp);
    auto val_loc = module::getLocLike(TopKOp, "buffer_val");
    auto val_buf_op = builder.create<tpu::BufferOp>(val_loc, val_buffer_type);
    auto idx_loc = module::getLocLike(TopKOp, "buffer_idx");
    auto idx_buf_op = builder.create<tpu::BufferOp>(idx_loc, idx_buffer_type);

    TopKOp.setOperand(TopKOp.getNumOperands() - 2, val_buf_op);
    TopKOp.setOperand(TopKOp.getNumOperands() - 1, idx_buf_op);
    return success();
  }
  bool shouldPrint(tpu::TopKOp TopKOp) const override { return false;}
};


class SortGlobalBuffer : public OpRewriterPatternEx<tpu::SortOp> {
public:
   SortGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::SortOp>(context, "SortGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::SortOp op,
                                    PatternRewriter &rewriter) const override {
    if (!module::isBM1684XFamily()) {
      return failure();
    }
    auto _op = op.getOperation();
    auto input_spec = BM168x::get_input_spec(_op);
    auto output_spec = BM168x::get_output_spec(_op);
    sort_per_dim_param_t param = {0};
    param.buffer_addr = module::getAddress(op.getBuffer());
    param.axis = op.getAxis();
    param.descending = op.getDescending();
    param.is_argsort = module::isNone(op.getValues());
    int64_t buffer_size = BM168x::call_global_bfsz_func(
      "backend_api_sort_per_dim_global_bfsz",
      &param, sizeof(param),
      input_spec->data(), output_spec->data());
    if (buffer_size) {
      std::vector<int64_t> buffer_shape = {buffer_size};
      auto buffer_type = RankedTensorType::get(buffer_shape, rewriter.getI8Type());
      OpBuilder builder(op->getContext());
      builder.setInsertionPoint(op);
      auto loc = module::getLocLike(op, "buffer");
      auto buf_op = builder.create<tpu::BufferOp>(loc, buffer_type);
      op.setOperand(op.getNumOperands() - 1, buf_op);
      return success();
    } else {
      return failure();
    }
  }
  bool shouldPrint(tpu::SortOp op) const override { return false;}
};

class CustomGlobalBuffer : public OpRewriterPatternEx<tpu::CustomOp> {
public:
   CustomGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::CustomOp>(context, "CustomGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::CustomOp customOp,
                                    PatternRewriter &rewriter) const override {
    if (!module::isBM1684XFamily()) {
      return failure();
    }
    auto op = customOp.getOperation();
    auto input_spec = BM168x::get_input_spec(op);
    auto output_spec = BM168x::get_output_spec(op);
    // call backend api according to the custom op name
    std::string op_name = customOp.getName().str();
    std::string api_name = "backend_api_" + op_name + "_global_bfsz";
    // parse param of the custom op
    auto params = customOp.getParams();
    std::vector<custom_param_t> values;
    values.push_back({0});
    customOpProcessParam(params, values);
    int64_t buffer_size = BM168x::call_global_bfsz_custom_func(
                            api_name.c_str(), values.data(), values.size() * sizeof(custom_param_t),
                            input_spec->data(), output_spec->data());

    if (buffer_size > 0) {
      std::vector<int64_t> buffer_shape = {buffer_size};
      auto buffer_type = RankedTensorType::get(buffer_shape, rewriter.getI8Type());
      OpBuilder builder(customOp->getContext());
      builder.setInsertionPoint(customOp);
      auto loc = module::getLocLike(customOp, "buffer");
      auto buf_op = builder.create<tpu::BufferOp>(loc, buffer_type);
      customOp.setOperand(customOp.getNumOperands() - 1, buf_op);
      return success();
    } else {
      return failure();
    }
  }
  bool shouldPrint(tpu::CustomOp customOp) const override { return false;}
};


class WhereGlobalBuffer : public OpRewriterPatternEx<tpu::WhereOp> {
public:
   WhereGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::WhereOp>(context, "WhereGlobalBuffer") {}

  LogicalResult matchAndRewriteImpl(tpu::WhereOp WhereOp,
                                    PatternRewriter &rewriter) const override {
    if (tpu::getRunMode(WhereOp) != tpu::RunMode::TPU_DYNAMIC) {
      return failure();
    }
    if (!module::isNone(WhereOp.getBuffer())) {
      return failure();
    }
    if (module::isOpInGroup(WhereOp)){
      return failure();
    }
    auto out_shape = module::getShape(WhereOp.getOutput());
    auto cond_shape = module::getShape(WhereOp.getCond());
    auto out_dim = out_shape.size();
    auto cond_reshape = shape_expand_dim(cond_shape, out_dim);
    int buffer_num = 0;
    for (int i = 0; i < out_dim; ++i) {
      if (out_shape[i] != cond_reshape[i]) {
        buffer_num ++;
        break;
      }
    }
    if (!WhereOp.getXIsConst()) {
      auto tbrn = WhereOp.getTbrn();
      auto tbrn_shape = module::getShape(tbrn);
      auto tbrn_reshape = shape_expand_dim(tbrn_shape, out_dim);
      for (int i = 0; i < out_dim; ++i) {
        if (out_shape[i] != tbrn_reshape[i]) {
          buffer_num ++;
          break;
        }
      }
    }
    if (!WhereOp.getYIsConst()) {
      auto fbrn = WhereOp.getFbrn();
      auto fbrn_shape = module::getShape(fbrn);
      auto fbrn_reshape = shape_expand_dim(fbrn_shape, out_dim);
      for (int i = 0; i < out_dim; ++i) {
        if (out_shape[i] != fbrn_reshape[i]) {
          buffer_num ++;
          break;
        }
      }
    }
    if (!buffer_num) {
      return failure();
    }
    auto elment_num = module::getNumElements(WhereOp.getOutput());
    auto type = module::getStorageType(WhereOp.getCond());
    // add buffer
    std::vector<int64_t> buffer_shape = {elment_num * 2 * buffer_num}; // double buffer for tile
    auto buffer_type = RankedTensorType::get(buffer_shape, type);
    auto buffer = tpu::BufferOp::create(WhereOp, buffer_type);
    WhereOp.setOperand(WhereOp.getNumOperands() - 1, buffer);
    return success();
  }
  bool shouldPrint(tpu::WhereOp WhereOp) const override { return false;}
};

class ConvbwdGlobalBuffer : public OpRewriterPatternEx<tpu::ConvbwdOp> {
public:
  ConvbwdGlobalBuffer(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::ConvbwdOp>(context, "ConvbwdGlobalBuffer") {}
  LogicalResult matchAndRewriteImpl(tpu::ConvbwdOp ConvbwdOp,
                                PatternRewriter &rewriter) const override {
    if (!module::isBM1690Family()) {
      return failure();
    }
    #define DIV_UP(a, b) ((a) == 0 ? 0 : ((a) - 1) / (b) + 1)
    auto kernel_shape = module::getI64Array(ConvbwdOp.getKernelShape());
    int _32oc_kernel_shape[4] = {0,0,0,0};

    // 32oc shape
    _32oc_kernel_shape[0] = kernel_shape->at(1);
    _32oc_kernel_shape[1] = kernel_shape->at(2)*kernel_shape->at(3);
    _32oc_kernel_shape[2] = DIV_UP( kernel_shape->at(0), 32 ); // =0 ?
    _32oc_kernel_shape[3] = 32;

    int kernel_size = 1;
    int dim = 4;
    for(int i = 0;i < dim;i++){
      kernel_size *= _32oc_kernel_shape[i];
    }
    int64_t buffer_size = kernel_size;
    if (buffer_size > 0) {
      auto type = module::getStorageType(ConvbwdOp.getInput());
      std::vector<int64_t> buffer_shape = {(int64_t)buffer_size};
      auto buffer_type = RankedTensorType::get(buffer_shape, type);
      auto buffer = tpu::BufferOp::create(ConvbwdOp, buffer_type);
      ConvbwdOp.setOperand(ConvbwdOp.getNumOperands() - 1,
                                  buffer);
      return success();
    }else{
      return failure();
    }
  }
  bool shouldPrint(tpu::ConvbwdOp ConvbwdOp) const override { return false;}
};
} // namespace bm168x

class MaskRCNNRPNGetBboxesGlobalBuffer : public OpRewritePattern<tpu::MaskRCNNRPNGetBboxesOp> {
public:
  using OpRewritePattern<tpu::MaskRCNNRPNGetBboxesOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::MaskRCNNRPNGetBboxesOp SuperiorOp,
                                PatternRewriter &rewriter) const override {

    const int batch_size     = module::getShape(SuperiorOp.getClsScores_0())[0];
    const int max_filter_num = SuperiorOp.getMAX_LENGTH_STATIC_STRECHED();
    const int num_indexes    = SuperiorOp.getNUM_INDEXES();
    const int num_classes    = SuperiorOp.getNUM_CLASSES();
    const int H_dyn_max      = SuperiorOp.getH_RPN_DYN_MAX();
    const int W_dyn_max      = SuperiorOp.getW_RPN_DYN_MAX();
    const int C              = SuperiorOp.getCHANNEL_RPN_SCORES();
    // const int C_bboxes       = SuperiorOp.getCHANNEL_RPN_BBOXES();
    const int NMS_PRE        = SuperiorOp.getNMS_PRE();
    const int HARDWARE_FACTOR_TOPK        = SuperiorOp.getHARDWARE_FACTOR_TOPK();
    const int TOPK_ONNX_NMS  = SuperiorOp.getTOPK_ONNX_NMS();
    const int num_prior_static_stretched =  SuperiorOp.getMAX_LENGTH_STATIC_STRECHED();
    const int H_dynamic_max[5] = {H_dyn_max,div_up(H_dyn_max,2),div_up(H_dyn_max,4),div_up(H_dyn_max,8),div_up(H_dyn_max,16)};
    const int W_dynamic_max[5] = {W_dyn_max,div_up(W_dyn_max,2),div_up(W_dyn_max,4),div_up(W_dyn_max,8),div_up(W_dyn_max,16)};
    const int max_scores_len = C* H_dynamic_max[0]*W_dynamic_max[0];

    const std::vector<int64_t> shape_batch_mlvl_num_indexes   = {batch_size, 1, max_filter_num, num_indexes};
    const std::vector<int64_t> shape_batch_mlvl_w_single      = {batch_size, 1, max_filter_num, num_classes};
    const std::vector<int64_t> shape_glb_topk_input               = {batch_size, 1, max_scores_len, 1}; //HARDWARE_FACTOR_TOPK
    const std::vector<int64_t> shape_glb_topk_output_refactor     = {batch_size, 1, HARDWARE_FACTOR_TOPK*max_scores_len, 1}; //TOPK is NMS_PRE
    const std::vector<int64_t> shape_glb_topk_refactor_single     = {1, 1, HARDWARE_FACTOR_TOPK*max_scores_len, 1}; //TOPK is NMS_PRE
    const std::vector<int64_t> shape_glb_buffer_topk_inds             = {1, 1, NMS_PRE, 1};
    const std::vector<int64_t> shape_glb_gather_buffer                = {1, 1, NMS_PRE, num_indexes};
    const std::vector<int64_t> shape_glb_buffer_rpn_bbox_permuted     = {batch_size, 1, max_scores_len, num_indexes};
    const std::vector<int64_t> shape_glb_buffer_nonzero                   = {1, 1, max_filter_num, 1};
    const std::vector<int64_t> shape_result_valid_ind                     = {1, 1, max_filter_num, 1};
    const std::vector<int64_t> shape_result_list               = {batch_size, 1, num_prior_static_stretched, num_classes+num_indexes};
    const std::vector<int64_t> shape_boxes_1b                  = {1,1,num_prior_static_stretched,num_indexes};
    const std::vector<int64_t> shape_single_1b                 = {1,1,num_prior_static_stretched,num_classes};
    const std::vector<int64_t> shape_keep_3nch                 = {1, 1,TOPK_ONNX_NMS*num_classes,3};
    const std::vector<int64_t> shape_keep_u32_1h               = {1, 1,TOPK_ONNX_NMS*num_classes,1};
    const std::vector<int64_t> shape_glb_buffer_nms        =  {1, 1, 1, 4*1024*1024};//fixed
    const std::vector<int64_t> shape_glb_buffer_result_list    = {batch_size, 1, num_prior_static_stretched, num_classes+num_indexes};

    OpBuilder builder(SuperiorOp->getContext());
    builder.setInsertionPoint(SuperiorOp);
    auto type_BatchMlvlScores = RankedTensorType::get(shape_batch_mlvl_w_single, rewriter.getF32Type());
    auto name_BatchMlvlScores = module::getName(SuperiorOp.getOperation()).str()+"_buffer_0";
    auto loc_BatchMlvlScores  = NameLoc::get(builder.getStringAttr(name_BatchMlvlScores));
    auto buf_op_BatchMlvlScores      = builder.create<tpu::BufferOp>(loc_BatchMlvlScores, type_BatchMlvlScores);
    SuperiorOp.setOperand(16, buf_op_BatchMlvlScores);

    auto type_BatchMlvlAnchors = RankedTensorType::get(shape_batch_mlvl_num_indexes, rewriter.getF32Type());
    auto name_BatchMlvlAnchors = module::getName(SuperiorOp.getOperation()).str()+"_buffer_1";
    auto loc_BatchMlvlAnchors  = NameLoc::get(builder.getStringAttr(name_BatchMlvlAnchors));
    auto buf_op_BatchMlvlAnchors      = builder.create<tpu::BufferOp>(loc_BatchMlvlAnchors, type_BatchMlvlAnchors);
    SuperiorOp.setOperand(17, buf_op_BatchMlvlAnchors);

    auto type_BatchMlvlRpnBboxPred = RankedTensorType::get(shape_batch_mlvl_num_indexes, rewriter.getF32Type());
    auto name_BatchMlvlRpnBboxPred = module::getName(SuperiorOp.getOperation()).str()+"_buffer_2";
    auto loc_BatchMlvlRpnBboxPred  = NameLoc::get(builder.getStringAttr(name_BatchMlvlRpnBboxPred));
    auto buf_op_BatchMlvlRpnBboxPred      = builder.create<tpu::BufferOp>(loc_BatchMlvlRpnBboxPred, type_BatchMlvlRpnBboxPred);
    SuperiorOp.setOperand(18, buf_op_BatchMlvlRpnBboxPred);

    auto type_BatchMlvlProposals = RankedTensorType::get(shape_batch_mlvl_num_indexes, rewriter.getF32Type());
    auto name_BatchMlvlProposals = module::getName(SuperiorOp.getOperation()).str()+"_buffer_3";
    auto loc_BatchMlvlProposals  = NameLoc::get(builder.getStringAttr(name_BatchMlvlProposals));
    auto buf_op_BatchMlvlProposals      = builder.create<tpu::BufferOp>(loc_BatchMlvlProposals, type_BatchMlvlProposals);
    SuperiorOp.setOperand(19, buf_op_BatchMlvlProposals);

    auto type_BatchMlvlIds = RankedTensorType::get(shape_batch_mlvl_w_single, rewriter.getI32Type());
    auto name_BatchMlvlIds = module::getName(SuperiorOp.getOperation()).str()+"_buffer_4";
    auto loc_BatchMlvlIds  = NameLoc::get(builder.getStringAttr(name_BatchMlvlIds));
    auto buf_op_BatchMlvlIds      = builder.create<tpu::BufferOp>(loc_BatchMlvlIds, type_BatchMlvlIds);
    SuperiorOp.setOperand(20, buf_op_BatchMlvlIds);

    auto type_GlbBufferTmpScoresStretched = RankedTensorType::get(shape_glb_topk_input, rewriter.getF32Type());
    auto name_GlbBufferTmpScoresStretched = module::getName(SuperiorOp.getOperation()).str()+"_buffer_5";
    auto loc_GlbBufferTmpScoresStretched  = NameLoc::get(builder.getStringAttr(name_GlbBufferTmpScoresStretched));
    auto buf_op_GlbBufferTmpScoresStretched      = builder.create<tpu::BufferOp>(loc_GlbBufferTmpScoresStretched, type_GlbBufferTmpScoresStretched);
    SuperiorOp.setOperand(21, buf_op_GlbBufferTmpScoresStretched);

    auto type_GlbBufferRankedScores = RankedTensorType::get(shape_glb_topk_output_refactor, rewriter.getF32Type());
    auto name_GlbBufferRankedScores = module::getName(SuperiorOp.getOperation()).str()+"_buffer_6";
    auto loc_GlbBufferRankedScores  = NameLoc::get(builder.getStringAttr(name_GlbBufferRankedScores));
    auto buf_op_GlbBufferRankedScores      = builder.create<tpu::BufferOp>(loc_GlbBufferRankedScores, type_GlbBufferRankedScores);
    SuperiorOp.setOperand(22, buf_op_GlbBufferRankedScores);

    auto type_GlbBufferRankIndsInt32 = RankedTensorType::get(shape_glb_topk_output_refactor, rewriter.getI32Type());
    auto name_GlbBufferRankIndsInt32 = module::getName(SuperiorOp.getOperation()).str()+"_buffer_7";
    auto loc_GlbBufferRankIndsInt32  = NameLoc::get(builder.getStringAttr(name_GlbBufferRankIndsInt32));
    auto buf_op_GlbBufferRankIndsInt32      = builder.create<tpu::BufferOp>(loc_GlbBufferRankIndsInt32, type_GlbBufferRankIndsInt32);
    SuperiorOp.setOperand(23, buf_op_GlbBufferRankIndsInt32);

    auto type_GlbBufferRankIndsU32 = RankedTensorType::get(shape_glb_topk_output_refactor, rewriter.getI32Type());
    auto name_GlbBufferRankIndsU32 = module::getName(SuperiorOp.getOperation()).str()+"_buffer_8";
    auto loc_GlbBufferRankIndsU32  = NameLoc::get(builder.getStringAttr(name_GlbBufferRankIndsU32));
    auto buf_op_GlbBufferRankIndsU32      = builder.create<tpu::BufferOp>(loc_GlbBufferRankIndsU32, type_GlbBufferRankIndsU32);
    SuperiorOp.setOperand(24, buf_op_GlbBufferRankIndsU32);

    auto type_GlbTopkInds = RankedTensorType::get(shape_glb_buffer_topk_inds, rewriter.getI32Type());
    auto name_GlbTopkInds = module::getName(SuperiorOp.getOperation()).str()+"_buffer_9";
    auto loc_GlbTopkInds  = NameLoc::get(builder.getStringAttr(name_GlbTopkInds));
    auto buf_op_GlbTopkInds      = builder.create<tpu::BufferOp>(loc_GlbTopkInds, type_GlbTopkInds);
    SuperiorOp.setOperand(25, buf_op_GlbTopkInds);

    auto type_GlbBufferGather_1 = RankedTensorType::get(shape_glb_gather_buffer, rewriter.getF32Type());
    auto name_GlbBufferGather_1 = module::getName(SuperiorOp.getOperation()).str()+"_buffer_10";
    auto loc_GlbBufferGather_1  = NameLoc::get(builder.getStringAttr(name_GlbBufferGather_1));
    auto buf_op_GlbBufferGather_1      = builder.create<tpu::BufferOp>(loc_GlbBufferGather_1, type_GlbBufferGather_1);
    SuperiorOp.setOperand(26, buf_op_GlbBufferGather_1);

    auto type_GlbBufferGather_2 = RankedTensorType::get(shape_glb_gather_buffer, rewriter.getF32Type());
    auto name_GlbBufferGather_2 = module::getName(SuperiorOp.getOperation()).str()+"_buffer_11";
    auto loc_GlbBufferGather_2  = NameLoc::get(builder.getStringAttr(name_GlbBufferGather_2));
    auto buf_op_GlbBufferGather_2      = builder.create<tpu::BufferOp>(loc_GlbBufferGather_2, type_GlbBufferGather_2);
    SuperiorOp.setOperand(27, buf_op_GlbBufferGather_2);

    auto type_GlbBufferRpnBboxPermuted = RankedTensorType::get(shape_glb_buffer_rpn_bbox_permuted, rewriter.getF32Type());
    auto name_GlbBufferRpnBboxPermuted = module::getName(SuperiorOp.getOperation()).str()+"_buffer_12";
    auto loc_GlbBufferRpnBboxPermuted  = NameLoc::get(builder.getStringAttr(name_GlbBufferRpnBboxPermuted));
    auto buf_op_GlbBufferRpnBboxPermuted      = builder.create<tpu::BufferOp>(loc_GlbBufferRpnBboxPermuted, type_GlbBufferRpnBboxPermuted);
    SuperiorOp.setOperand(28, buf_op_GlbBufferRpnBboxPermuted);

    auto type_GlbBufferNonzero = RankedTensorType::get(shape_glb_buffer_nonzero, rewriter.getI32Type());
    auto name_GlbBufferNonzero = module::getName(SuperiorOp.getOperation()).str()+"_buffer_13";
    auto loc_GlbBufferNonzero  = NameLoc::get(builder.getStringAttr(name_GlbBufferNonzero));
    auto buf_op_GlbBufferNonzero      = builder.create<tpu::BufferOp>(loc_GlbBufferNonzero, type_GlbBufferNonzero);
    SuperiorOp.setOperand(29, buf_op_GlbBufferNonzero);

    auto type_ResultValidInd = RankedTensorType::get(shape_result_valid_ind, rewriter.getI32Type());
    auto name_ResultValidInd = module::getName(SuperiorOp.getOperation()).str()+"_buffer_14";
    auto loc_ResultValidInd  = NameLoc::get(builder.getStringAttr(name_ResultValidInd));
    auto buf_op_ResultValidInd      = builder.create<tpu::BufferOp>(loc_ResultValidInd, type_ResultValidInd);
    SuperiorOp.setOperand(30, buf_op_ResultValidInd);

    auto type_GlbBufferGatherBoxes = RankedTensorType::get(shape_boxes_1b, rewriter.getF32Type());
    auto name_GlbBufferGatherBoxes = module::getName(SuperiorOp.getOperation()).str()+"_buffer_15";
    auto loc_GlbBufferGatherBoxes  = NameLoc::get(builder.getStringAttr(name_GlbBufferGatherBoxes));
    auto buf_op_GlbBufferGatherBoxes      = builder.create<tpu::BufferOp>(loc_GlbBufferGatherBoxes, type_GlbBufferGatherBoxes);
    SuperiorOp.setOperand(31, buf_op_GlbBufferGatherBoxes);

    auto type_GlbBufferGatherScores = RankedTensorType::get(shape_single_1b, rewriter.getF32Type());
    auto name_GlbBufferGatherScores = module::getName(SuperiorOp.getOperation()).str()+"_buffer_16";
    auto loc_GlbBufferGatherScores  = NameLoc::get(builder.getStringAttr(name_GlbBufferGatherScores));
    auto buf_op_GlbBufferGatherScores      = builder.create<tpu::BufferOp>(loc_GlbBufferGatherScores, type_GlbBufferGatherScores);
    SuperiorOp.setOperand(32, buf_op_GlbBufferGatherScores);

    auto type_Keep_3nch = RankedTensorType::get(shape_keep_3nch, rewriter.getF32Type());
    auto name_Keep_3nch = module::getName(SuperiorOp.getOperation()).str()+"_buffer_17";
    auto loc_Keep_3nch  = NameLoc::get(builder.getStringAttr(name_Keep_3nch));
    auto buf_op_Keep_3nch      = builder.create<tpu::BufferOp>(loc_Keep_3nch, type_Keep_3nch);
    SuperiorOp.setOperand(33, buf_op_Keep_3nch);

    auto type_KeepU32_1h = RankedTensorType::get(shape_keep_u32_1h, rewriter.getI32Type());
    auto name_KeepU32_1h = module::getName(SuperiorOp.getOperation()).str()+"_buffer_18";
    auto loc_KeepU32_1h  = NameLoc::get(builder.getStringAttr(name_KeepU32_1h));
    auto buf_op_KeepU32_1h      = builder.create<tpu::BufferOp>(loc_KeepU32_1h, type_KeepU32_1h);
    SuperiorOp.setOperand(34, buf_op_KeepU32_1h);

    auto type_GlbBufferBoxes = RankedTensorType::get(shape_boxes_1b, rewriter.getF32Type());
    auto name_GlbBufferBoxes = module::getName(SuperiorOp.getOperation()).str()+"_buffer_19";
    auto loc_GlbBufferBoxes  = NameLoc::get(builder.getStringAttr(name_GlbBufferBoxes));
    auto buf_op_GlbBufferBoxes      = builder.create<tpu::BufferOp>(loc_GlbBufferBoxes, type_GlbBufferBoxes);
    SuperiorOp.setOperand(35, buf_op_GlbBufferBoxes);

    auto type_GlbBufferScores = RankedTensorType::get(shape_single_1b, rewriter.getF32Type());
    auto name_GlbBufferScores = module::getName(SuperiorOp.getOperation()).str()+"_buffer_20";
    auto loc_GlbBufferScores  = NameLoc::get(builder.getStringAttr(name_GlbBufferScores));
    auto buf_op_GlbBufferScores      = builder.create<tpu::BufferOp>(loc_GlbBufferScores, type_GlbBufferScores);
    SuperiorOp.setOperand(36, buf_op_GlbBufferScores);

    auto type_GlbBufferNms = RankedTensorType::get(shape_glb_buffer_nms, rewriter.getF32Type());
    auto name_GlbBufferNms = module::getName(SuperiorOp.getOperation()).str()+"_buffer_21";
    auto loc_GlbBufferNms  = NameLoc::get(builder.getStringAttr(name_GlbBufferNms));
    auto buf_op_GlbBufferNms      = builder.create<tpu::BufferOp>(loc_GlbBufferNms, type_GlbBufferNms);
    SuperiorOp.setOperand(37, buf_op_GlbBufferNms);

    auto type_GatherMlvlProposals = RankedTensorType::get(shape_batch_mlvl_num_indexes, rewriter.getF32Type());
    auto name_GatherMlvlProposals = module::getName(SuperiorOp.getOperation()).str()+"_buffer_22";
    auto loc_GatherMlvlProposals  = NameLoc::get(builder.getStringAttr(name_GatherMlvlProposals));
    auto buf_op_GatherMlvlProposals      = builder.create<tpu::BufferOp>(loc_GatherMlvlProposals, type_GatherMlvlProposals);
    SuperiorOp.setOperand(38, buf_op_GatherMlvlProposals);

    auto type_GatherMlvlScores = RankedTensorType::get(shape_batch_mlvl_w_single, rewriter.getF32Type());
    auto name_GatherMlvlScores = module::getName(SuperiorOp.getOperation()).str()+"_buffer_23";
    auto loc_GatherMlvlScores  = NameLoc::get(builder.getStringAttr(name_GatherMlvlScores));
    auto buf_op_GatherMlvlScores      = builder.create<tpu::BufferOp>(loc_GatherMlvlScores, type_GatherMlvlScores);
    SuperiorOp.setOperand(39, buf_op_GatherMlvlScores);

    auto type_GatherMlvlIds = RankedTensorType::get(shape_batch_mlvl_w_single, rewriter.getI32Type());
    auto name_GatherMlvlIds = module::getName(SuperiorOp.getOperation()).str()+"_buffer_24";
    auto loc_GatherMlvlIds  = NameLoc::get(builder.getStringAttr(name_GatherMlvlIds));
    auto buf_op_GatherMlvlIds      = builder.create<tpu::BufferOp>(loc_GatherMlvlIds, type_GatherMlvlIds);
    SuperiorOp.setOperand(40, buf_op_GatherMlvlIds);

    auto type_GlbBufferResultList = RankedTensorType::get(shape_glb_buffer_result_list, rewriter.getF32Type());
    auto name_GlbBufferResultList = module::getName(SuperiorOp.getOperation()).str()+"_buffer_25";
    auto loc_GlbBufferResultList  = NameLoc::get(builder.getStringAttr(name_GlbBufferResultList));
    auto buf_op_GlbBufferResultList      = builder.create<tpu::BufferOp>(loc_GlbBufferResultList, type_GlbBufferResultList);
    SuperiorOp.setOperand(41, buf_op_GlbBufferResultList);

    return success();
  }
};

class MaskRCNNBboxPoolerGlobalBuffer : public OpRewritePattern<tpu::MaskRCNNBboxPoolerOp> {
public:
  using OpRewritePattern<tpu::MaskRCNNBboxPoolerOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::MaskRCNNBboxPoolerOp SuperiorOp,
                                PatternRewriter &rewriter) const override {
    const int batch_size     = module::getShape(SuperiorOp.getPtrFeat0())[0];
    const int C = SuperiorOp.getCHANNEL_ROI();
    const int roi_slice =  SuperiorOp.getROI_SLICE();
    const int roi_len   = SuperiorOp.getROI_LEN();
    const int roi_num = roi_slice*batch_size;
    const int PH = SuperiorOp.getROI_PH();
    const int PW = SuperiorOp.getROI_PW();
    const std::vector<int64_t> res_shape = {roi_num, C, PH, PW};
    const std::vector<int64_t> rois_slice_shape = {batch_size, roi_slice, 1, 1};
  //[Error] N should be GLOBAL_BATCH_SIZE, but now must compaitble with gsl_roi_pooler.pl
    const std::vector<int64_t> rois_shape3 = {batch_size, roi_slice, 1, roi_len};
    const std::vector<int64_t> bias_shape = {1, C, 1, 1};

    OpBuilder builder(SuperiorOp->getContext());
    builder.setInsertionPoint(SuperiorOp);
    auto type_PtrTmpRes = RankedTensorType::get(res_shape, rewriter.getF32Type());
    auto name_PtrTmpRes = module::getName(SuperiorOp.getOperation()).str()+"_buffer_0";
    auto loc_PtrTmpRes  = NameLoc::get(builder.getStringAttr(name_PtrTmpRes));
    auto buf_op_PtrTmpRes      = builder.create<tpu::BufferOp>(loc_PtrTmpRes, type_PtrTmpRes);
    SuperiorOp.setOperand(5, buf_op_PtrTmpRes);

    auto type_PtrRoisTmp = RankedTensorType::get(rois_slice_shape, rewriter.getF32Type());
    auto name_PtrRoisTmp = module::getName(SuperiorOp.getOperation()).str()+"_buffer_1";
    auto loc_PtrRoisTmp  = NameLoc::get(builder.getStringAttr(name_PtrRoisTmp));
    auto buf_op_PtrRoisTmp      = builder.create<tpu::BufferOp>(loc_PtrRoisTmp, type_PtrRoisTmp);
    SuperiorOp.setOperand(6, buf_op_PtrRoisTmp);

    return success();
  }
};

class MaskRCNNGetBboxBGlobalBuffer : public OpRewritePattern<tpu::MaskRCNNGetBboxBOp> {
public:
  using OpRewritePattern<tpu::MaskRCNNGetBboxBOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::MaskRCNNGetBboxBOp SuperiorOp,
                                PatternRewriter &rewriter) const override {

    std::vector<int64_t> input_shape = module::getShape(SuperiorOp.getPtrRois());
    assert (input_shape.size()==4);
    const int num_indexes = SuperiorOp.getNUM_INDEXES();
    const int num_classes = SuperiorOp.getNUM_CLASSES();
    const int roi_len = num_classes + num_indexes;
    const int batch_size = input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3]/roi_len/SuperiorOp.getMAX_PER_IMG();
    const int NUM_CLASSES_GetBboxB =  SuperiorOp.getNUM_CLASSESGetBboxB();
    const int MAX_PROPOSALS_PER_IMG = SuperiorOp.getMAX_PER_IMG();
    const int num_dets_w = num_classes + num_indexes;
    const int MAX_NMS_LENGTH_GetBboxB = MAX_PROPOSALS_PER_IMG * NUM_CLASSES_GetBboxB ;
    const int TOPK_ONNX_NMS_2nd =  SuperiorOp.getTOPK_ONNX_NMS();
    const int c = batch_size *MAX_PROPOSALS_PER_IMG ;
    const int w2 = NUM_CLASSES_GetBboxB * num_indexes;
    const int w3 = NUM_CLASSES_GetBboxB + num_classes;

    const std::vector<int64_t> shape2 = {1, c, 1, w2};
    const std::vector<int64_t> shape3 = {1, c, 1, w3};
    const std::vector<int64_t> shape6 = {1, 1, MAX_NMS_LENGTH_GetBboxB, num_dets_w};

    const std::vector<int64_t> shape7 = {1, 1, MAX_NMS_LENGTH_GetBboxB, num_classes};
    const std::vector<int64_t> shape8 = {1, 1, MAX_NMS_LENGTH_GetBboxB, num_indexes};
    const std::vector<int64_t> shape9 = {1, 1, TOPK_ONNX_NMS_2nd, 1};
    const std::vector<int64_t> shape10 = {1, 1, TOPK_ONNX_NMS_2nd, 3};
    const std::vector<int64_t> shape_glb_buffer_nms = {1, 1, 1, 4*1024*1024};
    const std::vector<int64_t> shape_attr = {1,1,1,4};
    const std::vector<int64_t> shape_mv = {1, 1, 1, 3};

    OpBuilder builder(SuperiorOp->getContext());
    builder.setInsertionPoint(SuperiorOp);
    auto type_Means = RankedTensorType::get(shape2, rewriter.getF32Type());
    auto name_Means = module::getName(SuperiorOp.getOperation()).str()+"_buffer_0";
    auto loc_Means  = NameLoc::get(builder.getStringAttr(name_Means));
    auto buf_op_Means      = builder.create<tpu::BufferOp>(loc_Means, type_Means);
    SuperiorOp.setOperand(5, buf_op_Means);

    auto type_Stds = RankedTensorType::get(shape2, rewriter.getF32Type());
    auto name_Stds = module::getName(SuperiorOp.getOperation()).str()+"_buffer_1";
    auto loc_Stds  = NameLoc::get(builder.getStringAttr(name_Stds));
    auto buf_op_Stds      = builder.create<tpu::BufferOp>(loc_Stds, type_Stds);
    SuperiorOp.setOperand(6, buf_op_Stds);

    auto type_ResBbox = RankedTensorType::get(shape2, rewriter.getF32Type());
    auto name_ResBbox = module::getName(SuperiorOp.getOperation()).str()+"_buffer_2";
    auto loc_ResBbox  = NameLoc::get(builder.getStringAttr(name_ResBbox));
    auto buf_op_ResBbox      = builder.create<tpu::BufferOp>(loc_ResBbox, type_ResBbox);
    SuperiorOp.setOperand(7, buf_op_ResBbox);

    auto type_ResBbox1 = RankedTensorType::get(shape8, rewriter.getF32Type());
    auto name_ResBbox1 = module::getName(SuperiorOp.getOperation()).str()+"_buffer_3";
    auto loc_ResBbox1  = NameLoc::get(builder.getStringAttr(name_ResBbox1));
    auto buf_op_ResBbox1      = builder.create<tpu::BufferOp>(loc_ResBbox1, type_ResBbox1);
    SuperiorOp.setOperand(8, buf_op_ResBbox1);

    auto type_ResBbox0 = RankedTensorType::get(shape2, rewriter.getF32Type());
    auto name_ResBbox0 = module::getName(SuperiorOp.getOperation()).str()+"_buffer_4";
    auto loc_ResBbox0  = NameLoc::get(builder.getStringAttr(name_ResBbox0));
    auto buf_op_ResBbox0      = builder.create<tpu::BufferOp>(loc_ResBbox0, type_ResBbox0);
    SuperiorOp.setOperand(9, buf_op_ResBbox0);

    auto type_ResScore0 = RankedTensorType::get(shape3, rewriter.getF32Type());
    auto name_ResScore0 = module::getName(SuperiorOp.getOperation()).str()+"_buffer_5";
    auto loc_ResScore0  = NameLoc::get(builder.getStringAttr(name_ResScore0));
    auto buf_op_ResScore0      = builder.create<tpu::BufferOp>(loc_ResScore0, type_ResScore0);
    SuperiorOp.setOperand(10, buf_op_ResScore0);

    auto type_ResScore1 = RankedTensorType::get(shape3, rewriter.getF32Type());
    auto name_ResScore1 = module::getName(SuperiorOp.getOperation()).str()+"_buffer_6";
    auto loc_ResScore1  = NameLoc::get(builder.getStringAttr(name_ResScore1));
    auto buf_op_ResScore1      = builder.create<tpu::BufferOp>(loc_ResScore1, type_ResScore1);
    SuperiorOp.setOperand(11, buf_op_ResScore1);

    auto type_ResScore2 = RankedTensorType::get(shape3, rewriter.getF32Type());
    auto name_ResScore2 = module::getName(SuperiorOp.getOperation()).str()+"_buffer_7";
    auto loc_ResScore2  = NameLoc::get(builder.getStringAttr(name_ResScore2));
    auto buf_op_ResScore2      = builder.create<tpu::BufferOp>(loc_ResScore2, type_ResScore2);
    SuperiorOp.setOperand(12, buf_op_ResScore2);

    auto type_ResScore3 = RankedTensorType::get(shape7, rewriter.getF32Type());
    auto name_ResScore3 = module::getName(SuperiorOp.getOperation()).str()+"_buffer_8";
    auto loc_ResScore3  = NameLoc::get(builder.getStringAttr(name_ResScore3));
    auto buf_op_ResScore3      = builder.create<tpu::BufferOp>(loc_ResScore3, type_ResScore3);
    SuperiorOp.setOperand(13, buf_op_ResScore3);

    auto type_ResLabel2 = RankedTensorType::get(shape7, rewriter.getI32Type());
    auto name_ResLabel2 = module::getName(SuperiorOp.getOperation()).str()+"_buffer_9";
    auto loc_ResLabel2  = NameLoc::get(builder.getStringAttr(name_ResLabel2));
    auto buf_op_ResLabel2      = builder.create<tpu::BufferOp>(loc_ResLabel2, type_ResLabel2);
    SuperiorOp.setOperand(14, buf_op_ResLabel2);

    auto type_ResultList = RankedTensorType::get(shape6, rewriter.getF32Type());
    auto name_ResultList = module::getName(SuperiorOp.getOperation()).str()+"_buffer_10";
    auto loc_ResultList  = NameLoc::get(builder.getStringAttr(name_ResultList));
    auto buf_op_ResultList      = builder.create<tpu::BufferOp>(loc_ResultList, type_ResultList);
    SuperiorOp.setOperand(15, buf_op_ResultList);

    auto type_Keep_3nch = RankedTensorType::get(shape10, rewriter.getF32Type());
    auto name_Keep_3nch = module::getName(SuperiorOp.getOperation()).str()+"_buffer_11";
    auto loc_Keep_3nch  = NameLoc::get(builder.getStringAttr(name_Keep_3nch));
    auto buf_op_Keep_3nch      = builder.create<tpu::BufferOp>(loc_Keep_3nch, type_Keep_3nch);
    SuperiorOp.setOperand(16, buf_op_Keep_3nch);

    auto type_KeepU32_1h = RankedTensorType::get(shape9, rewriter.getI32Type());
    auto name_KeepU32_1h = module::getName(SuperiorOp.getOperation()).str()+"_buffer_12";
    auto loc_KeepU32_1h  = NameLoc::get(builder.getStringAttr(name_KeepU32_1h));
    auto buf_op_KeepU32_1h      = builder.create<tpu::BufferOp>(loc_KeepU32_1h, type_KeepU32_1h);
    SuperiorOp.setOperand(17, buf_op_KeepU32_1h);

    auto type_GlbBufferBoxes = RankedTensorType::get(shape8, rewriter.getF32Type());
    auto name_GlbBufferBoxes = module::getName(SuperiorOp.getOperation()).str()+"_buffer_13";
    auto loc_GlbBufferBoxes  = NameLoc::get(builder.getStringAttr(name_GlbBufferBoxes));
    auto buf_op_GlbBufferBoxes      = builder.create<tpu::BufferOp>(loc_GlbBufferBoxes, type_GlbBufferBoxes);
    SuperiorOp.setOperand(18, buf_op_GlbBufferBoxes);

    auto type_GlbBufferScores = RankedTensorType::get(shape7, rewriter.getF32Type());
    auto name_GlbBufferScores = module::getName(SuperiorOp.getOperation()).str()+"_buffer_14";
    auto loc_GlbBufferScores  = NameLoc::get(builder.getStringAttr(name_GlbBufferScores));
    auto buf_op_GlbBufferScores      = builder.create<tpu::BufferOp>(loc_GlbBufferScores, type_GlbBufferScores);
    SuperiorOp.setOperand(19, buf_op_GlbBufferScores);

    auto type_GlbBufferNms = RankedTensorType::get(shape_glb_buffer_nms, rewriter.getF32Type());
    auto name_GlbBufferNms = module::getName(SuperiorOp.getOperation()).str()+"_buffer_15";
    auto loc_GlbBufferNms  = NameLoc::get(builder.getStringAttr(name_GlbBufferNms));
    auto buf_op_GlbBufferNms      = builder.create<tpu::BufferOp>(loc_GlbBufferNms, type_GlbBufferNms);
    SuperiorOp.setOperand(20, buf_op_GlbBufferNms);

    auto type_GlbBufferNonzero = RankedTensorType::get(shape3, rewriter.getI32Type());
    auto name_GlbBufferNonzero = module::getName(SuperiorOp.getOperation()).str()+"_buffer_16";
    auto loc_GlbBufferNonzero  = NameLoc::get(builder.getStringAttr(name_GlbBufferNonzero));
    auto buf_op_GlbBufferNonzero      = builder.create<tpu::BufferOp>(loc_GlbBufferNonzero, type_GlbBufferNonzero);
    SuperiorOp.setOperand(21, buf_op_GlbBufferNonzero);

    auto type_ResultValidInd = RankedTensorType::get(shape3, rewriter.getI32Type());
    auto name_ResultValidInd = module::getName(SuperiorOp.getOperation()).str()+"_buffer_17";
    auto loc_ResultValidInd  = NameLoc::get(builder.getStringAttr(name_ResultValidInd));
    auto buf_op_ResultValidInd      = builder.create<tpu::BufferOp>(loc_ResultValidInd, type_ResultValidInd);
    SuperiorOp.setOperand(22, buf_op_ResultValidInd);

    auto type_GlbLables = RankedTensorType::get(shape3, rewriter.getI32Type());
    auto name_GlbLables = module::getName(SuperiorOp.getOperation()).str()+"_buffer_18";
    auto loc_GlbLables  = NameLoc::get(builder.getStringAttr(name_GlbLables));
    auto buf_op_GlbLables      = builder.create<tpu::BufferOp>(loc_GlbLables, type_GlbLables);
    SuperiorOp.setOperand(23, buf_op_GlbLables);

    auto type_GlbLablesExpand = RankedTensorType::get(shape3, rewriter.getI32Type());
    auto name_GlbLablesExpand = module::getName(SuperiorOp.getOperation()).str()+"_buffer_19";
    auto loc_GlbLablesExpand  = NameLoc::get(builder.getStringAttr(name_GlbLablesExpand));
    auto buf_op_GlbLablesExpand      = builder.create<tpu::BufferOp>(loc_GlbLablesExpand, type_GlbLablesExpand);
    SuperiorOp.setOperand(24, buf_op_GlbLablesExpand);
   return success();
  }
};

class MaskRCNNMaskPoolerGlobalBuffer : public OpRewritePattern<tpu::MaskRCNNMaskPoolerOp> {
public:
  using OpRewritePattern<tpu::MaskRCNNMaskPoolerOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::MaskRCNNMaskPoolerOp SuperiorOp,
                                PatternRewriter &rewriter) const override {
    const int batch_size     = module::getShape(SuperiorOp.getX_0())[0];
    const int C = SuperiorOp.getCHANNEL_ROI();
    const int roi_slice =  SuperiorOp.getROI_SLICE();
    const int roi_len   = SuperiorOp.getROI_LEN();
    const int roi_num   = roi_slice*batch_size;
    const int PH = SuperiorOp.getROI_PH();
    const int PW = SuperiorOp.getROI_PW();
    const std::vector<int64_t> shape_mask_rois  = {batch_size, 1, roi_slice, roi_len};
    const std::vector<int64_t> shape_det_bboxes = {batch_size, 1, roi_slice ,roi_len};
    const std::vector<int64_t> shape_det_labels = {batch_size, 1, roi_slice, 1};

    const std::vector<int64_t> res_shape = {roi_num, C, PH, PW};
    const std::vector<int64_t> rois_slice_shape = {batch_size, roi_slice, 1, 1};
  //[Error] N should be GLOBAL_BATCH_SIZE, but now must compaitble with gsl_roi_pooler.pl
    const std::vector<int64_t> rois_shape3 = {batch_size, roi_slice, 1, roi_len};
    const std::vector<int64_t> bias_shape = {1, C, 1, 1};

    OpBuilder builder(SuperiorOp->getContext());
    builder.setInsertionPoint(SuperiorOp);
    auto type_PtrRoisBuff = RankedTensorType::get(rois_shape3, rewriter.getF32Type());
    auto name_PtrRoisBuff = module::getName(SuperiorOp.getOperation()).str()+"_buffer_0";
    auto loc_PtrRoisBuff  = NameLoc::get(builder.getStringAttr(name_PtrRoisBuff));
    auto buf_op_PtrRoisBuff      = builder.create<tpu::BufferOp>(loc_PtrRoisBuff, type_PtrRoisBuff);
    SuperiorOp.setOperand(7, buf_op_PtrRoisBuff);

    auto type_ResultFilledDetBboxes = RankedTensorType::get(shape_det_bboxes, rewriter.getF32Type());
    auto name_ResultFilledDetBboxes = module::getName(SuperiorOp.getOperation()).str()+"_buffer_1";
    auto loc_ResultFilledDetBboxes  = NameLoc::get(builder.getStringAttr(name_ResultFilledDetBboxes));
    auto buf_op_ResultFilledDetBboxes      = builder.create<tpu::BufferOp>(loc_ResultFilledDetBboxes, type_ResultFilledDetBboxes);
    SuperiorOp.setOperand(8, buf_op_ResultFilledDetBboxes);

    auto type_ResultFilledDetLabels = RankedTensorType::get(shape_det_labels, rewriter.getF32Type());
    auto name_ResultFilledDetLabels = module::getName(SuperiorOp.getOperation()).str()+"_buffer_2";
    auto loc_ResultFilledDetLabels  = NameLoc::get(builder.getStringAttr(name_ResultFilledDetLabels));
    auto buf_op_ResultFilledDetLabels      = builder.create<tpu::BufferOp>(loc_ResultFilledDetLabels, type_ResultFilledDetLabels);
    SuperiorOp.setOperand(9, buf_op_ResultFilledDetLabels);

    auto type_PtrTmpRes = RankedTensorType::get(res_shape, rewriter.getF32Type());
    auto name_PtrTmpRes = module::getName(SuperiorOp.getOperation()).str()+"_buffer_3";
    auto loc_PtrTmpRes  = NameLoc::get(builder.getStringAttr(name_PtrTmpRes));
    auto buf_op_PtrTmpRes      = builder.create<tpu::BufferOp>(loc_PtrTmpRes, type_PtrTmpRes);
    SuperiorOp.setOperand(10, buf_op_PtrTmpRes);

    auto type_PtrRoisTmp = RankedTensorType::get(rois_slice_shape, rewriter.getF32Type());
    auto name_PtrRoisTmp = module::getName(SuperiorOp.getOperation()).str()+"_buffer_4";
    auto loc_PtrRoisTmp  = NameLoc::get(builder.getStringAttr(name_PtrRoisTmp));
    auto buf_op_PtrRoisTmp      = builder.create<tpu::BufferOp>(loc_PtrRoisTmp, type_PtrRoisTmp);
    SuperiorOp.setOperand(11, buf_op_PtrRoisTmp);

    return success();
  }
};

namespace tpu {
using namespace bm168x;
void populateGlobalBufferBM168xPatterns(RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
      GatherGlobalBuffer,
      GatherElementsGlobalBuffer,
      GRUGlobalBuffer,
      LSTMGlobalBuffer,
      ReduceGlobalBuffer,
      SliceGlobalBuffer,
      ReshapeGlobalBuffer,
      SoftmaxGlobalBuffer,
      PermuteGlobalBuffer,
      InterpGlobalBuffer,
      GridSamplerBuffer,
      Pool3DGlobalBuffer,
      NonZeroGlobalBuffer,
      DeformGatherGlobalBuffer,
      TileGlobalBuffer,
      IndexPutGlobalBuffer,
      PadGlobalBuffer,
      ScatterElementsGlobalBuffer,
      Space2BatchGlobalBuffer,
      Batch2SpaceGlobalBuffer,
      ScatterNDGlobalBuffer,
      NmsGlobalBuffer,
      YoloDetectionGlobalBuffer,
      DetectionOutputGlobalBuffer,
      TopKGlobalBuffer,
      SortGlobalBuffer,
      CustomGlobalBuffer,
      WhereGlobalBuffer,
      MatMulGlobalBuffer,
      ConvbwdGlobalBuffer,
      MaskRCNNRPNGetBboxesGlobalBuffer,
      MaskRCNNBboxPoolerGlobalBuffer,
      MaskRCNNGetBboxBGlobalBuffer,
      MaskRCNNMaskPoolerGlobalBuffer
  >(patterns->getContext());
  // clang-format on
}

} // namespace tpu
} // namespace tpu_mlir
