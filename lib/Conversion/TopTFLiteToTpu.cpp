//===- TopTFLiteToTpu.cpp - Lower TOP TFLite constructs to TPU -------===//
//
// Part of the Sophgo Project, under the Apache License v2.0 with LLVM
// Exceptions.
//
//===----------------------------------------------------------------------===//

#include "sophgo/Conversion/TopTFLiteToTpu.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/Helper/Module.h"
#include "sophgo/Support/Helper/Quant.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Alignment.h"

using namespace mlir;
using namespace llvm;
using namespace mlir::quant;
namespace sophgo {
#define GEN_PASS_CLASSES
#include "sophgo/Conversion/Passes.h.inc"

Type getRankedTensorElementType(Value input) {
  return input.getType().cast<RankedTensorType>().getElementType();
}

template <typename ElementType>
ElementType getRankedTensorElementType(Value input) {
  return getRankedTensorElementType(input).cast<ElementType>();
}

template <class T>
bool isQuantized(T input) {
  if (!input.getType().template isa<RankedTensorType>())
    return false;
  if (!getRankedTensorElementType(input).template isa<quant::QuantizedType>()) {
    return false;
  }
  return true;
};

template <typename T, typename... Args>
bool isQuantized(T t, Args... args) {
  return isQuantized(t) && isQuantized(args...);
};

template <typename QuantizedType, typename T>
bool isQuantized(T input) {
  if (!input.getType().template isa<RankedTensorType>())
    return false;
  if (!getRankedTensorElementType(input).template isa<QuantizedType>()) {
    return false;
  }
  return true;
};

template <typename QuantizedType, typename T, typename... Args>
bool isQuantized(T t, Args... args) {
  return isQuantized<QuantizedType>(t) && isQuantized<QuantizedType>(args...);
};

// tensorflow/lite/kernels/internal/quantization_util.cc
// mlir/lib/Dialect/Tosa/Utils/QuantUtils.cpp
void QuantizeMultiplier(double double_multiplier, int64_t *quantized_multiplier,
                        int64_t *shift) {
  if (double_multiplier == 0.) {
    *quantized_multiplier = 0;
    *shift = 0;
    return;
  }
  int shift_tmp;
  const double q = std::frexp(double_multiplier, &shift_tmp);
  *shift = shift_tmp;
  auto q_fixed = static_cast<int64_t>(std::round(q * (1LL << 31)));
  assert(q_fixed <= (1LL << 31));
  if (q_fixed == (1LL << 31)) {
    q_fixed /= 2;
    ++*shift;
  }
  assert(q_fixed <= std::numeric_limits<int32_t>::max());
  // A shift amount smaller than -31 would cause all bits to be shifted out
  // and thus all results would be zero. We implement that instead with
  // q_fixed==0, so as to avoid hitting issues with right-shift
  // operations with shift amounts greater than 31. Note that this happens
  // roughly when abs(double_multiplier) < 2^-31 and the present handling means
  // that we're effectively flushing tiny double_multiplier's to zero.
  // We could conceivably handle values in the range (roughly) [32, 63]
  // as 'denormals' i.e. (shift==0, q_fixed < 2^30). In that point of view
  // the present handling is just doing 'flush denormals to zero'. We could
  // reconsider and actually generate nonzero denormals if a need arises.
  if (*shift < -31) {
    *shift = 0;
    q_fixed = 0;
  }
  // Single-rounding MultiplyByQuantizedMultiplier doesn't support a shift > 30,
  // saturate it.
  if (*shift > 30) {
    *shift = 30;
    q_fixed = (1LL << 31) - 1;
  }
  // Sophgo expects right shift to be positive, and embed (1 << 31) into right
  // shift bits.
  *shift = (-*shift) + 31;
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

//===----------------------------------------------------------------------===//
// TopTFLiteToTpuLoweringPass RewritePatterns: convolution operations
//===----------------------------------------------------------------------===//
class ConvOpLovering : public OpConversionPattern<top::ConvOp> {
public:
  using OpConversionPattern<top::ConvOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(top::ConvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isQuantized(op.input(), op.filter(), op.getResult()))
      return failure();
    if (isQuantized(adaptor.filter()) || isQuantized(adaptor.bias()))
      return failure(); // make sure filter and bias are storage type

    auto input_elem_type = getRankedTensorElementType(op.input());
    auto filter_elm_type = getRankedTensorElementType(op.filter());
    auto output_elm_type = getRankedTensorElementType(op.getResult());

    if (!input_elem_type.isa<UniformQuantizedType>() ||
        !output_elm_type.isa<UniformQuantizedType>())
      return failure();
    bool is_per_channel = filter_elm_type.isa<UniformQuantizedPerAxisType>();

    auto input_qtype = input_elem_type.cast<UniformQuantizedType>();
    auto output_qtype = output_elm_type.cast<UniformQuantizedType>();
    ArrayRef<double> filter_scales;

    if (is_per_channel) {
      filter_scales =
          filter_elm_type.cast<UniformQuantizedPerAxisType>().getScales();
    } else {
      filter_scales = ArrayRef<double>{
          filter_elm_type.cast<UniformQuantizedType>().getScale()};
    }

    SmallVector<int64_t> rshift(filter_scales.size());
    SmallVector<int64_t> multiplier(filter_scales.size());

    // tensorflow/lite/kernels/kernel_util.cc::PopulateConvolutionQuantizationParams
    // Populate multiplier and shift using affine quantization.
    const double input_scale = input_qtype.getScale();
    const double output_scale = output_qtype.getScale();
    for (auto filter : llvm::enumerate(filter_scales)) {
      const double effective_output_scale =
          input_scale * filter.value() / output_scale;
      QuantizeMultiplier(effective_output_scale, &multiplier[filter.index()],
                         &rshift[filter.index()]);
    }

    NamedAttrList attributes(adaptor.getAttributes());
    attributes.set("rshift", rewriter.getI64ArrayAttr(rshift));
    attributes.set("multiplier", rewriter.getI64ArrayAttr(multiplier));
    if (isQuantized(op.bias()))
      attributes.set("with_bias", rewriter.getBoolAttr(true));
    else
      attributes.set("with_bias", rewriter.getBoolAttr(false));

    int32_t input_zeroPoint = input_qtype.getZeroPoint();
    if (input_zeroPoint != 0 &&
        isa<top::WeightOp>(adaptor.filter().getDefiningOp()) &&
        isa<top::WeightOp, top::NoneOp>(adaptor.bias().getDefiningOp())) {
      // merge input_zeroPoint to bias
      std::shared_ptr<std::vector<int32_t>> bias_quant;
      std::shared_ptr<std::vector<int8_t>> filter_quant;
      filter_quant =
          cast<top::WeightOp>(adaptor.filter().getDefiningOp()).read<int8_t>();
      if (isa<top::WeightOp>(adaptor.bias().getDefiningOp())) {
        bias_quant =
            cast<top::WeightOp>(adaptor.bias().getDefiningOp()).read<int32_t>();
      }
      auto filter_type = adaptor.filter().getType().cast<RankedTensorType>();
      int64_t oc = filter_type.getShape()[0];
      int64_t kernel_size = filter_type.getNumElements() / oc;
      bias_quant->resize(oc, 0);
      for (size_t oc_ind = 0; oc_ind < oc; ++oc_ind) {
        for (size_t kernel_ind = 0; kernel_ind < kernel_size; ++kernel_ind) {
          bias_quant->data()[oc_ind] -=
              input_zeroPoint *
              filter_quant->at(kernel_ind + oc_ind * kernel_size);
        }
      }
      auto bias_type = RankedTensorType::get({oc}, rewriter.getI32Type());
      auto new_bias =
          top::WeightOp::create(adaptor.bias().getDefiningOp(),
                                "MergedInputZeroPoint", *bias_quant, bias_type);
      rewriter.replaceOpWithNewOp<tpu::ConvOp>(
          op, op->getResultTypes(),
          ValueRange({adaptor.input(), adaptor.filter(), new_bias}),
          attributes);

      return success();
    }

    rewriter.replaceOpWithNewOp<tpu::ConvOp>(op, op->getResultTypes(),
                                             adaptor.getOperands(), attributes);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TopTFLiteToTpuLoweringPass RewritePatterns: add operations
//===----------------------------------------------------------------------===//
// Reference TensorFlow Lite:
// gemmlowp/fixedpoint/fixedpoint.h::339
// tensorflow/lite/kernels/internal/reference/integer_ops/add.h::86
// tensorflow/compiler/mlir/tosa/transforms/legalize_tfl.cc::460
class AddOpLowering : public OpConversionPattern<top::AddOp> {
  using OpConversionPattern<top::AddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(top::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!llvm::all_of(op.inputs(), [](Value in) {
          return isQuantized<quant::UniformQuantizedType>(in);
        }))
      return failure();
    if (!isQuantized<quant::UniformQuantizedType>(op.getResult()))
      return failure();

    SmallVector<double> inputs_scale(op.inputs().size());
    for (auto item : llvm::enumerate(op.inputs())) {
      inputs_scale[item.index()] =
          getRankedTensorElementType<quant::UniformQuantizedType>(item.value())
              .getScale();
    }
    auto output_qtype =
        getRankedTensorElementType<quant::UniformQuantizedType>(op.getResult());

    // Following quantization described in tensorflow/lite/kernels/add.cc::172
    // In details it does (the size of inputs is 2):
    // 1. Rescale inputs to scale = 2.0 x max(lhs.scale, rhs.scale)
    // 2. Extra left shift to input to increase precision
    // Where input_shift = 20 if input is 8-bit
    // input_shift = 15 if input is 16-bit
    double output_scale = output_qtype.getScale();

    // tpu.Add supports the size of operands  more than 2. When more numbers
    // added together, we should consider the extra overflow.
    double max_scale_nx =
        *std::max_element(inputs_scale.begin(), inputs_scale.end()) *
        op.inputs().size();

    const int32_t SHIFT_8_BIT = 20;
    const int32_t SHIFT_16_BIT = 15;

    int32_t input_shift = (output_qtype.getStorageTypeIntegralWidth() == 16)
                              ? SHIFT_16_BIT
                              : SHIFT_8_BIT;

    llvm::for_each(inputs_scale, [&](double &scale) {
      scale = static_cast<double>(1 << input_shift) * scale / max_scale_nx;
    });
    double output_rescale_scale =
        max_scale_nx / (output_scale * static_cast<double>(1 << input_shift));

    SmallVector<double> scales(inputs_scale);
    scales.push_back(output_rescale_scale);
    SmallVector<int64_t> multiplier(scales.size());
    SmallVector<int64_t> shift(scales.size());
    for (auto scale : llvm::enumerate(scales)) {
      QuantizeMultiplier(scale.value(), &multiplier[scale.index()],
                         &shift[scale.index()]);
    }
    NamedAttrList attributes(adaptor.getAttributes());
    attributes.set("rshifts", rewriter.getI64ArrayAttr(shift));
    attributes.set("multipliers", rewriter.getI64ArrayAttr(multiplier));
    rewriter.replaceOpWithNewOp<tpu::AddOp>(op, op->getResultTypes(),
                                            adaptor.getOperands(), attributes);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TopTFLiteToTpuLoweringPass RewritePatterns: average_pooling operations
//===----------------------------------------------------------------------===//
// tensorflow/lite/kernels/reduce.cc::376
// tensorflow/lite/kernels/internal/reference/integer_ops/mean.h::24
class AvgPoolOpLowering : public OpConversionPattern<top::AvgPoolOp> {
  using OpConversionPattern<top::AvgPoolOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(top::AvgPoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isQuantized<quant::UniformQuantizedType>(op.input(), op.getResult()))
      return failure();

    auto input_qtype =
        getRankedTensorElementType<quant::UniformQuantizedType>(op.input());
    auto output_qtype =
        getRankedTensorElementType<quant::UniformQuantizedType>(op.getResult());

    int64_t num_elements = 1;
    for (auto x : adaptor.kernel_shape())
      num_elements *= x.cast<IntegerAttr>().getInt();

    const double real_multiplier = 1.0 / static_cast<double>(num_elements) *
                                   input_qtype.getScale() /
                                   output_qtype.getScale();
    int64_t multiplier, shift;
    QuantizeMultiplier(real_multiplier, &multiplier, &shift);
    NamedAttrList attributes(adaptor.getAttributes());
    attributes.set("rshift", rewriter.getI64IntegerAttr(shift));
    attributes.set("multiplier", rewriter.getI64IntegerAttr(multiplier));
    auto avg_in_type = adaptor.input().getType().cast<RankedTensorType>();
    auto avg_out_type = op.getResult().getType().cast<RankedTensorType>();

    if (avg_in_type.getShape().size() == avg_out_type.getShape().size()) {
      rewriter.replaceOpWithNewOp<tpu::AvgPoolOp>(op, op->getResultTypes(),
                                                  adaptor.input(), attributes);
      return success();
    }
    assert(avg_out_type.getShape().size() == 2);
    SmallVector<int64_t, 4> avg_out_shape(4, 1);
    avg_out_shape[0] = avg_out_type.getShape()[0];
    avg_out_shape[1] = avg_out_type.getShape()[1];
    auto avg_out_keep_dim_type =
        RankedTensorType::get(avg_out_shape, avg_out_type.getElementType());
    auto avg_pool_op = rewriter.create<tpu::AvgPoolOp>(
        op->getLoc(), avg_out_keep_dim_type, adaptor.input(), attributes);

    std::string reshape_name(adaptor.name());
    reshape_name += "/Reshape";
    auto reshape_op =
        rewriter.create<tpu::ReshapeOp>(op->getLoc(), op->getResultTypes(),
                                        avg_pool_op.getResult(), reshape_name);
    rewriter.replaceOp(op, reshape_op.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TopTFLiteToTpuLoweringPass RewritePatterns: max_pooling operations
//===----------------------------------------------------------------------===//
class MaxPoolOpLowering : public OpConversionPattern<top::MaxPoolOp> {
  using OpConversionPattern<top::MaxPoolOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(top::MaxPoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isQuantized<quant::UniformQuantizedType>(op.input(), op.getResult()))
      return failure();
    rewriter.replaceOpWithNewOp<tpu::MaxPoolOp>(
        op, op->getResultTypes(), adaptor.input(),
        NamedAttrList(adaptor.getAttributes()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TopTFLiteToTpuLoweringPass RewritePatterns: MatMul operations
//===----------------------------------------------------------------------===//
// tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h::23
// tensorflow/tensorflow/lite/kernels/fully_connected.cc::239
class MatMulOpLowering : public OpConversionPattern<top::MatMulOp> {
  using OpConversionPattern<top::MatMulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(top::MatMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isQuantized<quant::UniformQuantizedType>(op.input(), op.right(),
                                                  op.getResult()))
      return failure();
    if (isQuantized(adaptor.right()) || isQuantized(adaptor.bias()))
      return failure(); // make sure filter and bias are storage type

    auto input_qtype =
        getRankedTensorElementType<quant::UniformQuantizedType>(op.input());
    auto right_qtype =
        getRankedTensorElementType<quant::UniformQuantizedType>(op.right());
    auto output_qtype =
        getRankedTensorElementType<quant::UniformQuantizedType>(op.getResult());

    const double real_multiplier = input_qtype.getScale() *
                                   right_qtype.getScale() /
                                   output_qtype.getScale();
    int64_t multiplier, shift;
    QuantizeMultiplier(real_multiplier, &multiplier, &shift);
    NamedAttrList attributes(adaptor.getAttributes());
    attributes.set("rshift", rewriter.getI64IntegerAttr(shift));
    attributes.set("multiplier", rewriter.getI64IntegerAttr(multiplier));
    int32_t input_zeroPoint = input_qtype.getZeroPoint();
    if (input_zeroPoint != 0 &&
        isa<top::WeightOp>(adaptor.right().getDefiningOp()) &&
        isa<top::WeightOp, top::NoneOp>(adaptor.bias().getDefiningOp())) {
      // merge input_zeroPoint to bias
      std::shared_ptr<std::vector<int32_t>> bias_quant;
      std::shared_ptr<std::vector<int8_t>> right_quant;
      right_quant =
          cast<top::WeightOp>(adaptor.right().getDefiningOp()).read<int8_t>();
      if (isa<top::WeightOp>(adaptor.bias().getDefiningOp())) {
        bias_quant =
            cast<top::WeightOp>(adaptor.bias().getDefiningOp()).read<int32_t>();
      }
      auto right_type = adaptor.right().getType().cast<RankedTensorType>();
      int64_t row_size = right_type.getShape()[0];
      int64_t col_size = right_type.getShape()[1];
      bias_quant->resize(col_size, 0);
      for (size_t r_ind = 0; r_ind < row_size; ++r_ind) {
        for (size_t c_ind = 0; c_ind < col_size; ++c_ind) {
          bias_quant->data()[c_ind] -=
              input_zeroPoint * right_quant->at(c_ind + r_ind * col_size);
        }
      }
      auto bias_type = RankedTensorType::get({col_size}, rewriter.getI32Type());
      auto new_bias =
          top::WeightOp::create(adaptor.bias().getDefiningOp(),
                                "MergedInputZeroPoint", *bias_quant, bias_type);
      rewriter.replaceOpWithNewOp<tpu::MatMulOp>(
          op, op->getResultTypes(),
          ValueRange({adaptor.input(), adaptor.right(), new_bias}), attributes);
      return success();
    }
    rewriter.replaceOpWithNewOp<tpu::MatMulOp>(
        op, op->getResultTypes(), adaptor.getOperands(), attributes);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TopTFLiteToTpuLoweringPass RewritePatterns: softmax operations
//===----------------------------------------------------------------------===//
// tensorflow/lite/kernels/internal/reference/softmax.h::67
// tensorflow/compiler/mlir/tosa/transforms/legalize_common.cc::1313
class SoftmaxOpLowering : public OpConversionPattern<top::SoftmaxOp> {
  using OpConversionPattern<top::SoftmaxOp>::OpConversionPattern;
  // softmax = exp(logits) / reduce_sum(exp(logits), -1)
  //
  // or equivalently multiply exp(-max(logits)) to both numerator and
  // denominator we get:
  //
  // softmax = exp(logits - max(logits)) / reduce_sum(exp(logits -
  // max(logits)), -1).
  LogicalResult
  matchAndRewrite(top::SoftmaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isQuantized<quant::UniformQuantizedType>(op.input(), op.getResult()))
      return failure();

    const double input_scale =
        getRankedTensorElementType<UniformQuantizedType>(op.input()).getScale();
    const double output_scale =
        getRankedTensorElementType<UniformQuantizedType>(op.getResult())
            .getScale();
    const double beta = 1.0; // TODO
    const double real_multiplier = beta * input_scale / output_scale;
    int64_t multiplier, shift;
    QuantizeMultiplier(real_multiplier, &multiplier, &shift);

    SmallVector<int64_t, 513> table;

    int64_t input_zp =
        getRankedTensorElementType<UniformQuantizedType>(op.input())
            .getZeroPoint();
    const int sum_compacity = 12; // 2^12 - 1, should be the number of reduced.
    double output_inv_scale = static_cast<double>(1L << (31 - sum_compacity));
    // The (input - max(input)) is in [-256, 0].
    for (int32_t i = -256; i <= 256; i++) {
      double dequantized = input_scale * i;
      double transformed = std::exp(dequantized);
      double truncated = std::min(std::max(transformed, -1.0), 1.0);
      int64_t rescaled =
          static_cast<int64_t>(std::round(truncated * output_inv_scale));
      table.push_back(rescaled);
    }
    NamedAttrList attributes(adaptor.getAttributes());
    attributes.set("rshift", rewriter.getI64IntegerAttr(shift));
    attributes.set("multiplier", rewriter.getI64IntegerAttr(multiplier));
    attributes.set("table", rewriter.getI64ArrayAttr(table));

#ifdef SOFTMAX_USE_INT
    rewriter.replaceOpWithNewOp<tpu::SoftmaxOp>(op, op->getResultTypes(),
                                                adaptor.input(), attributes);
    return success();
#else
    // use float computation
    auto input_type = adaptor.input().getType().cast<RankedTensorType>();
    auto cast_type = RankedTensorType::get(
        input_type.getShape(),
        getRankedTensorElementType<quant::UniformQuantizedType>(adaptor.input())
            .getExpressedType());

    std::string cast_name_prefix(adaptor.name());
    NamedAttrList cast_name;
    cast_name.set("name",
                  rewriter.getStringAttr(cast_name_prefix + "/CastToFloat"));
    auto pre_cast_op = rewriter.create<tpu::CastOp>(op->getLoc(), cast_type,
                                                    adaptor.input(), cast_name);

    auto softmax_op = rewriter.create<tpu::SoftmaxOp>(
        op->getLoc(), cast_type, pre_cast_op.getResult(), attributes);

    cast_name.set("name",
                  rewriter.getStringAttr(cast_name_prefix + "/CastToInt8"));
    auto post_cast_op =
        rewriter.create<tpu::CastOp>(op->getLoc(), op.getResult().getType(),
                                     softmax_op.getResult(), cast_name);

    rewriter.replaceOp(op, post_cast_op.getResult());
    return success();
#endif
  }
};

//===----------------------------------------------------------------------===//
// TopTFLiteToTpuLoweringPass RewritePatterns: weight operations
//===----------------------------------------------------------------------===//
class WeightOpLowering : public OpConversionPattern<top::WeightOp> {
  using OpConversionPattern<top::WeightOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(top::WeightOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto input_tensor_type = getRankedTensorElementType(op.getResult());
    if (!input_tensor_type
             .isa<UniformQuantizedPerAxisType, UniformQuantizedType>())
      return failure();
    Type storage_type;
    if (auto quant_type =
            input_tensor_type.dyn_cast<UniformQuantizedPerAxisType>())
      storage_type = quant_type.getStorageType();
    else
      storage_type =
          input_tensor_type.cast<UniformQuantizedType>().getStorageType();

    auto output_type = RankedTensorType::get(
        op.getResult().getType().cast<RankedTensorType>().getShape(),
        storage_type);
    rewriter.replaceOpWithNewOp<top::WeightOp>(op, output_type, adaptor.name());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TopTFLiteToTpuLoweringPass RewritePatterns: quant.qcast operations
//===----------------------------------------------------------------------===//
class QuantizeCastOpLowering
    : public OpConversionPattern<quant::QuantizeCastOp> {
  using OpConversionPattern<quant::QuantizeCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(quant::QuantizeCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isQuantized(op.getResult())) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
        diag << "expected the output type of 'quant.qcast' is "
                "QuantizedType.";
      });
    }
    NamedAttrList attributes(adaptor.getAttributes());
    rewriter.replaceOpWithNewOp<tpu::CastOp>(op, op.getResult().getType(),
                                             adaptor.getOperands(), attributes);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TopTFLiteToTpuLoweringPass RewritePatterns: quant.dcast operations
//===----------------------------------------------------------------------===//
class DequantizeCastOpLowering
    : public OpConversionPattern<quant::DequantizeCastOp> {
  using OpConversionPattern<quant::DequantizeCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(quant::DequantizeCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isQuantized(op.getOperand())) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
        diag << "expected the input type of 'quant.dcast' is "
                "QuantizedType.";
      });
    }
    NamedAttrList attributes(adaptor.getAttributes());
    rewriter.replaceOpWithNewOp<tpu::CastOp>(op, op.getResult().getType(),
                                             adaptor.getOperands(), attributes);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TopTFLiteToTpuLoweringPass
//===----------------------------------------------------------------------===//
void populateTopToTpuConversionPatterns(RewritePatternSet &patterns) {
  // clang-format off
    patterns.add<
        AddOpLowering,
        AvgPoolOpLowering,
        ConvOpLovering,
        DequantizeCastOpLowering,
        MatMulOpLowering,
        MaxPoolOpLowering,
        QuantizeCastOpLowering,
        WeightOpLowering,
        SoftmaxOpLowering>(patterns.getContext());
  // clang-format on
}

namespace {
class LoweringTopTFLitePass
    : public ConvertTopTFLiteToTpuBase<LoweringTopTFLitePass> {
  void runOnOperation() {
    ConversionTarget target(getContext());
    target.addLegalDialect<tpu::TpuDialect>();
    target.addLegalOp<top::InputOp>();
    target
        .addIllegalDialect<mlir::quant::QuantizationDialect, top::TopDialect>();
    target.addDynamicallyLegalOp<top::WeightOp>(
        [](top::WeightOp op) { return !isQuantized(op.getResult()); });
    RewritePatternSet patterns(&getContext());
    populateTopToTpuConversionPatterns(patterns);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();

    auto module = getOperation();
    Module::setState(module, Module::State::TPU_LOWERED);
    Module::setChip(module, Module::Chip::BM1686);
  }
};
} // namespace

std::unique_ptr<Pass> createLowerTopTFLitePass() {
  return std::make_unique<LoweringTopTFLitePass>();
}

} // namespace sophgo
