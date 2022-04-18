#include "sophgo/Support/Helper/Quant.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Dialect/Quant/FakeQuantSupport.h"
#include "float.h"
#include <map>
using namespace llvm;
using namespace mlir;
namespace sophgo {
namespace helper {

constexpr double Quant::QMAX_INT8;
constexpr int Quant::BITS_INT8;
constexpr llvm::StringRef Quant::Type::INT8;
constexpr llvm::StringRef Quant::Type::BF16;
constexpr llvm::StringRef Quant::Type::FP16;
constexpr llvm::StringRef Quant::Type::FP32;

template bool Quant::isQuantizedType<quant::CalibratedQuantizedType>(Value v);
template quant::CalibratedQuantizedType
Quant::getQuantizedType<quant::CalibratedQuantizedType>(Value v);

void Quant::setQuantInt8Type(Value v, bool asymmetric, bool sighType) {
  auto type = v.getType().cast<RankedTensorType>();
  auto ctx = v.getContext();
  auto cali_type = type.getElementType().cast<quant::CalibratedQuantizedType>();
  auto max = cali_type.getMax();
  auto min = cali_type.getMin();
  if (asymmetric) {
    double scale = (max - min) / (127 - (-128));
    int64_t zeropoint = std::round(-min / scale);
    auto uniform_type = quant::UniformQuantizedType();
    if (sighType) {
      uniform_type = quant::UniformQuantizedType::get(
          quant::QuantizationFlags::Signed, IntegerType::get(ctx, 8),
          cali_type.getExpressedType(), scale, zeropoint, -128, 127);
    } else {
      uniform_type = quant::UniformQuantizedType::get(
          0, IntegerType::get(ctx, 8), cali_type.getExpressedType(), scale,
          zeropoint, 0, 255);
    }
    auto new_type = RankedTensorType::get(type.getShape(), uniform_type);
    v.setType(new_type);
  } else {
    assert(max == -min);
    double scale = max / 127.0;
    auto uniform_type = quant::UniformQuantizedType::get(
        quant::QuantizationFlags::Signed, IntegerType::get(ctx, 8),
        cali_type.getExpressedType(), scale, 0, -128, 127);
    auto new_type = RankedTensorType::get(type.getShape(), uniform_type);
    v.setType(new_type);
  }
}

void Quant::setQuantWeightInt8PerChannelType(Value v, ArrayRef<double> scales,
                                             ArrayRef<int64_t> zeroPoints,
                                             int32_t quantizedDimension,
                                             mlir::FloatType exptype) {
  auto type = v.getType().cast<RankedTensorType>();
  auto ctx = v.getContext();
  auto per_channel_int8_type = quant::UniformQuantizedPerAxisType::get(
      quant::QuantizationFlags::Signed, IntegerType::get(ctx, 8), exptype,
      scales, zeroPoints, quantizedDimension, -128, 127);
  auto new_type = RankedTensorType::get(type.getShape(), per_channel_int8_type);
  v.setType(new_type);
}

void Quant::setQuantExpressType(Value v) {
  if (!isUniformQuantized(v)) {
    return;
  }
  auto type = v.getType().cast<RankedTensorType>();
  auto expresstype =
      type.getElementType().cast<quant::QuantizedType>().getExpressedType();
  auto new_type = RankedTensorType::get(type.getShape(), expresstype);
  v.setType(new_type);
}

} // namespace helper
} // namespace sophgo
