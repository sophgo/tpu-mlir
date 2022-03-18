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
constexpr llvm::StringRef Quant::Type::INT8;
constexpr llvm::StringRef Quant::Type::BF16;
constexpr llvm::StringRef Quant::Type::FP16;
constexpr llvm::StringRef Quant::Type::FP32;

template bool Quant::isQuantizedType<quant::CalibratedQuantizedType>(Value v);
template quant::CalibratedQuantizedType
Quant::getQuantizedType<quant::CalibratedQuantizedType>(Value v);

void Quant::setQuantInt8Type(Value v, bool asymmetric) {
  auto type = v.getType().cast<RankedTensorType>();
  auto ctx = v.getContext();
  auto cali_type = type.getElementType().cast<quant::CalibratedQuantizedType>();
  auto max = cali_type.getMax();
  auto min = cali_type.getMin();
  if (asymmetric) {
    auto uniform_type = quant::fakeQuantAttrsToType(
        v.getLoc(), 8, min, max, false, cali_type.getExpressedType(), true);
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
