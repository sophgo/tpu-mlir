#include "sophgo/Support/Helper/Quant.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Quant/FakeQuantSupport.h"
#include "mlir/Dialect/Quant/QuantizeUtils.h"
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

void Quant::getScaleAndZeroPoint(int64_t qmin, int64_t qmax, double rmin,
                                 double rmax, double &scale,
                                 int64_t &zeroPoint) {
  // Determine the scale.
  double qminDouble = qmin;
  double qmaxDouble = qmax;
  scale = (rmax - rmin) / (qmaxDouble - qminDouble);
  double zeroPointFromMin = qminDouble - rmin / scale;

  // Now nudge the zero point to be an integer.
  zeroPoint = round(zeroPointFromMin);
  if (zeroPointFromMin < qminDouble) {
    zeroPoint = qmin;
  } else if (zeroPointFromMin > qmaxDouble) {
    zeroPoint = qmax;
  }
}

void Quant::setQuantInt8Type(Value v, bool asymmetric, bool signType) {
  auto type = v.getType().cast<RankedTensorType>();
  auto ctx = v.getContext();
  auto cali_type = getCalibratedType(v);
  auto max = cali_type.getMax();
  auto min = cali_type.getMin();
  double scale;
  int64_t zeropoint = 0;
  int64_t qmin = -128, qmax = 127;
  uint32_t flag = quant::QuantizationFlags::Signed;
  if (asymmetric) {
    if (signType == false) {
      flag = 0;
      qmin = 0;
      qmax = 255;
    }
    getScaleAndZeroPoint(qmin, qmax, min, max, scale, zeropoint);
  } else {
    assert(max == -min);
    scale = max / 127.0;
  }
  auto qtype = quant::UniformQuantizedType::get(flag, IntegerType::get(ctx, 8),
                                                cali_type.getExpressedType(),
                                                scale, zeropoint, qmin, qmax);
  auto new_type = RankedTensorType::get(type.getShape(), qtype);
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
