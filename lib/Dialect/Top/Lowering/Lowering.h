//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"

#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;
using namespace mlir;
using namespace tpu_mlir::helper;
namespace tpu_mlir {
namespace top {

// Lowering to a new Operation, with the same operands and same attrs, and
// newType
template <typename OpTy>
static mlir::Value lowering_common(Operation *from, Type newType) {
  OpBuilder builder(from);
  builder.setInsertionPointAfter(from);
  auto newOp = builder.create<OpTy>(from->getLoc(), newType,
                                    from->getOperands(), from->getAttrs());
  return newOp.output();
}

// lowering to a new Operation, with same operands and same attrs, and quantize
// f32 output to int8 output
template <typename OpTy>
static mlir::Value lowering_common_int8(Operation *from,
                                        bool asymmetric = false) {
  auto newType = Quant::getQuantInt8Type(from->getResult(0), asymmetric);
  return lowering_common<OpTy>(from, newType);
}

// lowering to f32/f16/bf16
template <typename OpTy, typename ElemTy = Float32Type>
static mlir::Value lowering_common_float(Operation *from) {
  auto output = from->getResult(0);
  auto sType = Module::getStorageType(output);
  Type newType = output.getType();
  if (sType.isa<ElemTy>() == false) {
    auto shape = Module::getShape(output);
    auto ctx = from->getContext();
    if (Quant::isCalibratedType(output)) {
      auto caliType = Quant::getCalibratedType(output);
      auto newCaliType = quant::CalibratedQuantizedType::get(
          ElemTy::get(ctx), caliType.getMin(), caliType.getMax());
      newType = RankedTensorType::get(shape, newCaliType);
    } else {
      newType = RankedTensorType::get(shape, ElemTy::get(ctx));
    }
  }
  return lowering_common<OpTy>(from, newType);
}

// cast Value to new type. if tensorType is true, new type will be "to", else
// {getShape(v), to}. support F32/F16/BF16/qint8
Value do_cast(Value v, Type to, bool tensorType);

// from f32 quant to qint8
Value do_quantize(Value v, bool asymmetric);

// from int8 to int8, convert one (scale zp) to another (scale zp)
Value do_transfer(Value in, Value out, bool asymmetric);

// from int8 to int32
Value do_dequant(Value input, Type to_type, int64_t multiplier, int64_t rshift,
                 tpu::DequantMode mode, int64_t lshift);

// from int8 to int32
Value do_requant(Location name_loc, Value input, Type to_type, bool tensorType,
                 int64_t multiplier, int64_t shift, tpu::RequantMode mode);

Value do_requant(Location name_loc, Value input, Value quant, Type to_type,
                 bool tensorType, tpu::RequantMode mode);

// create lookup table
typedef double (*activate_f)(double);

Value create_lookup_table(Value in, Value out, activate_f func,
                          bool asymmetric);

Value create_lookup_table(Operation *owner, const std::vector<float> &table);

} // namespace top
} // namespace tpu_mlir
