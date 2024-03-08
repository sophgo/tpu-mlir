//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "tpu-mlir/Dialect/BM1690/IR/BM1690.h"

namespace tpu_mlir {
namespace bm1690 {
using namespace mlir;

ArrayAttr buildIteratorTypes(Builder builder,
                             SmallVector<utils::IteratorType> &&iteratorTypes) {
  return builder.getArrayAttr(llvm::to_vector(llvm::map_range(
      iteratorTypes, [&](utils::IteratorType t) -> mlir::Attribute {
        return IteratorTypeAttr::get(builder.getContext(), t);
      })));
}

TPUISATraits MatMulOp::getStuctutedTraits(MLIRContext *context) {
  struct MatMulOpTraits {
    MatMulOpTraits(MLIRContext *context) : context(context){};

    AffineMap getLayout(OpOperand *opOperand) { return AffineMap(); };

    AffineMap getAccelerateMap() {
      const StringLiteral accelerateMap =
          R"mlir(affine_map<(d0, d1, d2) -> (64, 4, 32)>)mlir";
      return cast<AffineMapAttr>(mlir::parseAttribute(accelerateMap, context))
          .getValue();
    };

    ArithType getArithType() { return ArithType::add; };

    ArrayAttr getIteratorTypes() {
      return buildIteratorTypes(Builder(context),
                                {utils::IteratorType::parallel,
                                 utils::IteratorType::parallel,
                                 utils::IteratorType::reduction});
    };

    ArrayAttr getIndexingMaps() {
      const StringLiteral indexingMaps =
          R"mlir([affine_map<(d0, d1, d2) -> (d0, d2)>,
                  affine_map<(d0, d1, d2) -> (d1, d2)>,
                  affine_map<(d0, d1, d2) -> (d0, d1)>]
           )mlir";
      return cast<ArrayAttr>(mlir::parseAttribute(indexingMaps, context));
    };

    MLIRContext *context;
  };
  return MatMulOpTraits(context);
}

template <ArithType value>
struct ArithBinaryOpTraits {
  ArithBinaryOpTraits(MLIRContext *context) : context(context){};
  AffineMap getLayout(OpOperand *opOperand) { return AffineMap(); };

  ArithType getArithType() { return value; };

  AffineMap getAccelerateMap() {
    const StringLiteral accelerateMap =
        R"mlir(affine_map<(d0, d1, d2, d3) -> (1, 64, 1, 32)>)mlir";
    return cast<AffineMapAttr>(mlir::parseAttribute(accelerateMap, context))
        .getValue();
  };

  ArrayAttr getIteratorTypes() {
    return buildIteratorTypes(
        Builder(context),
        {utils::IteratorType::parallel, utils::IteratorType::parallel,
         utils::IteratorType::parallel, utils::IteratorType::parallel});
  };

  ArrayAttr getIndexingMaps() {
    const StringLiteral indexingMaps =
        R"mlir([affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]
         )mlir";
    return cast<ArrayAttr>(mlir::parseAttribute(indexingMaps, context));
  };

  MLIRContext *context;
};

template <ArithType value>
struct ArithUnaryOpTraits {
  ArithUnaryOpTraits(MLIRContext *context) : context(context){};
  AffineMap getLayout(OpOperand *opOperand) { return AffineMap(); };

  ArithType getArithType() { return value; };

  AffineMap getAccelerateMap() {
    const StringLiteral accelerateMap =
        R"mlir(affine_map<(d0, d1, d2, d3) -> (1, 64, 1, 32)>)mlir";
    return cast<AffineMapAttr>(mlir::parseAttribute(accelerateMap, context))
        .getValue();
  };

  ArrayAttr getIteratorTypes() {
    using namespace utils;
    return buildIteratorTypes(Builder(context),
                              {IteratorType::parallel, IteratorType::parallel,
                               IteratorType::parallel, IteratorType::parallel});
  };

  ArrayAttr getIndexingMaps() {
    const StringLiteral indexingMaps =
        R"mlir([affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]
         )mlir";
    return cast<ArrayAttr>(mlir::parseAttribute(indexingMaps, context));
  };

  MLIRContext *context;
};

template <ArithType value>
struct PoolingOpTraits {
  PoolingOpTraits(MLIRContext *context) : context(context){};
  AffineMap getLayout(OpOperand *opOperand) { return AffineMap(); };

  ArithType getArithType() { return value; };

  AffineMap getAccelerateMap() {
    const StringLiteral accelerateMap =
        R"mlir(affine_map<(d0, d1, d2, d3, d4, d5) -> (1, 64, 1, 32, 1, 1)>)mlir";
    return cast<AffineMapAttr>(mlir::parseAttribute(accelerateMap, context))
        .getValue();
  };

  ArrayAttr getIteratorTypes() {
    return buildIteratorTypes(
        Builder(context),
        {utils::IteratorType::parallel, utils::IteratorType::parallel,
         utils::IteratorType::parallel, utils::IteratorType::parallel,
         utils::IteratorType::reduction, utils::IteratorType::reduction});
  };

  ArrayAttr getIndexingMaps() {
    const StringLiteral indexingMaps = // n, c, h, w, kh, hw, hs, hd, ws,wd
        R"mlir([affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3] -> (d0, d1, d2 * s0 + d4 * s1, d3 * s2 + d5 * s3)>,
                affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3] -> (d0, d1, d2, d3)>]
         )mlir";
    return cast<ArrayAttr>(mlir::parseAttribute(indexingMaps, context));
  };

  MLIRContext *context;
};

TPUISATraits MaxOp::getStuctutedTraits(MLIRContext *context) {
  return ArithBinaryOpTraits<ArithType::max>(context);
}

TPUISATraits MinOp::getStuctutedTraits(MLIRContext *context) {
  return ArithBinaryOpTraits<ArithType::min>(context);
}

TPUISATraits AndOp::getStuctutedTraits(MLIRContext *context) {
  return ArithBinaryOpTraits<ArithType::AND>(context);
}

TPUISATraits OrOp::getStuctutedTraits(MLIRContext *context) {
  return ArithBinaryOpTraits<ArithType::OR>(context);
}

TPUISATraits XorOp::getStuctutedTraits(MLIRContext *context) {
  return ArithBinaryOpTraits<ArithType::XOR>(context);
}

TPUISATraits SubOp::getStuctutedTraits(MLIRContext *context) {
  return ArithBinaryOpTraits<ArithType::sub>(context);
}

TPUISATraits AddOp::getStuctutedTraits(MLIRContext *context) {
  return ArithBinaryOpTraits<ArithType::add>(context);
}

TPUISATraits MulOp::getStuctutedTraits(MLIRContext *context) {
  return ArithBinaryOpTraits<ArithType::mul>(context);
}

TPUISATraits AbsOp::getStuctutedTraits(MLIRContext *context) {
  return ArithUnaryOpTraits<ArithType::abs>(context);
}

TPUISATraits NotOp::getStuctutedTraits(MLIRContext *context) {
  return ArithUnaryOpTraits<ArithType::neg>(context);
}

TPUISATraits MaxPoolOp::getStuctutedTraits(MLIRContext *context) {
  return PoolingOpTraits<ArithType::max>(context);
}

TPUISATraits MinPoolOp::getStuctutedTraits(MLIRContext *context) {
  return PoolingOpTraits<ArithType::min>(context);
}

TPUISATraits AvgPoolOp::getStuctutedTraits(MLIRContext *context) {
  return PoolingOpTraits<ArithType::add>(context);
}

TPUISATraits DepthwiseOp::getStuctutedTraits(MLIRContext *context) {
  struct DepthwiseOpTraits : PoolingOpTraits<ArithType::add> {
    DepthwiseOpTraits(MLIRContext *contxt)
        : PoolingOpTraits<ArithType::add>(contxt) {}
    ArrayAttr getIndexingMaps() {
      const StringLiteral indexingMaps = // n, c, h, w, kh, hw, hs, hd, ws, wd
          R"mlir([affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3]
                                 -> (d0, d1, d2 * s0 + d4 * s1, d3 * s2 + d5 * s3)>,
                  affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3] -> (d1, d4, d5)>,
                  affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3] -> (d0, d1, d2, d3)>]
         )mlir";
      return cast<ArrayAttr>(mlir::parseAttribute(indexingMaps, context));
    };
  };
  return DepthwiseOpTraits(context);
}

struct OpTraits {
  OpTraits(){};
  template <typename... Args>
  void addTraits(MLIRContext *context) {
    (storage.insert(
         {Args::getOperationName(), Args::getStuctutedTraits(context)}),
     ...);
  }
  TPUISATraits &lookup(llvm::StringLiteral name) {
    if (storage.count(name))
      return storage[name];
    return null;
  };
  StructuredOpTraits &get() { return storage; };

private:
  StructuredOpTraits storage;
  TPUISATraits null;
};

StructuredOpTraits &registerTraits(MLIRContext *context) {
  static OpTraits opTraits;
  opTraits.addTraits<MatMulOp, MaxOp, MinOp, AndOp, OrOp, XorOp, SubOp, AddOp,
                     MulOp, AbsOp, NotOp, MaxPoolOp, MinPoolOp, AvgPoolOp,
                     DepthwiseOp>(context);
  return opTraits.get();
}

} // namespace bm1690
} // namespace tpu_mlir
