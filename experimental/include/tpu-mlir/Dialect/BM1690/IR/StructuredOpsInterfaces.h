//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "tpu-mlir/Dialect/BM1690/IR/BM1690Enum.h.inc"

namespace tpu_mlir {
namespace bm1690 {
using namespace mlir;
struct TPUISATraits;

using StructuredOpTraits = std::map<llvm::StringLiteral, TPUISATraits>;
StructuredOpTraits &registerTraits(MLIRContext *context);

// This class use an concept based polymorphism idioms which implements
// polymorphism without inheritance in C++.
// https://gracicot.github.io/conceptmodel/2017/09/13/concept-model-part1.html
// https://www.youtube.com/watch?v=QGcVXgEVMJg
struct TPUISATraits {
  template <typename ConcreteT>
  TPUISATraits(ConcreteT t) noexcept
      : self{std::make_unique<Model<ConcreteT>>(std::move(t))} {}

  TPUISATraits(std::nullptr_t) noexcept : self(nullptr) {}
  TPUISATraits() noexcept : self(nullptr) {}

  explicit operator bool() { return self != nullptr; }

  AffineMap getLayout(OpOperand *opOperand) {
    return self->getLayout(opOperand);
  };
  AffineMap getAccelerateMap() { return self->getAccelerateMap(); };
  ArithType getArithType() { return self->getArithType(); };
  ArrayAttr getIteratorTypes() { return self->getIteratorTypes(); };
  ArrayAttr getIndexingMaps() { return self->getIndexingMaps(); };

private:
  struct Concept {
    virtual ~Concept() = default;
    virtual AffineMap getLayout(OpOperand *opOperand) = 0;
    virtual AffineMap getAccelerateMap() = 0;
    virtual ArithType getArithType() = 0;
    virtual ArrayAttr getIteratorTypes() = 0;
    virtual ArrayAttr getIndexingMaps() = 0;
  };

  template <typename ConcreteT>
  struct Model : Concept {
    Model(ConcreteT s) noexcept : self{std::move(s)} {}
    AffineMap getLayout(OpOperand *opOperand) {
      return self.getLayout(opOperand);
    };
    AffineMap getAccelerateMap() { return self.getAccelerateMap(); };
    ArithType getArithType() { return self.getArithType(); };
    ArrayAttr getIteratorTypes() { return self.getIteratorTypes(); };
    ArrayAttr getIndexingMaps() { return self.getIndexingMaps(); };
    ConcreteT self;
  };

  std::shared_ptr<Concept> self;
};
} // namespace bm1690
} // namespace tpu_mlir
