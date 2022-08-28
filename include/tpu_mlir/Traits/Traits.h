//===------------------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// tpu-mlir-specific traits.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace tpu_mlir {
namespace trait {

namespace impl {
mlir::LogicalResult verifyTpuTypeRestrictTrait(mlir::Operation *op);
mlir::LogicalResult verifyInOutSameShapeTrait(mlir::Operation *op);
mlir::LogicalResult verifyInOutSameDimTrait(mlir::Operation *op);
} // namespace impl

// If a op has this trait, it means that relu follow this op can be fused to
// this op
template <typename ConcreteType>
class SupportFuseRelu
    : public ::mlir::OpTrait::TraitBase<ConcreteType, SupportFuseRelu> {};

// If a op has this trait, it should have nameAttr
template <typename ConcreteType>
class HasCommonAttributes
    : public ::mlir::OpTrait::TraitBase<ConcreteType, HasCommonAttributes> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    // static constexpr llvm::StringRef commonAttrs[] = {"name"};
    // for (auto attr : commonAttrs) {
    //   if (!op->hasAttrOfType<mlir::StringAttr>(attr)) {
    //     return op->emitError("expected operation to have attribute: " + attr);
    //   }
    // }
    return mlir::success();
  }
};

template <typename ConcreteType>
class TpuTypeRestrict
    : public ::mlir::OpTrait::TraitBase<ConcreteType, TpuTypeRestrict> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    return impl::verifyTpuTypeRestrictTrait(op);
  }
};

template <typename ConcreteType>
class InOutSameShape
    : public ::mlir::OpTrait::TraitBase<ConcreteType, InOutSameShape> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    return impl::verifyInOutSameShapeTrait(op);
  }
};

template <typename ConcreteType>
class InOutSameDim
    : public ::mlir::OpTrait::TraitBase<ConcreteType, InOutSameDim> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    return impl::verifyInOutSameDimTrait(op);
  }
};

} // namespace trait
} // namespace tpu_mlir
