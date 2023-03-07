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
} // namespace impl

// If a op has this trait, it means that relu follow this op can be fused to
// this op
template <typename ConcreteType>
class SupportFuseRelu
    : public ::mlir::OpTrait::TraitBase<ConcreteType, SupportFuseRelu> {};

template <typename ConcreteType>
class SupportEarlyStride
    : public ::mlir::OpTrait::TraitBase<ConcreteType, SupportEarlyStride> {};

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

} // namespace trait
} // namespace tpu_mlir
