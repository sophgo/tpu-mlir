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

// If a op has this trait, it means that some output(s) is(are) shape tensor(s)
template <typename ConcreteType>
class ShapeProducer
    : public ::mlir::OpTrait::TraitBase<ConcreteType, ShapeProducer> {};

// If a op has this trait, it means that some input(s) is(are) shape tensor(s)
template <typename ConcreteType>
class ShapeConsumer
    : public ::mlir::OpTrait::TraitBase<ConcreteType, ShapeConsumer> {};

template <typename ConcreteType>
class ScalarProducer
    : public ::mlir::OpTrait::TraitBase<ConcreteType, ScalarProducer> {};

template <typename ConcreteType>
class ScalarConsumer
    : public ::mlir::OpTrait::TraitBase<ConcreteType, ScalarConsumer> {};

// If a op has this trait, it means that relu follow this op can be fused to
// this op
template <typename ConcreteType>
class SupportFuseRelu
    : public ::mlir::OpTrait::TraitBase<ConcreteType, SupportFuseRelu> {};

template <typename ConcreteType>
class SupportPermuteMove
    : public ::mlir::OpTrait::TraitBase<ConcreteType, SupportPermuteMove> {};

template <typename ConcreteType>
class SupportConstant
    : public ::mlir::OpTrait::TraitBase<ConcreteType, SupportConstant> {};

template <typename ConcreteType>
class SupportEarlyStride
    : public ::mlir::OpTrait::TraitBase<ConcreteType, SupportEarlyStride> {};

template <typename ConcreteType>
class SupportElementwise
    : public ::mlir::OpTrait::TraitBase<ConcreteType, SupportElementwise> {};

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
