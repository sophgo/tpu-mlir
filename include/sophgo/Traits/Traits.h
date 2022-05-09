//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
//
// Sophgo-specific traits.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace sophgo {
namespace trait {

namespace impl {
mlir::LogicalResult verifyTpuTypeRestrictTrait(mlir::Operation *op);
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
    static constexpr llvm::StringRef commonAttrs[] = {"name"};
    for (auto attr : commonAttrs) {
      if (!op->hasAttrOfType<mlir::StringAttr>(attr)) {
        return op->emitError("expected operation to have attribute: " + attr);
      }
    }
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


} // namespace trait
} // namespace sophgo
