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

// If a op has this trait, it means that relu follow this op can be fused to
// this op
template <typename ConcreteType>
class SupportFuseRelu
    : public ::mlir::OpTrait::TraitBase<ConcreteType, SupportFuseRelu> {
// public:
//   static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
//     static constexpr llvm::StringRef kDoReluAttr = "do_relu";
//     if (!op->hasAttrOfType<mlir::BoolAttr>(kDoReluAttr)) {
//       return op->emitError("expected operation to have attribute: " +
//                            kDoReluAttr);
//     }
//     return mlir::success();
//   }
};

} // namespace OpTrait
} // namespace sophgo
