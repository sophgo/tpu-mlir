//===- InitAllDialects.h - MLIR Dialects Registration -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all dialects and
// passes to the system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INITALLDIALECTS_H_
#define MLIR_INITALLDIALECTS_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Dialect.h"

namespace mlir {

/// Add all the MLIR dialects to the provided registry.
inline void registerAllDialects(DialectRegistry &registry) {
  // clang-format off
  registry.insert<
                  affine::AffineDialect,
                  arith::ArithDialect,
                  bufferization::BufferizationDialect,
                  cf::ControlFlowDialect,
                  complex::ComplexDialect,
                  func::FuncDialect,
                  linalg::LinalgDialect,
                  memref::MemRefDialect,
                  scf::SCFDialect,
                  quant::QuantizationDialect,
                  tensor::TensorDialect,
                  tosa::TosaDialect>();
  // clang-format on
}

/// Append all the MLIR dialects to the registry contained in the given context.
inline void registerAllDialects(MLIRContext &context) {
  DialectRegistry registry;
  registerAllDialects(registry);
  context.appendDialectRegistry(registry);
}

} // namespace mlir

#endif // MLIR_INITALLDIALECTS_H_
