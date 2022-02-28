//===- Inference.h - Interface for tiling operations ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the TilingInterface defined in
// `TilingInterface.td`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_INFERENCEINTERFACE_H_
#define MLIR_INTERFACES_INFERENCEINTERFACE_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

/// Include the ODS generated interface header files.
#include "sophgo/Interfaces/InferenceInterface.h.inc"

#endif // MLIR_INTERFACES_INFERENCE_H_
