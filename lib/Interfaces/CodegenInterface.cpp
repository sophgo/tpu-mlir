//===- CodegenInterface.cpp - SideEffects in MLIR ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sophgo/Interfaces/CodegenInterface.h"

using namespace mlir;

/// Include the definitions of the side effect interfaces.
#include "sophgo/Interfaces/CodegenInterface.cpp.inc"
