//===- Passes.h - Conversion Pass Construction and Registration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SOPHGO_CONVERSION_PASSES_H
#define SOPHGO_CONVERSION_PASSES_H

#include "sophgo/Conversion/TopTFLiteToTpu.h"

namespace sophgo {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "sophgo/Conversion/Passes.h.inc"

} // namespace mlir

#endif // SOPHGO_CONVERSION_PASSES_H
