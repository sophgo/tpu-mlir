//===- Passes.h - Conversion Pass Construction and Registration -----------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
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
