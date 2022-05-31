//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef SOPHGO_TPU_BASE
#define SOPHGO_TPU_BASE

include "mlir/IR/OpBase.td"

def Tpu_Dialect : Dialect {
  let name = "tpu";
  let summary = "A tpu dialect for the SOPHGO AI chips";
  let cppNamespace = "::sophgo::tpu";
}

#endif
