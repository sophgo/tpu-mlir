//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace tpu_mlir::helper;
using namespace dnnl;

namespace tpu_mlir {
memory::data_type getDnnlType(mlir::Value v) {
  auto type = Module::getStorageType(v);
  if (type.isF32()) {
    return memory::data_type::f32;
  }
  if (type.isSignedInteger(8) || type.isSignlessInteger(8)) {
    return memory::data_type::s8;
  }
  if (type.isUnsignedInteger(8)) {
    return memory::data_type::u8;
  }
  if (type.isInteger(16) || type.isInteger(32)) {
    return memory::data_type::s32;
  }
  llvm::errs() << "Unsupport type: ";
  type.dump();
  return memory::data_type::f32;
}
} // namespace tpu_mlir
