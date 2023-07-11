//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"

using namespace dnnl;

namespace tpu_mlir {
memory::data_type getDnnlType(mlir::Value v) {
  auto type = module::getStorageType(v);
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
