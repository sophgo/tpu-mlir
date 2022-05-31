//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "sophgo/Traits/Traits.h"
#include "mlir/Dialect/Quant/QuantTypes.h"

using namespace mlir;

namespace sophgo {

namespace trait {
namespace impl {

static LogicalResult check_type(Value v) {
  auto type = v.getType();
  if (type.isa<NoneType>()) {
    return success();
  }
  if (auto tensor_type = type.dyn_cast<RankedTensorType>()) {
    auto etype = tensor_type.getElementType();
    if (etype.isIntOrFloat()) {
      return success();
    }
    if (etype.isa<quant::UniformQuantizedType>()) {
      return success();
    }
  }
  v.dump();
  return failure();
}

LogicalResult verifyTpuTypeRestrictTrait(Operation *op) {
  for (auto out : op->getResults()) {
    if (failed(check_type(out))) {
      return op->emitError("expected tpu supported type");
    }
  }
  return mlir::success();
}

} // namespace impl
} // namespace trait

} // namespace sophgo
