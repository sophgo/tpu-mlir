//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Support/Dnnl/Conv.h"
#include "tpu_mlir/Support/Dnnl/MatMul.h"
#include "tpu_mlir/Support/Dnnl/Pool.h"
#include "tpu_mlir/Support/Dnnl/Deconv.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

namespace tpu_mlir {

dnnl::memory::data_type getDnnlType(mlir::Value v);

}
