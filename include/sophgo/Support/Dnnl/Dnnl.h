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

#include "sophgo/Support/Dnnl/Conv.h"
#include "sophgo/Support/Dnnl/MatMul.h"
#include "sophgo/Support/Dnnl/Pool.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

namespace sophgo {

dnnl::memory::data_type getDnnlType(mlir::Value v);

}
