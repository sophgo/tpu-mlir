//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/OpDefinition.h"
#include "tpu_mlir/Support/Dnnl/Binary.h"
#include "tpu_mlir/Support/Dnnl/Conv.h"
#include "tpu_mlir/Support/Dnnl/Deconv.h"
#include "tpu_mlir/Support/Dnnl/LRN.h"
#include "tpu_mlir/Support/Dnnl/MatMul.h"
#include "tpu_mlir/Support/Dnnl/PRelu.h"
#include "tpu_mlir/Support/Dnnl/Pool.h"
#include "tpu_mlir/Support/Dnnl/ConvBwd.h"
namespace tpu_mlir {

dnnl::memory::data_type getDnnlType(mlir::Value v);

}
