//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/AttrStruct.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Module.h"
#include <vector>

using namespace std;

namespace tpu_mlir {

void parseGatherParam(const deform_conv2d_attr_t &attr,
                      deform_gather_attr_t &gattr);

void parseConvParam(const deform_conv2d_attr_t &attr, conv_attr_t &cattr);

void processDeformGather(InferenceParameter &p,
                         const deform_gather_attr_t &attr, float *data_out, bool top_flag);

void processDeformConv2D(InferenceParameter &p,
                                const deform_conv2d_attr_t &attr);
} // namespace tpu_mlir
