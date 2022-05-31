//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef SOPHGO_TPU_ATTRS
#define SOPHGO_TPU_ATTRS

include "sophgo/Dialect/Tpu/IR/TpuBase.td"

def Tpu_LayerGroupAttr :
    StructAttr<"LayerGroup", Tpu_Dialect, [
    StructFieldAttr<"out_addr", DefaultValuedAttr<I64Attr, "0">>,
    StructFieldAttr<"out_size", DefaultValuedAttr<I64Attr, "0">>,
    StructFieldAttr<"buffer_addr", DefaultValuedAttr<I64Attr, "0">>,
    StructFieldAttr<"buffer_size", DefaultValuedAttr<I64Attr, "0">>,
    StructFieldAttr<"h_idx", I64ArrayAttr>,
    StructFieldAttr<"h_slice", I64ArrayAttr>,
    StructFieldAttr<"n_idx", I64ArrayAttr>,
    StructFieldAttr<"n_slice", I64ArrayAttr>,
    StructFieldAttr<"timestep", I64Attr>
]> {
  let summary = "Structure of layer group parameters";
}

#endif
