//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypes.h"
#include "tpu_mlir/Backend/BM168x/BM1684x.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

typedef struct {
  unsigned long long *bottom_addrs;
  unsigned long long top_addr;
  int (*bottom_shapes)[MAX_SHAPE_DIMS];
  int *top_shape;
  int *is_st_concat_way;
  int concat_axis;
  int shape_dim;
  int bottom_num;
  DATA_TYPE_T dtype;
} concat_global_param_t;

void tpu::ConcatOp::codegen_global_int8_bm1684x() {
  codegen_global_float_bm1684x();
}

void tpu::ConcatOp::codegen_global_float_bm1684x() {

  concat_global_param_t param = {0};
  param.concat_axis = axis();
  param.bottom_num = inputs().size();
  param.shape_dim = Module::getShape(inputs()[0]).size();
  param.dtype = BM168x::getDataType(inputs()[0]);

  SmallVector<uint64_t> input_addr(inputs().size());
  std::vector<int[MAX_SHAPE_DIMS]> input_shape(param.bottom_num);
  for (auto v : llvm::enumerate(inputs())) {
    input_addr[v.index()] = Module::getAddress(v.value());
    for (auto dim : llvm::enumerate(Module::getShape(v.value())))
      input_shape[v.index()][dim.index()] = dim.value();
  }
  SmallVector<int> output_shape(param.shape_dim);
  for (auto v : llvm::enumerate(Module::getShape(output())))
    output_shape[v.index()] = v.value();

  SmallVector<int> is_st_concat_way(param.shape_dim, 0);
  param.bottom_addrs = (unsigned long long *)input_addr.data();
  param.top_addr = Module::getAddress(output());

  param.bottom_shapes = input_shape.data();
  param.top_shape = output_shape.data();
  param.is_st_concat_way = is_st_concat_way.data();

  BM1684x::instance().call_global_func("backend_api_concat_global", &param,
                                       sizeof(param));
}
