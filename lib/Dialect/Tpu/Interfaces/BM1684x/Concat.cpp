//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
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

#ifdef __cplusplus
extern "C" {
#endif
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

typedef struct {
  unsigned int *bottom_addrs;
  unsigned int top_addr;
  int **bottom_shapes;
  int *top_shape;
  int *is_st_concat_way;
  int concat_axis;
  int shape_dim;
  int bottom_num;
  DATA_TYPE_T dtype;
} concat_local_param_t;

#ifdef __cplusplus
}
#endif

// =========================================
// GlobalGenInterface
// =========================================

void tpu::ConcatOp::codegen_global_bm1684x() {
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

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::ConcatOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::ConcatOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step) {
  llvm::SmallVector<uint32_t, 16> input_addrs;
  int num_inputs = inputs().size();
  llvm::SmallVector<int, 16> is_st_concat_way(num_inputs, 0);
  std::vector<int *> input_shapes;
  int64_t n, c, h, w;
  for (auto in : inputs()) {
    auto in_gi = LocalGenInterface::getGroupInfo(in, n_step, h_step);
    input_addrs.push_back(in_gi.out_addr);
    Module::getNCHW(in, n, c, h, w);
    auto shape = new int[MAX_SHAPE_DIMS];
    memset(shape, 0, sizeof(int) * MAX_SHAPE_DIMS);
    shape[0] = (int)in_gi.n_slice;
    shape[1] = (int)c;
    shape[2] = (int)in_gi.h_slice;
    shape[3] = (int)w;
    input_shapes.push_back(shape);
  }
  Module::getNCHW(output(), n, c, h, w);
  auto gi = getGroupInfo(n_step, h_step);
  int output_shape[MAX_SHAPE_DIMS] = {(int)gi.n_slice, (int)c, (int)gi.h_slice,
                                      (int)w};
  concat_local_param_t param = {0};
  param.bottom_addrs = input_addrs.data();
  param.top_addr = gi.out_addr;
  param.bottom_shapes = input_shapes.data();
  param.top_shape = (int *)output_shape;
  param.is_st_concat_way = is_st_concat_way.data();
  param.concat_axis = axis();
  param.shape_dim = 4;
  param.bottom_num = num_inputs;
  param.dtype = BM168x::getDataType(output());
  BM1684x::instance().call_local_func("backend_api_concat_local", &param,
                                      sizeof(param));
  for (auto s : input_shapes) {
    delete[] s;
  }
}
