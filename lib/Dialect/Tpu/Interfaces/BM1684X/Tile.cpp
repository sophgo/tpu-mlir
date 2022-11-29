//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned int input_local_addr;
  unsigned int output_local_addr;
  int input_shape[MAX_SHAPE_DIMS];
  int tile_coeff[MAX_SHAPE_DIMS];
  int input_dim;
  int type;
  int dtype;
} tile_local_param_t;

typedef struct {
  unsigned long long input_global_addr;
  unsigned long long buffer_global_addr;
  unsigned long long output_global_addr;
  int input_shape[MAX_SHAPE_DIMS];
  int tile_coeff[MAX_SHAPE_DIMS];
  int input_dim;
  int type;
  int dtype;
} tile_global_param_t;

#ifdef __cplusplus
}
#endif

// =========================================
// GloballGenInterface
// =========================================
void tpu::TileOp::codegen_global_bm1684x() {
  auto in_shape = Module::getShape(input()).vec();
  auto out_shape = Module::getShape(output());
  if (in_shape.size() < out_shape.size())
    in_shape.insert(in_shape.begin(), 1);

  tile_global_param_t param = {0};
  for (int i = 0; i < in_shape.size(); ++i) {
    param.tile_coeff[i] = out_shape[i] / in_shape[i];
    param.input_shape[i] = in_shape[i];
  }

  param.input_global_addr = Module::getAddress(input());
  param.output_global_addr = Module::getAddress(output());
  param.input_dim = in_shape.size();
  param.type = 0;
  param.dtype = BM168x::getDataType(input());
  BM168x::call_global_func("backend_api_tile_global", &param,
                                       sizeof(param));
}


// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::TileOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  return 0;
}

void tpu::TileOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step) {
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);
  tile_local_param_t param{0};
  auto in_shape = Module::getShape(input()).vec();
  auto out_shape = Module::getShape(output());

  if (in_shape.size() < out_shape.size())
    in_shape.insert(in_shape.begin(), 1);

  param.input_local_addr = (uint32_t)in_gi.out_addr;
  param.output_local_addr = (uint32_t)gi.out_addr;
  param.input_shape[0] = in_gi.n_slice;
  param.input_shape[1] = in_shape[1];
  param.input_shape[2] = in_gi.h_slice;
  param.input_shape[3] = in_shape[3];
  for (int i = 0; i < in_shape.size(); ++i) {
    param.tile_coeff[i] = out_shape[i] / in_shape[i];
  }
  param.input_dim = in_shape.size();
  param.type = 0;
  param.dtype = BM168x::getDataType(input());
  BM168x::call_local_func("backend_api_tile_local", &param,
                                      sizeof(param));
}
