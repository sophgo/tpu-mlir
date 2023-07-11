//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

// int8
void tpu::LayerNormOp::codegen_global_cv18xx(int64_t layer_id) {
  bool has_weight = !getWeight().getType().isa<mlir::NoneType>();
  bool has_bias = !getBias().getType().isa<mlir::NoneType>();
  const auto input_shape = module::getShape(getInput());
  const float eps = getEps().convertToDouble();
  const int axis = getAxis();
  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  gaddr_t ga_table = module::getAddress(getTable());
  gaddr_t ga_mantissa_table = module::getAddress(getMantissaTable());
  gaddr_t ga_weight = module::getAddress(getWeight());
  gaddr_t ga_bias = module::getAddress(getBias());

  int outer_dim =
      std::accumulate(input_shape.begin(), input_shape.begin() + axis, 1,
                      std::multiplies<int>());
  int axes_dim = std::accumulate(input_shape.begin() + axis, input_shape.end(),
                                 1, std::multiplies<int>());

  cvi_backend_tg_bf16_layernorm_kernel(
      layer_id, ga_input, ga_table, ga_mantissa_table, ga_weight, ga_bias,
      ga_output, outer_dim, axes_dim, eps, has_weight, has_bias);
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::LayerNormOp::getBufferSize_cv18xx(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  auto input_shape = module::getShape(getInput()).vec();
  auto axis = getAxis();
  int axes_dim = std::accumulate(input_shape.begin() + axis, input_shape.end(),
                                 1, std::multiplies<int>());
  int eu_num = CV18xx::EU_BYTES / 2; // bf16 = 2 bytes
  int blob_num = 4;
  if (axes_dim >= 3 * eu_num) {
    blob_num = 2;
  } else if (2 * axes_dim >= 3 * eu_num) {
    blob_num = 3;
  }
  for (int i = input_shape.size(); i < 4; i++) {
    input_shape.push_back(1);
  }
  input_shape[0] = in_nslice;
  assert(in_hslice == input_shape[2]);
  auto fmt = CV18xx::getDataType(getInput());
  return CV18xx::lmem_woring_size(input_shape, blob_num, true, fmt);
}

void tpu::LayerNormOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                            int64_t d_step, int64_t w_step,
                                            group_type_t group_type,
                                            local_sec_info_t &sec_info,
                                            int64_t layer_id) {
  bool has_weight = !getWeight().getType().isa<mlir::NoneType>();
  bool has_bias = !getBias().getType().isa<mlir::NoneType>();
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  auto gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto out_gi = LocalGenInterface::getGroupInfo(getOutput(), n_step, h_step);
  auto w_gi = LocalGenInterface::getGroupInfo(getWeight());
  auto b_gi = LocalGenInterface::getGroupInfo(getBias());
  auto t_gi = LocalGenInterface::getGroupInfo(getTable());
  auto mt_gi = LocalGenInterface::getGroupInfo(getMantissaTable());

  laddr_t la_input = in_gi.out_addr;
  laddr_t la_output = out_gi.out_addr;
  laddr_t la_table = t_gi.out_addr;
  laddr_t la_mantissa_table = mt_gi.out_addr;
  laddr_t la_scale = w_gi.out_addr;
  laddr_t la_bias = b_gi.out_addr;
  laddr_t la_working = gi.buffer_addr;

  n = sec_info.n_slice;
  h = sec_info.h_slice;

  const float eps = getEps().convertToDouble();
  cvi_backend_tl_bf16_layernorm(
      layer_id, la_input, la_output, la_table, la_mantissa_table, la_scale,
      la_bias, la_working, has_weight, has_bias, eps, n, c, h, w);
}
