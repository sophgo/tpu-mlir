//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Support/LLVM.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"
#include <list>
#include <map>
#include <set>

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/BasicTimeStep.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupDefs.h"

namespace tpu_mlir {
namespace tpu {

shape_secs_t get_group_max_secs(const LgInfo &lg_info);
bool init_group_data_secs(const LgInfo &lg_info,
                                        shape_secs_t &shape_secs);

void update_tensor_infos(const LgInfo &lg_info, TensorInfo &tensor_infos);
bool update_data_split(BasicTimeStepPtr time_step, const LgInfo &lg_info,
                       shape_secs_t &shape_secs);

bool strip_back_judge(Value v, const LgInfo &lg_info,
                      const std::multiset<Operation *> &op_set,
                      const std::set<Value, value_compare> &out_tensor_set);
bool is_same_slice_info(const slice_info_t &si0, const slice_info_t &si1);
slice_info_t get_out_slice_info(const shape_secs_t &shape_secs, int64_t n,
                                int64_t c, int64_t h, int64_t d, int64_t w,
                                int64_t bitwidth);
bool get_backward_slice_info(slice_info_t &in_si, const slice_info_t &out_si,
                             Operation *op, Value in,
                             const shape_secs_t &shape_secs,
                             group_type_t group_type, bool &hold_in_lmem,
                             bool is_group_in);
bool stripe_mine_max_slice(const LgInfo &lg_info,
                           const shape_secs_t &shape_secs,
                           TensorInfo &tensor_infos);

void get_max_slice_nchdw(const slice_info_t &slice_info, int64_t &max_nslice,
                         int64_t &max_cslice, int64_t &max_hslice,
                         int64_t &max_dslice, int64_t &max_wslice);
void assign_dhwsecs(const LgInfo &lg_info, shape_secs_t &shape_secs,
                    int64_t &dhw_secs, const shape_secs_t &max_shape_secs);

int64_t get_buffer_size(Value v, const tensor_info_t &ti,
                        group_type_t group_type);

bool stripe_mine_idx_slice(const LgInfo &lg_info,
                           const shape_secs_t &shape_secs,
                           TensorInfo &tensor_infos);

void set_fake_local_layer_param(Operation *op, int64_t nidx, int64_t nslice,
                                int64_t hidx, int64_t hslice, int64_t didx,
                                int64_t cidx, int64_t cslice, int64_t dslice,
                                int64_t widx, int64_t wslice);
void delete_fake_local_layer_param(Operation *op);

void set_weight_allow_split_attr(Operation *op);

void delete_weight_allow_split_attr(Operation *op);

void generate_fake_global_addr(Operation *op);

void delete_fake_global_addr(Operation *op);

bool is_eu_align(Value opd);

bool need_bcast(Value opd);

int64_t use_3ic(Value opd);

std::vector<Value> get_input_values(Operation *op);
std::vector<Value> get_output_values(Operation *op);

} // namespace tpu
} // namespace tpu_mlir
