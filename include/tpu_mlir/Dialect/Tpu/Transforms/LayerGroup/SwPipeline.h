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
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupDefs.h"
#include <list>
#include <map>
#include <set>

namespace tpu_mlir {
namespace tpu {

class SoftwarePipeline {
public:
  SoftwarePipeline();
  void clear_all();
  void clear_swloop_buffer();
  void write_swloop_buffer(int64_t nstep, int64_t cstep, int64_t hstep,
                           int64_t dstep, int64_t wstep, int64_t stage_num);
  const tensor_step_t *read_swloop_buffer(int64_t stage);

  int64_t software_pipeline_schedule(std::vector<TimestepRow> &timestep_table);
  int64_t get_tensor_swpipl_stage(Value v);

private:
  std::list<tensor_step_t> tensor_swloop_buffer_;
  std::map<Value, int64_t, value_compare> tensor_swpipl_stage_;
};

} // namespace tpu
} // namespace tpu_mlir
