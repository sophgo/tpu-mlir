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

#include "mlir/Support/LLVM.h"
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"
#include <list>
#include <map>
#include <set>
namespace tpu_mlir {
namespace tpu {

struct value_compare {
  bool operator()(Value v0, Value v1) const {
    if (module::getName(v0).str() < module::getName(v1).str()) {
      return true;
    }
    return false;
  }
};

struct op_compare {
  bool operator()(Operation *op0, Operation *op1) const {
    if (op0 < op1) {
      return true;
    }
    return false;
  }
};

typedef struct {
  int64_t nstep;
  int64_t hstep;
} tensor_step_t;

using TpuTsField = std::vector<Operation *>;
using GdmaTsField = std::vector<Value>;
typedef struct {
  TpuTsField tpu0_ts_field;
  GdmaTsField gdma0_ts_field;
} TimestepRow;

class SoftwarePipeline {
public:
  SoftwarePipeline();
  void clear_all();
  void clear_swloop_buffer();
  void write_swloop_buffer(int64_t nstep, int64_t hstep, int stage_num);
  const tensor_step_t *read_swloop_buffer(int stage);

  int software_pipeline_schedule(std::vector<TimestepRow> &timestep_table);
  int get_tensor_swpipl_stage(Value v);

private:
  std::list<tensor_step_t> tensor_swloop_buffer_;
  std::map<Value, int, value_compare> tensor_swpipl_stage;
};

class BasicTimeStep {
public:
  BasicTimeStep();
  virtual ~BasicTimeStep() {}

  void add_tpu0_ts_field(const TpuTsField &field);
  void add_gdma0_ts_field(const GdmaTsField &field);
  void add_tpu0_gdma0_ts_field(const TpuTsField &tpu_field,
                               const GdmaTsField &gdma_field);

  std::shared_ptr<SoftwarePipeline> get_timestep_swpipl() { return swpipl; }
  int get_layer_swpipl_stage(Operation *op);
  int get_tensor_swpipl_stage(Value v);
  void software_pipeline();

  const TpuTsField &getOps(int ts) { return timestep_table[ts].tpu0_ts_field; }
  const GdmaTsField &getValues(int ts) {
    return timestep_table[ts].gdma0_ts_field;
  }

  int get_swpipl_stage_num() { return swpipl_stage_num; }
  int get_timestep_num() { return (int)timestep_table.size(); }
  void show_timestep();
  void clear();

  void gen_all_mem_buffer();

protected:
  std::shared_ptr<SoftwarePipeline> swpipl;
  std::vector<TimestepRow> timestep_table;
  int swpipl_stage_num;
};

} // namespace tpu
} // namespace tpu_mlir
