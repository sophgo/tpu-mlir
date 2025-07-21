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
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/CoreParallel/CoreParallel.hpp"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/BasicTimeStep.h"
#include "tpu_mlir/Support/Module.h"
#include <list>
#include <map>
#include <set>

namespace tpu_mlir {
namespace tpu {
struct group_cycle_info_t {
  int64_t out_addr;
  int64_t out_size;
  int64_t buffer_addr;
  int64_t buffer_size;
  int64_t n_idx;
  int64_t n_slice;
  int64_t c_idx;
  int64_t c_slice;
  int64_t h_idx;
  int64_t h_slice;
  int64_t d_idx;
  int64_t d_slice;
  int64_t w_idx;
  int64_t w_slice;
  int64_t h_idx_offset;
  int64_t id;
  int64_t stage;
  int64_t type;
  bool eu_align;
  bool overstepped;
};

class CycleCalculator {
public:
  CycleCalculator() : num_core_(module::getCoreNum()){};
  CycleCalculator(int num_core) : num_core_(num_core){};
  virtual ~CycleCalculator(){};
  virtual int64_t getGlobalLayerCycle(Operation *op) = 0;
  virtual int64_t getGroupCycle(BasicTimeStepPtr &time_step,
                                shape_secs_t &shape_secs,
                                group_type_t group_type);
  virtual int64_t getLocalLayerCycle(Operation *op, TensorInfo &tensor_infos,
                                     group_type_t group_type,
                                     bool calc_bdc_slack) = 0;
  virtual int64_t getGdmaCycle(Value v, tensor_info_t &tensor_info,
                               group_type_t group_type,
                               Operation *owner_op = nullptr, int mode = 0) = 0;
  virtual int64_t getLoadCycle(Value v, tensor_info_t &tensor_info,
                               group_type_t group_type,
                               Operation *owner_op = nullptr) = 0;
  virtual int64_t getStoreCycle(Value v, const tensor_info_t &tensor_info,
                                group_type_t group_type) = 0;

protected:
  void set_local_sec_info(local_sec_info_t &sec_info, Operation *op,
                          TensorInfo &tensor_infos, group_type_t group_type);
  int num_core_;
};

class Bm168xCycleCalculator : public CycleCalculator {
public:
  Bm168xCycleCalculator() {}
  Bm168xCycleCalculator(int num_core) : CycleCalculator(num_core) {}
  ~Bm168xCycleCalculator() {}
  int64_t getGroupCycle(BasicTimeStepPtr &time_step, shape_secs_t &shape_secs,
                        group_type_t group_type) override;
  int64_t getGlobalLayerCycle(Operation *op) override;
  int64_t getLocalLayerCycle(Operation *op, TensorInfo &tensor_infos,

                             group_type_t group_type,
                             bool calc_bdc_slack) override;
  int64_t getGdmaCycle(Value v, tensor_info_t &tensor_info,
                       group_type_t group_type, Operation *owner_op = nullptr,
                       int mode = 0) override;
  int64_t getLoadCycle(Value v, tensor_info_t &tensor_info,
                       group_type_t group_type,
                       Operation *owner_op = nullptr) override;
  int64_t getStoreCycle(Value v, const tensor_info_t &tensor_info,
                        group_type_t group_type) override;
  group_cycle_info_t getGdmaGroupInfo(Value v, tensor_info_t &tensor_info,
                                      group_type_t group_type, int64_t n_step,
                                      int64_t c_step, int64_t d_step,
                                      int64_t h_step, int64_t w_step,
                                      int64_t l_addr);
  int64_t getLoadCycleOpt(Value v, tensor_info_t &tensor_info,
                          group_type_t group_type, group_cycle_info_t &ginfo);
  int64_t getStoreCycleOpt(Value v, tensor_info_t &tensor_info,
                           group_type_t group_type, group_cycle_info_t &ginfo);
  int64_t getGdmaCycleOpt(Value v, tensor_info_t &tensor_info,
                          group_type_t group_type, group_cycle_info_t &ginfo);
  int64_t getLocalLayerCycleOpt(BasicTimeStepPtr &time_step, Operation *op,
                                TensorInfo &tensor_infos,
                                group_type_t group_type, bool calc_bdc_slack,
                                int64_t n_step = 0, int64_t c_step = 0,
                                int64_t d_step = 0, int64_t h_step = 0,
                                int64_t w_step = 0);
};

class Cv18xxCycleCalculator : public CycleCalculator {
public:
  Cv18xxCycleCalculator() {}
  Cv18xxCycleCalculator(int num_core) : CycleCalculator(num_core) {}
  ~Cv18xxCycleCalculator() {}
  int64_t getGlobalLayerCycle(Operation *op) override;
  int64_t getLocalLayerCycle(Operation *op, TensorInfo &tensor_infos,
                             group_type_t group_type,
                             bool calc_bdc_slack) override;
  int64_t getGdmaCycle(Value v, tensor_info_t &tensor_info,
                       group_type_t group_type, Operation *owner_op = nullptr,
                       int mode = 0) override;
  int64_t getLoadCycle(Value v, tensor_info_t &tensor_info,
                       group_type_t group_type,
                       Operation *owner_op = nullptr) override;
  int64_t getStoreCycle(Value v, const tensor_info_t &tensor_info,
                        group_type_t group_type) override;

private:
  bool check_lmem(Operation *op, const TensorInfo &tesnor_info,
                  group_type_t group_type);
};
} // namespace tpu
} // namespace tpu_mlir
