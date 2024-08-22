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
#include "tpu_mlir/Support/Module.h"
#include <list>
#include <map>
#include <set>
#include "tpu_mlir/Dialect/Tpu/Transforms/CoreParallel/CoreParallel.hpp"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/BasicTimeStep.h"

namespace tpu_mlir {
namespace tpu {

class CycleCalculator {
public:
  CycleCalculator(){};
  virtual ~CycleCalculator(){};
  virtual int64_t getGlobalLayerCycle(Operation *op) = 0;
  int64_t getGroupCycle(BasicTimeStepPtr &time_step, shape_secs_t &shape_secs,
                        group_type_t group_type);
  virtual int64_t getLocalLayerCycle(Operation *op, TensorInfo &tensor_infos,
                                     group_type_t group_type,
                                     bool calc_bdc_slack) = 0;
  virtual int64_t getGdmaCycle(Value v, tensor_info_t &tensor_info,
                               group_type_t group_type,
							   Operation* owner_op = nullptr, int mode = 0) = 0;
  virtual int64_t getLoadCycle(Value v, tensor_info_t &tensor_info,
                               group_type_t group_type, Operation* owner_op = nullptr) = 0;
  virtual int64_t getStoreCycle(Value v, const tensor_info_t &tensor_info,
                                group_type_t group_type) = 0;

protected:
  void set_local_sec_info(local_sec_info_t &sec_info, Operation *op,
                          TensorInfo &tensor_infos, group_type_t group_type);
};

class Bm168xCycleCalculator : public CycleCalculator {
public:
  Bm168xCycleCalculator() {}
  ~Bm168xCycleCalculator() {}
  int64_t getGlobalLayerCycle(Operation *op) override;
  int64_t getLocalLayerCycle(Operation *op, TensorInfo &tensor_infos,

                             group_type_t group_type,
                             bool calc_bdc_slack) override;
  int64_t getGdmaCycle(Value v, tensor_info_t &tensor_info,
                       group_type_t group_type,
					   Operation* owner_op = nullptr, int mode = 0) override;
  int64_t getLoadCycle(Value v, tensor_info_t &tensor_info,
                       group_type_t group_type, Operation* owner_op = nullptr) override;
  int64_t getStoreCycle(Value v, const tensor_info_t &tensor_info,
                        group_type_t group_type) override;
};

class Cv18xxCycleCalculator : public CycleCalculator {
public:
  Cv18xxCycleCalculator() {}
  ~Cv18xxCycleCalculator() {}
  int64_t getGlobalLayerCycle(Operation *op) override;
  int64_t getLocalLayerCycle(Operation *op, TensorInfo &tensor_infos,
                             group_type_t group_type,
                             bool calc_bdc_slack) override;
  int64_t getGdmaCycle(Value v, tensor_info_t &tensor_info,
                       group_type_t group_type,
					   Operation* owner_op = nullptr, int mode = 0) override;
  int64_t getLoadCycle(Value v, tensor_info_t &tensor_info,
                       group_type_t group_type, Operation* owner_op = nullptr) override;
  int64_t getStoreCycle(Value v, const tensor_info_t &tensor_info,
                        group_type_t group_type) override;

private:
  bool check_lmem(Operation *op, const TensorInfo &tesnor_info,
                  group_type_t group_type);
};
} // namespace tpu
} // namespace tpu_mlir
