
#pragma once

#include "mlir/Support/LLVM.h"
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"
#include <list>
#include <map>
#include <set>

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/BasicTimeStep.h"

namespace tpu_mlir {
namespace tpu {

class CycleCalculator {
public:
  CycleCalculator() {}
  int64_t getGlobalLayerCycle(Operation *op);
  int64_t getGroupCycle(BasicTimeStepPtr &time_step, shape_secs_t &shape_secs);
  int64_t getLocalLayerCycle(Operation *op, TensorInfo &tensor_infos,
                             bool calc_bdc_slack);
  int64_t getGdmaCycle(Value v, const tensor_info_t &tensor_info);
  int64_t getLoadCycle(Value v, const tensor_info_t &tensor_info);
  int64_t getStoreCycle(Value v, const tensor_info_t &tensor_info);

protected:
  void set_local_sec_info(local_sec_info_t *sec_info, Operation *op, TensorInfo &tensor_infos);
};

} // namespace tpu
} // namespace tpu_mlir
