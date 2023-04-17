//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/InternalOptimizer.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/GroupMethod.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/GroupPostTransform.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/TimeStepMethod.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LmemAllocator.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/TimeStepCombine.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/GroupOverlap.h"

// static bool use_partial_coeff_reload = true;

namespace tpu_mlir {
namespace tpu {

void InternalLgOptimizer::manage_passes(std::shared_ptr<LgPassManager> pm,
                                        const LgOptions &options) {
  /*
   * Add internal pass pipeline to the pass manager.
   */

  // Firstly, group layers
  pm->add_pass(CreateLayerGroupSearchPass(options));

  // Some transform after layer groups is determined
  pm->add_pass(CreateGroupPostTransformPass());

  // Then, time step assignment
  pm->add_pass(CreateTimeStepAssignmentPass());

  // Then, split the data
  // pm->add_pass(CreateDataSplitPass());

  // Then, allocate local memory for each layer group
  pm->add_pass(CreateLocalMemoryAllocationPass());

  // Decrease coeff reload if it is opened
  // if (use_partial_coeff_reload) {
  //   pm->add_pass(CreateCoeffReloadDereasePass());
  // }

  // Time step combination if it is opened
  pm->add_pass(CreateTimeStepCombinePass());
}

void InternalLgOptimizer::manage_post_passes(std::shared_ptr<LgPassManager> pm,
                                             const LgOptions &options) {
  pm->add_pass(CreateGroupDataMoveOverlapPass());
                                             }

} // namespace tpu
} // namespace tpu_mlir
