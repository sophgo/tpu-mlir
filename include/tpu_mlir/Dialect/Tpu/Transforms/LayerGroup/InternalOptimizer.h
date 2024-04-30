//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LgPass.h"

namespace tpu_mlir {
namespace tpu {

class InternalLgOptimizer : public LgOptimizer {
public:
  InternalLgOptimizer() {}
  virtual ~InternalLgOptimizer() {}

  virtual void manage_passes(std::shared_ptr<LgPassManager> pm,
                             const LgOptions& options) override;

  // Manage post passes after plugin
  void manage_post_passes(std::shared_ptr<LgPassManager> pm,
                          const LgOptions& options);

  virtual std::string brief() override {
    return "This is the internal optimizer of layer group";
  }
};

} // namespace tpu
} // namespace tpu_mlir
