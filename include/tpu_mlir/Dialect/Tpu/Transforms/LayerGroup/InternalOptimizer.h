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

  virtual std::string brief()
    { return "This is the internal optimizer of layer group"; }
};

} // namespace tpu
} // namespace tpu_mlir
