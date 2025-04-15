//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/GroupOps.h"
#include <fstream>
#include <iostream>

using namespace llvm;

namespace tpu_mlir {
namespace tpu {

bool force_group_by_cores(const std::string &option) {
  if (module::getCoreNum() < 2) {
    return false;
  }
  if (option == "true") {
    return true;
  } else if (option == "auto" || option == "false") {
    // auto as false
    return false;
  }
  llvm_unreachable("Unknown layer group options");
  return true;
}

bool force_lgcache(const std::string &option) {
  if (option == "true") {
    return true;
  } else if (option == "false") {
    // auto as false
    return false;
  }
  llvm_unreachable("Unknown layer group options");
  return true;
}

NnvlcMode force_nnvlc_mode(const std::string &nnvlc_mode) {
  if (nnvlc_mode == "none") {
    return NnvlcMode::NONE;
  } else if (nnvlc_mode == "weight") {
    return NnvlcMode::WEIGHT;
  } else if (nnvlc_mode == "activation") {
    return NnvlcMode::ACTIVATION;
  } else if (nnvlc_mode == "all") {
    return NnvlcMode::ALL;
  } else {
    llvm_unreachable("Unknown nnvlc mode");
  }
}

class LayerGroupPass : public LayerGroupBase<LayerGroupPass> {
public:
  LayerGroupPass() {}
  void runOnOperation() override {
    if (module::isDebugCmdEnable("disable_layer_group")) {
      return;
    }
    // init global options
    LgOptions options;
    options.opt = opt;
    options.group_by_cores = force_group_by_cores(group_by_cores);
    options.nnvlc_mode = force_nnvlc_mode(compress_mode);
    options.lgcache = force_lgcache(lgcache);
    options.num_core = module::getCoreNum();
    options.debugger = debugger;
    options.debugger_filename = debugger_filename;
    options.disable_group_overlap = disable_group_overlap;
    // group pass by modules
    auto modules = module::getAllModules();
    for (auto s : *modules) {
      for (auto f : s.getOps<FuncOp>()) {
        if (f.getName() == "main") {
          continue;
        }
        GroupOps gOps(f, options);
        gOps.process();
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createLayerGroupPass() {
  return std::make_unique<LayerGroupPass>();
}

class NetStatisticPass : public NetStatisticBase<NetStatisticPass> {
public:
  NetStatisticPass() {}
  void runOnOperation() override {
    auto modules = module::getAllModules();
    int64_t total_bytes = 0, total_bytes2 = 0;
    // std::ofstream file("group_after.txt", std::ios::ate | std::ios::out);
    for (auto s : *modules) {
      for (auto f : s.getOps<FuncOp>()) {
        if (f.getName() != "subfunc_0") {
          continue;
        }
        std::vector<Operation *> exclude_ops, exclude_ops2;
        f.walk([&](Operation *op) {
          if (isa<FuncOp, top::NoneOp, top::WeightOp>(op)) {
            // do nothing
          } else {
            if (std::find(exclude_ops.begin(), exclude_ops.end(), op) ==
                exclude_ops.end()) {
              if (isa<tpu::GroupOp>(op)) {
                auto nextOp = *(op->getUsers().begin());
                if (isa<tpu::SliceMergeOp>(nextOp)) {
                  for (auto v : nextOp->getOperands()) {
                    exclude_ops.push_back(v.getDefiningOp());
                  }
                }
              }

              if (!module::isOpInGroup(op) && !module::isOpInCoreParallel(op)) {
                for (auto v : op->getOperands()) {
                  if (v.getType().isa<NoneType>()) {
                    continue;
                  }
                  if (!v.getDefiningOp() || isa<ReturnOp>(op)) {
                    auto width =
                        module::getStorageType(v).getIntOrFloatBitWidth() / 8;
                    total_bytes +=
                        v.getType().cast<RankedTensorType>().getNumElements() *
                        width;
                  }
                }
              }
            }

            if (module::isOpInGroup(op)) {
              auto parent = op->getParentOp();
              if (std::find(exclude_ops2.begin(), exclude_ops2.end(), parent) ==
                  exclude_ops2.end()) {
                auto nextOp = *(parent->getUsers().begin());
                if (isa<tpu::SliceMergeOp>(nextOp)) {
                  for (auto v : nextOp->getOperands()) {
                    if (v.getDefiningOp() != parent) {
                      exclude_ops2.push_back(v.getDefiningOp());
                    }
                  }
                }
                if (!isa<tpu::ReshapeOp, tpu::LoadOp, tpu::StoreOp,
                         tpu::MoveOp>(op)) {
                  for (auto v : op->getOperands()) {
                    if (v.getType().isa<NoneType>()) {
                      continue;
                    }
                    if (module::isOpInGroup(v.getDefiningOp())) {
                      if (!isa<tpu::LoadOp>(v.getDefiningOp())) {
                        auto width =
                            module::getStorageType(v).getIntOrFloatBitWidth() /
                            8;
                        total_bytes2 += v.getType()
                                            .cast<RankedTensorType>()
                                            .getNumElements() *
                                        width;
                      }
                    }
                  }
                }
              }
            }
          }
        });
      }
    }
    // file.close();
    llvm::outs() << "NetStatisticPass total_bytes: " << total_bytes
                 << ", total_bytes2: " << total_bytes2 << "\n";
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createNetStatisticPass() {
  return std::make_unique<NetStatisticPass>();
}
} // namespace tpu
} // namespace tpu_mlir
