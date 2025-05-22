//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
// #include "ProcessorOptimize/json.hpp"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"
#include <fstream>
#include <iostream>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#define DEBUG_TYPE "shape_infer"

// using json = nlohmann::json;
using namespace llvm;
using namespace llvm::json;

namespace tpu_mlir {
namespace top {

class PruningPass : public PruningBase<PruningPass> {
private:
  void pruning_mm(MatMulOp op, int idx, int prun_dim,
                  std::vector<int> pruned_channel_in) {

    top::WeightOp wOp = dyn_cast<top::WeightOp>(op.getRight().getDefiningOp());
    auto data = wOp.read_as_float();
    int cutted_chls = pruned_channel_in.size();
    int dim = prun_dim;
    auto w_shape = module::getShape(wOp.getOutput());
    std::vector<int64_t> new_shape = {w_shape[0], w_shape[1]};
    new_shape[dim] -= cutted_chls;
    int off_nums = 0;
    std::vector<float> to_data(data->size(), 0);
    std::vector<float> new_data(new_shape[dim] * new_shape[1 - dim], 0);
    std::vector<int> pruned_channel;
    pruned_channel.push_back(-1);
    for (int p_chl : pruned_channel_in) {
      pruned_channel.push_back(p_chl);
    }
    pruned_channel.push_back(w_shape[1]);

    if (dim == 0) {
      std::vector<int64_t> order{0, 1};
      function_permute(data->data(), to_data.data(), w_shape, order);
      for (int k = 1; k < pruned_channel.size(); k++) {
        if (pruned_channel[k] - pruned_channel[k - 1] > 1) {
          std::copy(to_data.begin() +
                        (w_shape[1] * (pruned_channel[k - 1] + 1)),
                    to_data.begin() + (w_shape[1] * (pruned_channel[k])),
                    (new_data.begin() + off_nums));
          off_nums +=
              (w_shape[1] * (pruned_channel[k] - pruned_channel[k - 1] - 1));
        }
      }
    } else {
      std::vector<int64_t> order{1, 0};
      std::vector<float> new_data_copy(new_shape[dim] * new_shape[1 - dim], 0);
      function_permute(data->data(), to_data.data(), w_shape, order);
      for (int k = 1; k < pruned_channel.size(); k++) {
        if (pruned_channel[k] - pruned_channel[k - 1] > 1) {
          std::copy(to_data.begin() +
                        (w_shape[0] * (pruned_channel[k - 1] + 1)),
                    to_data.begin() + (w_shape[0] * (pruned_channel[k])),
                    (new_data_copy.begin() + off_nums));
          off_nums +=
              (w_shape[0] * (pruned_channel[k] - pruned_channel[k - 1] - 1));
        }
      }
      std::vector<int64_t> trans_shape = {new_shape[1], new_shape[0]};
      function_permute(new_data_copy.data(), new_data.data(), trans_shape,
                       order);
    }
    auto reop_out_type = module::getStorageType(wOp.getOutput());
    auto new_type = RankedTensorType::get(new_shape, reop_out_type);
    auto new_op = top::WeightOp::create(wOp, "weight1", new_data, new_type);
    wOp.replaceAllUsesWith(new_op.getDefiningOp());

    return;
  }

public:
  PruningPass() {}
  void runOnOperation() override {
    std::string config_path = this->config;
    if (config_path.empty()) {
      return;
    }
    auto MBOrErr = MemoryBuffer::getFile(config_path);
    if (!MBOrErr) {
      errs() << "Error int get jsonfile\n";
      return;
    }
    std::unique_ptr<MemoryBuffer> MB = std::move(*MBOrErr);
    auto JOrErr = parse(MB->getBuffer());
    if (!JOrErr) {
      errs() << "Error parsing JSON: " << toString(JOrErr.takeError()) << "\n";
      return;
    }
    auto &Root = *JOrErr;
    auto *Obj = Root.getAsObject();
    auto mOp = getOperation();
    auto modules = module::getAllModules();
    auto Main = module::getMainFuncOp(mOp);
    auto mm_idx = 0;
    Main.walk([&](Operation *op) {
      if (auto mOp = dyn_cast<top::MatMulOp>(op)) {
        if (auto wOp =
                dyn_cast<top::WeightOp>(mOp.getRight().getDefiningOp())) {
          for (auto value : *Obj->getArray("MatMul")) {
            auto obj_k = value.getAsObject();
            if (*obj_k->getInteger("idx") == mm_idx) {
              int prun_dim = int(*obj_k->getInteger("prun_dim"));
              std::vector<int> nums;
              auto chls = *obj_k->getArray("pruned_channel");
              for (auto chl : chls) {
                nums.push_back(static_cast<int>(chl.getAsInteger().value()));
              }
              pruning_mm(mOp, mm_idx, prun_dim, nums);
            }
          }
          mm_idx += 1;
        } else {
        }
      }
    });
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createPruningPass() {
  return std::make_unique<PruningPass>();
}
} // namespace top
} // namespace tpu_mlir
