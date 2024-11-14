//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "AddressAssign/BMAddressAssign.h"
#include "AddressAssign/CVAddressAssign.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"
using namespace llvm;

namespace tpu_mlir {
namespace tpu {
static void WeightFolder(Operation *op) {
  // if the op is in the region of other op, don't do WeightFolder

  if (isa<tpu::IfOp, tpu::LoopOp, top::LoopOp, top::IfOp>(
          op->getBlock()->getParentOp()))
    return;
  if (module::isAllWeight(op) == false) {
    return;
  }
  // avoid Weight2ActivationOp -> Gather be folded as Weight -> Gather
  if (isa<tpu::Weight2ActivationOp, tpu::Device2HostOp, tpu::ShapeCastOp,
          tpu::ShapePackOp>(op)) {
    return;
  }
  auto infer = dyn_cast<InferenceInterface>(op);
  if (!infer) {
    return;
  }
  for (auto user : op->getUsers()) {
    if (isa<ReturnOp>(user)) {
      return;
    }
  }
  auto ins = op->getOperands();
  auto outs = op->getResults();
  auto num_in = ins.size();
  auto num_out = outs.size();
  std::vector<float> datas[num_out];
  for (int i = 0; i < num_out; i++) {
    if (module::isNone(outs[i])) {
      continue;
    }
    auto num_elem = module::getNumElements(outs[i]);
    datas[i].assign(num_elem, 0.0f);
  }
  std::vector<std::shared_ptr<std::vector<float>>> inputs;
  for (int i = 0; i < num_in; i++) {
    if (false == module::isWeight(ins[i])) {
      inputs.push_back(nullptr);
      continue;
    }
    auto in_op = cast<top::WeightOp>(ins[i].getDefiningOp());

    auto d = in_op.read_as_float();
    inputs.push_back(d);
  }
  InferenceParameter p;
  for (int i = 0; i < num_in; i++) {
    if (inputs[i] == nullptr) {
      p.inputs.push_back(nullptr);
    } else {
      p.inputs.push_back(inputs[i]->data());
    }
  }
  for (int i = 0; i < num_out; i++) {
    p.outputs.push_back(datas[i].data());
  }
  auto ret = infer.init(p);
  assert(mlir::succeeded(ret));
  ret = infer.inference(p);
  assert(mlir::succeeded(ret));
  infer.deinit(p);
  OpBuilder builder(module::getCtx());
  builder.setInsertionPointAfter(op);
  for (int i = 0; i < num_out; i++) {
    if (datas[i].empty()) {
      continue;
    }
    std::string suffix = std::string("folder_") + std::to_string(i);
    auto out = outs[i];
    auto out_type = out.getType().cast<RankedTensorType>();
    Value new_op;

    auto dtype = module::getStorageType(out);
    if (dtype.isUnsignedInteger(8)) {
      auto castData = std::make_shared<std::vector<uint8_t>>(datas[i].size());
      std::transform(datas[i].begin(), datas[i].end(), (*castData).begin(),
                     [](float c) { return c; });
      new_op = top::WeightOp::create(op, "folder", *castData, out_type);
    } else if (dtype.isInteger(8)) {
      auto castData = std::make_shared<std::vector<int8_t>>(datas[i].size());
      std::transform(datas[i].begin(), datas[i].end(), (*castData).begin(),
                     [](float c) { return c; });
      new_op = top::WeightOp::create(op, "folder", *castData, out_type);
    } else if (dtype.isF32()) {
      new_op = top::WeightOp::create(op, "folder", datas[i], out_type);
    } else if (dtype.isF16()) {
      auto castData = std::make_shared<std::vector<uint16_t>>(datas[i].size());
      std::transform(datas[i].begin(), datas[i].end(), (*castData).begin(),
                     [](float c) { return f32_to_f16(c); });
      new_op = top::WeightOp::create(op, "folder", *castData, out_type);
    } else if (dtype.isBF16()) {
      auto castData = std::make_shared<std::vector<uint16_t>>(datas[i].size());
      std::transform(datas[i].begin(), datas[i].end(), (*castData).begin(),
                     [](float c) { return f32_to_bf16(c); });
      new_op = top::WeightOp::create(op, "folder", *castData, out_type);
    } else if (dtype.isUnsignedInteger(16)) {
      auto castData = std::make_shared<std::vector<uint16_t>>(datas[i].size());
      std::transform(datas[i].begin(), datas[i].end(), (*castData).begin(),
                     [](float c) { return c; });
      new_op = top::WeightOp::create(op, "folder", *castData, out_type);
    } else if (dtype.isInteger(16)) {
      auto castData = std::make_shared<std::vector<int16_t>>(datas[i].size());
      std::transform(datas[i].begin(), datas[i].end(), (*castData).begin(),
                     [](float c) { return c; });
      new_op = top::WeightOp::create(op, "folder", *castData, out_type);
    } else if (dtype.isUnsignedInteger(32)) {
      auto castData = std::make_shared<std::vector<uint32_t>>(datas[i].size());
      std::transform(datas[i].begin(), datas[i].end(), (*castData).begin(),
                     [](float c) { return c; });
      new_op = top::WeightOp::create(op, "folder", *castData, out_type);
    } else if (dtype.isInteger(32)) {
      auto castData = std::make_shared<std::vector<int32_t>>(datas[i].size());
      std::transform(datas[i].begin(), datas[i].end(), (*castData).begin(),
                     [](float c) { return c; });
      new_op = top::WeightOp::create(op, "folder", *castData, out_type);
      // } else if (auto quantType =
      //                dtype.dyn_cast<mlir::quant::UniformQuantizedType>()) {
      //   auto o_storageType_ = quantType.getStorageType();
      //   if (o_storageType_.isUnsignedInteger(8)) {
      //     auto castData =
      //     std::make_shared<std::vector<uint8_t>>(datas[i].size()); float
      //     scale = quantType.getScale(); int64_t zeroPoint =
      //     quantType.getZeroPoint(); std::transform(datas[i].begin(),
      //     datas[i].end(), castData->begin(),
      //                    [scale, zeroPoint, o_storageType_](float value) {
      //                      return static_cast<uint8_t>(saturate<float>(
      //                          (float)(value / scale + zeroPoint),
      //                          o_storageType_));
      //                    });
      //     new_op = top::WeightOp::create(op, "folder", *castData, out_type);
      //   } else {
      //       auto castData =
      //           std::make_shared<std::vector<int8_t>>(datas[i].size());
      //       float scale = quantType.getScale();
      //       int64_t zeroPoint = quantType.getZeroPoint();
      //       std::transform(datas[i].begin(), datas[i].end(),
      //       castData->begin(),
      //                      [scale, zeroPoint, o_storageType_](float value) {
      //                        return static_cast<int8_t>(saturate<float>(
      //                            (float)(value / scale + zeroPoint),
      //                            o_storageType_));
      //                      });
      //       new_op = top::WeightOp::create(op, "folder", *castData,
      //       out_type);
      //   }
    } else {
      UNREACHABLE_OP("Not supported type for weight folding",
                     out.getDefiningOp());
    }

    out.replaceAllUsesWith(new_op);
  }
}

class WeightFoldPass : public WeightFoldBase<WeightFoldPass> {
public:
  WeightFoldPass() {}
  void runOnOperation() override {
    auto mOp = getOperation();

    for (auto func : mOp.getOps<FuncOp>()) {
      func.walk([&](InferenceInterface op) {
        // LLVM_DEBUG(llvm::dbgs() << "weight infer: " << op << "\n";);
        if (false == removeIfNoUse(op)) {
          WeightFolder(op);
          removeIfNoUse(op);
        }
      });
    }
  }
  bool removeIfNoUse(Operation *op) {
    // if the op is in the region of other op, don't do removeIfNoUse
    if (op->getUsers().empty() &&
        !isa<tpu::IfOp, tpu::LoopOp, top::LoopOp, top::IfOp>(
            op->getBlock()->getParentOp())) {
      op->erase();
      return true;
    }
    return false;
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createWeightFoldPass() {
  return std::make_unique<WeightFoldPass>();
}
} // namespace tpu
} // namespace tpu_mlir
