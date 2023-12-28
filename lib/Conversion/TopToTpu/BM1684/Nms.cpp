//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"

namespace tpu_mlir {
namespace bm1684 {

void NmsLowering::LoweringF32(PatternRewriter &rewriter, top::NmsOp op) const {
  std::vector<NamedAttribute> attrs;
  std::vector<NamedAttribute> cpu_param;
  attrs.emplace_back(
      rewriter.getNamedAttr("cpu_op_name", rewriter.getStringAttr("onnx_nms")));

  for (auto &attr : op->getAttrs()) {
    cpu_param.push_back(attr);
  }

  attrs.emplace_back(
      rewriter.getNamedAttr("param", rewriter.getDictionaryAttr(cpu_param)));
  std::vector<Type> new_types;
  const auto shape = module::getShape(op.getOutput());
  const auto new_type = RankedTensorType::get(shape, rewriter.getI32Type());
  new_types.push_back(new_type);
  std::vector<Value> operands;
  for (auto v : llvm::enumerate(op->getOperands())) {
    if (v.index() != 2) {
      operands.push_back(v.value());
    } else {
      // convert fp32 into int32
      std::shared_ptr<std::vector<float>> max;
      i32_array_t max_int32;
      auto max_op = v.value().getDefiningOp<top::WeightOp>();
      max = max_op.read<float>();
      std::vector<int32_t> max_int32_v(1, 1);
      max_int32_v[0] = static_cast<int32_t>(max->at(0));
      max_int32 = std::make_shared<std::vector<int32_t>>(max_int32_v);
      std::vector<int64_t> max_shape(1, 1);
      auto new_type = RankedTensorType::get(max_shape, rewriter.getI32Type());
      // push_back into operands
      auto new_max_op =
          top::WeightOp::create(op, "max_int32", *max_int32, new_type);
      operands.push_back(new_max_op);
    }
  }
  rewriter.replaceOpWithNewOp<tpu::GenericCpuOp>(op, new_types, operands,
                                                 attrs);
}

void NmsLowering::LoweringINT8(PatternRewriter &rewriter, top::NmsOp op,
                               bool asymmetric) const {
  LoweringF32(rewriter, op);
}

} // namespace bm1684
} // namespace tpu_mlir
