//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../WeightReorder.h"
#include "tpu_mlir/Backend/Arch.h"

using namespace bm1684x;

template <>
LogicalResult WeightReorder<tpu::A16MatMulOp, Float16Type>::matchAndRewrite(
    tpu::A16MatMulOp op, PatternRewriter &rewriter) const {
  if (op.getWeightBits() != 4 || op.getQGroupSize() <= 0) {
    return failure();
  }
  auto scale_stype = module::getStorageType(op.getScale());
  auto scaleOp = op.getScale().getDefiningOp<top::WeightOp>();
  auto zpOp = op.getZp().getDefiningOp<top::WeightOp>();
  auto zp_stype = module::getStorageType(op.getZp());
  auto scale_shape = scaleOp.getType().getShape();
  auto ori_scale_data = scaleOp.read<uint16_t>();
  auto ori_zp_data = zpOp.read<uint8_t>();

  auto new_scale_data = std::make_shared<std::vector<uint16_t>>(
      scale_shape[0] * scale_shape[1], 0);
  auto new_zp_data = std::make_shared<std::vector<uint8_t>>(
      scale_shape[0] * scale_shape[1], 0);
  int64_t npu_num = backend::Arch::NPU_NUM;
  if (scale_shape[0] % npu_num) {
    llvm_unreachable("invalid scale channel");
  }
  auto w = scale_shape[1];
  auto h = scale_shape[0] / npu_num;

  for (auto i = 0; i < npu_num; i++) {
    for (auto j = 0; j < h; j++) {
      auto offset_new = i * h * w + j * w;
      auto offset_ori = i * w + npu_num * j * w;
      memcpy(new_scale_data->data() + offset_new,
             ori_scale_data->data() + offset_ori, w * sizeof(uint16_t));
      memcpy(new_zp_data->data() + offset_new, ori_zp_data->data() + offset_ori,
             w * sizeof(uint8_t));
    }
  }

  auto new_scale_type = RankedTensorType::get({npu_num, h, w}, scale_stype);
  auto new_zp_type = RankedTensorType::get({npu_num, h, w}, zp_stype);
  auto new_scaleOp =
      top::WeightOp::create(op, "reordered", *new_scale_data, new_scale_type);
  auto new_zpOp =
      top::WeightOp::create(op, "reordered", *new_zp_data, new_zp_type);
  op.setOperand(2, new_scaleOp);
  op.setOperand(3, new_zpOp);

  return success();
}
