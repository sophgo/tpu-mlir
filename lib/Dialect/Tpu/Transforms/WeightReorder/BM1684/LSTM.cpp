//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../WeightReorder.h"

using namespace bm1684;

// ======================================
// WeightReorderInterface
// ======================================
// W =  [num_directions, 4, hidden_size, input_size]
// R =  [num_directions, 4, hidden_size, hidden_size]
// => W0 = [num_directions, input_size,4, hidden_size] (0,3,2,1)
//    R0 = [num_directions, hidden_size,4, hidden_size] (0,3,2,1)
// => Merge in axis num_directions
// i o f g => i f o g
static void filter_merge(std::shared_ptr<std::vector<float>> &filter,
                         std::shared_ptr<std::vector<float>> &W,
                         std::shared_ptr<std::vector<float>> &R, int num_dir,
                         int input_size, int hidden_size) {
  int w_size = input_size * hidden_size;
  int r_size = hidden_size * hidden_size;
  int w_offset = 0, r_offset = 0;
  for (int d = 0; d < num_dir; d++) {
    // apple W
    for (int i = 0; i < 4; i++) {
      for (int h = 0; h < hidden_size; h++) {
        for (int x = 0; x < input_size; x++) {
          // gate i o f g => i f o g
          int gate = (i == 1) ? 2 : (i == 2 ? 1 : i);
          int dst_offset = d * 4 * (w_size + r_size) + gate * hidden_size +
                           x * 4 * hidden_size + h;
          filter->at(dst_offset) = W->at(w_offset);
          w_offset++;
        }
      }
    }

    // apply R
    for (int i = 0; i < 4; i++) {
      for (int h = 0; h < hidden_size; h++) {
        for (int x = 0; x < hidden_size; x++) {
          // gate i o f g => i f o g
          int gate = (i == 1) ? 2 : (i == 2 ? 1 : i);
          int dst_offset = d * 4 * (w_size + r_size) + 4 * w_size +
                           gate * hidden_size + x * 4 * hidden_size + h;
          filter->at(dst_offset) = R->at(r_offset);
          r_offset++;
        }
      }
    }
  }
}

// bias [num_dir, 8, hidden_size]
// onnx: i o f g
// pytorch: i f g o
// for comput easy, 1684&1684x pytorch lstm reshaped as: i f o g
// so need reorder as: i o f g => i f o g
static void iofg2ifog(std::shared_ptr<std::vector<float>> &filter, int num_dir,
                      int hsize) {
  auto filter_new = std::make_shared<std::vector<float>>(filter->size(), 0);
  int older[8] = {0, 2, 1, 3, 4, 6, 5, 7};
  for (int d = 0; d < num_dir; d++) {
    for (int i = 0; i < 8; ++i) {
      int l = older[i];
      int src_offset = d * 8 * hsize + l * hsize;
      int dst_offset = d * 8 * hsize + i * hsize;
      memcpy(filter_new->data() + dst_offset, filter->data() + src_offset,
             hsize * sizeof(float));
    }
  }
  filter = filter_new;
}

template <>
LogicalResult WeightReorder<tpu::LSTMOp, Float32Type>::matchAndRewriteImpl(
    tpu::LSTMOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isF32())
    return failure();

  auto attr = op.parseParam();
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  auto filter_f32 = filterOp.read<float>();

  auto recurrenceOp = op.getRecurrence().getDefiningOp<top::WeightOp>();
  auto recurrence_f32 = recurrenceOp.read<float>();
  auto num_filter = module::getNumElements(op.getFilter());
  auto num_recur = module::getNumElements(op.getRecurrence());
  auto filter_merged =
      std::make_shared<std::vector<float>>(num_filter + num_recur, 0);
  filter_merge(filter_merged, filter_f32, recurrence_f32, attr.num_direction,
               attr.input_size, attr.hidden_size);

  std::vector<int64_t> filter_reordered_shape = {
      attr.num_direction, 4 * attr.input_size + 4 * attr.hidden_size,
      attr.hidden_size};
  auto filter_type = module::getStorageType(op.getFilter());
  auto new_filter_type =
      RankedTensorType::get(filter_reordered_shape, filter_type);
  auto newFilterOp = top::WeightOp::create(op, "reordered_filter",
                                           *filter_merged, new_filter_type);
  op->setOperand(1, newFilterOp);
  op->setOperand(2, module::getNoneOp(op));
  if (attr.have_bias) {
    auto biasOp = op.getBias().getDefiningOp<top::WeightOp>();
    auto bias_f32 = biasOp.read<float>();
    auto type = op.getBias().getType().cast<RankedTensorType>();
    iofg2ifog(bias_f32, attr.num_direction, attr.hidden_size);
    auto newBiasOp =
        top::WeightOp::create(op, "reordered_bias", *bias_f32, type);
    op->setOperand(3, newBiasOp);
  }

  std::vector<int64_t> init_shape = {attr.num_direction, attr.batch_size,
                                     attr.hidden_size};
  if (!attr.have_h0) {
    auto stype = module::getStorageType(op.getInput());
    auto initial_h = std::make_shared<std::vector<float>>(
        attr.num_direction * attr.batch_size * attr.hidden_size, 0.0f);
    auto new_type = RankedTensorType::get(init_shape, stype);
    auto initial_h_Op =
        top::WeightOp::create(op, "initial_h", *initial_h, new_type);
    op->setOperand(4, initial_h_Op);
  }
  if (!attr.have_c0) {
    auto stype = module::getStorageType(op.getInput());
    auto initial_c = std::make_shared<std::vector<float>>(
        attr.num_direction * attr.batch_size * attr.hidden_size, 0.0f);
    auto new_type = RankedTensorType::get(init_shape, stype);
    auto initial_c_Op =
        top::WeightOp::create(op, "initial_c", *initial_c, new_type);
    op->setOperand(5, initial_c_Op);
  }
  return success();
}
