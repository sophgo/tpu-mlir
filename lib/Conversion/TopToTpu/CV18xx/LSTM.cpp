//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

namespace tpu_mlir {
namespace cv18xx {
static void spiltBias() {}
static void reorederWeight() {}
static double active_sigmoid(double val) { return 1.0 / (1 + expf(-val)); }
static double active_tanh(double val) { return tanh(val); }

void LSTMLowering::LoweringINT8(PatternRewriter &rewriter, top::LSTMOp op,
                                bool asymmetric) const {
  LoweringBF16(rewriter, op);
}

void LSTMLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::LSTMOp op) const {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  auto inpShape = module::getShape(op.getInput());
  assert(inpShape.size() == 3 && op.getBatchFirst() == false &&
         "Now just support [seq_len, batch_size, embed_size] as input.");
  operands.push_back(op.getInput());
  auto filterOp =
      cast<top::WeightOp>(op.getFilter().getDefiningOp()); // transpose later
  auto recurrenceOp = cast<top::WeightOp>(op.getRecurrence().getDefiningOp());
  auto recurrenceShape = module::getShape(op.getRecurrence());
  auto hiddenSize = recurrenceShape.back();
  auto biasOp =
      cast<top::WeightOp>(op.getBias().getDefiningOp()); // spilt later

  auto filterShape = module::getShape(op.getFilter());
  auto is_torch = module::isPlatform(module::Platform::TORCH);
  assert((filterShape.size() == 3 && !is_torch) ||
         (filterShape.size() == 2 && is_torch) && "please check filter shape.");
  auto filterF32 = filterOp.read<float>();
  auto N = filterShape[0];
  auto K = filterShape[1];
  if (filterShape.size() == 3) {
    N = filterShape[0] * filterShape[1];
    K = filterShape[2];
  }

  std::vector<int64_t> newFilterShape = {K, N};
  std::vector<float_t> newFilter(K * N);
  // transpose filter from (N, K) to (K, N)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      newFilter.at(j * N + i) = filterF32->at(i * K + j);
    }
  }
  // split bias to filterBias and recurrenceBias
  auto bias_size = 4 * hiddenSize;
  auto nofdir = op.getBidirectional() ? 2 : 1;
  auto biasF32 = biasOp.read<float>();
  std::vector<int64_t> newBiasShape = {nofdir, bias_size};
  std::vector<float_t> filterBias;
  std::vector<float_t> recurrenceBias;
  for (int ndir = 0; ndir < nofdir; ndir++) {
    auto offset = ndir * 2 * bias_size;
    filterBias.insert(filterBias.end(), biasF32->begin() + offset,
                      biasF32->begin() + offset + bias_size);
    recurrenceBias.insert(recurrenceBias.end(),
                          biasF32->begin() + offset + bias_size,
                          biasF32->begin() + offset + 2 * bias_size);
  }

  // for caffe which these is no recurrenceBias
  bool has_rBias = false;
  for (auto v : recurrenceBias) {
    if (v != 0) {
      has_rBias = true;
      break;
    }
  }

  // creat fcOp
  auto new_type = RankedTensorType::get(newFilterShape, rewriter.getF32Type());
  auto newFilterOp =
      top::WeightOp::create(op, "filter_trans", newFilter, new_type);
  new_type = RankedTensorType::get(newBiasShape, rewriter.getF32Type());
  auto newBiasOp = top::WeightOp::create(op, "w_bias", filterBias, new_type);
  operands.push_back(newFilterOp);
  operands.push_back(newBiasOp);
  std::vector<int64_t> fcShape = {inpShape[0], inpShape[1], N};
  new_type = RankedTensorType::get(fcShape, rewriter.getF32Type());
  auto op_name = module::getName(op.getOperation()).str();
  auto loc = NameLoc::get(rewriter.getStringAttr(op_name + "_fc"));
  auto fcOp = rewriter.create<top::MatMulOp>(loc, new_type, operands, attrs);

  // creat LSTM
  auto noneOp = module::getNoneOp(op);
  operands.clear();
  operands.push_back(fcOp.getOutput());
  operands.push_back(noneOp);

  // trans weight and bias in codegen
  if (recurrenceShape.size() == 2) {
    std::vector<int64_t> newRecurrenceShape = {
        nofdir, recurrenceShape[0] / nofdir, hiddenSize};
    module::setShape(op.getRecurrence(), newRecurrenceShape);
  }
  operands.push_back(recurrenceOp.clone_bf16(op));
  if (has_rBias) {
    std::vector<int64_t> newRecurrenceBiasShape = {nofdir, bias_size};
    new_type =
        RankedTensorType::get(newRecurrenceBiasShape, rewriter.getF32Type());
    newBiasOp = top::WeightOp::create(op, "r_bias", recurrenceBias, new_type);
    operands.push_back(newBiasOp);
  } else {
    // for caffe
    operands.push_back(module::getNoneOp(op));
  }

  // convert intit_h intit_c cont
  if (auto castOp = dyn_cast<top::WeightOp>(op.getInitialH().getDefiningOp())) {
    auto new_weight = castOp.clone_bf16(op, op_name + "_init_h_bf16");
    operands.push_back(new_weight);
  } else {
    operands.push_back(op.getInitialH());
  }
  if (auto castOp = dyn_cast<top::WeightOp>(op.getInitialC().getDefiningOp())) {
    auto new_weight = castOp.clone_bf16(op, op_name + "_init_c_bf16");
    operands.push_back(new_weight);
  } else {
    operands.push_back(op.getInitialC());
  }
  if (auto castOp = dyn_cast<top::WeightOp>(op.getCont().getDefiningOp())) {
    operands.push_back(castOp.clone_bf16(op));
  } else {
    operands.push_back(op.getCont());
  }
  operands.push_back(noneOp);
  // insert table
  int table_h = 32;
  int table_w = 8;
  float range_start = -12;
  float range_end = 12;
  int table_hw = table_h * table_w;
  std::vector<float> table(table_hw);
  std::vector<float> slope_table(table_hw);
  auto shape = std::vector<int64_t>{1, 1, table_h, table_w};
  auto table_type = RankedTensorType::get(shape, rewriter.getF32Type());

  // creat sigmoid table
  bf16_gen_base_slope_table(table.data(), slope_table.data(), range_start,
                            range_end, active_sigmoid);
  auto table_op = top::WeightOp::create(op, "sigmoid_table", table, table_type);
  auto slope_table_op =
      top::WeightOp::create(op, "sigmoid_slope_table", slope_table, table_type);
  auto table_weight_op = dyn_cast<top::WeightOp>(table_op.getDefiningOp());
  auto slope_table_weight_op =
      dyn_cast<top::WeightOp>(slope_table_op.getDefiningOp());
  operands.push_back(table_weight_op.clone_bf16(op));
  operands.push_back(slope_table_weight_op.clone_bf16(op));

  // creat tanh table
  range_start = -15;
  range_end = 15;
  bf16_gen_base_slope_table(table.data(), slope_table.data(), range_start,
                            range_end, active_tanh);
  table_op = top::WeightOp::create(op, "tanh_table", table, table_type);
  slope_table_op =
      top::WeightOp::create(op, "tanh_slope_table", slope_table, table_type);
  table_weight_op = dyn_cast<top::WeightOp>(table_op.getDefiningOp());
  slope_table_weight_op =
      dyn_cast<top::WeightOp>(slope_table_op.getDefiningOp());
  operands.push_back(table_weight_op.clone_bf16(op));
  operands.push_back(slope_table_weight_op.clone_bf16(op));
  attrs.clear();
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  std::vector<Type> new_types;
  for (auto out : op.getResults()) {
    new_types.push_back(getQuantBF16Type(out));
  }
  rewriter.replaceOpWithNewOp<tpu::LSTMOp>(op, new_types, operands, attrs);
}
} // namespace cv18xx
} // namespace tpu_mlir
