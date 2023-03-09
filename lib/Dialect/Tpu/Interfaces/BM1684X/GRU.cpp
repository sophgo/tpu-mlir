//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/WeightReorder.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/DynCompileCommon.hpp"

using namespace tpu_mlir::backend;
using namespace tpu_mlir::bm1684x;

// ======================================
// WeightReorderInterface
// ======================================
// W =  [num_directions, 3, hidden_size, input_size]
// R =  [num_directions, 3, hidden_size, hidden_size]
// => W0 = [num_directions, 3, input_size, hidden_size] (0,3,2,1)
//    R0 = [num_directions, 3, hidden_size, hidden_size] (0,3,2,1)
// => Merge in axis num_directions
// => z r h to r z h
template <typename T>
static void filter_merge(std::shared_ptr<std::vector<T>> &filter,
                         std::shared_ptr<std::vector<T>> &W,
                         std::shared_ptr<std::vector<T>> &R, int num_dir,
                         int input_size, int hidden_size) {
  int w_size = input_size * hidden_size;
  int r_size = hidden_size * hidden_size;
  int w_offset = 0, r_offset = 0;
  for (int d = 0; d < num_dir; d++) {
    // apple W
    for (int i = 0; i < 3; i++) {
      for (int h = 0; h < hidden_size; h++) {
        for (int x = 0; x < input_size; x++) {
          // gate z r h => z h r
          int gate = (i == 0) ? 1 : (i == 1 ? 0 : i);
          int dst_offset = d * 3 * (w_size + r_size) +
                           gate * input_size * hidden_size + x * hidden_size +
                           h;
          filter->at(dst_offset) = W->at(w_offset);
          w_offset++;
        }
      }
    }

    // apply R
    for (int i = 0; i < 3; i++) {
      for (int h = 0; h < hidden_size; h++) {
        for (int x = 0; x < hidden_size; x++) {
          // gate z r h => r z h
          int gate = (i == 0) ? 1 : (i == 1 ? 0 : i);
          int dst_offset = d * 3 * (w_size + r_size) + 3 * w_size +
                           gate * hidden_size * hidden_size + x * hidden_size +
                           h;
          filter->at(dst_offset) = R->at(r_offset);
          r_offset++;
        }
      }
    }
  }
}

// bias [num_dir, 3, hidden_size]
// onnx: z r h
// pytorch: r z h
template <typename T>
static void zrh2rzh(std::shared_ptr<std::vector<T>> &filter, int num_dir,
                    int hsize) {
  auto filter_new = std::make_shared<std::vector<T>>(filter->size(), 0);
  int older[6] = {1, 0, 2, 4, 3, 5};
  for (int d = 0; d < num_dir; d++) {
    for (int i = 0; i < 6; ++i) {
      int l = older[i];
      int src_offset = d * 6 * hsize + l * hsize;
      int dst_offset = d * 6 * hsize + i * hsize;
      memcpy(filter_new->data() + dst_offset, filter->data() + src_offset,
             hsize * sizeof(T));
    }
  }
  filter = filter_new;
}

template <>
LogicalResult WeightReorder<tpu::GRUOp, Float32Type>::matchAndRewrite(
    tpu::GRUOp op, PatternRewriter &rewriter) const {
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
      attr.num_direction, 3 * attr.input_size + 3 * attr.hidden_size,
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
    zrh2rzh(bias_f32, attr.num_direction, attr.hidden_size);
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
  return success();
}

#ifdef __cplusplus
extern "C" {
#endif

// BATCH_FIRST: x = [batch, seq_len, input_size], y = [batch, seq_len, num_dir,
// hidden_size] BATCH_TORCH: x = [seq_len, batch, input_size], y = [seq_len,
// batch, num_dir, hidden_size] BATCH_ONNX:  x = [seq_len, batch, input_size], y
// = [seq_len, num_dir, batch, hidden_size]
typedef enum {
  BATCH_TORCH = 0,
  BATCH_FIRST = 1,
  BATCH_ONNX = 2,
} gru_batch_t;

typedef struct {
  uint64_t xGlobalAddr;
  uint64_t h0GlobalAddr;
  uint64_t yGlobalAddr;
  uint64_t hnGlobalAddr;
  uint64_t wGlobalAddr;
  uint64_t bGlobalAddr;
  uint64_t zGlobalAddr;
  bool bias;
  bool outputY;
  bool outputYh;
  int sequence;
  int batch;
  int xSize;
  int hSize;
  int batchMode;
  bool bidirectional;
  int numLayers;
  int dtype;
} gru_param_t;

#ifdef __cplusplus
}
#endif
// =========================================
// GlobalGenInterface
// =========================================
void tpu::GRUOp::codegen_global_bm1684x() {
  auto attr = parseParam();
  gru_param_t p = {0};
  p.xGlobalAddr = module::getAddress(getInput());
  p.wGlobalAddr = module::getAddress(getFilter());
  p.bGlobalAddr = module::getAddress(getBias());
  p.h0GlobalAddr = module::getAddress(getInitialH());
  p.yGlobalAddr = module::getAddress(getY());
  p.hnGlobalAddr = module::getAddress(getYH());
  p.zGlobalAddr = module::getAddress(getBuffer());

  p.bias = attr.have_bias;
  p.outputY = attr.output_y;
  p.outputYh = attr.output_yh;
  p.sequence = attr.seq_len;
  p.batch = attr.batch_size;
  p.xSize = attr.input_size;
  p.hSize = attr.hidden_size;
  p.batchMode = attr.batch_first ? BATCH_FIRST : BATCH_ONNX;
  p.bidirectional = (attr.num_direction == 2);
  p.numLayers = 1;
  p.dtype = BM168x::getDataType(getInput());
  BM168x::call_global_func("backend_api_gru_global", &p, sizeof(gru_param_t));
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::GRUOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) return sizeof(dyn_glu_global_spec_t);
  auto attr = parseParam();
  dyn_glu_global_spec_t p = {0};
  p.xGlobalAddr = module::getAddress(getInput());
  p.wGlobalAddr = module::getAddress(getFilter());
  p.bGlobalAddr = module::getAddress(getBias());
  p.h0GlobalAddr = module::getAddress(getInitialH());
  p.yGlobalAddr = module::getAddress(getY());
  p.hnGlobalAddr = module::getAddress(getYH());
  p.zGlobalAddr = module::getAddress(getBuffer());
  p.common.bias = attr.have_bias;
  p.common.outputY = attr.output_y;
  p.common.outputYh = attr.output_yh;
  p.common.sequence = attr.seq_len;
  p.common.batch = attr.batch_size;
  p.common.xSize = attr.input_size;
  p.common.hSize = attr.hidden_size;
  p.common.batchMode = attr.batch_first ? BATCH_FIRST : BATCH_ONNX;
  p.common.bidirectional = (attr.num_direction == 2);
  p.common.numLayers = 1;
  p.common.dtype = BM168x::getDataType(getInput());
  return BM168x::dynamic_spec_to_buffer(buffer, p);
}

int64_t tpu::GRUOp::get_layer_type() {
  return FW_BMNET_GRU;
}
