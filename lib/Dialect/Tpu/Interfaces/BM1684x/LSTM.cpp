//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684x.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

// ======================================
// WeightReorderInterface
// ======================================
// W =  [num_directions, 4, hidden_size, input_size]
// R =  [num_directions, 4, hidden_size, hidden_size]
// => W0 = [num_directions, input_size,4, hidden_size] (0,3,2,1)
//    R0 = [num_directions, hidden_size,4, hidden_size] (0,3,2,1)
// => Merge in axis num_directions
// i o f g => i f o g
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
template <typename T>
static void iofg2ifog(std::shared_ptr<std::vector<T>> &filter, int num_dir,
                      int hsize) {
  auto filter_new = std::make_shared<std::vector<T>>(filter->size(), 0);
  int older[8] = {0, 2, 1, 3, 4, 6, 5, 7};
  for (int d = 0; d < num_dir; d++) {
    for (int i = 0; i < 8; ++i) {
      int l = older[i];
      int src_offset = d * 8 * hsize + l * hsize;
      int dst_offset = d * 8 * hsize + i * hsize;
      memcpy(filter_new->data() + dst_offset, filter->data() + src_offset,
             hsize * sizeof(T));
    }
  }
  filter = filter_new;
}

void tpu::LSTMOp::weight_reorder_f32_bm1684x() {
  auto attr = parseParam();
  auto op = getOperation();
  OpBuilder builder(getContext());

  auto filterOp = filter().getDefiningOp<top::WeightOp>();
  auto filter_f32 = filterOp.read<float>();

  auto recurrenceOp = recurrence().getDefiningOp<top::WeightOp>();
  auto recurrence_f32 = recurrenceOp.read<float>();
  auto num_filter = Module::getNumElements(filter());
  auto num_recur = Module::getNumElements(recurrence());
  auto filter_merged =
      std::make_shared<std::vector<float>>(num_filter + num_recur, 0);
  filter_merge(filter_merged, filter_f32, recurrence_f32, attr.num_direction,
               attr.input_size, attr.hidden_size);

  std::vector<int64_t> filter_reordered_shape = {
      attr.num_direction, 4 * attr.input_size + 4 * attr.hidden_size,
      attr.hidden_size};
  auto filter_type = Module::getStorageType(filter());
  auto new_filter_type =
      RankedTensorType::get(filter_reordered_shape, filter_type);
  auto newFilterOp = top::WeightOp::create(op, "reordered_filter",
                                           *filter_merged, new_filter_type);
  op->setOperand(1, newFilterOp);
  op->setOperand(2, Module::getNoneOp(op));
  if (attr.have_bias) {
    auto biasOp = bias().getDefiningOp<top::WeightOp>();
    auto bias_f32 = biasOp.read<float>();
    auto type = bias().getType().cast<RankedTensorType>();
    iofg2ifog(bias_f32, attr.num_direction, attr.hidden_size);
    auto newBiasOp =
        top::WeightOp::create(op, "reordered_bias", *bias_f32, type);
    op->setOperand(3, newBiasOp);
  }

  std::vector<int64_t> init_shape = {attr.num_direction, attr.batch_size,
                                     attr.hidden_size};
  if (!attr.have_h0) {
    auto stype = Module::getStorageType(input());
    auto initial_h = std::make_shared<std::vector<float>>(
        attr.num_direction * attr.batch_size * attr.hidden_size, 0.0f);
    auto new_type = RankedTensorType::get(init_shape, stype);
    auto initial_h_Op =
        top::WeightOp::create(op, "initial_h", *initial_h, new_type);
    op->setOperand(4, initial_h_Op);
  }
  if (!attr.have_c0) {
    auto stype = Module::getStorageType(input());
    auto initial_c = std::make_shared<std::vector<float>>(
        attr.num_direction * attr.batch_size * attr.hidden_size, 0.0f);
    auto new_type = RankedTensorType::get(init_shape, stype);
    auto initial_c_Op =
        top::WeightOp::create(op, "initial_c", *initial_c, new_type);
    op->setOperand(5, initial_c_Op);
  }
}

void tpu::LSTMOp::weight_reorder_f16_bm1684x() {}
void tpu::LSTMOp::weight_reorder_bf16_bm1684x() {}
void tpu::LSTMOp::weight_reorder_int8_bm1684x() {}

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
} lstm_batch_t;

typedef struct {
  unsigned long long x_global_addr;
  unsigned long long h0_global_addr;
  unsigned long long c0_global_addr;
  unsigned long long y_global_addr;
  unsigned long long hn_global_addr;
  unsigned long long cn_global_addr;
  unsigned long long w_global_addr;
  unsigned long long b_global_addr;
  unsigned long long z_global_addr;
  bool bias;
  bool output_h;
  bool output_c;
  int sequence;
  int batch;
  int x_size;
  int h_size;
  lstm_batch_t batch_mode;
  bool bidirection;
  int num_layers;
  int dtype;
} pytorch_lstm_param_t;

#ifdef __cplusplus
}
#endif
// =========================================
// GlobalGenInterface
// =========================================
void tpu::LSTMOp::codegen_global_bm1684x() {
  auto attr = parseParam();
  auto op = getOperation();
  auto input_spec = BM1684x::get_spec(
      ValueRange{input(), initial_h(), initial_c(), filter()});
  auto output_spec = BM1684x::get_output_spec(op);
  // 1684x pytorch lstm out is [seq_length, batch_size, num_dir * hidden_size]
  pytorch_lstm_param_t p = {0};
  p.x_global_addr = Module::getAddress(input());
  p.w_global_addr = Module::getAddress(filter());
  p.b_global_addr = attr.have_bias ? Module::getAddress(bias()) : 0;
  p.h0_global_addr = Module::getAddress(initial_h());
  p.c0_global_addr = Module::getAddress(initial_c());
  p.y_global_addr = Module::getAddress(Y());
  if (attr.output_h) {
    p.hn_global_addr = Module::getAddress(Y_h());
  }
  if (attr.output_c) {
    p.cn_global_addr = Module::getAddress(Y_c());
  }
  if (buffer().getType().isa<NoneType>() == false) {
    p.z_global_addr = Module::getAddress(buffer());
  }
  p.bias = attr.have_bias;
  p.output_h = attr.output_h;
  p.output_c = attr.output_c;
  p.sequence = attr.seq_len;
  p.batch = attr.batch_size;
  p.x_size = attr.input_size;
  p.h_size = attr.hidden_size;
  p.batch_mode = attr.batch_first ? BATCH_FIRST : BATCH_ONNX;
  p.bidirection = (attr.num_direction == 2);
  p.num_layers = 1;
  p.dtype = BM168x::getDataType(input());
  BM168x::call_global_func("backend_api_pytorch_lstm", &p,
                           sizeof(pytorch_lstm_param_t), input_spec->data(),
                           output_spec->data());
}
