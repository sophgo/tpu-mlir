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

// merge W&R => filter
template <typename T>
static void filter_merge(std::shared_ptr<std::vector<T>> &filter,
                         std::shared_ptr<std::vector<T>> &W,
                         std::shared_ptr<std::vector<T>> &R, int num_dir,
                         int input_size, int hidden_size) {
  for (int d = 0; d < num_dir; d++) {
    int w_size = input_size * hidden_size;
    int r_size = hidden_size * hidden_size;
    // apple W
    for (int i = 0; i < 4; i++) {
      for (int h = 0; h < hidden_size; h++) {
        for (int x = 0; x < input_size; x++) {
          int src_offset =
              d * w_size + i * input_size * hidden_size + h * input_size + x;
          int dst_offset = d * (w_size + r_size) +
                           i * input_size * hidden_size + x * hidden_size + h;
          filter->at(dst_offset) = W->at(src_offset);
        }
      }
    }

    // apply R
    for (int i = 0; i < 4; i++) {
      for (int h = 0; h < hidden_size; h++) {
        for (int x = 0; x < hidden_size; x++) {
          int src_offset =
              d * r_size + i * hidden_size * hidden_size + h * hidden_size + x;
          int dst_offset = d * (w_size + r_size) + 4 * w_size +
                           i * hidden_size * hidden_size + x * hidden_size + h;
          filter->at(dst_offset) = R->at(src_offset);
        }
      }
    }
  }
}

// weight&&bias layout:
// onnx: i o f g
// pytorch: i f g o
// for comput easy, 1684&1684x pytorch lstm reshaped as: i f o g
// so need reorder as: i o f g => i f o g
template <typename T>
static void filter_reorder(std::shared_ptr<std::vector<T>> &filter, int offset,
                           int xsize, int hsize) {
  auto filter_new = filter;
  for (int i = 0; i < 4; ++i) {
    int l = i == 1 ? 2 : (i == 2 ? 1 : i);
    for (int x = 0; x < xsize; ++x) {
      for (int h = 0; h < hsize; ++h) {
        int src_offset = offset + i * xsize * hsize + x * hsize + h;
        int dst_offset = offset + l * xsize * hsize + x * hsize + h;
        filter_new->at(dst_offset) = filter->at(src_offset);
      }
    }
  }
  filter = filter_new;
}

void tpu::LSTMOp::weight_reorder_f32_bm1684x() {
  auto op = getOperation();
  OpBuilder builder(getContext());
  int64_t in0_n, in0_c, in0_h, in0_w;
  int64_t in1_n, in1_c, in1_h, in1_w;
  int64_t in2_n, in2_c, in2_h, in2_w;
  Module::getNCHW(input(), in0_n, in0_c, in0_h, in0_w);
  Module::getNCHW(filter(), in1_n, in1_c, in1_h, in1_w);
  Module::getNCHW(recurrence(), in2_n, in2_c, in2_h, in2_w);
  int num_dir = bidirectional() ? 2 : 1;
  int batch_size = batch_first() ? in0_n : in0_h;
  int input_size = in0_h;
  int hidden_size = in2_h;

  auto filterOp = filter().getDefiningOp<top::WeightOp>();
  auto filter_f32 = filterOp.read<float>();

  auto recurrenceOp = recurrence().getDefiningOp<top::WeightOp>();
  auto recurrence_f32 = recurrenceOp.read<float>();

  auto filter_merged = std::make_shared<std::vector<float>>(
      Module::getNumElements(filter()) + Module::getNumElements(recurrence()),
      0);
  filter_merge(filter_merged, filter_f32, recurrence_f32, num_dir, input_size,
               hidden_size);

  int filter_offset = 0;
  for (int i = 0; i < num_layers() * num_dir; i++) {
    // trans w_x
    filter_reorder(filter_merged, filter_offset,
                   i < num_dir ? input_size : hidden_size * num_dir,
                   hidden_size);
    filter_offset +=
        (i < num_dir ? input_size : hidden_size * num_dir) * hidden_size * 4;
    // trans w_h
    filter_reorder(filter_merged, filter_offset, hidden_size, hidden_size);
    filter_offset += hidden_size * hidden_size * 4;
  }
  std::vector<int64_t> filter_reordered_shape = {
      num_dir, (in0_c * in0_h + in1_c * in1_h) / hidden_size, hidden_size};
  auto filter_type = Module::getStorageType(filter());
  auto new_filter_type =
      RankedTensorType::get(filter_reordered_shape, filter_type);
  auto newFilterOp = top::WeightOp::create(op, "reordered_filter",
                                           *filter_merged, new_filter_type);
  op->setOperand(1, newFilterOp);

  if (have_bias()) {
    auto biasOp = bias().getDefiningOp<top::WeightOp>();
    auto bias_f32 = biasOp.read<float>();
    auto bias_type = Module::getStorageType(bias());
    std::vector<int64_t> bias_shape = {num_dir, 8, hidden_size};

    int bias_offset = 0;
    for (int i = 0; i < num_layers() * num_dir * 2; i++) {
      filter_reorder(bias_f32, bias_offset, 1, hidden_size);
      bias_offset += 4 * hidden_size;
    }
    auto new_bias_type = RankedTensorType::get(bias_shape, bias_type);
    auto newBiasOp =
        top::WeightOp::create(op, "reordered_bias", *bias_f32, new_bias_type);
    op->setOperand(3, newBiasOp);
  }

  bool has_initial_h = !initial_h().getType().isa<NoneType>();
  bool has_initial_c = !initial_c().getType().isa<NoneType>();
  std::vector<int64_t> initial_hc_shape = {num_dir, batch_size, hidden_size};
  if (!has_initial_h) {
    auto initial_h_type = Module::getStorageType(initial_h());
    auto initial_h = std::make_shared<std::vector<float>>(
        num_dir * batch_size * hidden_size, 0.0f);
    auto new_initial_h_type =
        RankedTensorType::get(initial_hc_shape, initial_h_type);
    auto initial_h_Op =
        top::WeightOp::create(op, "initial_h", *initial_h, new_initial_h_type);
    op->setOperand(have_bias() ? 3 : 4, initial_h_Op);
  }
  if (!has_initial_c) {
    auto initial_c_type = Module::getStorageType(initial_c());
    auto initial_c = std::make_shared<std::vector<float>>(
        num_dir * batch_size * hidden_size, 0.0f);
    auto new_initial_c_type =
        RankedTensorType::get(initial_hc_shape, initial_c_type);
    auto initial_c_Op =
        top::WeightOp::create(op, "initial_c", *initial_c, new_initial_c_type);
    op->setOperand(have_bias() ? 4 : 5, initial_c_Op);
  }
}

void tpu::LSTMOp::weight_reorder_f16_bm1684x() {}
void tpu::LSTMOp::weight_reorder_bf16_bm1684x() {}
void tpu::LSTMOp::weight_reorder_int8_bm1684x() {}

#ifdef __cplusplus
extern "C" {
#endif

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
  int sequence;
  int batch;
  int x_size;
  int h_size;
  bool batch_first;
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
  auto op = getOperation();
  auto input_spec = BM1684x::get_input_spec(op);
  auto output_spec = BM1684x::get_output_spec(op);
  // 1684x pytorch lstm out is [seq_length, batch_size, num_dir * hidden_size]
  int64_t in0_n, in0_c, in0_h, in0_w;
  int64_t in1_n, in1_c, in1_h, in1_w;
  int64_t out_n, out_c, out_h, out_w;
  Module::getNCHW(input(), in0_n, in0_c, in0_h, in0_w);
  Module::getNCHW(filter(), in1_n, in1_c, in1_h, in1_w);
  Module::getNCHW(output(), out_n, out_c, out_h, out_w);
  pytorch_lstm_param_t p = {0};
  p.x_global_addr = Module::getAddress(input());
  p.w_global_addr = Module::getAddress(filter());
  p.b_global_addr = have_bias() ? Module::getAddress(bias()) : 0;
  p.h0_global_addr = Module::getAddress(initial_h());
  p.c0_global_addr = Module::getAddress(initial_c());
  p.y_global_addr = Module::getAddress(output());
  // (TODO)hn, cn, z(global_buffer) use true addr after
  p.hn_global_addr =
      p.y_global_addr + Module::getNumElements(output()) * sizeof(float);
  p.cn_global_addr =
      p.hn_global_addr + Module::getNumElements(initial_h()) * sizeof(float);
  p.z_global_addr =
      p.cn_global_addr + Module::getNumElements(initial_c()) * sizeof(float);
  p.bias = have_bias();
  p.sequence = batch_first() ? in0_c : in0_n;
  p.batch = batch_first() ? in0_n : in0_c;
  p.x_size = in0_h;
  p.h_size = in1_h;
  p.batch_first = batch_first();
  p.bidirection = bidirectional();
  p.num_layers = num_layers();
  p.dtype = BM168x::getDataType(input());
  BM1684x::instance().call_global_func("backend_api_pytorch_lstm", &p,
                                       sizeof(pytorch_lstm_param_t),
                                       input_spec->data(), output_spec->data());
}
