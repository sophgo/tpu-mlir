//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

template <typename Dtype>
void compute_permute(Dtype *data_inp, Dtype *data_oup, int64_t count,
                     std::vector<int64_t> &order, const int64_t *input_shape,
                     const int64_t *output_shape) {
  int64_t num_axes_ = order.size();

  std::vector<int64_t> new_steps_(order.size(), 1);
  std::vector<int64_t> old_steps_(order.size(), 1);
  for (int64_t i = 0; i < num_axes_ - 1; ++i) {
    for (int64_t j = i; j < num_axes_ - 1; ++j) {
      old_steps_[i] *= input_shape[j + 1];
      new_steps_[i] *= output_shape[j + 1];
    }
  }

  for (int64_t i = 0; i < count; ++i) {
    int64_t old_idx = 0;
    int64_t idx = i;
    for (int64_t j = 0; j < num_axes_; ++j) {
      old_idx += (idx / new_steps_[j]) * old_steps_[order[j]];
      idx %= new_steps_[j];
    }
    data_oup[i] = data_inp[old_idx];
  }
}

LogicalResult tpu::Batch2SpaceOp::init(InferenceParameter &p) {
  return success();
}
void tpu::Batch2SpaceOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::Batch2SpaceOp::inference(InferenceParameter &p) {
  int64_t in, ic, ih, iw, on, oc, oh, ow;
  module::getNCHW(getInput(), in, ic, ih, iw, false);
  module::getNCHW(getOutput(), on, oc, oh, ow, false);
  assert(in == on);
  int64_t block_h = getBlockH();
  int64_t block_w = getBlockW();
  auto pads_v = module::getI64Array(getCrops());
  int64_t pad_top = pads_v->at(0);
  int64_t pad_bottom = pads_v->at(1);
  int64_t pad_left = pads_v->at(2);
  int64_t pad_right = pads_v->at(3);
  std::vector<int64_t> in_shape(6, 0);
  in_shape[0] = in;
  in_shape[1] = ic;
  in_shape[2] = (ih + pad_top + pad_bottom) / block_h;
  in_shape[3] = block_h;
  in_shape[4] = (iw + pad_left + pad_right) / block_w;
  in_shape[5] = block_w;
  int64_t pad_count = std::accumulate(in_shape.begin(), in_shape.end(), 1,
                                      std::multiplies<int64_t>());
  std::vector<int64_t> order({3, 5, 0, 1, 2, 4});
  std::vector<int64_t> out_shape;
  for (int i = 0; i < in_shape.size(); ++i) {
    out_shape.push_back(in_shape[order[i]]);
  }
  float *in_addr_pad = new float[pad_count]{0};

  for (int index_n = 0; index_n < in; ++index_n)
    for (int index_c = 0; index_c < ic; ++index_c)
      for (int index_h = 0; index_h < ih; ++index_h)
        memcpy(in_addr_pad +
                   (index_n * ic + index_c) * (ih + pad_top + pad_bottom) *
                       (iw + pad_left + pad_right) +
                   (index_h + pad_top) * (iw + pad_left + pad_right) + pad_left,
               p.inputs[0] + (index_n * ic + index_c) * ih * iw + index_h * iw,
               iw * sizeof(float));

  compute_permute(in_addr_pad, p.outputs[0], pad_count, order, in_shape.data(),
                  out_shape.data());
  delete[] in_addr_pad;
  return success();
}

bool tpu::Batch2SpaceOp::support_multi_core() { return false; }
