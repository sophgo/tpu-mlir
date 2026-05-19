//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::StridedSliceOp::getFLOPs() { return 0; }

LogicalResult top::StridedSliceOp::init(InferenceParameter &p) {
  return success();
}
void top::StridedSliceOp::deinit(InferenceParameter &p) {}

LogicalResult top::StridedSliceOp::inference(InferenceParameter &p) {
  auto out_num_elem = module::getNumElements(getOutput());
  auto out_shape = module::getShape(getOutput());
  auto in_shape = module::getShape(getInput());
  auto in_dims = in_shape.size();
  auto out_dims = out_shape.size();

  auto input_ = p.inputs[0];
  auto output_ = p.outputs[0];
  auto starts_ = p.inputs[1];
  auto ends_ = p.inputs[2];
  auto strides_ = p.inputs[3];
  auto s_dims = module::getNumElements(getStrides());
  int32_t begin[8], end[8], step[8], input_shape[8];
  int32_t dims, b_mask, e_mask, shrink_mask;

  stride_slice_gen_params(in_shape.data(), in_dims, starts_, ends_, strides_,
                          s_dims, getBeginMask(), getEndMask(),
                          getEllipsisMask(), getNewAxisMask(),
                          getShrinkAxisMask(), input_shape, &dims, begin, end,
                          step, &b_mask, &e_mask, &shrink_mask);
  for (int i = 0; i < dims; ++i) {
    begin[i] = StartForAxis(begin, step, b_mask, input_shape, i);
    end[i] =
        StopForAxis(end, step, e_mask, shrink_mask, input_shape, i, begin[i]);
  }

  // just support the dims of input & input is equal and Stride Slice at one
  // axis now.
  ASSERT_THIS(in_dims == out_dims);

  if (in_dims == 2) {
#pragma omp parallel for schedule(static, omp_schedule(out_num_elem))
    for (int i = 0; i < out_shape[0]; i++) {
      for (int j = 0; j < out_shape[1]; j++) {
        int64_t o_index = i * out_shape[1] + j;
        int64_t i_index =
            (begin[0] + i * step[0]) * in_shape[1] + begin[1] + j * step[1];
        output_[o_index] = input_[i_index];
      }
    }
  } else if (in_dims == 3) {
#pragma omp parallel for schedule(static, omp_schedule(out_num_elem))
    for (int i = 0; i < out_shape[0]; i++) {
      for (int j = 0; j < out_shape[1]; j++) {
        for (int k = 0; k < out_shape[2]; k++) {
          int64_t o_index = (i * out_shape[1] + j) * out_shape[2] + k;
          int64_t i_index = ((begin[0] + i * step[0]) * in_shape[1] + begin[1] +
                             j * step[1]) *
                                in_shape[2] +
                            begin[2] + k * step[2];
          output_[o_index] = input_[i_index];
        }
      }
    }
  } else if (in_dims == 4) {
#pragma omp parallel for schedule(static, omp_schedule(out_num_elem))
    for (int i = 0; i < out_shape[0]; i++) {
      for (int j = 0; j < out_shape[1]; j++) {
        for (int k = 0; k < out_shape[2]; k++) {
          for (int z = 0; z < out_shape[3]; z++) {
            int64_t o_index =
                ((i * out_shape[1] + j) * out_shape[2] + k) * out_shape[3] + z;
            int64_t i_index = (((begin[0] + i * step[0]) * in_shape[1] +
                                begin[1] + j * step[1]) *
                                   in_shape[2] +
                               begin[2] + k * step[2]) *
                                  in_shape[3] +
                              begin[3] + z * step[3];
            output_[o_index] = input_[i_index];
          }
        }
      }
    }
  }

  return success();
}

void top::StridedSliceOp::shape_inference() {
  auto in_shape = module::getShape(getInput());
  auto num_dims = in_shape.size();
  auto begin_mask = getBeginMask();
  auto end_mask = getEndMask();
  auto starts_op = getStarts().getDefiningOp<top::WeightOp>();
  auto ends_op = getEnds().getDefiningOp<top::WeightOp>();
  auto strides_op = getStrides().getDefiningOp<top::WeightOp>();
  if (!starts_op || !ends_op || !strides_op) return;
  auto starts = starts_op.read_as_float();
  auto ends = ends_op.read_as_float();
  auto strides = strides_op.read_as_float();
  std::vector<int64_t> out_shape;
  for (size_t i = 0; i < num_dims; i++) {
    int64_t start_val = (int64_t)starts->at(i);
    int64_t end_val = (int64_t)ends->at(i);
    int64_t stride_val = (int64_t)strides->at(i);
    if (begin_mask & (1 << i)) start_val = (stride_val > 0) ? 0 : in_shape[i] - 1;
    if (end_mask & (1 << i)) end_val = (stride_val > 0) ? in_shape[i] : -1;
    if (start_val < 0) start_val += in_shape[i];
    if (end_val < 0) end_val += in_shape[i];
    int64_t dim_len = (end_val - start_val + (stride_val > 0 ? stride_val - 1 : stride_val + 1)) / stride_val;
    if (dim_len < 0) dim_len = 0;
    out_shape.push_back(dim_len);
  }
  auto out_type = RankedTensorType::get(out_shape, module::getStorageType(getInput()));
  getOutput().setType(out_type);
}
