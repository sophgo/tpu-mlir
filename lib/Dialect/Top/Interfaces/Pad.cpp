//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::PadOp::getFLOPs() { return 0; }

struct pad_info {
  std::vector<int64_t> shape_4;
  std::vector<int64_t> pads_4;
};

LogicalResult top::PadOp::init(InferenceParameter &p) {
  auto info = new pad_info();
  std::vector<int64_t> in_shape = module::getShape(getInput());
  i64_array_t pads_origin = module::getI64Array(getPaddings());
  std::vector<int64_t> pads(in_shape.size() * 2, 0);
  if (!getPaddingsT()) {
    pads = *pads_origin;
  }
  auto ret = pad_reset(in_shape, pads, info->shape_4, info->pads_4);
  if (ret == false) {
    dump();
    UNREACHABLE_THIS("Not Implemented");
  }
  p.handle = (void *)info;
  // set pads
  float *dst = p.outputs[0];
  auto total_num = module::getNumElements(getOutput());
  if (getMode() == "constant") {
    float val_ = getVal().convertToDouble();
    for (int i = 0; i < total_num; i++) {
      dst[i] = val_;
    }
  }

  return success();
}
void top::PadOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto info = (pad_info *)p.handle;
    delete info;
    p.handle = nullptr;
  }
}

LogicalResult top::PadOp::inference(InferenceParameter &p) {
  auto p_info = (pad_info *)p.handle;
  auto pad_mode = getMode();
  if (getPaddingsT()) {
    std::vector<int64_t> in_shape = module::getShape(getInput());
    int pad_dim = in_shape.size() * 2;
    std::vector<int64_t> pads(pad_dim, 0);
    for (int i = 0; i < pad_dim; i++) {
      pads[i] = p.inputs[1][i];
    }
    auto ret = pad_reset(in_shape, pads, p_info->shape_4, p_info->pads_4);
    if (ret == false) {
      dump();
      UNREACHABLE_THIS("Not Implemented");
    }
  }
  std::vector<int> pads(p_info->pads_4.begin(), p_info->pads_4.end());
  int64_t in = p_info->shape_4[0];
  int64_t ic = p_info->shape_4[1];
  int64_t ih = p_info->shape_4[2];
  int64_t iw = p_info->shape_4[3];
  int64_t oc = pads[1] + pads[5] + ic;
  int64_t oh = pads[2] + pads[6] + ih;
  int64_t ow = pads[3] + pads[7] + iw;

  const float *src = p.inputs[0];
  float *dst = p.outputs[0];

  if (pad_mode == "constant") {
    // when pads < 0 means cutoff
    int32_t start_in = pads[0] < 0 ? -pads[0] : 0;
    int32_t start_ic = pads[1] < 0 ? -pads[1] : 0;
    int32_t start_ih = pads[2] < 0 ? -pads[2] : 0;
    int32_t start_iw = pads[3] < 0 ? -pads[3] : 0;

    int32_t end_in = pads[4] < 0 ? in + pads[4] : in;
    int32_t end_ic = pads[5] < 0 ? ic + pads[5] : ic;
    int32_t end_ih = pads[6] < 0 ? ih + pads[6] : ih;
    int32_t end_iw = pads[7] < 0 ? iw + pads[7] : iw;

    int32_t pad_n_begin_size = pads[0] < 0 ? 0 : pads[0] * oc * oh * ow;
    int32_t pad_c_begin_size = pads[1] < 0 ? 0 : pads[1] * oh * ow;
    int32_t pad_h_begin_size = pads[2] < 0 ? 0 : pads[2] * ow;
    int32_t pad_w_begin_size = pads[3] < 0 ? 0 : pads[3];

    for (int out_idx = 0, in_idx = start_in; in_idx < end_in;
         in_idx++, out_idx++) {
      auto in_offset = in_idx * ic * ih * iw;
      auto out_offset =
          pad_n_begin_size + pad_c_begin_size + out_idx * oc * oh * ow;
      for (int oc_idx = 0, ic_idx = start_ic; ic_idx < end_ic;
           ic_idx++, oc_idx++) {
        auto in_ic_offset = in_offset + ic_idx * ih * iw;
        auto out_oc_offset = out_offset + pad_h_begin_size + oc_idx * oh * ow;
        for (int oh_idx = 0, ih_idx = start_ih; ih_idx < end_ih;
             ih_idx++, oh_idx++) {
          auto in_ih_offset = in_ic_offset + ih_idx * iw;
          auto out_oh_offset = out_oc_offset + pad_w_begin_size + oh_idx * ow;
          memcpy(dst + out_oh_offset, src + in_ih_offset + start_iw,
                 (end_iw - start_iw) * sizeof(float_t));
        }
      }
    }
  } else {
    ASSERT_THIS(pads[0] == pads[4] && pads[1] == pads[5] && pads[0] == 0 &&
                pads[1] == 0 && "only support hw pad");

    // comes from https://github.com/BVLC/caffe/pull/6506/files
    for (int n = 0; n < in; ++n) {
      for (int c = 0; c < ic; ++c) {
        // First copy the main body into place
        for (int h = 0; h < ih; ++h) {
          // copy the width part
          int input_offset = ((n * ic + c) * ih + h) * iw;
          int output_offset =
              ((n * oc + c) * oh + (h + pads[2])) * ow + pads[3];

          memcpy(dst + output_offset, src + input_offset, sizeof(float) * iw);
        }

        if (pad_mode == "reflect") {
          // Left and right. Loop over the rows not in the vertical padding
          for (int h = pads[2]; h < oh - pads[6]; ++h) {
            // Offset to current row start (in padding of this row)
            int off = ((n * oc + c) * oh + h) * ow;
            int loff = off + 2 * pads[3];
            int roff = off + ow - 1 - pads[7] - 1;

            // Left
            for (int wdst = 0; wdst < pads[3]; ++wdst) {
              *(dst + off + wdst) = *(dst + loff);
              loff--;
            }
            // Right
            for (int wdst = ow - pads[7]; wdst < ow; ++wdst) {
              *(dst + off + wdst) = *(dst + roff);
              roff--;
            }
          }

          // Top
          // Beginning of this image's data, including padding
          float *dstptr = dst + ((n * oc + c) * oh) * ow;
          // First row not in the vertical padding
          float *srcptr = dstptr + 2 * pads[2] * ow;
          for (int h = 0; h < pads[2]; ++h) {
            std::copy(srcptr, srcptr + ow, dstptr + h * ow);
            srcptr -= ow;
          }

          // Bottom
          // Start of last row not in the vertical padding
          srcptr = dst + ((n * oc + c) * oh + (oh - 2 - pads[6])) * ow;
          // Start of first row in bottom padding
          dstptr = srcptr + 2 * ow;
          for (int h = 0; h < pads[6]; ++h) {
            std::copy(srcptr, srcptr + ow, dstptr + h * ow);
            srcptr -= ow;
          }
        } else if (pad_mode == "edge") {
          // Edge pad to be implemented

          // Left and right. Loop over the rows not in the vertical padding
          for (int h = pads[2]; h < oh - pads[6]; ++h) {
            // Offset to current row start (in padding of this row)
            int off = ((n * oc + c) * oh + h) * ow;
            const float lval = *(dst + off + pads[3]),
                        rval = *(dst + off + ow - 1 - pads[7]);

            // Left
            for (int wdst = 0; wdst < pads[3]; ++wdst) {
              *(dst + off + wdst) = lval;
            }
            // Right
            for (int wdst = ow - pads[7]; wdst < ow; ++wdst) {
              *(dst + off + wdst) = rval;
            }
          }

          // Top
          // Beginning of this image's data, including padding
          float *dstptr = dst + ((n * oc + c) * oh) * ow;
          // First row not in the vertical padding
          float *srcptr = dstptr + pads[2] * ow;
          for (int h = 0; h < pads[2]; ++h) {
            std::copy(srcptr, srcptr + ow, dstptr + h * ow);
          }

          // Bottom
          // Start of last row not in the vertical padding
          srcptr = dst + ((n * oc + c) * oh + (oh - 1 - pads[6])) * ow;
          // Start of first row in bottom padding
          dstptr = srcptr + ow;
          for (int h = 0; h < pads[6]; ++h) {
            std::copy(srcptr, srcptr + ow, dstptr + h * ow);
          }
        } else {
          UNREACHABLE_THIS("Not Implemented");
        }
      }
    }
  }

  return success();
}

void top::PadOp::shape_inference() {
  auto pads_origin = module::getI64Array(getPaddings());
  auto input_shape = module::getShape(getInput());
  auto dim = input_shape.size();
  std::vector<int64_t> pads(dim * 2, 0);
  if (module::isPlatform(module::Platform::TORCH)) {
    if (pads_origin->size() >= 2) {
      // w pad
      pads[dim - 1] = pads_origin->at(0);
      pads[2 * dim - 1] = pads_origin->at(1);
    }
    if (pads_origin->size() >= 4) {
      pads[dim - 2] = pads_origin->at(2);
      pads[dim * 2 - 2] = pads_origin->at(3);
    }
    Builder builder(getContext());
    setPaddingsAttr(builder.getI64ArrayAttr(pads));
  } else {
    if (module::isShape(getPaddingsT())) {
      pads = module::getShapeTensorValue(getPaddingsT());
    } else {
      ASSERT_THIS(pads_origin->size() == dim * 2);
      pads = *pads_origin;
    }
  }
  std::vector<int64_t> out_shape(input_shape);
  for (int i = 0; i < dim; i++)
    out_shape[i] += pads[i] + pads[i + dim];
  module::setShapeOrVerify(getOutput(), out_shape);
}
