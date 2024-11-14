//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"

struct pad_info {
  std::vector<int64_t> shape_4;
  std::vector<int64_t> pads_4;
};

LogicalResult tpu::PadOp::init(InferenceParameter &p) {
  auto info = new pad_info();
  std::vector<int64_t> in_shape = module::getShape(getInput());
  i64_array_t pads_origin = module::getI64Array(getPaddings());
  std::vector<int64_t> pads(in_shape.size() * 2, 0);
  if (module::isNone(getPaddingsT())) {
    pads = *pads_origin;
  }
  auto ret = pad_reset(in_shape, pads, info->shape_4, info->pads_4);
  if (ret == false) {
    UNREACHABLE_THIS("Not Implemented");
  }
  p.handle = (void *)info;
  // set pads
  float *dst = p.outputs[0];
  auto total_num = module::getNumElements(getOutput());
  if (getMode() == tpu::PaddingMode::constant) {
    float val_ = getVal().convertToDouble();
    for (int i = 0; i < total_num; i++) {
      dst[i] = val_;
    }
  }
  return success();
}

void tpu::PadOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto info = (pad_info *)p.handle;
    delete info;
    p.handle = nullptr;
  }
}

LogicalResult tpu::PadOp::inference(InferenceParameter &p) {
  auto p_info = (pad_info *)p.handle;
  auto pad_mode = getMode();
  if (!module::isNone(getPaddingsT())) {
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

  if (pad_mode == tpu::PaddingMode::constant) {
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
    assert(pads[0] == pads[4] && pads[1] == pads[5] && pads[0] == 0 &&
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

        if (pad_mode == tpu::PaddingMode::reflect) {
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
        } else if (pad_mode == tpu::PaddingMode::edge) {
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

LogicalResult tpu::PadOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                    int64_t out_idx, int64_t out_slice) {
  i64_array_t pads = module::getI64Array(getPaddings());
  auto out_shape = module::getShape(getOutput());
  in_idx = out_idx ? out_idx - pads->at(2) : 0;
  if (out_idx == 0) {
    if (out_slice == out_shape[2]) {
      in_slice = out_slice - pads->at(2) - pads->at(6);
    } else {
      in_slice = out_slice - pads->at(2);
    }
  } else if (out_idx + out_slice == out_shape[2]) {
    in_slice = out_slice - pads->at(6);
  } else {
    in_slice = out_slice;
  }
  if (in_slice <= 0 || in_idx < 0) {
    return failure();
  }
  return success();
}

LogicalResult tpu::PadOp::BackwardW(int64_t &in_idx, int64_t &in_slice,
                                    int64_t out_idx, int64_t out_slice) {
  i64_array_t pads = module::getI64Array(getPaddings());
  auto out_shape = module::getShape(getOutput());
  in_idx = out_idx ? out_idx - pads->at(3) : 0;
  if (out_idx == 0) {
    if (out_slice == out_shape[3]) {
      in_slice = out_slice - pads->at(3) - pads->at(7);
    } else {
      in_slice = out_slice - pads->at(3);
    }
  } else if (out_idx + out_slice == out_shape[3]) {
    in_slice = out_slice - pads->at(7);
  } else {
    in_slice = out_slice;
  }
  if (in_slice <= 0 || in_idx < 0) {
    return failure();
  }
  return success();
}

LogicalResult tpu::PadOp::LocalGenSupport() {
  if (module::isCV18xx()) {
    if (getMode() != tpu::PaddingMode::constant ||
        module::getShape(getInput()).size() != 4) {
      return failure();
    }
    i64_array_t pads = module::getI64Array(getPaddings());
    if (pads->at(0) != 0 || pads->at(4) != 0) {
      return failure();
    }
    return success();
  }
  return failure();
}

void tpu::PadOp::assign_fw_param(void *param) {
  fw_pad_layer_param_t fw_pad_layer_param = {0};
  fw_pad_layer_param.ic = module::getShape(getInput())[1];
  fw_pad_layer_param.pad_val = getVal().convertToDouble();
  fw_pad_layer_param.pad_mode = (int)getMode();

  auto pads = module::getI64Array(getPaddings());
  if (pads->size() > 8 || fw_pad_layer_param.pad_mode > 1) {
    llvm_unreachable("not support");
  }
  auto dims = module::getShape(getInput()).size();
  assert(dims * 2 == pads->size());
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < dims; ++j) {
      fw_pad_layer_param.paddings[j][i] = pads->at(i * dims + j);
    }
  }
  if (module::isUniformQuantized(getInput())) {
    int in_shape[MAX_SHAPE_DIMS];
    module::getGlobalShape(getInput(), in_shape);
    auto buffer_offset = ceiling_func(in_shape[0], 4) * 4 * in_shape[1] *
                         in_shape[2] * in_shape[3];
    fw_pad_layer_param.input_global_offset_1N_buf =
        module::getAddress(getBuffer());
    fw_pad_layer_param.output_global_offset_1N_buf =
        module::getAddress(getBuffer()) + buffer_offset;
  }
  memcpy(param, &fw_pad_layer_param, sizeof(fw_pad_layer_param_t));
}

mlir::Type tpu::PadOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  auto op = getOperation();
  if (opd_idx == 1) {
    // indices
    auto opd = op->getOperand(1);
    auto in_op = opd.getDefiningOp();
    if (in_op != nullptr && isa<top::WeightOp, top::NoneOp>(in_op)) {
      return do_nothing(mode);
    }
    auto stype = module::getStorageType(opd);
    if (stype.isIntOrIndex()) {
      return do_nothing(mode);
    }
    mode = TypeCastMode::DO_CAST;
    auto bitwidth = 32;
    return Builder(op).getIntegerType(bitwidth);
  }
  return type_verify_case_same(op, opd_idx, mode);
}

bool tpu::PadOp::support_multi_core() { return false; }
