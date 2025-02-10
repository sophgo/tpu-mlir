//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"

inline static unsigned start_index(unsigned oidx, unsigned olen,
                                   unsigned ilen) {
  return oidx * ilen / olen;
}

inline static unsigned end_index(unsigned oidx, unsigned olen, unsigned ilen) {
  return ((oidx + 1) * ilen + olen - 1) / olen;
}
int64_t top::AvgPoolOp::getFLOPs() {
  auto attr = parseParam();
  return module::getNumElements(getOutput()) *
         (attr.kd * attr.kh * attr.kw + (attr.do_relu ? 1 : 0));
}

pool_attr_t top::AvgPoolOp::parseParam() {
  pool_attr_t p = {0};
  auto ishape = getInput().getType().dyn_cast<RankedTensorType>().getShape();
  auto oshape = getOutput().getType().dyn_cast<RankedTensorType>().getShape();
  auto kernel = module::getI64Array(getKernelShape());
  auto stride = module::getI64Array(getStrides());
  auto pad = module::getI64Array(getPads());
  // for dynamic case.
  auto kernel_shape = module::getI64Array(getKernelShape());
  std::vector<int64_t> new_pads(pad->begin(), pad->end());
  if (getCeilMode().has_value() && getCeilMode().value()) {
    auto kernel_len = kernel_shape->size();
    for (uint32_t i = 0; i < kernel_len; i++) {
      auto remain_pixel =
          (ishape[i + 2] + 2 * new_pads[i] - kernel_shape->at(i)) %
          stride->at(i);
      if (remain_pixel > 0) {
        new_pads[i + kernel_len] += (stride->at(i) - remain_pixel);
      }
    }
  }
  if (getKernelShape().size() == 3) {
    p.n = ishape[0];
    p.c = ishape[1];
    p.id = ishape[2];
    p.ih = ishape[3];
    p.iw = ishape[4];
    p.od = oshape[2];
    p.oh = oshape[3];
    p.ow = oshape[4];
    p.kd = kernel->at(0);
    p.kh = kernel->at(1);
    p.kw = kernel->at(2);
    p.sd = stride->at(0);
    p.sh = stride->at(1);
    p.sw = stride->at(2);
    p.pad_d = pad->at(0);
    p.pad_h = pad->at(1);
    p.pad_w = pad->at(2);
    p.pad_d_after = new_pads[3];
    p.pad_h_after = new_pads[4];
    p.pad_w_after = new_pads[5];
  } else if (getKernelShape().size() == 2) {
    p.id = 1;
    p.od = 1;
    p.kd = 1;
    p.sd = 1;
    module::getNCHW(ishape, p.n, p.c, p.ih, p.iw);
    module::getNCHW(oshape, p.n, p.c, p.oh, p.ow);
    p.kh = kernel->at(0);
    p.kw = kernel->at(1);
    p.sh = stride->at(0);
    p.sw = stride->at(1);
    p.pad_h = pad->at(0);
    p.pad_w = pad->at(1);
    p.pad_h_after = new_pads[2];
    p.pad_w_after = new_pads[3];
  } else if (getKernelShape().size() == 1) {
    p.id = 1;
    p.od = 1;
    p.kd = 1;
    p.kw = 1;
    p.sd = 1;
    p.sw = 1;
    module::getNCHW(ishape, p.n, p.c, p.ih, p.iw);
    module::getNCHW(oshape, p.n, p.c, p.oh, p.ow);
    p.kh = kernel->at(0);
    p.sh = stride->at(0);
    p.pad_h = pad->at(0);
    p.pad_h_after = new_pads[1];
  }
  p.pad_value = getPadValue();
  p.do_relu = getDoRelu();
  p.relu_limit = getReluLimit().convertToDouble();
  p.is_global = p.id == p.kd && p.ih == p.kh && p.iw == p.kw && p.od == 1 &&
                p.oh == 1 && p.ow == 1;
  p.count_include_pad = getCountIncludePad();
  p.is_adaptive = getIsAdaptive();
  return p;
}

LogicalResult top::AvgPoolOp::init(InferenceParameter &p) {
  auto pooling = new Pooling();
  auto attr = parseParam();
  pooling->setup(p.inputs[0], p.outputs[0], attr, true);
  p.handle = (void *)pooling;
  return success();
}

void top::AvgPoolOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto pooling = (Pooling *)p.handle;
    delete pooling;
    p.handle = nullptr;
  }
  return;
}

LogicalResult top::AvgPoolOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  if (getIsAdaptive()) {
    // Only consider adaptive_avgpool2d here, if encounter with other case,
    // please fix it
    auto input_shape = module::getShape(getInput());
    auto output_shape = module::getShape(getOutput());
    int input_dim = input_shape.size();
    int output_dim = output_shape.size();
    // auto input = input_.contiguous();
    // auto output = output_.contiguous();

    auto input_data = p.inputs[0];
    auto output_data = p.outputs[0];

    int64_t ndim = input_shape.size();
    // treat batch size and channels as one dimension
    // notice : 4D maybe (N, C, H, W)(pool2d) or (C, D, H, W)(pool3d)
    ASSERT_THIS(ndim == 3 || ndim == 4);
    int64_t channels =
        ndim == 3 ? input_shape[0] : input_shape[0] * input_shape[1];
    int64_t input_height = input_shape[input_dim - 2];
    int64_t input_width = input_shape[input_dim - 1];
    int64_t output_height = output_shape[output_dim - 2];
    int64_t output_width = output_shape[output_dim - 1];

// parallel on dim of N, C
#pragma omp parallel for schedule(static, omp_schedule(channels))
    for (int c_idx = 0; c_idx < channels; c_idx++) {
      int64_t input_idx = c_idx * input_height * input_width;
      int64_t output_idx = c_idx * output_height * output_width;

      for (int oh_idx = 0; oh_idx < output_height; oh_idx++) {
        int64_t ih0 = start_index(oh_idx, output_height, input_height);
        int64_t ih1 = end_index(oh_idx, output_height, input_height);
        int64_t kh = ih1 - ih0;

        for (int ow_idx = 0; ow_idx < output_width; ow_idx++) {
          int64_t iw0 = start_index(ow_idx, output_width, input_width);
          int64_t iw1 = end_index(ow_idx, output_width, input_width);
          int64_t kw = iw1 - iw0;

          // compute local average
          float sum = 0.f;
          for (int ih_idx = ih0; ih_idx < ih1; ih_idx++)
            for (int iw_idx = iw0; iw_idx < iw1; iw_idx++) {
              sum += input_data[input_idx + ih_idx * input_width + iw_idx];
            }
          output_data[output_idx + oh_idx * output_width + ow_idx] =
              (sum / kh / kw);
        }
      }
    }
  } else {
    auto pooling = (Pooling *)p.handle;
    pooling->run();
  }
  if (getDoRelu()) {
    auto limit = getReluLimit().convertToDouble();
    function_relu(p.outputs[0], p.outputs[0],
                  module::getNumElements(getOutput()), limit);
  }
  return success();
}

void top::AvgPoolOp::shape_inference() {
  auto input_shape = module::getShape(getInput());
  auto kernel_shape = module::getI64Array(getKernelShape());
  if (kernel_shape->size() == 0) {
    // for onnx GlobalAvgPool
    auto num_dim = input_shape.size() - 2;
    ASSERT_THIS(num_dim > 0);
    std::vector<int64_t> vkernel_shape;
    std::vector<int64_t> vstrides(num_dim, 1);
    std::vector<int64_t> vpads(2 * num_dim, 0);
    for (uint32_t i = 2; i < input_shape.size(); i++) {
      vkernel_shape.push_back(input_shape[i]);
    }
    auto builder = OpBuilder(getContext());
    setKernelShapeAttr(builder.getI64ArrayAttr(vkernel_shape));
    setStridesAttr(builder.getI64ArrayAttr(vstrides));
    setPadsAttr(builder.getI64ArrayAttr(vpads));
    kernel_shape = module::getI64Array(getKernelShape());
  }
  ASSERT_THIS(input_shape.size() > 2);
  int spacial_rank = input_shape.size() - 2;
  ASSERT_THIS(spacial_rank == getKernelShape().size());
  ASSERT_THIS(getPads().size() == spacial_rank * 2);
  llvm::SmallVector<int64_t> out_shape;
  out_shape.push_back(input_shape[0]);
  out_shape.push_back(input_shape[1]);
  auto input_spacial_shape = llvm::ArrayRef(&input_shape[2], spacial_rank);
  auto pads = module::getI64Array(getPads());
  auto strides = module::getI64Array(getStrides());
  // for AutoPad
  std::vector<int64_t> new_pads(pads->begin(), pads->end());
  if (getAutoPad().has_value()) {
    if (module::isDynamic()) {
      auto auto_pad_mode = getAutoPad().value();
      if (auto_pad_mode == "SAME_UPPER" || auto_pad_mode == "SAME_LOWER")
        llvm_unreachable("Auto_pad is a DEPRECATED attribute. It's not support "
                         "in BM1688 backend.");
    }
    set_auto_pad(getAutoPad().value(), input_shape, *kernel_shape, *strides,
                 new_pads);
    removeAutoPadAttr();
  }

  // for CeilMode
  if (getCeilMode().has_value() && getCeilMode().value()) {
    auto kernel_len = kernel_shape->size();
    for (uint32_t i = 0; i < kernel_len; i++) {
      auto remain_pixel =
          (input_shape[i + 2] + 2 * new_pads[i] - kernel_shape->at(i)) %
          strides->at(i);
      if (remain_pixel > 0) {
        new_pads[i + kernel_len] += (strides->at(i) - remain_pixel);
      }
    }
  }
  if (!module::isDynamic()) {
    removeCeilModeAttr();
    auto builder = OpBuilder(getContext());
    setPadsAttr(builder.getI64ArrayAttr(new_pads));
  }

  for (int i = 0; i < spacial_rank; i++) {
    auto out_dim = (input_spacial_shape[i] + new_pads[i] +
                    new_pads[i + spacial_rank] - kernel_shape->at(i)) /
                       strides->at(i) +
                   1;
    out_shape.push_back(out_dim);
  }
  if (getKeepdims() == false) {
    while (out_shape.size() > 2) {
      if (out_shape.back() == 1) {
        out_shape.pop_back();
      } else {
        break;
      }
    }
  }
  module::setShapeOrVerify(getOutput(), out_shape);
}
