//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"

pool_attr_t top::MaxPoolOp::parseParam() {
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
  return p;
}

int64_t top::MaxPoolOp::getFLOPs() {
  auto attr = parseParam();
  auto extra = attr.do_relu ? 1 : 0;
  return module::getNumElements(getOutput()) *
         (attr.kd * attr.kh * attr.kw + extra);
}

LogicalResult top::MaxPoolOp::init(InferenceParameter &p) {
  auto pooling = new Pooling();
  auto attr = parseParam();
  pooling->setup(p.inputs[0], p.outputs[0], attr, false);
  p.handle = (void *)pooling;
  return success();
}

void top::MaxPoolOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto pooling = (Pooling *)p.handle;
    delete pooling;
    p.handle = nullptr;
  }
  return;
}

LogicalResult top::MaxPoolOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto pooling = (Pooling *)p.handle;
  pooling->run();
  if (getDoRelu()) {
    auto limit = getReluLimit().convertToDouble();
    function_relu(p.outputs[0], p.outputs[0],
                  module::getNumElements(getOutput()), limit);
  }
  return success();
}

void top::MaxPoolOp::shape_inference() {
  auto input_shape = module::getShape(getInput());
  auto kernel_shape = module::getI64Array(getKernelShape());
  if (kernel_shape->size() == 0) {
    // for onnx GlobalMaxPool
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
  module::setShapeOrVerify(getOutput(), out_shape);
}
