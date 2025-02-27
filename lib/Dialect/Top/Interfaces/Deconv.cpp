//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"

deconv_attr_t top::DeconvOp::parseParam() {
  deconv_attr_t p = {0};
  auto ishape = getInput().getType().cast<RankedTensorType>().getShape();
  auto oshape = getOutput().getType().cast<RankedTensorType>().getShape();
  auto kernel = module::getI64Array(getKernelShape());
  auto stride = module::getI64Array(getStrides());
  auto dilation =
      module::getI64Array(getDilations(), getKernelShape().size(), 1);
  auto pad = module::getI64Array(getPads());
  auto output_padding =
      module::getI64Array(getOutputPadding(), getKernelShape().size(), 0);
  p.do_relu = getDoRelu();
  p.relu_limit = getReluLimit().convertToDouble();
  p.with_bias = !getBias().getType().isa<NoneType>();
  p.g = getGroup();
  p.n = ishape[0];
  p.ic = ishape[1];
  p.oc = oshape[1];
  auto dims = ishape.size() - 2;
  if (dims == 3) {
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
    p.dd = dilation->at(0);
    p.dh = dilation->at(1);
    p.dw = dilation->at(2);
    p.pad_d = pad->at(0);
    p.pad_h = pad->at(1);
    p.pad_w = pad->at(2);
    p.pad_d_after = pad->at(3);
    p.pad_h_after = pad->at(4);
    p.pad_w_after = pad->at(5);
    p.output_pad_d = output_padding->at(0);
    p.output_pad_h = output_padding->at(1);
    p.output_pad_w = output_padding->at(2);
  } else if (dims == 2) {
    p.ih = ishape[2];
    p.iw = ishape[3];
    p.oh = oshape[2];
    p.ow = oshape[3];
    p.kh = kernel->at(0);
    p.kw = kernel->at(1);
    p.sh = stride->at(0);
    p.sw = stride->at(1);
    p.dh = dilation->at(0);
    p.dw = dilation->at(1);
    p.pad_h = pad->at(0);
    p.pad_w = pad->at(1);
    p.pad_h_after = pad->at(2);
    p.pad_w_after = pad->at(3);
    p.output_pad_h = output_padding->at(0);
    p.output_pad_w = output_padding->at(1);
    p.id = 1;
    p.od = 1;
    p.kd = 1;
    p.sd = 1;
    p.dd = 1;
  } else if (dims == 1) {
    p.id = p.od = p.kd = p.dd = p.sd = 1;
    p.iw = p.ow = p.kw = p.dw = p.sw = 1;
    p.ih = ishape[2];
    p.oh = oshape[2];
    p.kh = kernel->at(0);
    p.pad_h = pad->at(0);
    p.pad_h_after = pad->size() > 2 ? pad->at(2) : pad->at(1);
    p.sh = stride->at(0);
    p.dh = dilation->at(0);
    p.output_pad_h = output_padding->at(0);
  }
  p.is_dw = (p.oc == p.ic && p.oc == p.g && p.g > 1);
  return p;
}

deconv_attr_t top::DeconvOp::dynparseParam() {
  auto input_shape = module::getShape(getInput());
  auto filter_shape = module::getShape(getFilter());
  std::vector<int64_t> in_shape{input_shape.begin(), input_shape.end()};

  ASSERT_THIS(in_shape.size() > 2);
  int spacial_rank = in_shape.size() - 2;
  if (spacial_rank != getKernelShape().size()) {
    // have 1d to 2d
    in_shape.push_back(1);
    spacial_rank = in_shape.size() - 2;
    ASSERT_THIS(module::isUnranked(getOutput()) == false);
    // return;
  }
  ASSERT_THIS(in_shape.size() == filter_shape.size());
  ASSERT_THIS(getPads().size() == spacial_rank * 2);
  std::vector<int64_t> out_shape;
  out_shape.push_back(in_shape[0]); // batch size
  bool dyn_weight_reorderd = getDynweightReorderd();

  if (module::isWeight(getOperand(1)) || module::isTrain() ||
      dyn_weight_reorderd) {
    // WeightOp, weight reorder will be finished in origin.mlir generation phase
    // oc, ic
    out_shape.push_back(filter_shape[0] * getGroup());
  } else {
    // ic,
    out_shape.push_back(filter_shape[1] * getGroup());
  }

  auto input_spacial_shape = llvm::ArrayRef(&in_shape[2], spacial_rank);
  auto filter_spacial_shape = llvm::ArrayRef(&filter_shape[2], spacial_rank);
  auto pads = module::getI64Array(getPads());
  auto strides = module::getI64Array(getStrides());
  auto dilation = module::getI64Array(getDilations(), spacial_rank, 1);
  auto output_paddding =
      module::getI64Array(getOutputPadding(), spacial_rank, 0);
  for (int i = 0; i < spacial_rank; i++) {
    auto out_dim = (input_spacial_shape[i] - 1) * strides->at(i) - pads->at(i) -
                   pads->at(i + spacial_rank) +
                   dilation->at(i) * (filter_spacial_shape[i] - 1) +
                   output_paddding->at(i) + 1;
    out_shape.push_back(out_dim);
  }
  module::setShape(getOutput(), out_shape);

  deconv_attr_t p = {0};
  auto ishape = getInput().getType().cast<RankedTensorType>().getShape();
  auto oshape = getOutput().getType().cast<RankedTensorType>().getShape();
  auto kernel = module::getI64Array(getKernelShape());
  auto stride = module::getI64Array(getStrides());
  dilation = module::getI64Array(getDilations(), getKernelShape().size(), 1);
  auto pad = module::getI64Array(getPads());
  auto output_padding =
      module::getI64Array(getOutputPadding(), getKernelShape().size(), 0);
  p.do_relu = getDoRelu();
  p.relu_limit = getReluLimit().convertToDouble();
  p.with_bias = !getBias().getType().isa<NoneType>();
  p.g = getGroup();
  p.n = ishape[0];
  p.ic = ishape[1];
  p.oc = oshape[1];
  auto dims = ishape.size() - 2;
  if (dims == 3) {
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
    p.dd = dilation->at(0);
    p.dh = dilation->at(1);
    p.dw = dilation->at(2);
    p.pad_d = pad->at(0);
    p.pad_h = pad->at(1);
    p.pad_w = pad->at(2);
    p.pad_d_after = pad->at(3);
    p.pad_h_after = pad->at(4);
    p.pad_w_after = pad->at(5);
    p.output_pad_d = output_padding->at(0);
    p.output_pad_h = output_padding->at(1);
    p.output_pad_w = output_padding->at(2);
  } else if (dims == 2) {
    p.ih = ishape[2];
    p.iw = ishape[3];
    p.oh = oshape[2];
    p.ow = oshape[3];
    p.kh = kernel->at(0);
    p.kw = kernel->at(1);
    p.sh = stride->at(0);
    p.sw = stride->at(1);
    p.dh = dilation->at(0);
    p.dw = dilation->at(1);
    p.pad_h = pad->at(0);
    p.pad_w = pad->at(1);
    p.pad_h_after = pad->at(2);
    p.pad_w_after = pad->at(3);
    p.output_pad_h = output_padding->at(0);
    p.output_pad_w = output_padding->at(1);
    p.id = 1;
    p.od = 1;
    p.kd = 1;
    p.sd = 1;
    p.dd = 1;
  } else if (dims == 1) {
    p.id = p.od = p.kd = p.dd = p.sd = 1;
    p.iw = p.ow = p.kw = p.dw = p.sw = 1;
    p.ih = ishape[2];
    p.oh = oshape[2];
    p.kh = kernel->at(0);
    p.pad_h = pad->at(0);
    p.pad_h_after = pad->size() > 2 ? pad->at(2) : pad->at(1);
    p.sh = stride->at(0);
    p.dh = dilation->at(0);
    p.output_pad_h = output_padding->at(0);
  }
  p.is_dw = (p.oc == p.ic && p.oc == p.g && p.g > 1);
  return p;
}

int64_t top::DeconvOp::getFLOPs() {
  auto attr = parseParam();
  auto extra = (attr.with_bias ? 1 : 0) + (attr.do_relu ? 1 : 0);
  return module::getNumElements(getInput()) *
         (attr.kw * attr.kw * attr.oc / attr.g * 2 + extra);
}

LogicalResult top::DeconvOp::init(InferenceParameter &p) {
  auto deconv = new Deconv();
  p.handle = (void *)deconv;
  return success();
}

void top::DeconvOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto deconv = (Deconv *)p.handle;
    delete deconv;
    p.handle = nullptr;
  }
}

LogicalResult top::DeconvOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto deconv = (Deconv *)p.handle;
  auto attr = dynparseParam();
  deconv->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], attr);
  deconv->run();
  return success();
}

void top::DeconvOp::shape_inference() {
  // n, c, w | n, c, h, w | n, c, d, h, w
  auto input_shape = module::getShape(getInput());
  auto filter_shape = module::getShape(getFilter());
  ASSERT_THIS(input_shape.size() == filter_shape.size());
  ASSERT_THIS(input_shape.size() > 2);
  int spacial_rank = input_shape.size() - 2;
  if (spacial_rank != getKernelShape().size()) {
    // have 1d to 2d
    ASSERT_THIS(module::isUnranked(getOutput()) == false);
    return;
  }
  ASSERT_THIS(getPads().size() == spacial_rank * 2);
  llvm::SmallVector<int64_t> out_shape;
  out_shape.push_back(input_shape[0]); // batch size

  if (module::isWeight(getOperand(1)) || module::isTrain()) {
    // WeightOp, weight reorder will be finished in origin.mlir generation phase
    // oc, ic
    out_shape.push_back(filter_shape[0] * getGroup());
  } else {
    // ic,
    out_shape.push_back(filter_shape[1] * getGroup());
  }

  auto input_spacial_shape = llvm::ArrayRef(&input_shape[2], spacial_rank);
  auto filter_spacial_shape = llvm::ArrayRef(&filter_shape[2], spacial_rank);
  auto pads = module::getI64Array(getPads());
  auto strides = module::getI64Array(getStrides());
  auto dilation = module::getI64Array(getDilations(), spacial_rank, 1);
  auto output_paddding =
      module::getI64Array(getOutputPadding(), spacial_rank, 0);
  for (int i = 0; i < spacial_rank; i++) {
    auto out_dim = (input_spacial_shape[i] - 1) * strides->at(i) - pads->at(i) -
                   pads->at(i + spacial_rank) +
                   dilation->at(i) * (filter_spacial_shape[i] - 1) +
                   output_paddding->at(i) + 1;
    out_shape.push_back(out_dim);
  }
  module::setShapeOrVerify(getOutput(), out_shape);
}
