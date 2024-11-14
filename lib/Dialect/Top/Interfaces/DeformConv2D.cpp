//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/DeformConv2D.h"

deform_conv2d_attr_t top::DeformConv2DOp::parseParam() {
  deform_conv2d_attr_t attr = {0};
  auto i_s = getInput().getType().cast<RankedTensorType>().getShape();
  auto o_s = getOutput().getType().cast<RankedTensorType>().getShape();
  auto of_s = getOffset().getType().cast<RankedTensorType>().getShape();
  attr.use_mask = getUseMask();
  attr.do_relu = getDoRelu();
  attr.relu_limit = getReluLimit().convertToDouble();
  attr.has_bias = !getBias().getType().isa<NoneType>();
  auto kernel = module::getI64Array(getKernelShape());
  auto pads_v = module::getI64Array(getPads());
  auto strides_v = module::getI64Array(getStrides());
  auto dilation = module::getI64Array(getDilations(), kernel->size(), 1);
  attr.n = i_s[0];
  attr.ic = i_s[1];
  attr.oc = o_s[1];

  attr.id = attr.od = attr.ofd = attr.mkd = attr.kd = attr.dd = attr.sd = 1;
  attr.ih = i_s.size() > 2 ? i_s[2] : 1;
  attr.iw = i_s.size() > 3 ? i_s[3] : 1;
  attr.oh = o_s.size() > 2 ? o_s[2] : 1;
  attr.ow = o_s.size() > 3 ? o_s[3] : 1;
  attr.ofc = of_s[1];
  attr.ofh = of_s[2];
  attr.ofw = of_s[3];
  if (attr.use_mask) {
    auto mk_s = getMask().getType().cast<RankedTensorType>().getShape();
    attr.mkc = mk_s[1];
    attr.mkh = mk_s[2];
    attr.mkw = mk_s[3];
  }
  attr.kh = kernel->at(0);
  attr.kw = kernel->at(1);
  attr.pht = pads_v->at(0);
  attr.pwl = pads_v->at(1);
  attr.phb = pads_v->at(2);
  attr.pwr = pads_v->at(3);
  attr.sh = strides_v->at(0);
  attr.sw = strides_v->at(1);
  attr.dh = dilation->at(0);
  attr.dw = dilation->at(1);

  attr.groups = getGroup();
  attr.deform_groups = getDeformGroup();
  return attr;
}

int64_t top::DeformConv2DOp::getFLOPs() {
  auto attr = parseParam();
  auto extra = (attr.has_bias ? 1 : 0) + (attr.do_relu ? 1 : 0);
  return module::getNumElements(getOutput()) *
         (attr.kd * attr.kh * attr.kw * attr.ic / attr.groups * 2 + 1 + extra);
}

LogicalResult top::DeformConv2DOp::init(InferenceParameter &p) {
  auto num = module::getNumElements(getOutput());
  float *buffer = new float[num];
  p.handle = (void *)buffer;
  return success();
}

void top::DeformConv2DOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    float *buffer = (float *)p.handle;
    delete[] buffer;
    p.handle = nullptr;
  }
}

LogicalResult top::DeformConv2DOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto attr = parseParam();
  processDeformConv2D(p, attr);
  return success();
}

void top::DeformConv2DOp::shape_inference() {
  auto input_shape = module::getShape(getInput());
  auto filter_shape = module::getShape(getFilter());
  auto offset_shape = module::getShape(getOffset());
  ASSERT_THIS(input_shape.size() == filter_shape.size());
  if (getUseMask()) {
    auto mask_shape = module::getShape(getMask());
    ASSERT_THIS(offset_shape.size() == mask_shape.size());
  }
  ASSERT_THIS(input_shape.size() == offset_shape.size());
  ASSERT_THIS(input_shape.size() > 2);
  int spacial_rank = input_shape.size() - 2;
  ASSERT_THIS(spacial_rank == getKernelShape().size());
  ASSERT_THIS(getPads().size() == spacial_rank * 2);
  llvm::SmallVector<int64_t> out_shape;
  out_shape.push_back(input_shape[0]);
  out_shape.push_back(filter_shape[0]);
  auto input_spacial_shape = llvm::ArrayRef(&input_shape[2], spacial_rank);
  auto filter_spacial_shape = llvm::ArrayRef(&filter_shape[2], spacial_rank);
  auto pads = module::getI64Array(getPads());
  auto strides = module::getI64Array(getStrides());
  auto dilation = module::getI64Array(getDilations(), spacial_rank, 1);
  for (int i = 0; i < spacial_rank; i++) {
    auto out_dim =
        (input_spacial_shape[i] + pads->at(i) + pads->at(i + spacial_rank) -
         dilation->at(i) * (filter_spacial_shape[i] - 1) - 1) /
            strides->at(i) +
        1;
    out_shape.push_back(out_dim);
  }
  module::setShapeOrVerify(getOutput(), out_shape);
}
