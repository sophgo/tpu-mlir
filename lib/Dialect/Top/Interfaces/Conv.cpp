//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"

conv_attr_t top::ConvOp::parseParam() {
  conv_attr_t p = {0};
  auto i_s = getInput().getType().cast<RankedTensorType>().getShape();
  auto o_s = getOutput().getType().cast<RankedTensorType>().getShape();
  p.do_relu = getDoRelu();
  p.relu_limit = getReluLimit().convertToDouble();
  p.has_bias = !getBias().getType().isa<NoneType>();
  p.weight_is_coeff = getWeightIsCoeff();
  auto kernel = module::getI64Array(getKernelShape());
  auto pads_v = module::getI64Array(getPads());
  auto strides_v = module::getI64Array(getStrides());
  auto dilation = module::getI64Array(getDilations(), kernel->size(), 1);
  auto ins = module::getI64Array(getInserts(), kernel->size(), 0);
  p.n = i_s[0];
  p.ic = i_s[1];
  p.oc = o_s[1];
  p.dims = i_s.size() - 2;
  if (p.dims == 3) {
    // 3d conv
    p.id = i_s[2];
    p.ih = i_s[3];
    p.iw = i_s[4];
    p.od = o_s[2];
    p.oh = o_s[3];
    p.ow = o_s[4];
    p.kd = kernel->at(0);
    p.kh = kernel->at(1);
    p.kw = kernel->at(2);
    p.pdf = pads_v->at(0);
    p.pht = pads_v->at(1);
    p.pwl = pads_v->at(2);
    p.pdb = pads_v->at(3);
    p.phb = pads_v->at(4);
    p.pwr = pads_v->at(5);
    p.sd = strides_v->at(0);
    p.sh = strides_v->at(1);
    p.sw = strides_v->at(2);
    p.dd = dilation->at(0);
    p.dh = dilation->at(1);
    p.dw = dilation->at(2);
    p.ins_d = ins->at(0);
    p.ins_h = ins->at(1);
    p.ins_w = ins->at(2);
  } else if (p.dims == 2) {
    // 2d conv
    p.id = p.od = p.kd = p.dd = p.sd = 1;
    p.ih = i_s[2];
    p.iw = i_s[3];
    p.oh = o_s[2];
    p.ow = o_s[3];
    p.kh = kernel->at(0);
    p.kw = kernel->at(1);
    p.pht = pads_v->at(0);
    p.pwl = pads_v->at(1);
    p.phb = pads_v->at(2);
    p.pwr = pads_v->at(3);
    p.sh = strides_v->at(0);
    p.sw = strides_v->at(1);
    p.dh = dilation->at(0);
    p.dw = dilation->at(1);
    p.ins_h = ins->at(0);
    p.ins_w = ins->at(1);
  } else if (p.dims == 1) {
    p.id = p.od = p.kd = p.dd = p.sd = 1;
    p.iw = p.ow = p.kw = p.dw = p.sw = 1;
    p.ih = i_s[2];
    p.oh = o_s[2];
    p.kh = kernel->at(0);
    p.pht = pads_v->at(0);
    p.phb = pads_v->size() > 2 ? pads_v->at(2) : pads_v->at(1);
    p.sh = strides_v->at(0);
    p.dh = dilation->at(0);
    p.ins_h = ins->at(0);
  }
  p.groups = getGroup();
  p.is_dw = (p.oc == p.ic && p.oc == p.groups && p.groups > 1);
  return p;
}

conv_attr_t top::ConvOp::dynparseParam() {
  auto input_shape = module::getShape(getInput());
  auto filter_shape = module::getShape(getFilter());
  int spacial_rank = input_shape.size() - 2;
  std::vector<int64_t> out_shape;
  out_shape.push_back(input_shape[0]);
  out_shape.push_back(filter_shape[0]);
  auto input_spacial_shape = llvm::ArrayRef(&input_shape[2], spacial_rank);
  auto filter_spacial_shape = llvm::ArrayRef(&filter_shape[2], spacial_rank);
  auto pads = module::getI64Array(getPads());
  auto strides = module::getI64Array(getStrides());
  auto dilation =
      module::getI64Array(getDilations(), getKernelShape().size(), 1);
  std::vector<int64_t> new_pads(pads->begin(), pads->end());
  bool flag = input_shape.size() == filter_shape.size() ? 1 : 0;
  for (int i = 0; i < spacial_rank; i++) {
    auto out_dim =
        (input_spacial_shape[i] + pads->at(i) + pads->at(i + spacial_rank) -
         dilation->at(i) * (filter_spacial_shape[i] - 1) - 1) /
            strides->at(i) +
        1;
    if (!flag) {
      out_dim = (input_spacial_shape[i] + pads->at(i) +
                 pads->at(i + spacial_rank + 1) -
                 dilation->at(i) * (filter_spacial_shape[i] - 1) - 1) /
                    strides->at(i) +
                1;
    }
    out_shape.push_back(out_dim);
  }
  module::setShape(getOutput(), out_shape);

  conv_attr_t p = {0};
  auto i_s = getInput().getType().cast<RankedTensorType>().getShape();
  auto o_s = getOutput().getType().cast<RankedTensorType>().getShape();
  p.do_relu = getDoRelu();
  p.relu_limit = getReluLimit().convertToDouble();
  p.has_bias = !getBias().getType().isa<NoneType>();
  p.weight_is_coeff = getWeightIsCoeff();
  auto kernel = module::getI64Array(getKernelShape());
  auto pads_v = module::getI64Array(getPads());
  auto strides_v = module::getI64Array(getStrides());
  dilation = module::getI64Array(getDilations(), kernel->size(), 1);
  auto ins = module::getI64Array(getInserts(), kernel->size(), 0);
  p.n = i_s[0];
  p.ic = i_s[1];
  p.oc = o_s[1];
  p.dims = i_s.size() - 2;
  if (p.dims == 3) {
    // 3d conv
    p.id = i_s[2];
    p.ih = i_s[3];
    p.iw = i_s[4];
    p.od = o_s[2];
    p.oh = o_s[3];
    p.ow = o_s[4];
    p.kd = kernel->at(0);
    p.kh = kernel->at(1);
    p.kw = kernel->at(2);
    p.pdf = pads_v->at(0);
    p.pht = pads_v->at(1);
    p.pwl = pads_v->at(2);
    p.pdb = pads_v->at(3);
    p.phb = pads_v->at(4);
    p.pwr = pads_v->at(5);
    p.sd = strides_v->at(0);
    p.sh = strides_v->at(1);
    p.sw = strides_v->at(2);
    p.dd = dilation->at(0);
    p.dh = dilation->at(1);
    p.dw = dilation->at(2);
    p.ins_d = ins->at(0);
    p.ins_h = ins->at(1);
    p.ins_w = ins->at(2);
  } else if (p.dims == 2) {
    // 2d conv
    p.id = p.od = p.kd = p.dd = p.sd = 1;
    p.ih = i_s[2];
    p.iw = i_s[3];
    p.oh = o_s[2];
    p.ow = o_s[3];
    p.kh = kernel->at(0);
    p.kw = kernel->at(1);
    p.pht = pads_v->at(0);
    p.pwl = pads_v->at(1);
    p.phb = pads_v->at(2);
    p.pwr = pads_v->at(3);
    p.sh = strides_v->at(0);
    p.sw = strides_v->at(1);
    p.dh = dilation->at(0);
    p.dw = dilation->at(1);
    p.ins_h = ins->at(0);
    p.ins_w = ins->at(1);
  } else if (p.dims == 1) {
    p.id = p.od = p.kd = p.dd = p.sd = 1;
    p.iw = p.ow = p.kw = p.dw = p.sw = 1;
    p.ih = i_s[2];
    p.oh = o_s[2];
    p.kh = kernel->at(0);
    p.pht = pads_v->at(0);
    p.phb = pads_v->size() > 2 ? pads_v->at(2) : pads_v->at(1);
    p.sh = strides_v->at(0);
    p.dh = dilation->at(0);
    p.ins_h = ins->at(0);
  }
  p.groups = getGroup();
  p.is_dw = (p.oc == p.ic && p.oc == p.groups && p.groups > 1);
  return p;
}

int64_t top::ConvOp::getFLOPs() {
  auto attr = parseParam();
  auto extra = (attr.has_bias ? 1 : 0) + (attr.do_relu ? 1 : 0);
  return module::getNumElements(getOutput()) *
         (attr.kd * attr.kh * attr.kw * attr.ic / attr.groups * 2 + extra);
}

LogicalResult top::ConvOp::init(InferenceParameter &p) {
  auto conv = new Conv();
  p.handle = (void *)conv;
  return success();
}

void top::ConvOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto conv = (Conv *)p.handle;
    delete conv;
    p.handle = nullptr;
  }
}

LogicalResult top::ConvOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto conv = (Conv *)p.handle;
  auto attr = dynparseParam();
  conv->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], attr);
  conv->run();
  return success();
}

LogicalResult top::ConvOp::backward_weight(InferenceParameter &p,
                                           InferenceParameter &p_back) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto conv = (Conv *)p.handle;
  conv->run_backw(p_back.inputs[0], p_back.outputs[0]);

  return success();
}

void top::ConvOp::shape_inference() {
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
  out_shape.push_back(input_shape[0]);
  out_shape.push_back(filter_shape[0]);
  auto input_spacial_shape = llvm::ArrayRef(&input_shape[2], spacial_rank);
  auto filter_spacial_shape = llvm::ArrayRef(&filter_shape[2], spacial_rank);
  auto pads = module::getI64Array(getPads());
  auto strides = module::getI64Array(getStrides());
  auto dilation = module::getI64Array(getDilations(), spacial_rank, 1);
  std::vector<int64_t> new_pads(pads->begin(), pads->end());
  if (getAutoPad().has_value()) {
    auto kernel_shape = module::getI64Array(getKernelShapeAttr());
    set_auto_pad(getAutoPad().value(), input_shape, *kernel_shape, *strides,
                 new_pads);
    auto builder = OpBuilder(getContext());
    setPadsAttr(builder.getI64ArrayAttr(new_pads));
    removeAutoPadAttr();
    pads = module::getI64Array(getPads());
  }

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