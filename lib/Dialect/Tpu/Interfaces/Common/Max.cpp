//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Interfaces/IndexingMapsInterface.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"

LogicalResult tpu::MaxOp::init(InferenceParameter &p) {
  auto output_shape = computer_broadcast_shape(getOperation());
  module::setShape(getOutput(), output_shape);
  auto binary = new Binary();
  auto lhs_shape = module::getShape(getInputs()[0]);
  auto rhs_shape = module::getShape(getInputs()[1]);

  (*binary)
      .hs(p.inputs[0], p.inputs[1], lhs_shape, rhs_shape)
      .dst(p.outputs[0], module::getShape(getOutput()))
      .algorithem(algorithm::binary_max)
      .setup();
  p.handle = (void *)binary;
  return success();
}

void tpu::MaxOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto binary = (Binary *)p.handle;
    delete binary;
    p.handle = nullptr;
  }
}

LogicalResult tpu::MaxOp::inference(InferenceParameter &p) {
  auto output_shape = computer_broadcast_shape(getOperation());
  module::setShape(getOutput(), output_shape);
  auto binary = new Binary();
  auto lhs_shape = module::getShape(getInputs()[0]);
  auto rhs_shape = module::getShape(getInputs()[1]);

  (*binary)
      .hs(p.inputs[0], p.inputs[1], lhs_shape, rhs_shape)
      .dst(p.outputs[0], module::getShape(getOutput()))
      .algorithem(algorithm::binary_max)
      .setup();

  binary->run();
  return success();
}

LogicalResult tpu::MaxOp::LocalGenSupport() {
  if (module::isCV18xx()) {
    return failure();
  }
  return BroadCastBinaryLocalGenSupport(getOperation());
}

void tpu::MaxOp::assign_sec_info_kernel(
    group_type_t group_type, local_sec_info_t &sec_info,
    std::vector<group_info_t> &in_group_infos,
    std::vector<group_info_t> &out_group_infos) {
  memset(&sec_info, 0, sizeof(local_sec_info_t));
  sec_info.group_type = group_type;
  int64_t n0, c0, d0, h0, w0, n1, c1, d1, h1, w1, on, oc, od, oh, ow;
  auto input0 = getOperand(0);
  auto input1 = getOperand(1);
  auto output = getResult();
  module::getNCDHW(input0, n0, c0, d0, h0, w0, group_type);
  module::getNCDHW(input1, n1, c1, d1, h1, w1, group_type);
  module::getNCDHW(output, on, oc, od, oh, ow, group_type);
  auto gi = out_group_infos[0];
  auto in0_gi = in_group_infos[0];
  auto in1_gi = in_group_infos[1];
  sec_info.n_slice = std::max(in0_gi.n_slice, in1_gi.n_slice);
  sec_info.c_slice = std::max(in0_gi.c_slice, in1_gi.c_slice);
  sec_info.d_slice = std::max(in0_gi.d_slice, in1_gi.d_slice);
  sec_info.h_slice = gi.h_slice;
  // set "w_slice" for 5d broadcast (i.e., broadcast across both h and w dims)
  sec_info.w_slice = std::max(in0_gi.w_slice, in1_gi.w_slice) >= gi.w_slice
                         ? gi.w_slice
                         : std::max(in0_gi.w_slice, in1_gi.w_slice);
  sec_info.n_idx = std::max(in0_gi.n_idx, in1_gi.n_idx);
  sec_info.d_idx = std::max(in0_gi.d_idx, in1_gi.d_idx);
  sec_info.c_idx = std::max(in0_gi.c_idx, in1_gi.c_idx);
  sec_info.is_c_split =
      !(std::max(in0_gi.c_idx, in1_gi.c_idx) == 0 &&
        std::max(in0_gi.c_slice, in1_gi.c_slice) == std::max(c0, c1));
  sec_info.h_idx = std::max(in0_gi.h_idx, in1_gi.h_idx);
  sec_info.is_h_split =
      !(std::max(in0_gi.h_idx, in1_gi.h_idx) == 0 &&
        std::max(in0_gi.h_slice, in1_gi.h_slice) == std::max(h0, h1));
  sec_info.w_idx = std::max(in0_gi.w_idx, in1_gi.w_idx);
  sec_info.is_w_split =
      !(std::max(in0_gi.w_idx, in1_gi.w_idx) == 0 &&
        std::max(in0_gi.w_slice, in1_gi.w_slice) == std::max(w0, w1));
  sec_info.out_n_slice = gi.n_slice;
  sec_info.out_h_idx = gi.h_idx;
  sec_info.out_h_slice = gi.h_slice;
  sec_info.out_w_idx = gi.w_idx;
  sec_info.out_w_slice = gi.w_slice;
}

void tpu::MaxOp::assign_fw_param(void *param) {
  IR_PARAM_BROADCAST_BINARY(BINARY_MAX);
}

ArrayAttr tpu::MaxOp::getIndexingMaps() {
  return getBinaryIndexingMaps(getOperation());
};

bool tpu::MaxOp::support_multi_core() { return false; }
