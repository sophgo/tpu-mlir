//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Deconv.h"
#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

void tpu::DeconvOp::weight_reorder_int8_bm1684() {
  deconv_attr_t attr;
  parseParam(&attr);
  auto filterOp = cast<top::WeightOp>(filter().getDefiningOp());
  auto filter_int8 = filterOp.read<int8_t>();
  int new_size = attr.oc * (align_up(attr.ic, 4l)) * attr.kh * attr.kw;
  auto filter_new = std::make_shared<std::vector<int8_t>>(new_size, 0);
  for (int oc_idx = 0; oc_idx < attr.oc; oc_idx++) {
    for (int ic_idx = 0; ic_idx < attr.ic; ic_idx++) {
      for (int k_idx = 0; k_idx < attr.kh * attr.kw; k_idx++) {
        int orig_offset = ic_idx * attr.kh * attr.kw + k_idx +
                          oc_idx * attr.kh * attr.kw * attr.ic;
        int trans_offset = ic_idx + k_idx * align_up(attr.ic, 4l) +
                           oc_idx * (attr.kh * attr.kw * align_up(attr.ic, 4l));
        filter_new->at(trans_offset) = filter_int8->at(orig_offset);
      }
    }
  }
  auto filter_type = filterOp.getType().cast<RankedTensorType>();
  std::vector<int64_t> new_shape = {
      1, attr.oc, attr.kh * attr.kw * align_up(attr.ic, 4l), 1};
  auto new_type =
      RankedTensorType::get(new_shape, filter_type.getElementType());
  auto new_filter = top::WeightOp::create(filter().getDefiningOp(), "reorderd",
                                          *filter_new, new_type);
  setOperand(1, new_filter);
}

void tpu::DeconvOp::codegen_global_int8_bm1684() {
  llvm_unreachable("support later");
}

int64_t tpu::DeconvOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice) {
  // TODO for spicial situation
  return 0;
}

void tpu::DeconvOp::codegen_local_int8_bm1684(int64_t n_step, int64_t h_step) {
  llvm_unreachable("support later");
}
