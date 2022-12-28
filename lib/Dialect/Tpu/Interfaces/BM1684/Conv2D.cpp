//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM1684/WeightReorder.h"
#include "tpu_mlir/Support/Dnnl/Conv.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;
using namespace tpu_mlir::bm1684;

template <>
LogicalResult WeightReorder<tpu::Conv2DOp, int8_t>::matchAndRewrite(
    tpu::Conv2DOp op, PatternRewriter &rewriter) const {
  if (!Module::getStorageType(op.filter()).isInteger(8))
    return failure();

  auto &attr = op.parseParam();
  auto filterOp = cast<top::WeightOp>(op.filter().getDefiningOp());
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
  auto new_filter = top::WeightOp::create(op.filter().getDefiningOp(),
                                          "reorderd", *filter_new, new_type);
  op->setOperand(1, new_filter);
  return success();
}

void tpu::Conv2DOp::codegen_global_bm1684() {
  auto &attr = parseParam();
  if (attr.is_dw) {
    BM1684::instance().dl_nodechip_depthwise_fix8b_forward_parallel(
        Module::getAddress(input()), Module::getAddress(output()),
        Module::getAddress(filter()),
        attr.has_bias ? Module::getAddress(bias()) : 0, attr.n, attr.ic,
        attr.ih, attr.iw, attr.kh, attr.kw, attr.pht, attr.phb, attr.pwl,
        attr.pwr, attr.sh, attr.sw, attr.ins_h, attr.ins_w,
        rshift().value()[0].cast<IntegerAttr>().getInt(), attr.has_bias ? 1 : 0,
        0, 1, 1, 1, 1, attr.do_relu ? 1 : 0,
        (CMD_ID_NODE *)BM1684::instance().cmdid_node);
  } else {
    BM1684::instance().dl_nodechip_conv_forward_parallel_fix8b_with_data_split(
        Module::getAddress(input()), Module::getAddress(output()),
        Module::getAddress(filter()),
        attr.has_bias ? Module::getAddress(bias()) : 0, attr.n, attr.ic,
        attr.ih, attr.iw, attr.groups, attr.oc, attr.kh, attr.kw, attr.dh,
        attr.dw, attr.pht, attr.phb, attr.pwl, attr.pwr, attr.sh, attr.sw,
        attr.has_bias ? 1 : 0, 0, attr.do_relu ? 1 : 0, 0, 1, 0, 0,
        rshift().value()[0].cast<IntegerAttr>().getInt(), 1, 1, 1, 3, 0, 0, 0,
        0, 0, (CMD_ID_NODE *)BM1684::instance().cmdid_node);
  }
}

int64_t tpu::Conv2DOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  // TODO for spicial situation
  return 0;
}

void tpu::Conv2DOp::codegen_local_bm1684(int64_t n_step, int64_t h_step) {
  llvm_unreachable("Not Implemented");
}
