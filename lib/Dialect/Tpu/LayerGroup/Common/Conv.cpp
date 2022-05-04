#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

LogicalResult tpu::ConvOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                     int64_t out_idx, int64_t out_slice) {
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, do_relu;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, do_relu);
  int kh_with_dh = (kh - 1) * dh + 1;
  in_slice = (out_slice - 1) * sh + (kh_with_dh >= sh ? kh_with_dh : sh);
  in_idx = out_idx * sh - pt;
  LayerGroupInterface::fixSlice(in_idx, in_slice, ih);
  return success();
}

int64_t tpu::ConvOp::getBufferSize(int64_t out_n, int64_t out_c, int64_t out_h,
                                   int64_t out_w, int64_t out_lmem_bytes) {
  if (coeff_merged() == false) {
    return 0;
  }
  return out_lmem_bytes * sizeof(int32_t);
}
