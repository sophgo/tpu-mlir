#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

LogicalResult tpu::AvgPoolOp::Verify() {
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool relu, is_global, count_include_pad;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             relu, is_global, count_include_pad);
  if (is_global == false && (sh > 15 || sw > 15)) {
    return failure();
  }
  return  success();
}

LogicalResult tpu::AvgPoolOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                        int64_t out_idx, int64_t out_slice) {
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool relu, is_global, count_include_pad;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             relu, is_global, count_include_pad);
  in_slice = (out_slice - 1) * sh + kh;
  in_idx = out_idx * sh - pt;
  return success();
}

LogicalResult tpu::MaxPoolOp::Verify() {
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool relu, is_global, count_include_pad;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             relu, is_global, count_include_pad);
  if (is_global == false && (sh > 15 || sw > 15)) {
    return failure();
  }
  return  success();
}

LogicalResult tpu::MaxPoolOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                        int64_t out_idx, int64_t out_slice) {
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool relu, is_global, count_include_pad;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             relu, is_global, count_include_pad);
  in_slice = (out_slice - 1) * sh + kh;
  in_idx = out_idx * sh - pt;
  return success();
}
