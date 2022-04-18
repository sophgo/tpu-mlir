#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Backend/BM168x/BM1684.h"
#include "sophgo/Support/MathUtils.h"
#include "sophgo/Support/Helper/Module.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;
using namespace sophgo::backend;

void tpu::ConvOp::weight_reorder_int8_bm1684() {
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, relu;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, relu);
  auto filterOp = cast<top::WeightOp>(filter().getDefiningOp());
  auto filter_int8 = filterOp.read<int8_t>();
  int new_size = oc * (align_up(ic, 4l)) * kh * kw;
  auto filter_new = std::make_shared<std::vector<int8_t>>(new_size, 0);
  for (int oc_idx = 0; oc_idx < oc; oc_idx++) {
    for (int ic_idx = 0; ic_idx < ic; ic_idx++) {
      for (int k_idx = 0; k_idx < kh * kw; k_idx++) {
        int orig_offset = ic_idx * kh * kw + k_idx + oc_idx * kh * kw * ic;
        int trans_offset =
            ic_idx + k_idx * align_up(ic, 4l) + oc_idx * (kh * kw * align_up(ic, 4l));
        filter_new->at(trans_offset) = filter_int8->at(orig_offset);
      }
    }
  }
  auto filter_type = filterOp.getType().cast<RankedTensorType>();
  std::vector<int64_t> new_shape = {1, oc, kh * kw * align_up(ic, 4l), 1};
  auto new_type =
      RankedTensorType::get(new_shape, filter_type.getElementType());
  auto new_filter = top::WeightOp::create(filter().getDefiningOp(), "reorderd",
                                          *filter_new, new_type);
  setOperand(1, new_filter);
}
