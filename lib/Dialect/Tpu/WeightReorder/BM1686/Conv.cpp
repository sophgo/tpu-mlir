#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Backend/BM168x/BM1686.h"
#include "sophgo/Support/MathUtils.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;
using namespace sophgo::backend;

// convert (1, oc, 1, w) to (1, NPU_NUM, 1, DIV_UP(oc, NPU_NUM) * w)
template <typename T>
static void
reshape_coeff_for_broadcast_channel(std::shared_ptr<std::vector<T>> &coeff,
                                    std::vector<int64_t> &shape,
                                    bool align = false) {
  int64_t n, c, h, w;
  Module::getNCHW(shape, n, c, h, w);
  if (n != 1 || h != 1 || c <= BM1686::NPU_NUM) {
    return;
  }
  // convert (1, oc, 1, w) to (1, NPU_NUM, 1, DIV_UP(oc, NPU_NUM) * w)
  int64_t new_c = BM1686::NPU_NUM;
  int type_len = sizeof(T);
  auto c2w = ceiling_func(c, new_c);
  auto old_w_align = align_up(w, BM1686::instance().get_eu_num(type_len));
  int new_w = (align ? old_w_align : w) * (c2w - 1) + w;
  int64_t new_size = new_w * new_c * type_len;
  auto filter_new = std::make_shared<std::vector<T>>(new_size, 0);
  for (int i = 0; i < c2w; i++) {
    for (int j = 0; j < new_c; j++) {
      for (int k = 0; k < w; k++) {
        int src_idx = i * new_c * w + j * w + k;
        int dst_idx = j * new_w + i * (align ? old_w_align : w) + k;
        filter_new->at(dst_idx) = coeff->at(src_idx);
      }
    }
  }
  shape = {1, new_c, 1, new_w};
  coeff = filter_new;
}

// refer to net_compiler: bool BM1686CoeffArranger::ConvWeightArr(GraphEdge*
// edge)
void tpu::ConvOp::weight_reorder_int8_bm1686() {
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, relu;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, relu);
  auto filterOp = filter().getDefiningOp<top::WeightOp>();
  if (is_dw) {
    llvm_unreachable("depthwise should support !!");
  }
  auto elem_type = Module::getStorageType(filter());
  int64_t IC_PARALLEL = 64;
  int64_t merge_w = 0;
  auto type_bytes = elem_type.getIntOrFloatBitWidth() / 8;
  size_t new_bytes = align_up(ic, IC_PARALLEL) * oc * kh * kw * type_bytes;
  auto filter_i8 = filterOp.read_as_byte();
  auto filter_new = std::make_shared<std::vector<uint8_t>>(new_bytes, 0);
  auto kernel_hw = kh * kw;
  int64_t new_ic = ceiling_func(ic, IC_PARALLEL);
  int64_t new_hw = kernel_hw * IC_PARALLEL;
  for (int oc_idx = 0; oc_idx < oc; oc_idx++) {
    for (int ic_idx = 0; ic_idx < new_ic; ic_idx++) {
      for (int k_idx = 0; k_idx < kernel_hw; k_idx++) {
        for (int inner = 0; inner < IC_PARALLEL; inner++) {
          if (ic_idx * IC_PARALLEL + inner >= ic)
            break;
          int orig_offset = oc_idx * ic * kh * kw +
                            (ic_idx * IC_PARALLEL + inner) * kernel_hw + k_idx;
          int trans_offset = oc_idx * new_ic * new_hw + ic_idx * new_hw +
                             k_idx * IC_PARALLEL + inner;
          filter_new->at(trans_offset) = filter_i8->at(orig_offset);
        }
      }
    }
  }
  auto filter_type = filterOp.getType().cast<RankedTensorType>();
  std::vector<int64_t> filter_shape = {1, oc, 1, new_ic * new_hw};
  // refer to net_compier: reshape_coeff_for_broadcast_channel(weight, false);
  reshape_coeff_for_broadcast_channel(filter_new, filter_shape);

  auto filter_w_bytes = filter_shape[3];
  merge_w += filter_w_bytes;
  std::shared_ptr<std::vector<int32_t>> bias_new;
  std::vector<int64_t> bias_shape = {1, oc, 1, 1};
  int64_t bias_w_bytes = 0;
  if (with_bias) {
    auto biasOp = bias().getDefiningOp<top::WeightOp>();
    bias_new = biasOp.read<int32_t>();
    reshape_coeff_for_broadcast_channel(bias_new, bias_shape, false);
    bias_w_bytes = bias_shape[3] * sizeof(int32_t);
    merge_w += bias_w_bytes;
  }

  // add requant op
  auto op = getOperation();
  auto qtype = Quant::getQuantizedType<quant::UniformQuantizedType>(output());
  std::vector<int64_t> quant_shape = {1, oc, 1, 3};
  auto quant_data = std::make_shared<std::vector<int32_t>>(oc * 3, 0);
  auto m_data = Module::getI64Array(multiplier().getValue());
  auto r_data = Module::getI64Array(rshift().getValue());
  for (int i = 0; i < oc; i++) {
    quant_data->at(i * 3) = m_data->at(i);
    quant_data->at(i * 3 + 1) = r_data->at(i);
    quant_data->at(i * 3 + 2) = qtype.getZeroPoint();
  }
  reshape_coeff_for_broadcast_channel(quant_data, quant_shape, true);
  auto quant_w_bytes = quant_shape[3] * sizeof(int32_t);
  merge_w += quant_w_bytes;
  // merge requant/bias/filter
  auto new_coeff =
      std::make_shared<std::vector<int8_t>>(BM1686::NPU_NUM * merge_w, 0);
  std::vector<int64_t> coeff_shape = {1, BM1686::NPU_NUM, 1, merge_w};
  for (int i = 0; i < BM1686::NPU_NUM; i++) {
    auto coeff_ptr = new_coeff->data() + i * merge_w;
    auto quant_ptr = quant_data->data() + i * quant_shape[3];
    auto bias_ptr = with_bias ? bias_new->data() + i * bias_shape[3] : nullptr;
    auto filter_ptr = filter_new->data() + i * filter_shape[3];
    // copy quant
    memcpy(coeff_ptr, quant_ptr, quant_w_bytes);
    coeff_ptr += quant_w_bytes;
    if (with_bias) {
      memcpy(coeff_ptr, bias_ptr, bias_w_bytes);
      coeff_ptr += bias_w_bytes;
    }
    memcpy(coeff_ptr, filter_ptr, filter_w_bytes);
  }
  OpBuilder builder(getContext());
  auto coeff_type = RankedTensorType::get(coeff_shape, builder.getI8Type());
  auto coeff_op = top::WeightOp::create(op, "merge", *new_coeff, coeff_type);
  op->removeAttr("rshift");
  op->removeAttr("multiplier");
  op->setAttr("coeff_merged", builder.getBoolAttr(true));
  op->setOperand(1, coeff_op);
  auto none = Module::getNoneOp(op);
  op->setOperand(2, none.getResult());
}
