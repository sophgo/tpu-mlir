#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Backend/BM168x/BM1684.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"
#include "sophgo/Support/MathUtils.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;
using namespace sophgo::backend;

void tpu::ConvOp::codegen_int8_bm1684() {
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, relu;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, relu);
  if (is_dw) {
    BM1684::instance().dl_nodechip_depthwise_fix8b_forward_parallel(
        Module::getAddress(input()), Module::getAddress(output()),
        Module::getAddress(filter()),
        with_bias ? Module::getAddress(bias()) : 0, n, ic, ih, iw, kh, kw, pt,
        pb, pl, pr, sh, sw, ins_h, ins_w, rshift().getValue()[0].cast<IntegerAttr>().getInt(), with_bias ? 1 : 0, 0, 1, 1,
        1, 1, relu ? 1 : 0, BM1684::instance().get_cmd_id_node());
  } else {
    auto weight_addr = Module::getAddress(filter());
    auto bias_offset = align_up(ic / g, 4l) * kh * kw;
    BM1684::instance().dl_nodechip_conv_forward_parallel_fix8b_with_data_split(
        Module::getAddress(input()), Module::getAddress(output()),
        Module::getAddress(filter()),
        with_bias ? Module::getAddress(bias()) : 0, n, ic, ih, iw, g, oc, kh,
        kw, dh, dw, pt, pb, pl, pr, sh, sw, with_bias ? 1 : 0, 0, relu ? 1 : 0,
        0, 1, 0, 0, rshift().getValue()[0].cast<IntegerAttr>().getInt(), 1, 1, 1, 3, 0, 0, 0, 0, 0,
        BM1684::instance().get_cmd_id_node());
  }
}
