#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Backend/BM168x/BM1684.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;
using namespace sophgo::backend;

void tpu::MaxPoolOp::codegen_int8_bm1684() {
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool is_global, count_include_pad;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             is_global, count_include_pad);
  BM1684::instance().dl_nodechip_pooling_fix8b_forward_parallel_with_data_split(
      Module::getAddress(input()), Module::getAddress(output()), n, c, ih, iw,
      kh, kw, pt, pb, pl, pr, sh, sw, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
      do_relu() ? 1 : 0, BM1684::instance().get_cmd_id_node());
}
