
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

void tpu::AddOp::codegen_int8_bm1684() {
  int input_num = inputs().size();
  assert(input_num == 2);
  int64_t n, c, h, w;
  Module::getNCHW(output(), n, c, h, w);
  int op_code = 1; // (0: Product; 1: Sum; 2: Max)

  std::vector<float> coeff_v;
  if (coeff().hasValue()) {
    for (auto data : coeff().getValue()) {
      coeff_v.push_back(data.cast<FloatAttr>().getValueAsDouble());
    }
  } else {
    coeff_v = {0, 0};
  }
  std::vector<int> rshift_v;
  for (auto data : rshifts()) {
    rshift_v.push_back(data.cast<IntegerAttr>().getInt());
  }

  BM1684::instance().dl_nodechip_eltwise_fix8b_forward_parallel(
      Module::getAddress(inputs()[0]),     // u64    bottom_A_global_addr,
      Module::getAddress(inputs()[1]),     // u64    bottom_B_global_addr,
      Module::getAddress(output()),        // u64    top_global_addr,
      n,                                   // int    tensor_n,
      c,                                   // int    tensor_c,
      h,                                   // int    tensor_h,
      w,                                   // int    tensor_w,
      op_code,                             // int    op_code,
      (int8_t)coeff_v[0],                  // int    scale_A,
      (int8_t)coeff_v[1],                  // int    scale_B,
      1,                                   // int    sign_A,
      1,                                   // int    sign_B,
      rshift_v[0],                         // int    rshift_A,
      rshift_v[1],                         // int    rshift_B,
      do_relu() ? 1 : 0,                   // int    do_relu(),
      BM1684::instance().get_cmd_id_node() // CMD_ID_NODE *id_node
  );
}
