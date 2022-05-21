#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Backend/BM168x/BM1686.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"
#include "sophgo/Support/MathUtils.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;
using namespace sophgo::backend;

void tpu::LoadOp::codegen_global_int8_bm1686() {
  llvm_unreachable("global not support");
}

// f32
void tpu::LoadOp::codegen_global_float_bm1686() {
  llvm_unreachable("global not support");
}

int64_t tpu::LoadOp::getBufferSize_bm1686(int64_t out_n, int64_t out_c,
                                          int64_t out_h, int64_t out_w,
                                          int64_t out_lmem_bytes) {
  return 0;
}

void tpu::LoadOp::codegen_local_int8_bm1686(int64_t n_step, int64_t h_step) {
  char prefix[64];
  CMD_ID_NODE *pid_node = (CMD_ID_NODE *)BM1686::instance().gdma_node;
  auto gi = getGroupInfo(n_step, h_step);
  assert(false == gi.overstepped);
  sprintf(prefix, "LD_%s", name().data());
  BM1686::instance().dl_set_cmd_id_prefix(pid_node, prefix);
  auto data_type = BM168x::getDataType(output());
  auto gdma_format = BM168x::getGdmaFormat(data_type);
  auto fmt_bytes = BM168x::getFmtBytes(data_type);
  int64_t N, C, H, W;
  Module::getNCHW(output(), N, C, H, W);
  auto g_stride = BM1686::instance().getGlobalStride(N, C, H, W);
  auto s_stride = BM1686::instance().getLocalStride(gi.n_slice, C, gi.h_slice,
                                                    W, fmt_bytes);
  auto g_addr = Module::getAddress(input());
  int64_t g_offset =
      (gi.n_idx * g_stride.N + gi.h_idx * g_stride.H) * fmt_bytes;
  BM1686::instance().dl_tensor_stride_move_gen_cmd(
      gi.out_addr, 0, g_addr + g_offset, gi.n_slice, C, gi.h_slice, W,
      g_stride.N, g_stride.C, g_stride.H, g_stride.W, s_stride.N, s_stride.C,
      s_stride.H, s_stride.W, gdma_format, GDMA_VALUE_DIR_S2L, 0, pid_node);
}

void tpu::LoadOp::codegen_local_float_bm1686(int64_t n_step, int64_t h_step) {
  codegen_local_int8_bm1686(n_step, h_step);
}
