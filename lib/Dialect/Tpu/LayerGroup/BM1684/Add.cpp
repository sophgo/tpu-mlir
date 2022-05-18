#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

int64_t tpu::AddOp::getBufferSize_bm1684(int64_t out_n, int64_t out_c,
                                         int64_t out_h, int64_t out_w,
                                         int64_t out_lmem_bytes) {
  auto stype = Module::getStorageType(output());
  if (stype.isF32()) {
    return 0;
  }
  return out_lmem_bytes;
}
