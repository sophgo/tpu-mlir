//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684.h"

#include "tpu_mlir/Support/Module.h"



using namespace tpu_mlir::backend;

void tpu::UpsampleOp::codegen_global_bm1684() {
  int64_t n, c, h, w;
  assert(scale_h() == scale_w());
  module::getNCHW(input(), n, c, h, w);
  BM1684::instance().dl_nodechip_upsample_forward_parallel_fix8b(
      module::getAddress(input()),
      module::getAddress(output()),
      n, c, h, w,
      scale_h(),
      do_relu() ? 1 : 0,
      (CMD_ID_NODE*)BM1684::instance().cmdid_node
  );
}

int64_t tpu::UpsampleOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                         int64_t out_lmem_bytes,
                                         int64_t in_nslice, int64_t in_hslice,
                                         int64_t out_nslice,
                                         int64_t out_hslice) {
  return 0;
}

void tpu::UpsampleOp::codegen_local_bm1684(int64_t n_step, int64_t h_step) {
  auto out_ginfo = LocalGenInterface::getGroupInfo(output());
  int64_t n, c, h, w;
  module::getNCHW(input(), n, c, h, w);
  int scale = scale_h();
  assert(scale == scale_w());

  llvm::SmallVector<int, 4> input_shape;
  llvm::SmallVector<int, 4> output_shape;
  input_shape.push_back(n);
  input_shape.push_back(c);
  input_shape.push_back(h);
  input_shape.push_back(w);
  output_shape.push_back(n);
  output_shape.push_back(c);
  output_shape.push_back(scale_h() * h);
  output_shape.push_back(scale_w() * w);
  BM1684::instance().dl_nodechip_upsample_fix8b_forward_local(
      module::getAddress(input()),
      module::getAddress(output()),
      input_shape.data(),
      output_shape.data(),
      scale,
      0, 1, 1, 1, 1,
      (CMD_ID_NODE*)BM1684::instance().cmdid_node,
      do_relu() ? 1 : 0
  );
}
