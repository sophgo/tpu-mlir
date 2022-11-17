//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::backend;

static void parsePadParam(Operation *op, std::vector<int64_t> &is_4,
                          std::vector<int64_t> &os_4, std::vector<int> &pad_4) {
  std::vector<int64_t> is, os, pads;
  auto castOp = llvm::dyn_cast<tpu::PadOp>(op);
  auto _is = Module::getShape(castOp.input());
  auto _os = Module::getShape(castOp.output());
  auto _pads = Module::getI64Array(castOp.paddings());
  is.assign(_is.begin(), _is.end());
  os.assign(_os.begin(), _os.end());
  pads.assign(_pads->begin(), _pads->end());

  int num_dims = is.size();

  assert(is.size() * 2 == pads.size());
  assert(is.size() == os.size());

  if (num_dims > 4) {
    // remove continous
    while (num_dims > 4) {
      bool done = false;
      for (int i = 0; i < num_dims - 1; i++) {
        if (is[i] == os[i] && is[i + 1] == os[i + 1]) {
          is[i] *= is[i + 1];
          os[i] *= os[i + 1];
          is.erase(is.begin() + i + 1);
          os.erase(os.begin() + i + 1);
          pads.erase(pads.begin() + i + 1);
          num_dims--;
          done = true;
          break;
        }
      }
      if (done == false) {
        break;
      }
    }
    if (num_dims > 4) {
      llvm_unreachable("Pad shape not support");
    }
  }
  is_4 = {1, 1, 1, 1};
  os_4 = {1, 1, 1, 1};
  pad_4 = {0, 0, 0, 0, 0, 0, 0, 0};
  switch (num_dims) {
  case 1:
    is_4[3] = is[0];
    os_4[3] = os[0];
    pad_4[3] = pads[0];
    pad_4[7] = pads[1];
    break;
  case 2:
    is_4[1] = is[0];
    is_4[3] = is[1];
    os_4[1] = os[0];
    os_4[3] = os[1];
    pad_4[1] = pads[0];
    pad_4[3] = pads[1];
    pad_4[5] = pads[2];
    pad_4[7] = pads[3];
    break;
  default:
    for (int idx = 0; idx < num_dims; idx++) {
      is_4[idx] = is[idx];
      os_4[idx] = os[idx];
      pad_4[idx] = pads[idx];
      pad_4[idx + 4] = pads[idx + num_dims];
    }
    break;
  }
}

// =========================================
// GlobalGenInterface
// =========================================
void tpu::PadOp::codegen_global_cv18xx(void *ctx, int64_t layer_id) {
  CviBackendContext *backend_ctx = (CviBackendContext *)ctx;
  std::string s_model;
  std::vector<int64_t> i_s;
  std::vector<int64_t> o_s;
  std::vector<int> pads;
  gaddr_t ga_input = Module::getAddress(input());
  gaddr_t ga_output = Module::getAddress(output());
  cvk_fmt_t fmt =
      Quant::isUniformQuantized(output()) ? CVK_FMT_I8 : CVK_FMT_BF16;
  parsePadParam(getOperation(), i_s, o_s, pads);
  float const_val = val().convertToDouble();

  switch (mode()) {
  case 0:
    s_model = "constant";
    break;
  case 1:
    s_model = "reflect"; // todo
    llvm_unreachable("Unsupport now.");
    break;
  case 3:
    s_model = "edge";
    break;
  default:
    llvm_unreachable("Unsupport pad type.");
  }

  cvi_backend_tg_pad_kernel(*backend_ctx, layer_id, ga_input, ga_output, i_s[0],
                            i_s[1], i_s[2], i_s[3], pads.data(), const_val,
                            s_model.c_str(), fmt);
}
