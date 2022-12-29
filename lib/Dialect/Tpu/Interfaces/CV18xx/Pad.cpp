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
#include "tpu_mlir/Support/Module.h"



using namespace tpu_mlir::backend;

static void parsePadParam(Operation *op, std::vector<int64_t> &is_4,
                          std::vector<int64_t> &os_4, std::vector<int> &pad_4) {
  std::vector<int64_t> is, os, pads;
  auto castOp = llvm::dyn_cast<tpu::PadOp>(op);
  module::getShapeVec(castOp.input(), is);
  module::getShapeVec(castOp.output(), os);
  auto _pads = module::getI64Array(castOp.paddings());
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
void tpu::PadOp::codegen_global_cv18xx(int64_t layer_id) {

  std::string s_model;
  std::vector<int64_t> i_s;
  std::vector<int64_t> o_s;
  std::vector<int> pads;
  gaddr_t ga_input = module::getAddress(input());
  gaddr_t ga_output = module::getAddress(output());
  cvk_fmt_t fmt =
      module::isUniformQuantized(output()) ? CVK_FMT_I8 : CVK_FMT_BF16;
  if (mode() == 0) {
    parsePadParam(getOperation(), i_s, o_s, pads);
    float const_val = val().convertToDouble();
    cvi_backend_tg_pad_kernel(layer_id, ga_input, ga_output, i_s[0], i_s[1],
                              i_s[2], i_s[3], pads.data(), const_val,
                              "constant", fmt);
  } else if (mode() == 1) {
    // reflect
    std::vector<int> pads(4, 0);
    auto num_dims = module::getShape(input()).size();
    auto _pads = module::getI64Array(paddings());
    pads[0] = _pads->at(num_dims - 1);
    pads[1] = _pads->at(num_dims * 2 - 1);
    pads[num_dims * 2 - 1] = 0;
    if (num_dims == 4 || num_dims == 3) {
      pads[2] = _pads->at(num_dims - 2);
      pads[3] = _pads->at(num_dims * 2 - 2);
    }

    module::getShapeVec(input(), i_s);
    int outer_size = std::accumulate(i_s.begin(), i_s.end() - 2, 1,
                                     std::multiplies<int64_t>());
    int ih = *(i_s.end() - 2);
    int iw = i_s.back();
    gaddr_t ga_left_select = module::getAddress(left_select());
    gaddr_t ga_right_select = module::getAddress(right_select());
    cvi_backend_tg_reflectionpad_kernel(layer_id, ga_input, ga_output,
                                        ga_left_select, ga_right_select,
                                        outer_size, ih, iw, pads, fmt);

  } else if (mode() == 3) {
    // edge
    llvm_unreachable("Unsupport pad type.");
  } else {
    llvm_unreachable("Unsupport pad type.");
  }
}
