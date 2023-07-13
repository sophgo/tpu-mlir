//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

using namespace tpu_mlir::backend;

static void parsePadParam(Operation *op, std::vector<int64_t> &is_4,
                          std::vector<int64_t> &os_4, std::vector<int> &pad_4) {
  std::vector<int64_t> pads;
  auto castOp = llvm::dyn_cast<tpu::PadOp>(op);
  std::vector<int64_t> is = module::getShape(castOp.getInput());
  std::vector<int64_t> os = module::getShape(castOp.getOutput());
  auto _pads = module::getI64Array(castOp.getPaddings());
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
          pads.erase(pads.begin() + i + 1 + num_dims);
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
  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getOutput());
  cvk_fmt_t fmt =
      module::isUniformQuantized(getOutput()) ? CVK_FMT_I8 : CVK_FMT_BF16;
  auto mode = getMode();
  if (mode == tpu::PaddingMode::constant || mode == tpu::PaddingMode::edge) {
    parsePadParam(getOperation(), i_s, o_s, pads);
    float const_val = getVal().convertToDouble();
    cvi_backend_tg_pad_kernel(layer_id, ga_input, ga_output, i_s[0], i_s[1],
                              i_s[2], i_s[3], pads.data(), const_val, (int)mode,
                              fmt);
  } else if (mode == tpu::PaddingMode::reflect) {
    // reflect
    std::vector<int> pads(4, 0);
    auto num_dims = module::getShape(getInput()).size();
    auto _pads = module::getI64Array(getPaddings());
    pads[0] = _pads->at(num_dims - 1);
    pads[1] = _pads->at(num_dims * 2 - 1);
    pads[num_dims * 2 - 1] = 0;
    if (num_dims == 4 || num_dims == 3) {
      pads[2] = _pads->at(num_dims - 2);
      pads[3] = _pads->at(num_dims * 2 - 2);
    }

    i_s = module::getShape(getInput());
    int outer_size = std::accumulate(i_s.begin(), i_s.end() - 2, 1,
                                     std::multiplies<int64_t>());
    int ih = *(i_s.end() - 2);
    int iw = i_s.back();
    gaddr_t ga_left_select = module::getAddress(getLeftSelect());
    gaddr_t ga_right_select = module::getAddress(getRightSelect());
    cvi_backend_tg_reflectionpad_kernel(layer_id, ga_input, ga_output,
                                        ga_left_select, ga_right_select,
                                        outer_size, ih, iw, pads, fmt);
  } else {
    llvm_unreachable("Unsupport pad type.");
  }
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::PadOp::getBufferSize_cv18xx(int64_t in_lmem_bytes,
                                         int64_t out_lmem_bytes,
                                         int64_t in_nslice, int64_t in_hslice,
                                         int64_t out_nslice,
                                         int64_t out_hslice) {
  return 0;
}

void tpu::PadOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step,
                                      int64_t d_step, int64_t w_step,
                                      group_type_t group_type,
                                      local_sec_info_t &sec_info,
                                      int64_t layer_id) {
  std::vector<int64_t> i_s;
  std::vector<int64_t> o_s;
  std::vector<int> pads;
  parsePadParam(getOperation(), i_s, o_s, pads);

  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto out_gi = LocalGenInterface::getGroupInfo(getOutput(), n_step, h_step);
  laddr_t la_input = in_gi.out_addr;
  laddr_t la_output = out_gi.out_addr;

  pads[2] = (sec_info.h_idx == 0 ? pads[2] : 0);
  pads[6] = (sec_info.h_idx + sec_info.h_slice == i_s[2] ? pads[6] : 0);

  i_s[0] = sec_info.n_slice;
  i_s[2] = sec_info.h_slice;

  o_s[0] = sec_info.out_n_slice;
  o_s[2] = sec_info.out_h_slice;

  float const_val = getVal().convertToDouble();
  if (module::isUniformQuantized(getOutput())) {
    cvi_backend_tl_pad(layer_id, // layer_id,
                       i_s.data(), o_s.data(), la_input, la_output, const_val,
                       pads.data());
  } else {
    cvi_backend_tl_bf16_pad(layer_id, // layer_id,
                            i_s.data(), o_s.data(), la_input, la_output,
                            const_val, pads.data());
  }
  return;
}
