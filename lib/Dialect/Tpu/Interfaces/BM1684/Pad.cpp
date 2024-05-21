//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================
void tpu::PadOp::codegen_global_bm1684() {
  uint64_t in_addr = module::getAddress(getInput());
  uint64_t out_addr = module::getAddress(getOutput());
  int dims = getPaddings().size() / 2;
  assert(dims == 2 || dims == 3 || dims == 4 || dims == 5);
  std::vector<int64_t> pads_n;
  auto pads = module::getI64Array(getPaddings());
  for (int i = 0; i < 2 * dims; i++) {
    pads_n.push_back(pads->at(i));
  }
  int type = (int)getMode();
  float constant = getVal().convertToDouble();
  if (dims <= 4) {
    int in_shape[4] = {0};
    module::getGlobalShape(getInput(), in_shape);
    int(*p_pad)[2] = new int[4][2];
    for (int i = 0; i < 4; i++) {
      if (i < dims) {
        p_pad[i][0] = pads_n[i];
        p_pad[i][1] = pads_n[i + dims];
      } else {
        p_pad[i][0] = 0;
        p_pad[i][1] = 0;
      }
    }
    if (false == module::isUniformQuantized(getOutput())) {
      BM1684::instance().dl_nodechip_pad(
          in_addr, out_addr, in_shape[0], in_shape[1], in_shape[2], in_shape[3],
          p_pad, type, constant, (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
    } else {
      STORE_MODE_T in_stmode = BM1684::getStoreMode(getInput());
      STORE_MODE_T out_stmode = BM1684::getStoreMode(getOutput());
      assert(in_stmode != STORE_MODE_2N && out_stmode != STORE_MODE_2N);
      uint64_t input_1N_global_offset = 0;
      uint64_t output_1N_global_offset = 0;
      uint64_t buffer_global_addr = module::getAddress(getBuffer());
      if (in_stmode != STORE_MODE_4N || out_stmode != STORE_MODE_4N) {
        if (in_stmode == STORE_MODE_4N) {
          input_1N_global_offset = buffer_global_addr;
          if (out_stmode == STORE_MODE_4N) {
            uint64_t input_1N_buffer_size = ceiling_func(in_shape[0], 4) * 4 *
                                            in_shape[1] * in_shape[2] *
                                            in_shape[3];
            output_1N_global_offset = buffer_global_addr + input_1N_buffer_size;
          }
        } else if (out_stmode == STORE_MODE_4N) {
          output_1N_global_offset = buffer_global_addr;
        }
      }
      BM1684::instance().dl_nodechip_pad_fix8b(
          in_addr, out_addr, in_stmode == STORE_MODE_4N ? 1 : 0,
          input_1N_global_offset, out_stmode == STORE_MODE_4N ? 1 : 0,
          output_1N_global_offset, in_shape[0], in_shape[1], in_shape[2],
          in_shape[3], p_pad, type, constant,
          (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
    }
  } else if (dims == 5) {
    int in_shape[5] = {0};
    module::getGlobalShape(getInput(), in_shape);
    int(*p_pad)[2] = new int[dims][2];
    for (int i = 0; i < dims; i++) {
      p_pad[i][0] = pads_n[i];
      p_pad[i][1] = pads_n[i + dims];
    }
    uint64_t buffer_addr = module::getAddress(getBuffer());
    if (false == module::isUniformQuantized(getOutput())) {
      BM1684::instance().dl_nodechip_pad3d(
          in_addr, out_addr, buffer_addr, in_shape[0], in_shape[1], in_shape[2],
          in_shape[3], in_shape[4], p_pad, type, constant,
          (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
    } else {
      STORE_MODE_T in_stmode = BM1684::getStoreMode(getInput());
      STORE_MODE_T out_stmode = BM1684::getStoreMode(getOutput());
      uint64_t buffer_addr = module::getAddress(getBuffer());
      BM1684::instance().dl_nodechip_pad3d_fix8b(
          in_addr, out_addr, buffer_addr, buffer_addr,
          in_stmode == STORE_MODE_4N ? 1 : 0,
          out_stmode == STORE_MODE_4N ? 1 : 0, in_shape[0], in_shape[1],
          in_shape[2], in_shape[3], in_shape[4], p_pad, type, constant,
          (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
    }
  }
}

int64_t tpu::PadOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                         int64_t out_lmem_bytes,
                                         int64_t in_nslice, int64_t in_hslice,
                                         int64_t out_nslice,
                                         int64_t out_hslice) {
  return 0;
}

void tpu::PadOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                      local_sec_info_t &sec_info) {
  auto in_g_info = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  int dims = getPaddings().size() / 2;
  assert(dims == 2 || dims == 3 || dims == 4 || dims == 5);
  std::vector<int64_t> pads_n;
  auto pads = module::getI64Array(getPaddings());
  for (int i = 0; i < 2 * dims; i++) {
    pads_n.push_back(pads->at(i));
  }
  int type = (int)getMode();
  float constant = getVal().convertToDouble();
  if (dims <= 4) {
    int in_shape[4] = {0};
    module::getGlobalShape(getInput(), in_shape);
    int(*p_pad)[2] = new int[4][2];
    for (int i = 0; i < 4; i++) {
      if (i < dims) {
        p_pad[i][0] = pads_n[i];
        p_pad[i][1] = pads_n[i + dims];
      } else {
        p_pad[i][0] = 0;
        p_pad[i][1] = 0;
      }
    }
    if (false == module::isUniformQuantized(getOutput())) {
      BM1684::instance().dl_nodechip_pad_local(
          (int)in_g_info.out_addr, (int)gi.out_addr, in_shape[0], in_shape[1],
          in_shape[2], in_shape[3], p_pad, type, constant,
          (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
    } else {
      STORE_MODE_T in_stmode = BM1684::getStoreMode(getInput());
      STORE_MODE_T out_stmode = BM1684::getStoreMode(getOutput());
      assert(in_stmode != STORE_MODE_2N && out_stmode != STORE_MODE_2N);
      uint64_t input_1N_global_offset = 0;
      uint64_t output_1N_global_offset = 0;
      uint64_t buffer_global_addr = module::getAddress(getBuffer());
      if (in_stmode != STORE_MODE_4N || out_stmode != STORE_MODE_4N) {
        if (in_stmode == STORE_MODE_4N) {
          input_1N_global_offset = buffer_global_addr;
          if (out_stmode == STORE_MODE_4N) {
            uint64_t input_1N_buffer_size = ceiling_func(in_shape[0], 4) * 4 *
                                            in_shape[1] * in_shape[2] *
                                            in_shape[3];
            output_1N_global_offset = buffer_global_addr + input_1N_buffer_size;
          }
        } else if (out_stmode == STORE_MODE_4N) {
          output_1N_global_offset = buffer_global_addr;
        }
      }
      BM1684::instance().dl_nodechip_pad_fix8b_local(
          (int)in_g_info.out_addr, (int)gi.out_addr, in_shape[0], in_shape[1],
          in_shape[2], in_shape[3], p_pad, type, (unsigned char)constant,
          (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
    }
  } else if (dims == 5) {
    UNREACHABLE_THIS("Not Implemented");
  }
}

uint32_t tpu::PadOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  GLOBAL_IR_COMMON(pad);
}

int64_t tpu::PadOp::get_fw_type_bm1684() { return FW_BMNET_PAD; }
