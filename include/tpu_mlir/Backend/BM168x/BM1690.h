//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Backend/BM168x/BackendInterfaces.h"
#include "tpu_mlir/Support/Module.h"

typedef void (*set_tiu_freq)(float freq);
typedef void (*set_gdma_bw_s2s)(float GBps);
typedef void (*set_gdma_bw_s2l)(float GBps);
typedef void (*set_gdma_bw_l2s)(float GBps);
typedef void (*set_gdma_bw_l2l)(float GBps);
typedef void (*tpu_sync_all)();
typedef void (*tpu_core_context_setup)(int, int, int);


typedef void (*sdma_tensor_general_move_gen_cmd)(
    uint64_t src_addr,
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    uint32_t src_N_stride,
    uint32_t src_C_stride,
    uint32_t src_H_stride,
    uint32_t src_W_stride,
    int src_format,
    uint64_t dst_addr,
    int dst_N,
    int dst_C,
    int dst_H,
    int dst_W,
    uint32_t dst_N_stride,
    uint32_t dst_C_stride,
    uint32_t dst_H_stride,
    uint32_t dst_W_stride,
    int transpose,  // N/C transpose
    int port_id,
    CMD_ID_NODE * pid_node);

namespace tpu_mlir {
namespace backend {
#define BUFFER_SIZE (4 * 1024 * 1024)
class BM1690 : public BM1684X, MultiCoreInterface::Base<BM1690> {
public:
  static bool classof(const BM168x *bm168x) {
    return bm168x->getTypeID() == TypeID::get<BM1690>();
  }

  static BM1690 &instance() {
    static BM1690 bm1690;
    return bm1690;
  }
  virtual void before_codegen() override;
  virtual void after_codegen(int64_t flops = 0) override;

  set_tiu_freq dl_set_tiu_freq;
  set_gdma_bw_s2s dl_set_gdma_bw_s2s;
  set_gdma_bw_s2l dl_set_gdma_bw_s2l;
  set_gdma_bw_l2s dl_set_gdma_bw_l2s;
  set_gdma_bw_l2l dl_set_gdma_bw_l2l;
  tpu_sync_all dl_tpu_sync_all;
  tpu_core_context_setup dl_tpu_core_context_setup;
  sdma_tensor_general_move_gen_cmd dl_sdma_tensor_general_move_gen_cmd;

  void setCoreNum(int core = 1) final;
  int getCoreNum() final { return multiCode.size(); };
  int getCurrentCoreID() final;
  void useCore(int coreID = 0) final;
  void setupMultiCoreContext(int core_idx, int core_num,
                             int core_msg_id) final {
    dl_tpu_core_context_setup(core_idx, core_num, core_msg_id);
  }
  void syncAll() final {
    dl_tpu_set_id_node(code->cmdid_node);
    dl_tpu_sync_all();
    dl_tpu_get_id_node(code->cmdid_node);
  }


  std::vector<std::shared_ptr<BM168x::Code>> const &getCodebuffer() final {
    return multiCode;
  }

  // specific global info
  static constexpr llvm::StringRef LIB_KERNEL_NAME =
      "libbm1690_kernel_module.so";
private:
  enum TagType {
    TAG_USERS = 0,
    TAG_WEIGHT = (1ul << 40),
    TAG_ACTIVATION = (2ul << 40),
    TAG_L2MEM = (30ul << 40),
  };

protected:
  BM1690() {
    typeID = TypeID::get<BM1690>();
    NPU_NUM = 64;
    EU_BYTES = 64;        // vector length 512bit
    LMEM_BYTES = 1 << 18; // 256KB
    LMEM_BANKS = 16;
    IC_PARALLEL = 64;
    ALIGNMENT = 0x1000;
    LMEM_BANK_BYTES = LMEM_BYTES / LMEM_BANKS;

    // have 30 memory section, now don't kown use which section
    GMEM_START_ADDR = 0; // tag for global memory address
    COEFF_START_ADDR = GMEM_START_ADDR | TAG_WEIGHT;
    CTX_START_ADDR = GMEM_START_ADDR | TAG_ACTIVATION;
    L2_SRAM_START_ADDR = 0x6980000000 | TAG_L2MEM;
    L2_SRAM_SIZE = 0x8000000;
    LIB_BACKEND_NAME = "libbackend_bm1690.so";
    // GDMA format
    GDMA_VALUE_FORMAT_INT8 = 0;
    GDMA_VALUE_FORMAT_FLOAT16 = 1;
    GDMA_VALUE_FORMAT_FLOAT32 = 2;
    GDMA_VALUE_FORMAT_INT16 = 3;
    GDMA_VALUE_FORMAT_INT32 = 4;
    GDMA_VALUE_FORMAT_BFLOAT16 = 5;
    GDMA_VALUE_FORMAT_FLOAT20 = 6;
    GDMA_VALUE_FORMAT_NUM = 7;
    multiCode.push_back(std::make_unique<BM168x::Code>());
    code = multiCode.back();
    core_num = module::getCoreNum();
    start_env();
  };
  virtual void load_functions() override;
  virtual ~BM1690(){};
  bool useCode0 = true;
  std::vector<std::shared_ptr<BM168x::Code>> multiCode;
};

} // namespace backend
} // namespace tpu_mlir
