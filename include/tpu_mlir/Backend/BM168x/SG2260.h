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

namespace tpu_mlir {
namespace backend {
#define BUFFER_SIZE (4 * 1024 * 1024)
class SG2260 : public BM1684X, MultiCoreInterface::Base<SG2260> {
public:
  static bool classof(const BM168x *bm168x) {
    return bm168x->getTypeID() == TypeID::get<SG2260>();
  }

  static SG2260 &instance() {
    static SG2260 sg2260;
    return sg2260;
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

private:
  enum TagType {
    TAG_USERS = 0,
    TAG_WEIGHT = (1ul << 40),
    TAG_ACTIVATION = (2ul << 40),
  };

protected:
  SG2260() {
    typeID = TypeID::get<SG2260>();
    NPU_NUM = 64;
    EU_BYTES = 128;
    LMEM_BYTES = 1 << 18; // 256KB
    LMEM_BANKS = 16;
    IC_PARALLEL = 64;
    ALIGNMENT = 0x1000;
    LMEM_BANK_BYTES = LMEM_BYTES / LMEM_BANKS;

    //have 30 memory section, now don't kown use which section
    GMEM_START_ADDR = 0; // tag for global memory address
    COEFF_START_ADDR = GMEM_START_ADDR | TAG_WEIGHT;
    CTX_START_ADDR = GMEM_START_ADDR | TAG_ACTIVATION;
    LIB_BACKEND_NAME = "libbackend_sg2260.so";
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
    start_env();
  };
  virtual void load_functions() override;
  virtual ~SG2260(){};
  bool useCode0 = true;
  std::vector<std::shared_ptr<BM168x::Code>> multiCode;
};

} // namespace backend
} // namespace tpu_mlir
