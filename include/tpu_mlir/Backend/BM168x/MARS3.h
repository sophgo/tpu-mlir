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
class MARS3 : public BM1684X, MultiCoreInterface::Base<MARS3> {
public:
  static bool classof(const BM168x *bm168x) {
    return bm168x->getTypeID() == TypeID::get<MARS3>();
  }

  static MARS3 &instance(int frequency) {
    static MARS3 MARS3;
    MARS3.set_simulation_freq(frequency);
    return MARS3;
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
  void set_simulation_freq(int frequency) {
    CAST_FUNCTION(set_tiu_freq);
    if (get_frequance() == 0) {
      CAST_FUNCTION(set_gdma_bw_s2s);
      CAST_FUNCTION(set_gdma_bw_s2l);
      CAST_FUNCTION(set_gdma_bw_l2s);
      CAST_FUNCTION(set_gdma_bw_l2l);
      if (frequency == A2_2::value) // cv186
      {
        dl_set_gdma_bw_s2s(12.0f);
        dl_set_gdma_bw_s2l(12.0f);
        dl_set_gdma_bw_l2s(12.0f);
        dl_set_gdma_bw_l2l(10.0f);
        dl_set_tiu_freq(static_cast<float>(A2_2::value));
      } else {
        dl_set_gdma_bw_s2s(12.0f);
        dl_set_gdma_bw_s2l(24.0f);
        dl_set_gdma_bw_l2s(24.0f);
        dl_set_gdma_bw_l2l(12.0f);
        dl_set_tiu_freq(static_cast<float>(A2_1::value));
      }

    } else {
      dl_set_tiu_freq(static_cast<float>(get_frequance()));
    }
  }

  enum TagType {
    TAG_USERS = 0,
    TAG_WEIGHT = (1ul << 36),
    TAG_ACTIVATION = (2ul << 36),
  };

protected:
  MARS3() {
    typeID = TypeID::get<MARS3>();
    NPU_NUM = NPU_NUM_test_fp16; // origin=32;
    EU_BYTES = (EU_NUM_test_fp16)*2; // origin=16;
    LMEM_BYTES = 1 << LOCAL_MEM_SHIFT; // 128KB
    LMEM_BANKS = 16;
    IC_PARALLEL = (IC_PARALLEL_test_fp16)*2; // origin=32;
    ALIGNMENT = 0x1000;
    LMEM_BANK_BYTES = LMEM_BYTES / LMEM_BANKS;
    GMEM_START_ADDR = 0x1ul << 39; // tag for global memory address
    COEFF_START_ADDR = GMEM_START_ADDR | TAG_WEIGHT;
    CTX_START_ADDR = GMEM_START_ADDR | TAG_ACTIVATION;
    LIB_BACKEND_NAME = "libbackend_mars3.so";
    // GDMA format
    GDMA_VALUE_FORMAT_INT8 = 0;
    GDMA_VALUE_FORMAT_FLOAT16 = 1;
    GDMA_VALUE_FORMAT_FLOAT32 = 2;
    GDMA_VALUE_FORMAT_INT16 = 3;
    GDMA_VALUE_FORMAT_INT32 = 4;
    GDMA_VALUE_FORMAT_BFLOAT16 = 5;
    GDMA_VALUE_FORMAT_INT4 = 6;
    GDMA_VALUE_FORMAT_NUM = 7;
    multiCode.push_back(std::make_unique<BM168x::Code>());
    code = multiCode.back();
    start_env();
  };
  virtual void load_functions() override;
  virtual ~MARS3(){};
  bool useCode0 = true;
  std::vector<std::shared_ptr<BM168x::Code>> multiCode;
};

} // namespace backend
} // namespace tpu_mlir
