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
#define BUFFER_SIZE (4 *1024 * 1024)
class BM1686 : public BM1684X {
public:
  static bool classof(const BM168x *bm168x) {
    return bm168x->getTypeID() == TypeID::get<BM1686>();
  }

  static BM1686 &instance(int frequency) {
    static BM1686 BM1686;
    BM1686.set_simulation_freq(frequency);
    return BM1686;
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

  void setCoreNum(int core = 1);
  void useCore(int coreID = 0);

  std::vector<std::shared_ptr<BM168x::Code>> const &getCodebuffer() {
    return multiCodes;
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
  BM1686() {
    typeID = TypeID::get<BM1686>();
    NPU_NUM = 32;
    EU_BYTES = 16;
    LMEM_BYTES = 1 << 17; // 128KB
    LMEM_BANKS = 16;
    IC_PARALLEL = 32;
    ALIGNMENT = 0x1000;
    LMEM_BANK_BYTES = LMEM_BYTES / LMEM_BANKS;
    GMEM_START_ADDR = 0x1ul << 39; // tag for global memory address
    COEFF_START_ADDR = GMEM_START_ADDR | TAG_WEIGHT;
    CTX_START_ADDR = GMEM_START_ADDR | TAG_ACTIVATION;
    LIB_BACKEND_NAME = "libbackend_1686.so";
    // GDMA format
    GDMA_VALUE_FORMAT_INT8 = 0;
    GDMA_VALUE_FORMAT_FLOAT16 = 1;
    GDMA_VALUE_FORMAT_FLOAT32 = 2;
    GDMA_VALUE_FORMAT_INT16 = 3;
    GDMA_VALUE_FORMAT_INT32 = 4;
    GDMA_VALUE_FORMAT_BFLOAT16 = 5;
    GDMA_VALUE_FORMAT_INT4 = 6;
    GDMA_VALUE_FORMAT_NUM = 7;
    multiCodes.push_back(std::make_unique<BM168x::Code>());
    code = multiCodes.back();
    start_env();
  };
  virtual void load_functions() override;
  virtual ~BM1686(){};
  bool useCode0 = true;
  std::vector<std::shared_ptr<BM168x::Code>> multiCodes;
};

} // namespace backend
} // namespace tpu_mlir
