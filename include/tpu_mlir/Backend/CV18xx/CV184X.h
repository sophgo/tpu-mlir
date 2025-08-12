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
class CV184X : public BM1684X {
public:
  static constexpr llvm::StringRef LIB_KERNEL_NAME =
      "libcv184x_kernel_module.so";
  static bool classof(const BM168x *bm168x) {
    return bm168x->getTypeID() == TypeID::get<CV184X>();
  }

  static CV184X &instance(int frequency) {
    static CV184X CV184X;
    // CV184X.set_simulation_freq(frequency);
    return CV184X;
  }
  virtual void before_codegen() override;
  virtual void after_codegen(int64_t flops = 0) override;

  set_tiu_freq dl_set_tiu_freq;
  set_gdma_bw_s2s dl_set_gdma_bw_s2s;
  set_gdma_bw_s2l dl_set_gdma_bw_s2l;
  set_gdma_bw_l2s dl_set_gdma_bw_l2s;
  set_gdma_bw_l2l dl_set_gdma_bw_l2l;
  set_id_node dl_set_id_node;

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
    TAG_WEIGHT = (1ul << 40),
    TAG_ACTIVATION = (2ul << 40),
    TAG_IO0 = (3ul << 40),
    TAG_IO1 = (4ul << 40),
    TAG_IO2 = (5ul << 40),
    TAG_IO3 = (6ul << 40),
    TAG_IO4 = (7ul << 40),
  };

protected:
  CV184X() {
    typeID = TypeID::get<CV184X>();
    code = std::make_unique<BM168x::Code>();
    NPU_NUM = 8;
    EU_BYTES = 16;
    LMEM_BYTES = 65536;
    LMEM_BANKS = 16;
    DMA_ALGN_BYTES = 256;
    IC_PARALLEL = 16;
    ALIGNMENT = 0x1000;
    LMEM_BANK_BYTES = LMEM_BYTES / LMEM_BANKS;
    GMEM_START_ADDR = 0x80000000UL;
    COEFF_START_ADDR = GMEM_START_ADDR | TAG_WEIGHT;
    CTX_START_ADDR = GMEM_START_ADDR | TAG_ACTIVATION;
    IO_ADDR[0] = TAG_IO0;
    IO_ADDR[1] = TAG_IO1;
    IO_ADDR[2] = TAG_IO2;
    IO_ADDR[3] = TAG_IO3;
    IO_ADDR[4] = TAG_IO4;
    SUPPORT_MEM_TAG = true;
    LIB_BACKEND_NAME = "libbackend_cv184x.so";
    // GDMA format
    GDMA_VALUE_FORMAT_INT8 = 0;
    GDMA_VALUE_FORMAT_FLOAT16 = 1;
    GDMA_VALUE_FORMAT_FLOAT32 = 2;
    GDMA_VALUE_FORMAT_INT16 = 3;
    GDMA_VALUE_FORMAT_INT32 = 4;
    GDMA_VALUE_FORMAT_BFLOAT16 = 5;
    GDMA_VALUE_FORMAT_INT4 = 6;
    GDMA_VALUE_FORMAT_NUM = 7;
    core_num = module::getCoreNum();
    start_env();
    dl_tpu_set_id_node(code->cmdid_node);
  };
  virtual void load_functions() override;
  virtual ~CV184X(){};
};

} // namespace backend
} // namespace tpu_mlir
