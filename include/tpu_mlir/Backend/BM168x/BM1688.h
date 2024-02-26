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
class BM1688 : public BM1684X, MultiCoreInterface::Base<BM1688> {
public:
  static bool classof(const BM168x *bm168x) {
    return bm168x->getTypeID() == TypeID::get<BM1688>();
  }

  static BM1688 &instance(int frequency) {
    static BM1688 BM1688;
    BM1688.set_simulation_freq(frequency);
    return BM1688;
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
      // 64bit, for cv186x 32bit device, use (6.0f, 12.0f, 12.0f, 10.0f)
      dl_set_gdma_bw_s2s(12.0f);
      dl_set_gdma_bw_s2l(24.0f);
      dl_set_gdma_bw_l2s(24.0f);
      dl_set_gdma_bw_l2l(12.0f);
      // frequency == A2_2::value || frequency == A2_1::value
      dl_set_tiu_freq(static_cast<float>(frequency));
    } else {
      dl_set_tiu_freq(static_cast<float>(get_frequance()));
    }
  }

  enum TagType {
    TAG_USERS = 0,
    TAG_WEIGHT = (1ul << 36),
    TAG_ACTIVATION = (2ul << 36),
    TAG_IO0 = (3ul << 36),
    TAG_IO1 = (4ul << 36),
    TAG_IO2 = (5ul << 36),
    TAG_IO3 = (6ul << 36),
    TAG_IO4 = (7ul << 36),
  };

public:
  // specific global info
  static constexpr llvm::StringRef LIB_KERNEL_NAME =
      "libbmtpulv60_kernel_module.so";

protected:
  BM1688() {
    typeID = TypeID::get<BM1688>();
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
    IO_ADDR[0] = GMEM_START_ADDR | TAG_IO0;
    IO_ADDR[1] = GMEM_START_ADDR | TAG_IO1;
    IO_ADDR[2] = GMEM_START_ADDR | TAG_IO2;
    IO_ADDR[3] = GMEM_START_ADDR | TAG_IO3;
    IO_ADDR[4] = GMEM_START_ADDR | TAG_IO4;
    LIB_BACKEND_NAME = "libbackend_1688.so";
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
    core_num = module::getCoreNum();
    start_env();
  };
  virtual void load_functions() override;
  virtual ~BM1688(){};
  bool useCode0 = true;
  std::vector<std::shared_ptr<BM168x::Code>> multiCode;
};

} // namespace backend
} // namespace tpu_mlir
