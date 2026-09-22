//===----------------------------------------------------------------------===//
//
// Copyright (C) 2024 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Backend/BM168x/BackendInterfaces.h"
#include "tpu_mlir/Support/Module.h"

typedef void (*tpu_sync_all)();
typedef void (*tpu_core_context_setup)(int, int, int);
typedef void (*gen_riscv_code_begin)();
typedef void (*gen_riscv_code_end)();

namespace tpu_mlir {
namespace backend {
#define BUFFER_SIZE (4 * 1024 * 1024)
class CV84X6 : public BM1684X, MultiCoreInterface::Base<CV84X6> {
public:
  static bool classof(const BM168x *bm168x) {
    return bm168x->getTypeID() == TypeID::get<CV84X6>();
  }

  static CV84X6 &instance() {
    static CV84X6 cv84x6;
    return cv84x6;
  }
  virtual void before_codegen() override;
  virtual void after_codegen(int64_t flops = 0) override;

  tpu_sync_all dl_tpu_sync_all;
  tpu_core_context_setup dl_tpu_core_context_setup;
  gen_riscv_code_begin dl_gen_riscv_code_begin;
  gen_riscv_code_end dl_gen_riscv_code_end;

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
    TAG_IOALONE = (3ul << 40),
    TAG_L2MEM = (30ul << 40),
    // 4 ~ 29 for user defined tag
  };

public:
  // specific global info
  static constexpr llvm::StringRef LIB_KERNEL_NAME =
      "libcv84x6_kernel_module.so";
  static constexpr llvm::StringRef LIB_RVTI_KERNEL_NAME =
      "libcv84x6rv_kernel_module.so";

protected:
  CV84X6() {
    typeID = TypeID::get<CV84X6>();
    NPU_NUM = 16;
    EU_BYTES = 32;
    LMEM_BYTES = 1 << 18; // 256KB
    LMEM_BANKS = 16;
    IC_PARALLEL = 32;
    ALIGNMENT = 0x1000;
    LMEM_BANK_BYTES = LMEM_BYTES / LMEM_BANKS;
    GMEM_START_ADDR = 0x1000000000UL;
    COEFF_START_ADDR = GMEM_START_ADDR | TAG_WEIGHT;
    CTX_START_ADDR = GMEM_START_ADDR | TAG_ACTIVATION;
    IO_START_ADDR = GMEM_START_ADDR | TAG_IOALONE;
    L2_SRAM_START_ADDR = 0x26d20000UL | TAG_L2MEM;
    L2_SRAM_SIZE = 0x100000;
    USER_TAG_START = 4;
    USER_TAG_END = 29;
    SUPPORT_MEM_TAG = true;
    LIB_BACKEND_NAME = "libbackend_cv84x6.so";
    LIB_PPL_DYN_HOST_NAME = "libppl_dyn_host_cv84x6.so";
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
  virtual ~CV84X6(){};
  bool useCode0 = true;
  std::vector<std::shared_ptr<BM168x::Code>> multiCode;
};

} // namespace backend
} // namespace tpu_mlir
