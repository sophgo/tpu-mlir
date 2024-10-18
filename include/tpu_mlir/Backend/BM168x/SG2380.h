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

typedef struct
{
    struct
    {
        int s, i;
    } Y_row;
    struct
    {
        int s, i;
    } Y_col;
    struct
    {
        int s, i;
    } inner_num;
    struct
    {
        bool activ, weight;
    } hold_in_lmem;
} a16mm_slice_info_t;

typedef struct
{
    int activ_size;
    int weight_size;
    int bias_size;
    int scale_size;
    int zp_size;
    int output_size;
    // the buffer for untransposed output
    int buffer_size1;
    // the buffer for f16 weight data
    int buffer_size2;
    // the buffer for transposed input data
    int buffer_size3;
    // the buffer for unsqueeze int4 data
    // if inner_dim is not sliced, buffer4 shares the same addr with buffer1
    int buffer_size4;
    int half_offset;
    bool load_full_scale;
    int align_size;
} a16mm_size_info_t;

typedef enum
{
    DT_INT8 = (0 << 1) | 1,
    DT_UINT8 = (0 << 1) | 0,
    DT_INT16 = (3 << 1) | 1,
    DT_UINT16 = (3 << 1) | 0,
    DT_FP16 = (1 << 1) | 1,
    DT_BFP16 = (5 << 1) | 1,
    DT_INT32 = (4 << 1) | 1,
    DT_UINT32 = (4 << 1) | 0,
    DT_FP32 = (2 << 1) | 1,
    DT_INT4 = (6 << 1) | 1,
    DT_UINT4 = (6 << 1) | 0,
    DT_FP8E5M2 = (0 << 5) | (7 << 1) | 1,
    DT_FP8E4M3 = (1 << 5) | (7 << 1) | 1,
    DT_FP20 = (8 << 1) | 1,
    DT_TF32 = (9 << 1) | 1,
} data_type_t;

typedef void (*tpu_sync_all)();
typedef void (*tpu_core_context_setup)(int, int, int);
typedef bool (*a16mm_data_split_trans)(
    int, int, int, int, bool, bool, bool, int, int, data_type_t,
    data_type_t, a16mm_slice_info_t *, a16mm_size_info_t *);
typedef void (*gen_riscv_code_begin)();
typedef void (*gen_riscv_code_end)();

namespace tpu_mlir {
namespace backend {
#define BUFFER_SIZE (4 * 1024 * 1024)
class SG2380 : public BM1684X, MultiCoreInterface::Base<SG2380> {
public:
  static bool classof(const BM168x *bm168x) {
    return bm168x->getTypeID() == TypeID::get<SG2380>();
  }

  static SG2380 &instance() {
    static SG2380 SG2380;
    return SG2380;
  }
  virtual void before_codegen() override;
  virtual void after_codegen(int64_t flops = 0) override;

  tpu_sync_all dl_tpu_sync_all;
  tpu_core_context_setup dl_tpu_core_context_setup;
  a16mm_data_split_trans dl_a16mm_data_split_trans;
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
    TAG_IO0 = (3ul << 40),
    TAG_IO1 = (4ul << 40),
    TAG_IO2 = (5ul << 40),
    TAG_IO3 = (6ul << 40),
    TAG_IO4 = (7ul << 40),

  };

public:
  // specific global info
  static constexpr llvm::StringRef LIB_KERNEL_NAME =
      "libbmtpulv60_kernel_module.so";

protected:
  SG2380() {
    typeID = TypeID::get<SG2380>();
    NPU_NUM = 32;
    EU_BYTES = 16;
    LMEM_BYTES = 1 << 17; // 128KB
    LMEM_BANKS = 16;
    IC_PARALLEL = 32;
    ALIGNMENT = 0x1000;
    LMEM_BANK_BYTES = LMEM_BYTES / LMEM_BANKS;
    GMEM_START_ADDR = 0;
    COEFF_START_ADDR = GMEM_START_ADDR | TAG_WEIGHT;
    CTX_START_ADDR = GMEM_START_ADDR | TAG_ACTIVATION;
    IO_ADDR[0] = GMEM_START_ADDR | TAG_IO0;
    IO_ADDR[1] = GMEM_START_ADDR | TAG_IO1;
    IO_ADDR[2] = GMEM_START_ADDR | TAG_IO2;
    IO_ADDR[3] = GMEM_START_ADDR | TAG_IO3;
    IO_ADDR[4] = GMEM_START_ADDR | TAG_IO4;
    LIB_BACKEND_NAME = "libbackend_sg2380.so";
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
  virtual ~SG2380(){};
  bool useCode0 = true;
  std::vector<std::shared_ptr<BM168x::Code>> multiCode;
};

} // namespace backend
} // namespace tpu_mlir
