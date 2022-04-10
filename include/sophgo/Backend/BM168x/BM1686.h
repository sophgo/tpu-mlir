#pragma once

#include "sophgo/Backend/BM168x/BM168x.h"
// typedef struct {
//   uint32_t n;
//   uint32_t c;
//   uint32_t h;
//   uint32_t w;
// } shape_t;

// typedef enum {
//   INT8 = 0,
//   FP16 = 1,
//   FP32 = 2,
//   INT16 = 3,
//   INT32 = 4,
//   BFP16 = 5,
//   PREC_END
// } PREC;
// void (*load_lookup_tables)();
// void (*store_cmd_end)();
// void (*set_cmd_len_ptr)(void *gdma_cmd_len_ptr, void *bdc_cmd_len_ptr);
// void (*sg_set_profile_path)(const char *path);
// void (*sg_set_profile_dump)(bool enable);
// void (*tensor_general_move_gen_cmd)(uint64_t src_addr, int src_local_idx, int src_N, int src_C, int src_H, int src_W, uint32_t src_N_stride, uint32_t src_C_stride, uint32_t src_H_stride, uint32_t src_W_stride, int src_format, uint64_t dst_addr, int dst_local_idx, int dst_N, int dst_C, int dst_H, int dst_W, uint32_t dst_N_stride, uint32_t dst_C_stride, uint32_t dst_H_stride, uint32_t dst_W_stride, int direction, int transpose, CMD_ID_NODE *pid_node);
// void (*tensor_align_move_gen_cmd)(int local_mem_start_addr, int local_mem_idx, uint64_t sys_mem_start_addr, int src_N, int src_C, int src_H, int src_W, int src_format, int direction, int transpose, CMD_ID_NODE *pid_node);
// void (*general_matrix_move_gen_cmd)(int local_mem_start_addr, int local_mem_idx, uint64_t sys_mem_start_addr, int sec_size, int row_num, int col_num, uint32_t row_stride, int src_format, int direction, int transpose, CMD_ID_NODE *pid_node);
// void (*tensor_broadcast_move_gen_cmd)(uint64_t src_addr, int src_local_idx, int dst_lmem_start_addr, int dst_local_idx, int src_N, int src_H, int src_W, int dst_C, uint32_t src_N_stride, uint32_t src_H_stride, uint32_t dst_N_stride, uint32_t dst_H_stride, int data_format, int stride_enable, int direction, CMD_ID_NODE *pid_node);
// void (*get_local_mem_tensor_data)(void *data_ptr, int node_idx, uint32_t N, uint32_t C, uint32_t H, uint32_t W, uint32_t laddr, int short_str, shape_t *stride, PREC precision, bool transpose);
// void (*set_local_mem_tensor_data)(void *data_ptr, int node_idx, uint32_t N, uint32_t C, uint32_t H, uint32_t W, uint32_t laddr, int short_str, shape_t *stride, PREC precision, uint64_t lane_mask, bool enable_lane_mask, bool transpose);
// void (*allow_atomic_cmodel_assert)();
// void (*forbid_atomic_cmodel_assert)();

namespace sophgo {
namespace backend {
class BM1686 : public BM168x {
public:
  static BM1686 &instance() {
    static BM1686 inst;
    return inst;
  }
public:
  virtual uint64_t get_gmem_start() override;
  virtual uint64_t get_ctx_start_addr() override;

protected:
  BM1686();
  ~BM1686();

  template <typename FPtrTy> FPtrTy CastToFPtr(const char *symbolName);

};
}
}
