//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include <llvm/Support/Debug.h>

namespace tpu_mlir {
namespace backend {
class TgLstmKernel {
public:
  TgLstmKernel() {}

  void init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_recurrence,
            gaddr_t ga_bias, gaddr_t ga_initial_h, gaddr_t ga_initial_c,
            gaddr_t ga_cont, gaddr_t ga_sigmoid_lut,
            gaddr_t ga_sigmoid_slope_lut, gaddr_t ga_tanh_lut,
            gaddr_t ga_tanh_slope_lut, gaddr_t ga_output, gaddr_t ga_last_h,
            gaddr_t ga_last_c, int seq_length, int num_dir, int batch_size,
            int hidden_size, bool do_bias, bool with_initial_h,
            bool with_initial_c, bool with_cont, bool bidirectional,
            bool with_final_h, bool with_final_c, bool with_final_y,
            bool is_torch);

  void schedule();

protected:
  typedef struct {
    int pos_h;
    int h;
  } tiling_t;
  void init_table();
  bool need_tiling();
  void compute_without_tiling(bool forward = true);
  void compute_with_tiling(bool forward = true);
  void tiling();
  void init_gaddr(bool forward = true);
  void init_h0c0();
  void compute(int seq_idx, bool forward = true);
  uint8_t ps32_mode(int step_idx);
  void assign_matrix(cvk_ml_t *ml_mem, const cvk_ml_shape_t &shape);
  void fill_matrix(cvk_ml_t *ml_mem, uint32_t row, uint32_t col, uint32_t addr);
  void assign_addr(cvk_tl_t *tl_mem, uint32_t size);
  void matrix_to_tensor(cvk_tl_t *tensor, const cvk_ml_t &matrix);
  void matrix_for_tiu(cvk_ml_t *matrix);
  void matrix_mul(const cvk_ml_t &ml_res, const cvk_ml_t &ml_left,
                  const cvk_ml_t &ml_right, const cvk_ml_t &ml_bias,
                  uint8_t ps32_mode = 0);
  void matrix_recurrence(const cvk_ml_t &ml_res, int flip, const tiling_t &tile,
                         gaddr_t ga_weight, gaddr_t ga_bias);
  void zeros(const cvk_ml_t &matrix);
  void eltwise_add(const cvk_ml_t &ml_res, const cvk_ml_t &ml_right);
  void eltwise_mul(const cvk_ml_t &ml_res, const cvk_ml_t &ml_right);
  void eltwise_mul(const cvk_ml_t &ml_res, const cvk_ml_t &ml_left,
                   const cvk_ml_t &ml_right);
  void eltwise_sub(const cvk_ml_t &ml_res, const cvk_ml_t &ml_right);
  void eltwise_sub(const cvk_ml_t &ml_res, const cvk_ml_t &ml_left,
                   const cvk_ml_t &ml_right);
  void sigmoid(const cvk_ml_t &ml_out, const cvk_ml_t &ml_in,
               const cvk_ml_t &ml_buff);
  void tanh(const cvk_ml_t &ml_out, const cvk_ml_t &ml_in,
            const cvk_ml_t &ml_buff);
  void load_cont(const cvk_ml_t &ml_cont, gaddr_t cont_addr);

protected:
  uint32_t layer_id;
  gaddr_t ga_input;
  gaddr_t ga_recurrence;
  gaddr_t ga_bias;
  gaddr_t ga_init_h;
  gaddr_t ga_init_c;
  gaddr_t ga_cont;
  gaddr_t ga_sigmoid_lut;
  gaddr_t ga_sigmoid_slope_lut;
  gaddr_t ga_tanh_lut;
  gaddr_t ga_tanh_slope_lut;
  gaddr_t ga_output;
  gaddr_t ga_last_h;
  gaddr_t ga_last_c;
  // for bidirectional
  gaddr_t ga_store, ga_h0, ga_c0, ga_store_h, ga_store_c;
  gaddr_t ga_xi, ga_xf, ga_xc, ga_xo, ga_ri, ga_rf, ga_rc, ga_ro;
  gaddr_t ga_rbi, ga_rbf, ga_rbc, ga_rbo;
  int seq_length;
  int batch_size;
  int hidden_size;
  int num_dir;
  uint32_t input_bytes;
  uint32_t recurrence_bytes;
  uint32_t hidden_bytes;
  uint32_t x_bytes;
  bool is_torch_bidir;
  bool do_bias;
  bool with_initial_h;
  bool with_initial_c;
  bool with_cont;
  bool bidirectional;
  bool with_final_h;
  bool with_final_c;
  bool with_final_y;
  cvk_fmt_t fmt;
  int fmt_size;
  uint32_t num_recurrence;

  // for lmem addr alloc
  cvk_tl_shape_t table_shape;
  uint32_t table_size;
  uint32_t addr_sigmoid;
  uint32_t addr_sigmoid_slope;
  uint32_t addr_tanh;
  uint32_t addr_tanh_slope;
  uint32_t lmem_used;

  // for tiling
  int step_size;
  int step_num;
  std::vector<tiling_t> tiles;         // tilng hidden_size
  std::vector<cvk_ml_t> ml_hiddens[2]; // one for backup
  std::vector<cvk_ml_t> ml_cells;      // cell state
  uint32_t addr_recurrence;            // for recurrence
  uint32_t addr_bias;
  uint32_t addr_cont;  // for cont addr
  uint32_t addr_work0; // for lut buffer and ps32 bias buffer
  uint32_t addr_work1; // for dot(h_t,r) result
  uint32_t addr_work2; // for gate buffer
  cvk_mg_stride_t x_gstride;
  cvk_mg_stride_t h_gstride;
  cvk_mg_stride_t oh_gstride;
};
} // namespace backend
} // namespace tpu_mlir
