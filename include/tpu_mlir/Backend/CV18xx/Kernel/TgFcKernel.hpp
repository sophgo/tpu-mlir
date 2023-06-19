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
  // Y[M, N] = L[M,K] * R[K,N] + B[4,N]
class TgFcKernel {
public:
  TgFcKernel() {}

  void init(uint32_t layer_id, gaddr_t ga_input, gaddr_t ga_weight,
            gaddr_t ga_bias, gaddr_t ga_output, int M, int K, int N,
            bool do_bias, bool do_relu, std::vector<int> *rshift_width,
            std::vector<int> *multiplier,
            const std::vector<uint8_t> *old_filter,
            std::vector<uint8_t> *new_filter, int batch_high, int batch_low,
            bool lstride, bool rstride, bool ostride, cvk_fmt_t fmt,
            bool do_quant_bf16 = false, gaddr_t ga_scale = 0,
            gaddr_t ga_zeropoint = 0);

  void selectTilePolicy();
  void schedule();

protected:
  void compute(int32_t step_idx);
  void load(int32_t step_idx);
  void store(int32_t step_idx);
  uint32_t lmem_matrix_size(uint32_t row, uint32_t col,
                            bool ps32 = false) const;
  void load_L(int32_t step_idx);
  void load_R(int32_t step_idx);
  void load_B(int32_t step_idx);
  void load_Q(int32_t step_idx);
  void quant_bf16(int32_t step_idx);
  void matrix_to_tensor(cvk_tl_t *tensor, const cvk_ml_t &matrix);
  void update_tl_matrix(int32_t step_idx);
  void set_laddr();
  void matrix_for_tiu();
  bool is_last_k(int32_t step_idx) const;
  inline uint32_t slice_m() const { return (M + tile_M - 1) / tile_M; }
  inline uint32_t slice_k() const { return (K + tile_K - 1) / tile_K; }
  inline uint32_t slice_n() const { return (N + tile_N - 1) / tile_N; }
  typedef struct {
    uint32_t L, R, B, Y, Q;
    uint32_t blob_L, blob_R, blob_B, blob_Y, blob_Q;
  } lmem_size_t;
  lmem_size_t get_lmem_size() const;
  uint32_t total_lmem_size() const;
  void update_batch_info(int high_idx, int low_idx);

  bool try_tiling_group_parallel();
  bool try_no_tiling();
  bool try_tiling_parallel_mn();
  bool try_tiling_parallel_kn();
  bool try_tiling_no_parallel();
  void tiling_generic();
  void schedule_parallel();
  void schedule_group_parallel();
  void schedule_no_parallel();

  void filter_optimize();
  bool try_optimize();

protected:

  gaddr_t ga_input;
  gaddr_t ga_weight;
  gaddr_t ga_bias;
  gaddr_t ga_output;
  gaddr_t ga_scale;
  gaddr_t ga_zeropoint;

  gaddr_t ga_i, ga_w, ga_o, ga_b; // for origin addr

  uint32_t M;
  uint32_t K;
  uint32_t N;

  bool do_bias;
  bool do_relu;
  bool do_quant_bf16;
  bool do_qdm;
  std::vector<int> rshift;
  std::vector<int> multiplier;
  int cur_rshift;
  int cur_multiplier;
  const std::vector<uint8_t> *old_filter;
  std::vector<uint8_t> *new_filter;
  cvk_fmt_t fmt;
  int fmt_size;
  cvk_fmt_t r_fmt; // filter fmt
  int r_fmt_size;  // filter fmt bytes
  uint32_t layer_id;

  cvk_ml_t tl_Y;
  cvk_ml_t tl_L;
  cvk_ml_t tl_B;
  cvk_ml_t tl_R;
  cvk_ml_t tl_scale;
  cvk_ml_t tl_zeropoint;

  int batch_high;
  int batch_low;
  int batch; // high * low
  bool lstride;
  bool rstride;
  bool ostride;
  cvk_mg_stride_t left_gstride;
  cvk_mg_stride_t right_gstride;
  cvk_mg_stride_t output_gstride;

  bool do_parallel;
  bool bias_loaded;
  bool quant_loaded;
  uint32_t maxM, maxK, maxN;
  uint32_t TOTAL_EU;
  uint32_t tile_M;
  uint32_t tile_K;
  uint32_t tile_N;
  typedef enum {
    FC_OPT_COMPRESS,
    FC_OPT_REPOSE,
    FC_NO_OPT,
  } fc_opt_t;
  fc_opt_t opt_mode;
  int opt_offset; // for batch compress pos
  std::vector<int> opt_pos;
  typedef struct TileInfo {
    uint32_t pos_m;
    uint32_t pos_k;
    uint32_t pos_n;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    int RB_idx;
    int L_idx;
    int Y_idx;
    int batch_high;
    int batch_low;
    int opt_idx; // compress pos
  } tile_info_t;
  std::vector<tile_info_t> tiles;
  typedef enum {
    FC_NO_TILING,      // no tile
    FC_GROUP_PARALLEL, // only tile N
    FC_PARALLEL_KN,    // tile K, N
    FC_PARALLEL_MN,    // tile M, N
    FC_NO_PARALLEL,    // tile M, K, N
  } fc_mode_t;
  fc_mode_t mode;
  int total_steps;
  std::vector<uint32_t> Y_laddr; // [M, tile_N]
  uint32_t L_laddr[2];           // [M, tile_K]
  uint32_t R_laddr[2];           // [tile_K, tile_N]
  uint32_t B_laddr[2];           // [4, tile_N]
  uint32_t Q_laddr[4];           // [1, tile_N]
};
} // namespace backend
} // namespace tpu_mlir
