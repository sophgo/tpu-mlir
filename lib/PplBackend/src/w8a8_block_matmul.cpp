//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "w8a8_block_matmul.h"
#include "ppl_static_host.h"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ======================= decode family =======================

// TILE_N fixed at block_size_n; the design search never chose otherwise.
constexpr int kTileN = 128;
constexpr double kCv84x4PeakTflops = 48.0; // CV84X4, 48 TFLOPS @ 1.5 GHz
constexpr double kCv84x4DdrGbs = 136.0;    // CV84X4 measured stream
constexpr int kDecMaxWeightKSpan = 2048;
constexpr int kActKSpan = 512;
constexpr int kTileMMax = 4096;
constexpr double kActHandshakeUs = 0.51; // per act chunk, mixed splits only
constexpr double kReduceTailUs = 2.3;    // once, p_k > 1

static int64_t div_up_i64(int64_t a, int64_t b) { return (a + b - 1) / b; }

static int weight_k_span_of(int k, int bsk) {
  int span = std::min(k, kDecMaxWeightKSpan);
  return std::max(bsk, (span / bsk) * bsk);
}

// Board arm switches (PPL_FORCE_SPAN/ACTK/SPLIT): force one factor per arm.
static int env_span() {
  const char *s = getenv("PPL_FORCE_SPAN");
  return s ? atoi(s) : 0;
}

static int env_actk() {
  const char *s = getenv("PPL_FORCE_ACTK");
  return s ? atoi(s) : 0;
}

struct MemQuery {
  uint64_t output, input, weight, weight_scale;
  int g, m, k, n, core_num, lane;
  int block_size_k, block_size_n;
};

// Largest lane-aligned TILE_M <= kTileMMax accepted by the allocator probe.
template <typename FitsFn>
static int fit_tile_m(FitsFn fits, int64_t num_rows, int lane) {
  int64_t hi = std::min<int64_t>(num_rows, kTileMMax);
  hi = (hi / lane) * lane;
  if (hi < lane) {
    hi = lane;
  }
  if (!fits(lane)) {
    return 0;
  }
  int lo = lane;
  int best = lane;
  while (lo <= hi) {
    const int mid = static_cast<int>(((lo + hi) / 2 / lane) * lane);
    if (mid < lane) {
      break;
    }
    if (fits(mid)) {
      best = mid;
      lo = mid + lane;
    } else {
      hi = mid - lane;
    }
  }
  return best;
}

// check_mem gained a trailing l2tensor list in some ppl versions.
template <typename... A>
static auto check_mem_bf16(int, A... a)
    -> decltype(w8a8_block_matmul_decode_bf16_check_mem(a...,
                                                        (int32_t *)nullptr)) {
  return w8a8_block_matmul_decode_bf16_check_mem(a..., (int32_t *)nullptr);
}

template <typename... A>
static auto check_mem_bf16(long, A... a)
    -> decltype(w8a8_block_matmul_decode_bf16_check_mem(a...)) {
  return w8a8_block_matmul_decode_bf16_check_mem(a...);
}

static bool env_split(int *p_m, int *p_n, int *p_k) {
  const char *s = getenv("PPL_FORCE_SPLIT");
  if (s == nullptr) {
    return false;
  }
  return sscanf(s, "%d,%d,%d", p_m, p_n, p_k) == 3;
}

static bool alpha_fits(const MemQuery &q, int p_m, int p_n, int p_k, int tile_m,
                       int tile_n, int weight_k_span, int act_k_span) {
  return check_mem_bf16(0, q.output, q.input, q.weight, q.weight_scale, q.g,
                        q.m, q.k, q.n, q.core_num, p_m, p_n, p_k, tile_m,
                        tile_n, weight_k_span, act_k_span, q.block_size_k,
                        q.block_size_n) == 0;
}

struct Tiling {
  int p_m;
  int p_n;
  int p_k;
  int tile_m;
  int tile_n;
  int weight_k_span;
  int act_k_span;
};

// Cost model for one split point; false = infeasible.
static bool eval_point(const MemQuery &q, int p_m, int p_n, int p_k,
                       double *t_out, Tiling *out) {
  const int core_num = q.core_num, g = q.g, m = q.m, k = q.k, n = q.n;
  const int bsk = q.block_size_k, bsn = q.block_size_n;
  if (p_m * p_n * p_k != core_num) {
    return false;
  }
  // One unit per core minimum per axis; no separate P_G (row split covers
  // batch boundaries).
  if (int64_t(g) * m < p_m || k < p_k * bsk || n < p_n * bsn) {
    return false;
  }
  // K-split L2 slab: p_k*num_rows*N*2B <= 1 MiB inside (num_rows<=16,
  // N<=8192); over capacity falls back to p_k = 1.
  if (p_k > 1) {
    const int64_t slab_bytes = int64_t(p_k) * g * m * n * 2;
    if (slab_bytes > (1 << 20) || int64_t(g) * m > 16 || n > 8192) {
      return false;
    }
  }
  const int64_t num_rows = div_up_i64(int64_t(g) * m, p_m);
  // K shard bound matches the kernel's bal_first/bal_size exactly
  // (ceil((K/bsk)/P_K)*bsk); imbalance is priced by ser below.
  const int kc = static_cast<int>(div_up_i64(k / bsk, p_k)) * bsk;
  const int n_shard_len = static_cast<int>(div_up_i64(n, p_n));
  int weight_k_span = weight_k_span_of(kc, bsk);
  if (env_span() >= bsk) {
    weight_k_span = std::min(kc, (env_span() / bsk) * bsk);
  }
  int act_k_span = std::max(bsk, std::min(kActKSpan, (kc / bsk) * bsk));
  // Forced act_k_span: clamp to [bsk, kc], round down to a bsk multiple.
  if (env_actk() >= bsk) {
    act_k_span = std::max(bsk, std::min((env_actk() / bsk) * bsk, kc));
  }
  if (weight_k_span < bsk || kTileN > n_shard_len) {
    return false;
  }

  int tile_m = fit_tile_m(
      [&](int tm) {
        return alpha_fits(q, p_m, p_n, p_k, tm, kTileN, weight_k_span,
                          act_k_span);
      },
      num_rows, q.lane);
  if (tile_m < q.lane) {
    return false;
  }

  // Traffic (bytes) and per-core engine time (ms).
  const int64_t num_m_tiles = div_up_i64(num_rows, tile_m);
  const double weight_bytes = double(p_m) * double(num_m_tiles) * n * k * 1.0;
  const double act_bytes = double(p_n) * g * m * k * 2.0;
  // Reduce priced in the GLOBAL form; not rebased to the kernel's L2 staging
  // until the cost model gains its tail term (split_strategy.md).
  const double out_bytes = double(p_k) * g * m * n * 2.0;
  const double red_bytes = (p_k > 1) ? double(p_k + 1) * g * m * n * 4.0 : 0.0;
  const double ddr = (weight_bytes + act_bytes + out_bytes + red_bytes) /
                     (kCv84x4DdrGbs * 1e9) * 1e3;
  const double cube = 2.0 * g * m * k * n / (kCv84x4PeakTflops * 1e12) * 1e3;
  // Requant: 4 elementwise passes, 256 elem/cycle per core.
  const double requant_elems = double(p_n) * g * m * k;
  const double vec = (requant_elems / core_num) * 4.0 / 256.0 / 1.5e9 * 1e3;

  // ser = slowest-core work / mean; engines are per-core, the ddr pipe is.
  const double imbal_m = double(p_m) * double(num_rows) / (double(g) * m);
  const double imbal_n = double(p_n) * double(div_up_i64(n, p_n)) / double(n);
  const double imbal_k = double(p_k) * double(kc) / double(k);
  const double ser = imbal_m * imbal_n * imbal_k;

  // Engines overlap (max); requant and two board penalties add serially.
  const int act_chunks = static_cast<int>(div_up_i64(kc, act_k_span));
  const double act_hs =
      (p_n > 1 && p_k > 1) ? kActHandshakeUs * act_chunks * 1e-3 : 0.0;
  const double red_tail = (p_k > 1) ? kReduceTailUs * 1e-3 : 0.0;
  const double main = std::max(std::max(ddr, ser * cube), ser * vec);
  *t_out = main + ser * vec + act_hs + red_tail;
  out->p_m = p_m;
  out->p_n = p_n;
  out->p_k = p_k;
  out->tile_m = tile_m;
  out->tile_n = kTileN;
  out->weight_k_span = weight_k_span;
  out->act_k_span = act_k_span;
  return true;
}

static Tiling decide_tiling(const MemQuery &q) {
  const int core_num = q.core_num;
  {
    int fm = 0, fn = 0, fk = 0;
    if (env_split(&fm, &fn, &fk)) {
      Tiling forced{};
      double t = 0.0;
      if (eval_point(q, fm, fn, fk, &t, &forced)) {
        return forced;
      }
    }
  }
  static const int kSplits[][3] = {{1, core_num, 1},
                                   {2, core_num / 2, 1},
                                   {core_num, 1, 1},
                                   {1, core_num / 2, 2},
                                   {1, 1, core_num}};
  Tiling best{};
  double best_t = 0.0;
  bool have = false;
  for (const auto &sp : kSplits) {
    Tiling cand{};
    double t = 0.0;
    if (!eval_point(q, sp[0], sp[1], sp[2], &t, &cand)) {
      continue;
    }
    if (!have || t < best_t - 1e-12) {
      best = cand;
      best_t = t;
      have = true;
    }
  }
  if (!have) {
    // Nothing feasible: pure-N fallback, shrunk by the launch() retry loop.
    best =
        Tiling{1, core_num, 1, q.lane, kTileN, q.block_size_k, q.block_size_k};
  }
  return best;
}

// Allocator-rejection recovery: halve TILE_M -> weight_k_span -> act_k_span.
static void shrink(Tiling &t, int bsk, int lane) {
  if (t.tile_m > lane) {
    t.tile_m = std::max(lane, (t.tile_m / 2 / lane) * lane);
  } else if (t.weight_k_span > bsk) {
    t.weight_k_span = std::max(bsk, (t.weight_k_span / 2 / bsk) * bsk);
  } else if (t.act_k_span > bsk) {
    t.act_k_span = std::max(bsk, (t.act_k_span / 2 / bsk) * bsk);
  }
}

static bool exhausted(const Tiling &t, int bsk, int lane) {
  return t.tile_m <= lane && t.weight_k_span <= bsk && t.act_k_span <= bsk;
}

static int launch(const MemQuery &q, int32_t dtype) {
  Tiling t = decide_tiling(q);
  for (;;) {
    int ret = -1;
    switch (dtype) {
    case DTYPE_BFP16:
      ret = w8a8_block_matmul_decode_bf16(
          q.output, q.input, q.weight, q.weight_scale, q.g, q.m, q.k, q.n,
          q.core_num, t.p_m, t.p_n, t.p_k, t.tile_m, t.tile_n, t.weight_k_span,
          t.act_k_span, q.block_size_k, q.block_size_n);
      break;
    case DTYPE_FP16:
      ret = w8a8_block_matmul_decode_f16(
          q.output, q.input, q.weight, q.weight_scale, q.g, q.m, q.k, q.n,
          q.core_num, t.p_m, t.p_n, t.p_k, t.tile_m, t.tile_n, t.weight_k_span,
          t.act_k_span, q.block_size_k, q.block_size_n);
      break;
    default:
      assert(0 && "not supported, need fix w8a8_block_matmul.pl\n");
      return -1;
    }
    if (getenv("PPL_TILING_DEBUG")) {
      // Arm log (parse anchor for regress/parse_verdict.py); flush now.
      printf("[tiling] P=%d.%d.%d TILE_M=%d TILE_N=%d "
             "weight_k_span=%d act_k_span=%d ret=%d\n",
             t.p_m, t.p_n, t.p_k, t.tile_m, t.tile_n, t.weight_k_span,
             t.act_k_span, ret);
      fflush(stdout);
    }
    if (ret == 0 || exhausted(t, q.block_size_k, q.lane)) {
      return ret;
    }
    shrink(t, q.block_size_k, q.lane);
  }
}

static int fill_tiling_struct(const MemQuery &q, const Tiling &t, int32_t dtype,
                              void *buffer) {
  switch (dtype) {
  case DTYPE_BFP16:
    return fill_w8a8_block_matmul_decode_bf16_struct(
        q.output, q.input, q.weight, q.weight_scale, q.g, q.m, q.k, q.n,
        q.core_num, t.p_m, t.p_n, t.p_k, t.tile_m, t.tile_n, t.weight_k_span,
        t.act_k_span, q.block_size_k, q.block_size_n, buffer);
  case DTYPE_FP16:
    return fill_w8a8_block_matmul_decode_f16_struct(
        q.output, q.input, q.weight, q.weight_scale, q.g, q.m, q.k, q.n,
        q.core_num, t.p_m, t.p_n, t.p_k, t.tile_m, t.tile_n, t.weight_k_span,
        t.act_k_span, q.block_size_k, q.block_size_n, buffer);
  default:
    assert(0 && "not supported, need fix w8a8_block_matmul.pl\n");
    return 0;
  }
}

static MemQuery query_of(tensor_spec_t *in_spec, tensor_spec_t *out_spec,
                         int core_num, const w8a8_block_matmul_spec_t *param) {
  MemQuery q{};
  int npu_num = 0, npu_size = 0;
  get_chip_info(&npu_num, &npu_size); // 84X2: 16 lanes per core, 256K LMEM
  q.lane = npu_num;
  q.output = out_spec[0].addr;
  q.input = in_spec[0].addr;
  q.weight = in_spec[1].addr;
  q.weight_scale = in_spec[2].addr;
  q.g = in_spec[0].shape[0];
  q.m = in_spec[0].shape[1];
  // K is on the last non-trivial axis (3D and 4D specs both reach here).
  q.k = in_spec[0].shape[in_spec[0].dims - 1];
  q.n = in_spec[1].shape[0];
  q.core_num = core_num;
  q.block_size_k = param->block_size_k;
  q.block_size_n = param->block_size_n;
  return q;
}

// ======================= prefill family =======================

// Independent calibration (see file header).
constexpr int kPreMaxWeightKSpan = 4096;

// Keep enough k-chunks per shard to overlap.
constexpr int kMinKChunks = 4;

static int pre_weight_k_span_of(int k, int bsk) {
  int span = std::min(k, kPreMaxWeightKSpan);
  span = std::min(span, std::max(bsk, k / kMinKChunks));
  return std::max(bsk, (span / bsk) * bsk);
}

// Prefill arm switch: filters the enumeration (guards still apply).
static void pre_env_split(int *p_m, int *p_n, int *p_k) {
  const char *s = getenv("PPL_FORCE_SPLIT");
  if (!s || sscanf(s, "%d,%d,%d", p_m, p_n, p_k) != 3)
    return;
}

static int env_tilem() {
  const char *s = getenv("PPL_FORCE_TILEM");
  return s ? atoi(s) : 0;
}

static int env_tkw() {
  const char *s = getenv("PPL_FORCE_TKW");
  return s ? atoi(s) : 0;
}

template <typename... A>
static auto pre_check_mem_bf16(int, A... a)
    -> decltype(w8a8_block_matmul_prefill_bf16_check_mem(a...,
                                                         (int32_t *)nullptr)) {
  return w8a8_block_matmul_prefill_bf16_check_mem(a..., (int32_t *)nullptr);
}

template <typename... A>
static auto pre_check_mem_bf16(long, A... a)
    -> decltype(w8a8_block_matmul_prefill_bf16_check_mem(a...)) {
  return w8a8_block_matmul_prefill_bf16_check_mem(a...);
}

static bool pre_alpha_fits(const MemQuery &q, int p_m, int p_n, int p_k,
                           int tile_m, int tile_n, int weight_k_span,
                           int act_k_span) {
  return pre_check_mem_bf16(0, q.output, q.input, q.weight, q.weight_scale, q.g,
                            q.m, q.k, q.n, q.core_num, p_m, p_n, p_k, tile_m,
                            tile_n, weight_k_span, act_k_span, q.block_size_k,
                            q.block_size_n) == 0;
}

static bool pre_eval_point(const MemQuery &q, int p_m, int p_n, int p_k,
                           double *t_out, Tiling *out) {
  const int core_num = q.core_num, g = q.g, m = q.m, k = q.k, n = q.n;
  const int bsk = q.block_size_k, bsn = q.block_size_n;
  if (p_m * p_n * p_k != core_num) {
    return false;
  }
  if (int64_t(g) * m < p_m || k < p_k * bsk || n < p_n * bsn) {
    return false;
  }
  const int64_t num_rows = div_up_i64(int64_t(g) * m, p_m);
  // Largest per-core shard under the kernel's balanced split.
  const int kc = static_cast<int>(div_up_i64(k, p_k));
  const int n_shard_len = static_cast<int>(div_up_i64(n, p_n));
  // Fixed-bsk chunking would starve the DMA burst (board 2026-09-23).
  int weight_k_span = pre_weight_k_span_of(kc, bsk);
  int act_k_span = std::max(bsk, std::min(kActKSpan, (kc / bsk) * bsk));
  if (weight_k_span < bsk || kTileN > n_shard_len) {
    return false;
  }
  // PPL_FORCE_TKW / PPL_FORCE_ACTK arm switches.
  {
    const int ftkw = env_tkw();
    if (ftkw >= bsk) {
      weight_k_span = (ftkw / bsk) * bsk;
    }
  }
  {
    const int factk = env_actk();
    if (factk >= bsk) {
      act_k_span = std::max(bsk, std::min((factk / bsk) * bsk, kc));
    }
  }

  int tile_m;
  // PPL_FORCE_TILEM: skip the search; forced point still probes.
  const int forced_tm = env_tilem();
  if (forced_tm >= q.lane) {
    tile_m = (forced_tm / q.lane) * q.lane;
    if (tile_m < q.lane || tile_m > kTileMMax) {
      return false;
    }
    if (!pre_alpha_fits(q, p_m, p_n, p_k, tile_m, kTileN, weight_k_span,
                        act_k_span)) {
      return false;
    }
  } else {
    tile_m = fit_tile_m(
        [&](int tm) {
          return pre_alpha_fits(q, p_m, p_n, p_k, tm, kTileN, weight_k_span,
                                act_k_span);
        },
        num_rows, q.lane);
  }
  if (tile_m < q.lane) {
    return false;
  }
  // K-split parity slots need 2*core_num*tile_m*kTileN*2B <= 1 MiB (the
  // kernel allocates them before check_mem; tile_m <= 512). Clamp; a forced
  // arm over capacity is infeasible, not re-parameterized.
  {
    const int tm_slot_max = (1 << 20) / (2 * core_num * kTileN * 2);
    if (p_k > 1 && tile_m > tm_slot_max) {
      if (forced_tm >= q.lane) {
        return false;
      }
      tile_m = tm_slot_max;
    }
  }

  // T74-A (board -92 us at K8192): raise act_k_span to the largest rung
  // passing the probe (longer segments burst faster, 25.5->27.6 GB/s/core);
  // capped at 1024 (2048 passes check_mem but fails the real link, ret=17).
  if (env_actk() < bsk) {
    for (int ak = std::min(1024, (kc / bsk) * bsk); ak > act_k_span; ak /= 2) {
      if (pre_alpha_fits(q, p_m, p_n, p_k, tile_m, kTileN, weight_k_span, ak)) {
        act_k_span = ak;
        break;
      }
    }
  }

  // Traffic (bytes) and per-core engine time (ms).
  const int64_t num_m_tiles = div_up_i64(num_rows, tile_m);
  const double weight_bytes = double(p_m) * double(num_m_tiles) * n * k * 1.0;
  const double act_bytes = double(p_n) * g * m * k * 2.0;
  const double out_bytes = double(p_k) * g * m * n * 2.0;
  const double red_bytes = (p_k > 1) ? double(p_k + 1) * g * m * n * 4.0 : 0.0;
  const double ddr = (weight_bytes + act_bytes + out_bytes + red_bytes) /
                     (kCv84x4DdrGbs * 1e9) * 1e3;
  const double cube = 2.0 * g * m * k * n / (kCv84x4PeakTflops * 1e12) * 1e3;
  // Requant: 4 elementwise passes, 256 elem/cycle per core.
  const double requant_elems = double(p_n) * g * m * k;
  const double vec = (requant_elems / core_num) * 4.0 / 256.0 / 1.5e9 * 1e3;

  // ser = slowest-core work / mean; engines are per-core, the ddr pipe is.
  const double imbal_m = double(p_m) * double(num_rows) / (double(g) * m);
  const double imbal_n = double(p_n) * double(div_up_i64(n, p_n)) / double(n);
  const double imbal_k =
      double(p_k) * double(div_up_i64(k / bsk, p_k) * bsk) / double(k);
  const double ser = imbal_m * imbal_n * imbal_k;

  const double main = std::max(std::max(ddr, ser * cube), ser * vec);
  *t_out = main + ser * vec;
  out->p_m = p_m;
  out->p_n = p_n;
  out->p_k = p_k;
  out->tile_m = tile_m;
  out->tile_n = kTileN;
  out->weight_k_span = weight_k_span;
  out->act_k_span = act_k_span;
  return true;
}

// Board guard rules; evidence table in split_strategy.md.
static bool board_guard_ok(int p_m, int p_n, int p_k, int k, int n, int m) {
  if (p_k == 4 && k < 20000) {
    return false;
  }
  if (p_k == 2 && n >= 4096 && k < 16384) {
    return false;
  }
  if (p_k == 2 && k >= 16384 && k < 20000 && n < 5120) {
    return false;
  }
  // pk1 ban measured at M=512 only (1.2.2 beat all rivals 9/9, 1.8-5.1%).
  if (p_k == 1 && k >= 16384 && k < 20000 && n >= 5120 && m <= 512) {
    return false;
  }
  return true;
}

static Tiling pre_decide_tiling(const MemQuery &q) {
  const int core_num = q.core_num;
  static const int kSplits[][3] = {{1, core_num, 1},
                                   {2, core_num / 2, 1},
                                   {core_num, 1, 1},
                                   {1, core_num / 2, 2},
                                   {1, 1, core_num}};
  int fp_m = 0, fp_n = 0, fp_k = 0;
  pre_env_split(&fp_m, &fp_n, &fp_k);
  Tiling best{};
  double best_t = 0.0;
  bool have = false;
  for (const auto &sp : kSplits) {
    if (fp_m && (sp[0] != fp_m || sp[1] != fp_n || sp[2] != fp_k)) {
      continue;
    }
    if (!fp_m && !board_guard_ok(sp[0], sp[1], sp[2], q.k, q.n, q.m)) {
      continue;
    }
    Tiling cand{};
    double t = 0.0;
    if (!pre_eval_point(q, sp[0], sp[1], sp[2], &t, &cand)) {
      continue;
    }
    if (!have || t < best_t - 1e-12) {
      best = cand;
      best_t = t;
      have = true;
    }
  }
  if (!have) {
    // Nothing feasible: pure-N fallback, shrunk by the retry loop.
    best =
        Tiling{1, core_num, 1, q.lane, kTileN, q.block_size_k, q.block_size_k};
  }
  return best;
}

static int pre_launch(const MemQuery &q, int32_t dtype) {
  Tiling t = pre_decide_tiling(q);
  for (;;) {
    int ret = -1;
    switch (dtype) {
    case DTYPE_BFP16:
      ret = w8a8_block_matmul_prefill_bf16(
          q.output, q.input, q.weight, q.weight_scale, q.g, q.m, q.k, q.n,
          q.core_num, t.p_m, t.p_n, t.p_k, t.tile_m, t.tile_n, t.weight_k_span,
          t.act_k_span, q.block_size_k, q.block_size_n);
      break;
    case DTYPE_FP16:
      ret = w8a8_block_matmul_prefill_f16(
          q.output, q.input, q.weight, q.weight_scale, q.g, q.m, q.k, q.n,
          q.core_num, t.p_m, t.p_n, t.p_k, t.tile_m, t.tile_n, t.weight_k_span,
          t.act_k_span, q.block_size_k, q.block_size_n);
      break;
    default:
      assert(0 && "not supported, need fix w8a8_block_matmul.pl\n");
      return -1;
    }
    if (getenv("PPL_TILING_DEBUG")) {
      // Arm log. L2 = reduce-landing guard (g*m*n*2 <= L2); parse anchor.
      const int l2 =
          t.p_k > 1 && int64_t(q.g) * q.m * q.n * 2 <= (int64_t(1) << 20);
      printf("[tiling] P=%d.%d.%d TILE_M=%d TILE_N=%d "
             "weight_k_span=%d act_k_span=%d ret=%d L2=%d\n",
             t.p_m, t.p_n, t.p_k, t.tile_m, t.tile_n, t.weight_k_span,
             t.act_k_span, ret, l2);
      fflush(stdout);
    }
    if (ret == 0 || exhausted(t, q.block_size_k, q.lane)) {
      return ret;
    }
    shrink(t, q.block_size_k, q.lane);
  }
}

static int pre_fill_tiling_struct(const MemQuery &q, const Tiling &t,
                                  int32_t dtype, void *buffer) {
  switch (dtype) {
  case DTYPE_BFP16:
    return fill_w8a8_block_matmul_prefill_bf16_struct(
        q.output, q.input, q.weight, q.weight_scale, q.g, q.m, q.k, q.n,
        q.core_num, t.p_m, t.p_n, t.p_k, t.tile_m, t.tile_n, t.weight_k_span,
        t.act_k_span, q.block_size_k, q.block_size_n, buffer);
  case DTYPE_FP16:
    return fill_w8a8_block_matmul_prefill_f16_struct(
        q.output, q.input, q.weight, q.weight_scale, q.g, q.m, q.k, q.n,
        q.core_num, t.p_m, t.p_n, t.p_k, t.tile_m, t.tile_n, t.weight_k_span,
        t.act_k_span, q.block_size_k, q.block_size_n, buffer);
  default:
    assert(0 && "not supported, need fix w8a8_block_matmul.pl\n");
    return 0;
  }
}

static MemQuery pre_query_of(tensor_spec_t *in_spec, tensor_spec_t *out_spec,
                             int core_num,
                             const w8a8_block_matmul_spec_t *param) {
  MemQuery q{};
  int npu_num = 0, npu_size = 0;
  get_chip_info(&npu_num, &npu_size); // 84X2: 16 lanes per core, 256K LMEM
  q.lane = npu_num;
  q.output = out_spec[0].addr;
  q.input = in_spec[0].addr;
  q.weight = in_spec[1].addr;
  q.weight_scale = in_spec[2].addr;
  q.g = in_spec[0].shape[0];
  q.m = in_spec[0].shape[1];
  // K is on the last non-trivial axis (3D and 4D specs both reach here).
  q.k = in_spec[0].shape[in_spec[0].dims - 1];
  q.n = in_spec[1].shape[0];
  q.core_num = core_num;
  q.block_size_k = param->block_size_k;
  q.block_size_n = param->block_size_n;
  return q;
}

// ======================= exported API =======================
// Routing: M == 1 -> decode (bandwidth-bound), M > 1 -> prefill.

#ifdef __cplusplus
extern "C" {
#endif

static bool use_prefill_family(tensor_spec_t *in_spec) {
  const char *force_kernel = getenv("PPL_FORCE_FP8_KERNEL");
  if (force_kernel)
    return strcmp(force_kernel, "prefill") == 0;
  return in_spec[0].shape[1] > 1;
}

static void decode_global_impl(void *param, size_t param_size, void *input,
                               void *output) {
  const auto *spec = (w8a8_block_matmul_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input;
  tensor_spec_t *out_spec = (tensor_spec_t *)output;
  const MemQuery q = query_of(in_spec, out_spec, get_core_num(), spec);
  const int ret = launch(q, in_spec[0].dtype);
  if (ret != 0) {
    printf("Error: w8a8_block_matmul kernel failed due to address assignment "
           "failure\n");
    exit(-1);
  }
}

static int decode_dyn_impl(void *param, void *input, void *output,
                           void *buffer) {
  const auto *spec = (w8a8_block_matmul_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input;
  tensor_spec_t *out_spec = (tensor_spec_t *)output;
  const MemQuery q = query_of(in_spec, out_spec, get_core_num(), spec);
  const Tiling t = decide_tiling(q);
  const int len = fill_tiling_struct(q, t, in_spec[0].dtype, buffer);
  if (getenv("PPL_TILING_DEBUG")) {
    printf("[tiling-dyn] buffer=%d len=%d P=%d.%d.%d TILE_M=%d "
           "TILE_N=%d weight_k_span=%d act_k_span=%d dtype=%d\n",
           buffer == nullptr ? 0 : 1, len, t.p_m, t.p_n, t.p_k, t.tile_m,
           t.tile_n, t.weight_k_span, t.act_k_span, in_spec[0].dtype);
    fflush(stdout);
  }
  return len;
}

static void prefill_global_impl(void *param, size_t param_size, void *input,
                                void *output) {
  const auto *spec = (w8a8_block_matmul_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input;
  tensor_spec_t *out_spec = (tensor_spec_t *)output;
  const MemQuery q = pre_query_of(in_spec, out_spec, get_core_num(), spec);
  const int ret = pre_launch(q, in_spec[0].dtype);
  if (ret != 0) {
    printf("Error: w8a8_block_matmul_prefill kernel failed due to address "
           "assignment failure\n");
    exit(-1);
  }
}

static int prefill_dyn_impl(void *param, void *input, void *output,
                            void *buffer) {
  const auto *spec = (w8a8_block_matmul_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input;
  tensor_spec_t *out_spec = (tensor_spec_t *)output;
  const MemQuery q = pre_query_of(in_spec, out_spec, get_core_num(), spec);
  const Tiling t = pre_decide_tiling(q);
  const int len = pre_fill_tiling_struct(q, t, in_spec[0].dtype, buffer);
  if (getenv("PPL_TILING_DEBUG")) {
    printf("[tiling-dyn] buffer=%d len=%d P=%d.%d.%d TILE_M=%d "
           "TILE_N=%d weight_k_span=%d act_k_span=%d dtype=%d\n",
           buffer == nullptr ? 0 : 1, len, t.p_m, t.p_n, t.p_k, t.tile_m,
           t.tile_n, t.weight_k_span, t.act_k_span, in_spec[0].dtype);
    fflush(stdout);
  }
  return len;
}

void api_w8a8_block_matmul_global(void *param, size_t param_size, void *input,
                                  void *output) {
  if (use_prefill_family((tensor_spec_t *)input))
    prefill_global_impl(param, param_size, input, output);
  else
    decode_global_impl(param, param_size, input, output);
}

// Split decided at compile time and baked into the fw payload; the firmware
// ctrl only fills addresses and shapes, picking the entry by the same M rule.
int api_dyn_w8a8_block_matmul_global(void *param, void *input, void *output,
                                     void *buffer) {
  if (use_prefill_family((tensor_spec_t *)input))
    return prefill_dyn_impl(param, input, output, buffer);
  return decode_dyn_impl(param, input, output, buffer);
}

#ifdef __cplusplus
}
#endif
