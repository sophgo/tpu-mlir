//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

// Shared affine helpers used by BM1684X Load/Store codegen.

#ifndef TPU_MLIR_AFFINE_UTILS_H
#define TPU_MLIR_AFFINE_UTILS_H

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace mlir;

struct FreeTensorDmaInfo {
  int64_t gOffset;
  int64_t lOffset;
  int64_t entry_N;
  int64_t entry_C;
  int64_t entry_H;
  int64_t entry_W;
  int64_t g_stride_n; // global
  int64_t g_stride_c;
  int64_t g_stride_h;

  int64_t l_nidx;
  int64_t l_cidx;
  // Note: is_pad removed - padding blocks are filtered out before this stage
};

// Cache key for offset-based lookup
struct AffineOffsetKey {
  int64_t n_idx;
  int64_t c_idx;
  int64_t h_idx;
  int64_t w_idx;

  bool operator==(const AffineOffsetKey &other) const {
    return n_idx == other.n_idx && c_idx == other.c_idx &&
           h_idx == other.h_idx && w_idx == other.w_idx;
  }
};

struct AffineOffsetKeyHash {
  size_t operator()(const AffineOffsetKey &k) const {
    auto h = std::hash<int64_t>{}(k.n_idx);
    h ^= std::hash<int64_t>{}(k.c_idx) + 0x9e3779b97f4a7c15ULL + (h << 6) +
         (h >> 2);
    h ^= std::hash<int64_t>{}(k.h_idx) + 0x9e3779b97f4a7c15ULL + (h << 6) +
         (h >> 2);
    h ^= std::hash<int64_t>{}(k.w_idx) + 0x9e3779b97f4a7c15ULL + (h << 6) +
         (h >> 2);
    return h;
  }
};

using AffineDmaCache =
    std::unordered_map<AffineOffsetKey, std::vector<FreeTensorDmaInfo>,
                       AffineOffsetKeyHash>;
using AffineSharedCache = std::unordered_map<std::string, AffineDmaCache>;

constexpr size_t kAffineFullCacheLimit = 16384;
constexpr size_t kAffineSharedCacheLimit = 4096;

static inline void append_i64_key(std::string &key, int64_t v) {
  key.append(std::to_string(v));
  key.push_back(';');
}

static inline std::string build_affine_shared_cache_key(
    AffineMap indexingMap, const std::vector<int64_t> &iShape,
    const std::vector<int64_t> &oShape, int64_t n_slice, int64_t c_slice,
    int64_t h_slice, int64_t w_slice) {
  std::string key;
  key.reserve(256);
  key.append("map=");
  {
    std::string map_str;
    llvm::raw_string_ostream os(map_str);
    indexingMap.print(os);
    os.flush();
    key.append(map_str);
  }
  key.append("|i=");
  for (auto v : iShape) {
    append_i64_key(key, v);
  }
  key.append("|o=");
  for (auto v : oShape) {
    append_i64_key(key, v);
  }
  key.append("|slice=");
  append_i64_key(key, n_slice);
  append_i64_key(key, c_slice);
  append_i64_key(key, h_slice);
  append_i64_key(key, w_slice);
  return key;
}

enum AffineFuseDim {
  DIM_N,
  DIM_C,
  DIM_H,
};

static inline std::vector<FreeTensorDmaInfo>
fuse_dma_info_with_stride(std::vector<FreeTensorDmaInfo> &dma_infos,
                          const AffineFuseDim merge_dim,
                          const std::vector<int64_t> &lStride_ext) {

  std::vector<std::pair<bool, int64_t>> can_fuse_with_r_neighbor(
      dma_infos.size(), std::make_pair(false, -1));

  auto can_fuse = [&](FreeTensorDmaInfo &dma_info1,
                      FreeTensorDmaInfo &dma_info2) {
    if (dma_info1.entry_N != dma_info2.entry_N ||
        dma_info1.entry_C != dma_info2.entry_C ||
        dma_info1.entry_H != dma_info2.entry_H ||
        dma_info1.entry_W != dma_info2.entry_W) {
      return false;
    }

    int length = dma_info1.entry_N * dma_info1.entry_C * dma_info1.entry_H *
                 dma_info1.entry_W;

    if (merge_dim > DIM_N) {
      if (dma_info1.l_nidx != dma_info2.l_nidx || length >= lStride_ext[0])
        return false;
    }
    if (merge_dim > DIM_C) {
      if (dma_info1.l_cidx != dma_info2.l_cidx || length >= lStride_ext[1])
        return false;
    }

    if (merge_dim == DIM_N) {
      if (length != lStride_ext[0]) {
        return false;
      }
    }
    if (merge_dim == DIM_C && length != lStride_ext[1]) {
      return false;
    }
    if (dma_info2.gOffset <= dma_info1.gOffset) {
      return false;
    }
    return true;
  };

  for (int i = 0; i < (int)dma_infos.size() - 1; ++i) {
    if (can_fuse(dma_infos[i], dma_infos[i + 1])) {
      can_fuse_with_r_neighbor[i] =
          std::make_pair(true, dma_infos[i + 1].gOffset - dma_infos[i].gOffset);
    }
  }

  std::vector<FreeTensorDmaInfo> fused_dma_infos;
  for (int i = 0; i < (int)dma_infos.size();) {
    if (i == (int)dma_infos.size() - 1 || !can_fuse_with_r_neighbor[i].first) {
      fused_dma_infos.push_back(dma_infos[i]);
      i++;
      continue;
    }

    int64_t expected_stride = can_fuse_with_r_neighbor[i].second;
    int j = i + 1;
    while (j < (int)dma_infos.size() && can_fuse_with_r_neighbor[j].first &&
           can_fuse_with_r_neighbor[j].second == expected_stride) {
      j++;
    }

    int fuse_num = j - i + 1;
    FreeTensorDmaInfo fused_dma_info = dma_infos[i];
    switch (merge_dim) {
    case DIM_H:
      assert(fused_dma_info.entry_H == 1 && fused_dma_info.entry_C == 1 &&
             fused_dma_info.entry_N == 1);
      fused_dma_info.entry_H = fuse_num;
      fused_dma_info.g_stride_h = fused_dma_info.g_stride_c =
          fused_dma_info.g_stride_n = expected_stride;
      break;
    case DIM_C:
      assert(fused_dma_info.entry_C == 1 && fused_dma_info.entry_N == 1);
      fused_dma_info.entry_C = fuse_num;
      fused_dma_info.g_stride_c = fused_dma_info.g_stride_n = expected_stride;
      break;
    case DIM_N:
      assert(fused_dma_info.entry_N == 1);
      fused_dma_info.entry_N = fuse_num;
      fused_dma_info.g_stride_n = expected_stride;
      break;
    default:
      llvm_unreachable("not supported");
    }

    fused_dma_infos.push_back(fused_dma_info);
    i = j + 1;
  }
  return fused_dma_infos;
}

static inline bool
affine_cache_lookup(AffineSharedCache &shared_cache,
                    const std::string &shared_cache_key,
                    const AffineOffsetKey &offset_key,
                    std::vector<FreeTensorDmaInfo> &cached_infos) {
  auto shared_it = shared_cache.find(shared_cache_key);
  if (shared_it == shared_cache.end()) {
    return false;
  }
  auto full_it = shared_it->second.find(offset_key);
  if (full_it == shared_it->second.end()) {
    return false;
  }
  cached_infos = full_it->second;
  return true;
}

static inline void
affine_cache_store(AffineSharedCache &shared_cache,
                   const std::string &shared_cache_key,
                   const AffineOffsetKey &offset_key,
                   const std::vector<FreeTensorDmaInfo> &infos) {
  auto shared_it = shared_cache.find(shared_cache_key);
  if (shared_it == shared_cache.end()) {
    if (shared_cache.size() > kAffineSharedCacheLimit) {
      shared_cache.clear();
    }
    shared_it = shared_cache.emplace(shared_cache_key, AffineDmaCache{}).first;
  }
  auto &full_cache = shared_it->second;
  if (full_cache.size() > kAffineFullCacheLimit) {
    full_cache.clear();
  }
  full_cache[offset_key] = infos;
}

static inline int64_t
evaluateAffineExpr(AffineExpr expr, const std::vector<int64_t> &dimValues) {
  if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
    return constExpr.getValue();
  }

  if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
    int64_t dim = dimExpr.getPosition();
    assert(dim < dimValues.size());
    return dimValues[dim];
  }

  if (auto binaryExpr = expr.dyn_cast<AffineBinaryOpExpr>()) {
    int64_t lhs = evaluateAffineExpr(binaryExpr.getLHS(), dimValues);
    int64_t rhs = evaluateAffineExpr(binaryExpr.getRHS(), dimValues);

    switch (binaryExpr.getKind()) {
    case AffineExprKind::Add:
      return lhs + rhs;
    case AffineExprKind::Mul:
      return lhs * rhs;
    case AffineExprKind::FloorDiv:
      assert(rhs != 0 && "FloorDiv by zero in affine expr evaluation");
      return lhs / rhs;
    case AffineExprKind::CeilDiv:
      assert(rhs != 0 && "CeilDiv by zero in affine expr evaluation");
      return (lhs + rhs - 1) / rhs;
    case AffineExprKind::Mod:
      assert(rhs != 0 && "Mod by zero in affine expr evaluation");
      return lhs % rhs;
    default:
      return 0;
    }
  }

  return 0;
}

// ============ Compiled Affine Evaluator ============
// Instead of recursively walking AffineExpr trees for every element,
// compile each expression into a flat instruction sequence (bytecode)
// that can be evaluated with a simple loop over an array.

enum class AffineOp : uint8_t {
  LoadDim,    // push dimValues[arg]
  LoadConst,  // push constant value
  Add,
  Mul,
  FloorDiv,
  Mod,
  CeilDiv,
};

struct AffineInstr {
  AffineOp op;
  int64_t arg; // dim index or constant value
};

// Compiled evaluator for one AffineMap result expression
struct CompiledAffineExpr {
  std::vector<AffineInstr> instrs;
};

// Compiled evaluator for an entire AffineMap
struct CompiledAffineMap {
  std::vector<CompiledAffineExpr> results;
  int64_t num_inputs;
};

static inline void compileExpr(AffineExpr expr,
                               std::vector<AffineInstr> &instrs) {
  if (auto c = expr.dyn_cast<AffineConstantExpr>()) {
    instrs.push_back({AffineOp::LoadConst, c.getValue()});
    return;
  }
  if (auto d = expr.dyn_cast<AffineDimExpr>()) {
    instrs.push_back({AffineOp::LoadDim, (int64_t)d.getPosition()});
    return;
  }
  if (auto bin = expr.dyn_cast<AffineBinaryOpExpr>()) {
    compileExpr(bin.getLHS(), instrs);
    compileExpr(bin.getRHS(), instrs);
    switch (bin.getKind()) {
    case AffineExprKind::Add:
      instrs.push_back({AffineOp::Add, 0});
      break;
    case AffineExprKind::Mul:
      instrs.push_back({AffineOp::Mul, 0});
      break;
    case AffineExprKind::FloorDiv:
      instrs.push_back({AffineOp::FloorDiv, 0});
      break;
    case AffineExprKind::CeilDiv:
      instrs.push_back({AffineOp::CeilDiv, 0});
      break;
    case AffineExprKind::Mod:
      instrs.push_back({AffineOp::Mod, 0});
      break;
    default:
      instrs.push_back({AffineOp::LoadConst, 0});
      break;
    }
  }
}

static inline CompiledAffineMap compileAffineMap(AffineMap map) {
  CompiledAffineMap compiled;
  compiled.num_inputs = map.getNumInputs();
  compiled.results.resize(map.getNumResults());
  for (unsigned i = 0; i < map.getNumResults(); ++i) {
    compileExpr(map.getResult(i), compiled.results[i].instrs);
  }
  return compiled;
}

static inline int64_t
evalCompiled(const CompiledAffineExpr &compiled,
             const std::vector<int64_t> &dimValues) {
  // Use a small fixed-size stack to avoid heap allocations
  int64_t stack[32];
  int sp = 0;
  for (const auto &instr : compiled.instrs) {
    switch (instr.op) {
    case AffineOp::LoadDim:
      stack[sp++] = dimValues[instr.arg];
      break;
    case AffineOp::LoadConst:
      stack[sp++] = instr.arg;
      break;
    case AffineOp::Add: {
      int64_t b = stack[--sp];
      stack[sp - 1] += b;
      break;
    }
    case AffineOp::Mul: {
      int64_t b = stack[--sp];
      stack[sp - 1] *= b;
      break;
    }
    case AffineOp::FloorDiv: {
      int64_t b = stack[--sp];
      stack[sp - 1] /= b;
      break;
    }
    case AffineOp::CeilDiv: {
      int64_t b = stack[--sp];
      int64_t a = stack[sp - 1];
      stack[sp - 1] = (a + b - 1) / b;
      break;
    }
    case AffineOp::Mod: {
      int64_t b = stack[--sp];
      stack[sp - 1] %= b;
      break;
    }
    }
  }
  return stack[0];
}

static inline std::vector<int64_t>
evalCompiledMap(const CompiledAffineMap &compiled,
                const std::vector<int64_t> &dimValues) {
  std::vector<int64_t> results(compiled.results.size());
  for (size_t i = 0; i < compiled.results.size(); ++i) {
    results[i] = evalCompiled(compiled.results[i], dimValues);
  }
  return results;
}

// Fast inline version that writes to pre-allocated output
static inline void evalCompiledMapInto(const CompiledAffineMap &compiled,
                                       const std::vector<int64_t> &dimValues,
                                       std::vector<int64_t> &outCoords) {
  for (size_t i = 0; i < compiled.results.size(); ++i) {
    outCoords[i] = evalCompiled(compiled.results[i], dimValues);
  }
}

// Evaluate an AffineMap using recursive tree walking (used by estimation path).
static inline std::vector<int64_t>
do_affine_map(const std::vector<int64_t> &inputCoords, AffineMap map) {
  assert(inputCoords.size() == map.getNumInputs());
  std::vector<int64_t> outputCoords;
  outputCoords.reserve(map.getNumResults());

  for (unsigned i = 0; i < map.getNumResults(); ++i) {
    AffineExpr expr = map.getResult(i);
    int64_t value = evaluateAffineExpr(expr, inputCoords);
    outputCoords.push_back(value);
  }

  return outputCoords;
}

// Compute compact (row-major) strides from a shape vector.
static inline std::vector<int64_t>
getCompactStrideFromShape(const std::vector<int64_t> &shape) {
  std::vector<int64_t> stride(shape.size());
  assert(!shape.empty());
  stride.back() = 1;
  for (int i = shape.size() - 2; i >= 0; --i) {
    stride[i] = stride[i + 1] * shape[i + 1];
  }
  return stride;
}

// Decompose a linear offset into multi-dimensional coordinates.
static inline std::vector<int64_t>
offset_2_coords(int64_t offset, const std::vector<int64_t> &stride) {
  std::vector<int64_t> coords(stride.size());
  for (size_t i = 0; i < stride.size(); ++i) {
    coords[i] = offset / stride[i];
    offset = offset % stride[i];
  }
  return coords;
}

// Linearize multi-dimensional coordinates into a flat offset.
static inline int64_t coords_2_offset(const std::vector<int64_t> &coords,
                                      const std::vector<int64_t> &stride) {
  assert(coords.size() == stride.size());
  int64_t offset = 0;
  for (size_t i = 0; i < coords.size(); ++i) {
    offset += coords[i] * stride[i];
  }
  return offset;
}

// Check if coordinates are within valid range [0, shape[i]).
static inline bool is_coords_valid(const std::vector<int64_t> &coords,
                                   const std::vector<int64_t> &shape) {
  assert(coords.size() == shape.size());
  for (size_t i = 0; i < coords.size(); ++i) {
    if (coords[i] < 0 || coords[i] >= shape[i]) {
      return false;
    }
  }
  return true;
}

// ============ Safe delta cache with linearity analysis ============
// Only allow delta reuse when ALL changed input dimensions are provably
// linear (no floordiv/mod) across ALL result expressions of the affine map.

// Check if an AffineExpr is linear in a given dimension (no floordiv/mod).
static inline bool isLinearInDim(AffineExpr expr, unsigned dimIdx) {
  if (auto d = expr.dyn_cast<AffineDimExpr>()) {
    return true; // A bare dimension reference is always linear
  }
  if (expr.dyn_cast<AffineConstantExpr>()) {
    return true; // Constants don't involve any dimension
  }
  if (auto bin = expr.dyn_cast<AffineBinaryOpExpr>()) {
    // For floordiv/mod: if the LHS involves the target dimension, NOT linear
    if (bin.getKind() == AffineExprKind::FloorDiv ||
        bin.getKind() == AffineExprKind::CeilDiv ||
        bin.getKind() == AffineExprKind::Mod) {
      // Check if LHS involves our target dimension
      std::function<bool(AffineExpr)> involvesDim = [&](AffineExpr e) -> bool {
        if (auto d = e.dyn_cast<AffineDimExpr>())
          return d.getPosition() == dimIdx;
        if (auto b = e.dyn_cast<AffineBinaryOpExpr>())
          return involvesDim(b.getLHS()) || involvesDim(b.getRHS());
        return false;
      };
      if (involvesDim(bin.getLHS())) {
        return false; // floordiv/mod on an expression involving our dim
      }
      // RHS typically doesn't matter (usually a constant), but check anyway
      return isLinearInDim(bin.getRHS(), dimIdx);
    }
    // Add and Mul are linear-safe: recurse into both sides
    return isLinearInDim(bin.getLHS(), dimIdx) &&
           isLinearInDim(bin.getRHS(), dimIdx);
  }
  return true;
}

// Check if the entire AffineMap is linear in a given input dimension
// across ALL result expressions.
static inline bool isMapLinearInDim(AffineMap map, unsigned dimIdx) {
  for (unsigned i = 0; i < map.getNumResults(); ++i) {
    if (!isLinearInDim(map.getResult(i), dimIdx)) {
      return false;
    }
  }
  return true;
}

// Precompute which input dimensions are safe for delta reuse
static inline std::vector<bool> computeSafeDeltaDims(AffineMap map) {
  std::vector<bool> safe(map.getNumInputs(), true);
  for (unsigned d = 0; d < map.getNumInputs(); ++d) {
    safe[d] = isMapLinearInDim(map, d);
  }
  return safe;
}

struct AffineAnchorInfo {
  AffineOffsetKey anchor_key;
  std::vector<FreeTensorDmaInfo> anchor_infos;
};

using AffineAnchorCache =
    std::unordered_map<std::string, AffineAnchorInfo>;

static inline bool
affine_cache_lookup_with_delta(AffineSharedCache &shared_cache,
                               AffineAnchorCache &anchor_cache,
                               const std::string &shared_cache_key,
                               const AffineOffsetKey &offset_key,
                               const std::vector<int64_t> &iStride,
                               const std::vector<int64_t> &oStride,
                               const std::vector<int64_t> &iShape,
                               const std::vector<int64_t> &oShape,
                               const CompiledAffineMap &compiled_map,
                               const std::vector<bool> &safeDeltaDims,
                               bool is_load, bool same_o_shape,
                               std::vector<FreeTensorDmaInfo> &cached_infos) {
  // Level 1: exact match
  if (affine_cache_lookup(shared_cache, shared_cache_key, offset_key,
                          cached_infos)) {
    return true;
  }

  // Level 2: anchor-based delta reuse with linearity safety check
  // Only safe when map_num_inputs == 4 && same_o_shape, ensuring
  // offset_key coords map directly to affine map input dims.
  int64_t map_num_inputs = compiled_map.num_inputs;
  if (map_num_inputs != 4 || !same_o_shape) {
    return false;
  }

  auto anchor_it = anchor_cache.find(shared_cache_key);
  if (anchor_it == anchor_cache.end()) {
    return false;
  }

  const auto &anchor = anchor_it->second;
  if (anchor.anchor_infos.empty()) {
    return false;
  }

  // Direct 4D coord construction (guaranteed by map_num_inputs==4 &&
  // same_o_shape check above)
  auto build_first_coords = [](const AffineOffsetKey &key) {
    std::vector<int64_t> coords(4);
    coords[0] = key.n_idx;
    coords[1] = key.c_idx;
    coords[2] = key.h_idx;
    coords[3] = key.w_idx;
    return coords;
  };

  auto anchor_coords = build_first_coords(anchor.anchor_key);
  auto current_coords = build_first_coords(offset_key);

  // Safety check: only allow delta if ALL changed dimensions are linear
  for (int64_t d = 0; d < map_num_inputs; ++d) {
    if (anchor_coords[d] != current_coords[d]) {
      if (d >= (int64_t)safeDeltaDims.size() || !safeDeltaDims[d]) {
        return false; // Changed dim has floordiv/mod — unsafe
      }
    }
  }

  // All changed dims are linear → delta is guaranteed uniform
  const auto &gStride = is_load ? iStride : oStride;
  const auto &gShape = is_load ? iShape : oShape;

  auto anchor_mapped = evalCompiledMap(compiled_map, anchor_coords);
  auto current_mapped = evalCompiledMap(compiled_map, current_coords);

  if (!is_coords_valid(anchor_mapped, gShape) ||
      !is_coords_valid(current_mapped, gShape)) {
    return false;
  }

  int64_t anchor_gOff = coords_2_offset(anchor_mapped, gStride);
  int64_t current_gOff = coords_2_offset(current_mapped, gStride);
  int64_t delta = current_gOff - anchor_gOff;

  // Apply delta to all cached entries
  cached_infos = anchor.anchor_infos;
  for (auto &info : cached_infos) {
    info.gOffset += delta;
  }

  affine_cache_store(shared_cache, shared_cache_key, offset_key, cached_infos);
  return true;
}

static inline void
affine_anchor_store(AffineAnchorCache &anchor_cache,
                    const std::string &shared_cache_key,
                    const AffineOffsetKey &offset_key,
                    const std::vector<FreeTensorDmaInfo> &infos) {
  if (anchor_cache.find(shared_cache_key) != anchor_cache.end()) {
    return;
  }
  AffineAnchorInfo anchor;
  anchor.anchor_key = offset_key;
  anchor.anchor_infos = infos;
  anchor_cache[shared_cache_key] = std::move(anchor);
}


// ============ Analytical contiguous block detection ============
struct AffineBlockAnalysis {
  bool can_predict = false;
  int64_t max_contig = 0;
  int fast_result_idx = -1;
};

static inline AffineBlockAnalysis
analyzeContiguousBlocks(AffineMap indexingMap, int scan_dim_idx,
                        const std::vector<int64_t> &targetStride) {
  AffineBlockAnalysis result;
  int numResults = indexingMap.getNumResults();
  if (numResults == 0 || targetStride.empty())
    return result;

  int last_idx = numResults - 1;
  AffineExpr lastExpr = indexingMap.getResult(last_idx);

  std::function<bool(AffineExpr, int &, int64_t &)> matchLinear;
  matchLinear = [&](AffineExpr expr, int &dim, int64_t &offset) -> bool {
    if (auto d = expr.dyn_cast<AffineDimExpr>()) {
      dim = d.getPosition();
      offset = 0;
      return true;
    }
    if (auto bin = expr.dyn_cast<AffineBinaryOpExpr>()) {
      if (bin.getKind() == AffineExprKind::Add) {
        int64_t cval = 0;
        int tmpDim = -1;
        int64_t tmpOff = 0;
        if (auto c = bin.getRHS().dyn_cast<AffineConstantExpr>()) {
          cval = c.getValue();
          if (matchLinear(bin.getLHS(), tmpDim, tmpOff)) {
            dim = tmpDim;
            offset = tmpOff + cval;
            return true;
          }
        }
        if (auto c = bin.getLHS().dyn_cast<AffineConstantExpr>()) {
          cval = c.getValue();
          if (matchLinear(bin.getRHS(), tmpDim, tmpOff)) {
            dim = tmpDim;
            offset = tmpOff + cval;
            return true;
          }
        }
      }
    }
    return false;
  };

  int dim = -1;
  int64_t offset = 0;
  if (!matchLinear(lastExpr, dim, offset) || dim != scan_dim_idx) {
    return result;
  }

  int64_t min_break = INT64_MAX;
  std::function<void(AffineExpr, int)> findBreaks;
  findBreaks = [&](AffineExpr expr, int target_dim) {
    if (auto bin = expr.dyn_cast<AffineBinaryOpExpr>()) {
      if (bin.getKind() == AffineExprKind::FloorDiv ||
          bin.getKind() == AffineExprKind::Mod) {
        std::function<bool(AffineExpr)> involvesDim;
        involvesDim = [&](AffineExpr e) -> bool {
          if (auto d = e.dyn_cast<AffineDimExpr>())
            return (int)d.getPosition() == target_dim;
          if (auto b = e.dyn_cast<AffineBinaryOpExpr>())
            return involvesDim(b.getLHS()) || involvesDim(b.getRHS());
          return false;
        };
        if (involvesDim(bin.getLHS())) {
          if (auto c = bin.getRHS().dyn_cast<AffineConstantExpr>()) {
            min_break = std::min(min_break, c.getValue());
          }
        }
      }
      findBreaks(bin.getLHS(), target_dim);
      findBreaks(bin.getRHS(), target_dim);
    }
  };

  for (int i = 0; i < numResults; ++i) {
    if (i == last_idx)
      continue;
    findBreaks(indexingMap.getResult(i), scan_dim_idx);
  }

  if (min_break == INT64_MAX) {
    result.can_predict = true;
    result.max_contig = INT64_MAX;
    result.fast_result_idx = last_idx;
  } else if (min_break > 1) {
    result.can_predict = true;
    result.max_contig = min_break;
    result.fast_result_idx = last_idx;
  }

  return result;
}

// Find contiguous block length starting at scan_pos using the optimal search
// strategy based on block analysis. get_mapped_gOffset is a callable
// (int64_t pos, int64_t &gOff) -> bool that maps a scan position to a global
// offset.
template <typename OffsetFunc>
static inline int64_t
findContiguousLength(const AffineBlockAnalysis &block_analysis,
                     int64_t scan_pos, int64_t scan_size, int64_t gOffset,
                     OffsetFunc &&get_mapped_gOffset) {
  int64_t entry_W = 1;

  if (block_analysis.can_predict && block_analysis.max_contig > 1) {
    // Fast path: use block analysis to predict contiguous size
    int64_t pos_in_block = scan_pos % block_analysis.max_contig;
    int64_t remain = block_analysis.max_contig - pos_in_block;
    int64_t max_entry = std::min(remain, scan_size - scan_pos);
    if (max_entry > 1) {
      int64_t end_gOffset = 0;
      if (get_mapped_gOffset(scan_pos + max_entry - 1, end_gOffset) &&
          end_gOffset == gOffset + max_entry - 1) {
        entry_W = max_entry;
      } else {
        // Binary search for contiguous length within predicted block
        int64_t lo = 1, hi = max_entry;
        while (lo < hi) {
          int64_t mid = (lo + hi + 1) / 2;
          int64_t mid_gOffset = 0;
          if (get_mapped_gOffset(scan_pos + mid - 1, mid_gOffset) &&
              mid_gOffset == gOffset + mid - 1) {
            lo = mid;
          } else {
            hi = mid - 1;
          }
        }
        entry_W = lo;
      }
    }
  } else {
    // Slow path: exponential doubling + binary search
    int64_t max_remaining = scan_size - scan_pos;
    if (max_remaining > 1) {
      int64_t good = 1, probe = 2;
      while (probe < max_remaining) {
        int64_t probe_gOffset = 0;
        if (get_mapped_gOffset(scan_pos + probe - 1, probe_gOffset) &&
            probe_gOffset == gOffset + probe - 1) {
          good = probe;
          probe *= 2;
        } else {
          break;
        }
      }
      int64_t hi = std::min(probe, max_remaining);
      int64_t lo = good;
      while (lo < hi) {
        int64_t mid = (lo + hi + 1) / 2;
        int64_t mid_gOffset = 0;
        if (get_mapped_gOffset(scan_pos + mid - 1, mid_gOffset) &&
            mid_gOffset == gOffset + mid - 1) {
          lo = mid;
        } else {
          hi = mid - 1;
        }
      }
      entry_W = lo;
    }
  }

  return entry_W;
}

// ============ Row pattern cache for intra-codegen reuse ============
// Caches the contiguous block pattern for a given (c_local, h_local_base)
// so rows with different n_local can reuse the pattern with a gOffset delta.

struct RowBlockEntry {
  int64_t scan_pos;
  int64_t entry_W;
  int64_t gOffset;
  int64_t lOffset;
  int64_t l_nidx;
  int64_t l_cidx;
};

struct RowPatternKey {
  int64_t c_local;
  int64_t h_local_base;

  bool operator==(const RowPatternKey &other) const {
    return c_local == other.c_local && h_local_base == other.h_local_base;
  }
};

struct RowPatternKeyHash {
  size_t operator()(const RowPatternKey &k) const {
    auto h = std::hash<int64_t>{}(k.c_local);
    h ^= std::hash<int64_t>{}(k.h_local_base) + 0x9e3779b97f4a7c15ULL +
         (h << 6) + (h >> 2);
    return h;
  }
};

struct RowPattern {
  int64_t ref_n_local;          // n_local of the row that created this pattern
  int64_t ref_first_gOffset;    // gOffset of the first block in reference row
  std::vector<RowBlockEntry> blocks;
};

using RowPatternCache =
    std::unordered_map<RowPatternKey, RowPattern, RowPatternKeyHash>;

#endif // TPU_MLIR_AFFINE_UTILS_H
