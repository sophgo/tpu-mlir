//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Support/Module.h"
#include <list>
#include <map>
#include <set>
#include <vector>

// using namespace tpu_mlir::helper;
namespace tpu_mlir {
namespace tpu {

typedef struct {
  int64_t nstep;
  int64_t hstep;
} tensor_step_t;

typedef enum ld_st_type {
  TIMESTEP_LOAD = 0,    // load from gmem to lmem
  TIMESTEP_STORE = 1,   // store from lmem to gmem
  TIMESTEP_MOVE = 2,    // move between global mem
  TIMESTEP_LD_G2L2 = 3, // load from gmem to l2mem
  TIMESTEP_LDST_UNKNOWN
} TIMESTEP_LD_ST;

inline bool is_timestep_load(TIMESTEP_LD_ST type) {
  return (type == TIMESTEP_LOAD || type == TIMESTEP_LD_G2L2);
}

typedef enum {
  LMEM_WEIGHT,
  LMEM_ACTIVATION,
  LMEM_OPERATION,
  LMEM_ANY,
} lmem_type_t;

typedef std::pair<int64_t, int64_t> slice_pair_t; // idx and slice
struct slice_info_t {
  std::vector<slice_pair_t> h; // h_idx and h_slice
  std::vector<slice_pair_t> n; // n_idx and n_slice
};

typedef struct mem_buffer_key {
  lmem_type_t type;
  Value value;
  Operation *op;
  int64_t conflict;
  bool operator<(const mem_buffer_key &other) const {
    if (type < other.type) {
      return true;
    } else if (type == other.type) {
      if (type == LMEM_OPERATION) {
        return op < other.op;
      } else {
        return value.getImpl() < other.value.getImpl();
      }
    }
    return false;
  }
} mem_buffer_key_t;

typedef struct mem_buffer_value {
  int64_t start_ts;
  int64_t end_ts;
  int64_t addr;
  int64_t size;
  int64_t align_bytes;
} mem_buffer_value_t;

struct value_compare {
  bool operator()(Value v0, Value v1) const {
    if (v0.getImpl() < v1.getImpl()) {
      return true;
    }
    return false;
  }
};

struct tensor_info_t {
  TIMESTEP_LD_ST mode;
  slice_info_t slice_info;
  int64_t stage;
  int64_t use_3ic_opt;
  bool eu_align;
  bool need_bcast;
  // init
  tensor_info_t()
      : mode(TIMESTEP_LOAD), stage(0), use_3ic_opt(0), eu_align(false),
        need_bcast(false) {}
  tensor_info_t(TIMESTEP_LD_ST mode)
      : mode(mode), stage(0), use_3ic_opt(0), eu_align(false),
        need_bcast(false) {}
  tensor_info_t(slice_info_t slice_info)
      : slice_info(slice_info), mode(mode), stage(0), use_3ic_opt(0),
        eu_align(false), need_bcast(false) {}
};

using ValueSet = std::set<Value, value_compare>;
using ValueIntMap = std::map<Value, int64_t, value_compare>;
using TensorInfo = std::map<Value, tensor_info_t, value_compare>;
using MemBuff = std::map<mem_buffer_key_t, mem_buffer_value_t>;
using MemBuffElt = std::pair<mem_buffer_key_t, mem_buffer_value_t>;
using TpuTsField = std::vector<Operation *>;
using GdmaElt = std::pair<Value, tensor_info_t>;
using GdmaTsField = std::vector<GdmaElt>;
using MemBlock = std::pair<int64_t, int64_t>; // <addr, size>

typedef struct {
  TpuTsField tpu0_ts_field;
  GdmaTsField gdma0_ts_field;
} TimestepRow;

typedef struct {
  int64_t nsecs;
  int64_t hsecs;
} shape_secs_t;

struct LgInfo {
  LgInfo() { this->clear(); }
  ~LgInfo() { this->clear(); }
  void clear() {
    this->group_ops.clear();
    this->group_ins.clear();
    this->group_outs.clear();
    this->type = GROUP_NORMAL;
  }
  void update_group_io() {
    this->group_ins.clear();
    this->group_outs.clear();

    for (auto op : group_ops) {
      // update group_ins
      for (auto in : op->getOperands()) {
        auto src_op = in.getDefiningOp();
        if ((src_op == nullptr ||
             (!isa<top::WeightOp, top::NoneOp>(src_op) &&
              (std::find(group_ops.begin(), group_ops.end(), src_op) ==
               group_ops.end()))) &&
            std::find(group_ins.begin(), group_ins.end(), in) ==
                group_ins.end()) {
          group_ins.push_back(in);
        }
      }
      // update group_outs
      for (auto out : op->getResults()) {
        if (module::isNone(out)) {
          continue;
        }
        for (auto dst_op : out.getUsers()) {
          if (std::find(group_ops.begin(), group_ops.end(), dst_op) ==
                  group_ops.end() &&
              std::find(group_outs.begin(), group_outs.end(), out) ==
                  group_outs.end()) {
            group_outs.push_back(out);
            break;
          }
        }
      }
    }
  }

  // group layers
  std::vector<Operation *> group_ops;
  // in tensors
  std::vector<Value> group_ins;
  // out tensors
  std::vector<Value> group_outs;

  shape_secs_t shape_secs;
  // layer group type
  group_type_t type;
};

} // namespace tpu
} // namespace tpu_mlir
