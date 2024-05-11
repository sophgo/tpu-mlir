//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LgPass.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "llvm/Support/FormatVariadic.h"

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/IlpTimeStep.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/TimeStepMethod.h"

using namespace tpu_mlir::backend;

namespace tpu_mlir {
namespace tpu {

bool isLgSupport(Operation *op) {
  bool res = false;
  if (isa<top::WeightOp, top::InputOp>(op)) {
    res = true;
  }
  if (auto lg_op = dyn_cast<tpu_mlir::LocalGenInterface>(op)) {
    res = mlir::succeeded(lg_op.LocalGenSupport());
  }
  return res;
}

void find_all_pre_ops(Operation * op, std::vector<Operation*>& glayer_pre_ops) {
  glayer_pre_ops.push_back(op);
  int i = 0;
  for (auto v : op->getOperands()) {
    auto pre_op = v.getDefiningOp();
    if (pre_op == nullptr || isa<top::NoneOp, top::WeightOp, top::InputOp>(pre_op)) {
      continue;
    }
    if (i > 0 && !isa<tpu::AddOp, tpu::ConcatOp>(pre_op)) {
      continue;
    }

    if (std::find(glayer_pre_ops.begin(), glayer_pre_ops.end(), pre_op) == glayer_pre_ops.end()){
      find_all_pre_ops(pre_op, glayer_pre_ops);
    }
    i++;
  }
}

void find_all_next_ops(Operation * op, std::vector<Operation*>& glayer_next_ops) {
  glayer_next_ops.push_back(op);
  for (auto user : op->getUsers()) {
    if (isa<ReturnOp>(user)) {
      continue;
    }

    if (std::find(glayer_next_ops.begin(), glayer_next_ops.end(), user) == glayer_next_ops.end()){
      find_all_next_ops(user, glayer_next_ops);
    }
  }
}

std::string replaceChars_for_dot(std::string str) {
  if (std::isdigit(str[0])) {
    str = "s_" + str;
  }
  std::string toReplace = "./:";
  for (size_t i = 0; i < toReplace.size(); ++i) {
      std::replace(str.begin(), str.end(), toReplace[i], '_');
  }
  return str;
}

bool isPreOpHaveAComputeOp(Operation * op) {
  if (!isa<tpu::ReshapeOp, tpu::PermuteOp>(op)) {
    llvm::errs()<<"meet compute op:"<<module::getName(op).str()<<"\n";
    return true;
  }
  llvm::errs()<<"meet not_compute op:"<<module::getName(op).str()<<"\n";
  bool have_compute = false;
  for (auto v : op->getOperands()) {
    auto pre_op = v.getDefiningOp();
    if (pre_op == nullptr || isa<top::NoneOp, top::WeightOp, top::InputOp>(pre_op)) {
      continue;
    }
    if (isPreOpHaveAComputeOp(pre_op)) {
      have_compute = true;
      break;
    }
  }
  return have_compute;
}

shape_secs_t get_group_max_secs(const LgInfo &lg_info, std::vector<std::pair<Operation*, int>>& vec_op_hsecs) {
  int64_t n, c, d, h, w;
  module::getNCDHW(lg_info.group_ops[0]->getOperand(0), n, c, d, h, w,
                   lg_info.type);
  int64_t max_nsecs = n;
  if (isa<tpu::AddOp, tpu::SubOp, tpu::MulOp, tpu::DivOp, tpu::MaxOp,
          tpu::MinOp, tpu::MatMulOp>(lg_info.group_ops[0])) {
    module::getNCDHW(lg_info.group_ops[0]->getOperand(1), n, c, d, h, w,
                     lg_info.type);
    if (isa<tpu::MatMulOp>(lg_info.group_ops[0]) && n != max_nsecs) {
      max_nsecs = 1;
    } else {
      max_nsecs = std::max(n, max_nsecs);
    }
  }
  int64_t max_csecs = llvm::maxIntN(64);
  int64_t max_hsecs = llvm::maxIntN(64);
  int64_t max_dsecs = llvm::maxIntN(64);
  int64_t max_wsecs = llvm::maxIntN(64);
  // Need consider n_align if backend is BM1684
  int64_t n_align = 1;
  for (auto op : lg_info.group_ops) {
    auto mode = getRunMode(dyn_cast<func::FuncOp>(op->getParentOp()));
    auto lgOp = cast<LocalGenInterface>(op);
    for (auto v : get_output_values(op)) {
      module::getNCDHW(v, n, c, d, h, w, lg_info.type);
      auto stype = module::getStorageType(v);
      int dtype_bytes = stype.getIntOrFloatBitWidth() / 8;
      if (Arch::ALIGN_4N) {
        auto stype = module::getStorageType(v);
        n_align = 32 / stype.getIntOrFloatBitWidth();
      }
      max_nsecs = std::min(max_nsecs, ceiling_func(n, n_align));

      if (mode != RunMode::TPU_DYNAMIC && (lg_info.type == GROUP_MM || lg_info.type == GROUP_SMALL_C) &&
          succeeded(lgOp.AllowDataSplit(1, lg_info.type))) {
        int64_t csecs = ceiling_func(c, Arch::NPU_NUM);
        max_csecs = std::min(max_csecs, csecs);
      } else {
        max_csecs = 1;
      }

      // split d now only supports BM1684X and not int4, not dynamic
      if ((module::isBM1684XFamily() || module::isBM1690Family()) &&
          (!stype.isInteger(4)) && lg_info.type == GROUP_3D &&
          mode != RunMode::TPU_DYNAMIC &&
          succeeded(lgOp.AllowDataSplit(2, lg_info.type))) {
        max_dsecs = std::min(max_dsecs, d);
      } else {
        max_dsecs = 1;
      }
      if (succeeded(lgOp.AllowDataSplit(2 + (lg_info.type == GROUP_3D ? 1 : 0),
                                        lg_info.type))) {
        max_hsecs = std::min(max_hsecs, h);
        vec_op_hsecs.push_back(std::make_pair(op, h));
      } else {
        max_hsecs = 1;
      }
      // split w now only supports BM1684X and not int4, not dynamic
      if ((module::isBM1684XFamily() || module::isBM1690Family()) &&
          (!stype.isInteger(4)) && mode != RunMode::TPU_DYNAMIC &&
          succeeded(lgOp.AllowDataSplit(3 + (lg_info.type == GROUP_3D ? 1 : 0),
                                        lg_info.type))) {
        // make min_wslice >= 256 to avoid gdma performance drop
        max_wsecs = std::min(max_wsecs, w);
        // avoid A2 chip ddr interleave
        if (w * dtype_bytes == 512 && module::isBM1688()) {
          max_wsecs = 1;
        }
      } else {
        max_wsecs = 1;
      }
    }
  }

  return shape_secs_t{.nsecs = max_nsecs,
                      .hsecs = max_hsecs,
                      .dsecs = max_dsecs,
                      .wsecs = max_wsecs,
                      .csecs = max_csecs};
}

void update_multi_core_secs(const shape_secs_t max_shape_secs, shape_secs_t &shape_secs) {
  auto core_num = module::getCoreNum();
  int64_t secs = shape_secs.nsecs * shape_secs.csecs * shape_secs.hsecs;
  int64_t max_secs =
      max_shape_secs.nsecs * max_shape_secs.csecs * max_shape_secs.hsecs;
  if (max_secs < core_num || secs >= core_num)
    return;

  shape_secs.nsecs = max_shape_secs.nsecs;
  secs = core_num / shape_secs.nsecs;
  if (shape_secs.csecs < secs && max_shape_secs.csecs >= secs) {
    shape_secs.csecs = secs;
  } else if (shape_secs.csecs < secs && max_shape_secs.csecs >= secs / 2) {
    shape_secs.csecs = secs / 2;
  }

  secs /= shape_secs.csecs;
  if (shape_secs.hsecs < secs && max_shape_secs.hsecs >= secs) {
    shape_secs.hsecs = secs;
  } else if (shape_secs.hsecs < secs && max_shape_secs.hsecs >= secs / 2) {
    shape_secs.hsecs = secs / 2;
  }
}

bool init_group_data_secs(const LgInfo &lg_info, shape_secs_t &shape_secs,
                          std::vector<std::pair<Value, int64_t>>& value_size) {
  shape_secs = {1, 1, 1, 1, 1};
  if (lg_info.group_ops.size() == 1 && false == LgPass::OPTIONS.group_by_cores) {
    return true;
  }

  std::vector<std::pair<Operation*, int>> vec_op_hsecs;
  shape_secs_t max_shape_secs = get_group_max_secs(lg_info, vec_op_hsecs);

  int64_t in_n, in_c, in_d, in_h, in_w;
  int64_t out_n, out_c, out_d, out_h, out_w;
  for (auto op : lg_info.group_ops) {
    auto ins = get_input_values(op);
    auto outs = get_output_values(op);
    module::getNCDHW(ins[0], in_n, in_c, in_d, in_h, in_w, lg_info.type);
    module::getNCDHW(outs[0], out_n, out_c, out_d, out_h, out_w, lg_info.type);
    int64_t in0_lmem_bytes =
        Arch::get_tensor_lmem_bytes(ins[0], in_n, in_c, in_d, in_h, in_w);
    int64_t out0_lmem_bytes =
        Arch::get_tensor_lmem_bytes(outs[0], out_n, out_c, out_d, out_h, out_w);

    int64_t total_size = in0_lmem_bytes + out0_lmem_bytes;
    auto lg_op = cast<LocalGenInterface>(op);
    total_size += lg_op.getBufferSize(in0_lmem_bytes, out0_lmem_bytes, in_n,
                                      in_c, in_h, in_d, in_w, out_n, out_c,
                                      out_h, out_d, out_w, lg_info.type);
    for (size_t i = 1; i < ins.size(); ++i) {
      if ((module::isTrain() && !isa<tpu::AddOp, tpu::ConcatOp>(op))
          || module::isWeight(ins[i])) {
        bool eu_align = is_eu_align(ins[i]);
        if (module::isTrain()) {
          eu_align = !is_value_weight(ins[i]);
        }
        int w_size = Arch::get_weight_lmem_bytes(ins[i], lg_info.type,
                                                  eu_align);
        total_size += w_size;
        value_size.push_back(std::make_pair(ins[i], (w_size + 63)/64*64));
      } else {
        module::getNCDHW(ins[i], in_n, in_c, in_d, in_h, in_w, lg_info.type);
        total_size +=
            Arch::get_tensor_lmem_bytes(ins[i], in_n, in_c, in_d, in_h, in_w);
      }
    }
    for (size_t i = 1; i < outs.size(); ++i) {
      module::getNCDHW(outs[i], out_n, out_c, out_d, out_h, out_w,
                       lg_info.type);
      total_size += Arch::get_tensor_lmem_bytes(outs[i], out_n, out_c, out_d,
                                                out_h, out_w);
    }

    // Need consider different backends
    int64_t total_secs = ceiling_func(total_size, Arch::LMEM_BYTES);
    shape_secs.nsecs =
        std::max(std::min(total_secs, max_shape_secs.nsecs), shape_secs.nsecs);
    total_secs = ceiling_func(total_secs, shape_secs.nsecs);
    if (lg_info.type == GROUP_MM || lg_info.type == GROUP_SMALL_C) {
      if (total_secs > max_shape_secs.csecs) {
        shape_secs.csecs = max_shape_secs.csecs;
      } else {
        int64_t cslice_per_npu = max_shape_secs.csecs / total_secs;
        shape_secs.csecs =
            std::max(ceiling_func(max_shape_secs.csecs, cslice_per_npu),
                     shape_secs.csecs);
      }
      total_secs = ceiling_func(total_secs, shape_secs.csecs);
    }
    shape_secs.dsecs =
        std::max(std::min(total_secs, max_shape_secs.dsecs), shape_secs.dsecs);
    total_secs = ceiling_func(total_secs, shape_secs.dsecs);
    shape_secs.hsecs = std::max(total_secs, shape_secs.hsecs);
    if (shape_secs.hsecs > max_shape_secs.hsecs) {
      shape_secs.wsecs = ceiling_func(shape_secs.hsecs, max_shape_secs.hsecs);
      if (shape_secs.wsecs > max_shape_secs.wsecs) {
        llvm::errs() << "fail at op:"<<module::getName(op).str()<<"\n";
        return false;
      }
      shape_secs.hsecs = max_shape_secs.hsecs;
    }
  }
  return true;
}

static int64_t get_split_max_secs(BasicTimeStepPtr time_step) {
  int64_t timestep_num = time_step->get_timestep_num();
  if (timestep_num == 0) {
    return 0;
  }
  std::vector<int64_t> lmem_req(timestep_num, 0);
  const MemBuff &lmem_buffer = time_step->get_lmem_buffer();

  auto update_lmem_req = [&lmem_req, &timestep_num](int64_t start_ts,
                                                    int64_t end_ts,
                                                    int64_t lmem_size) {
    if (start_ts <= end_ts) {
      for (int64_t ts = start_ts; ts <= end_ts; ++ts) {
        lmem_req[ts] += lmem_size;
      }
    } else {
      for (int64_t ts = 0; ts <= end_ts; ++ts) {
        lmem_req[ts] += lmem_size;
      }
      for (int64_t ts = start_ts; ts <= timestep_num - 1; ++ts) {
        lmem_req[ts] += lmem_size;
      }
    }
  };

  for (auto iter = lmem_buffer.begin(); iter != lmem_buffer.end(); ++iter) {
    int64_t start_ts = iter->second.start_ts;
    int64_t end_ts = iter->second.end_ts;
    update_lmem_req(start_ts, end_ts, (iter->second).size);
  }

  std::stable_sort(lmem_req.begin(), lmem_req.end(), std::greater<int64_t>());
  return ceiling_func(lmem_req[0], Arch::LMEM_BYTES);
}

void update_tensor_infos(const LgInfo &lg_info, TensorInfo &tensor_infos) {
  for (auto &iter : tensor_infos) {
    auto v = iter.first;
    iter.second.use_3ic_opt = use_3ic(v);
    if (module::isTrain()) {
      iter.second.eu_align = !is_value_weight(v);
    } else {
      iter.second.eu_align = is_eu_align(v);
    }
    iter.second.need_bcast = need_bcast(v);
  }

  tensor_info_t ti(TIMESTEP_LOAD);
  int64_t n, c, d, h, w;
  for (auto op : lg_info.group_ops) {
    auto ins = get_input_values(op);
    for (auto in : ins) {
      if (auto src_op = dyn_cast_or_null<top::WeightOp>(in.getDefiningOp())) {
        bool allow_split = false;
        if (src_op.getAllowSplitAttr() != nullptr) {
          allow_split = true;
        }
        if (allow_split == false) {
          ti.eu_align = is_eu_align(in);
          ti.need_bcast = need_bcast(in);
          module::getNCDHW(in, n, c, d, h, w, lg_info.type);
          ti.slice_info.n.clear();
          ti.slice_info.h.clear();
          ti.slice_info.d.clear();
          ti.slice_info.w.clear();
          ti.slice_info.c.clear();
          ti.slice_info.n.push_back(std::make_pair((int64_t)0, (int64_t)n));
          ti.slice_info.h.push_back(std::make_pair((int64_t)0, (int64_t)h));
          ti.slice_info.d.push_back(std::make_pair((int64_t)0, (int64_t)d));
          ti.slice_info.w.push_back(std::make_pair((int64_t)0, (int64_t)w));
          ti.slice_info.c.push_back(std::make_pair((int64_t)0, (int64_t)c));
          tensor_infos[in] = ti;
        }
      }
    }
  }
}

bool can_split_w(const LgInfo &lg_info, int64_t dhw_secs, int64_t height_min,
                 int64_t wsecs) {
  if (dhw_secs < height_min && wsecs > 1) {
    for (auto out : lg_info.group_outs) {
      int64_t n, c, d, h, w;
      module::getNCDHW(out, n, c, d, h, w, lg_info.type);
      int dtype_size = module::getDtypeSize(out);
      if (ceiling_func(w, wsecs) * dtype_size < 256) {
        return false;
      }
    }
  }
  return true;
}

// make sure group secs can be devided by num_core
static void force_group_by_cores(shape_secs_t &shape_secs,
                                 const shape_secs_t &max_shape_secs) {
  auto num_cores = module::getCoreNum();
  if (num_cores < 2) {
    return;
  }
  auto pre_secs = shape_secs.nsecs * shape_secs.csecs * shape_secs.dsecs;
  if (pre_secs * shape_secs.hsecs % num_cores == 0) {
    return;
  }
  for (int i = 1; i < num_cores; i++) {
    if ((shape_secs.hsecs + i) > max_shape_secs.hsecs) {
      return;
    }
    if (pre_secs * (shape_secs.hsecs + i) % num_cores == 0) {
      shape_secs.hsecs += i;
      return;
    }
  }
  return;
}

void assign_dhwsecs(const LgInfo &lg_info, shape_secs_t &shape_secs,
                    int64_t &dhw_secs, const shape_secs_t &max_shape_secs) {
  shape_secs.dsecs = 1;
  shape_secs.hsecs = dhw_secs;
  shape_secs.wsecs = 1;
  ValueSet group_out_tensors;
  for (auto op : lg_info.group_ops) {
    auto outs = get_output_values(op);
    group_out_tensors.insert(outs.begin(), outs.end());
  }
  if (max_shape_secs.dsecs == 1) {
    shape_secs.dsecs = 1;
    shape_secs.hsecs = dhw_secs;
    shape_secs.wsecs = 1;
    // split height and width
    float h_len = 0.f, w_len = 0.f;
    for (auto out : group_out_tensors) {
      if (out.use_empty()) {
        continue;
      }
      int64_t n, c, d, h, w;
      module::getNCDHW(out, n, c, d, h, w, lg_info.type);
      h_len += (float)h;
      w_len += (float)w;
    }

    float min_len = __FLT_MAX__;
    for (int64_t i = max_shape_secs.hsecs; i > 0; --i) {
      int64_t hsecs = i;
      int64_t wsecs = ceiling_func(dhw_secs, i);

      int cur_len = (hsecs - 1) + (wsecs - 1) * (h_len / w_len);
      bool split_w =
          can_split_w(lg_info, dhw_secs, max_shape_secs.hsecs, wsecs);

      if (cur_len < min_len && split_w && wsecs <= max_shape_secs.wsecs) {
        min_len = cur_len;
        shape_secs.hsecs = hsecs;
        shape_secs.wsecs = wsecs;
      }
    }
  } else {
    // split depth and height
    if (module::isBM1688()) {
      float d_len = 0.f, h_len = 0.f;
      for (auto out : group_out_tensors) {
        int64_t n, c, d, h, w;
        module::getNCDHW(out, n, c, d, h, w, lg_info.type);
        d_len += (float)d;
        h_len += (float)h;
      }
      float min_len = __FLT_MAX__;
      for (int64_t i = max_shape_secs.dsecs; i > 0; --i) {
        int64_t dsecs = i;
        int64_t hsecs = ceiling_func(dhw_secs, i);

        int cur_len = (dsecs - 1) + (hsecs - 1) * (d_len / h_len);
        if (cur_len < min_len) {
          min_len = cur_len;
          shape_secs.dsecs = dsecs;
          shape_secs.hsecs = hsecs;
        }
      }
    } else {
      /// if split h or w, gdma band width may be lowered due to ddr 4k channel
      /// interleave
      shape_secs.dsecs = std::min(max_shape_secs.dsecs, dhw_secs);
      shape_secs.hsecs = ceiling_func(dhw_secs, shape_secs.dsecs);
    }
    // d split is max but h max still not enough, split w
    if (shape_secs.hsecs > max_shape_secs.hsecs) {
      shape_secs.wsecs = ceiling_func(shape_secs.hsecs, max_shape_secs.hsecs);
      shape_secs.hsecs = max_shape_secs.hsecs;
    }
  }
  if (LgPass::OPTIONS.group_by_cores) {
    force_group_by_cores(shape_secs, max_shape_secs);
  }
  dhw_secs = shape_secs.dsecs * shape_secs.hsecs * shape_secs.wsecs;
}

bool update_data_split(BasicTimeStepPtr time_step, const LgInfo &lg_info,
                       shape_secs_t &shape_secs) {
  shape_secs.nsecs = 1;
  shape_secs.hsecs = 1;
  shape_secs.dsecs = 1;
  shape_secs.wsecs = 1;
  shape_secs.csecs = 1;
  bool status = false;
  auto &tensor_infos = time_step->get_tensor_infos();
  std::vector<std::pair<Operation*, int>> vec_op_hsecs;
  shape_secs_t max_shape_secs = get_group_max_secs(lg_info, vec_op_hsecs);
  for (int64_t nsec = 1; nsec <= max_shape_secs.nsecs; ++nsec) {
    shape_secs.nsecs = nsec;
    tensor_infos.clear();
    if (stripe_mine_max_slice(lg_info, shape_secs, tensor_infos) == false) {
      return false;
    }
    time_step->update_all_mem_buffer_size(lg_info);

    // update nsecs
    int64_t total_secs = get_split_max_secs(time_step);
    if (total_secs == 0) {
      return false;
    }
    shape_secs.nsecs =
        std::max(shape_secs.nsecs, std::min(max_shape_secs.nsecs, total_secs));
    if (shape_secs.nsecs > nsec)
      continue;
    // update csecs
    int64_t cdhw_secs = ceiling_func(total_secs, shape_secs.nsecs);
    shape_secs.csecs =
        std::max(shape_secs.csecs, std::min(max_shape_secs.csecs, cdhw_secs));
    // update d/h/w secs
    int64_t dhw_secs = ceiling_func(cdhw_secs, shape_secs.csecs);
    if (dhw_secs > 1) {
      if (shape_secs.nsecs == max_shape_secs.nsecs) {
        assign_dhwsecs(lg_info, shape_secs, dhw_secs, max_shape_secs);
      } else {
        continue;
      }
    }
    if (shape_secs.dsecs <= max_shape_secs.dsecs &&
        shape_secs.hsecs <= max_shape_secs.hsecs &&
        shape_secs.wsecs <= max_shape_secs.wsecs) {
      status = true;
      break;
    }
  }


  update_tensor_infos(lg_info, tensor_infos);
  return status;
}

bool strip_back_judge(Value v, const LgInfo &lg_info,
                      const std::multiset<Operation *> &op_set,
                      const std::set<Value, value_compare> &out_tensor_set) {
  auto users = v.getUsers();
  bool res = true;
  bool has_outer_group_user = false;
  for (auto op : users) {
    if (std::find(lg_info.group_ops.begin(), lg_info.group_ops.end(), op) !=
        lg_info.group_ops.end()) {
      if (op_set.count(op) == 0) {
        return false;
      }
    } else {
      has_outer_group_user = true;
    }
  }

  if (has_outer_group_user) {
    res = out_tensor_set.find(v) != out_tensor_set.end();
  }
  return res;
}

bool strip_back_judge2(Value v, const LgInfo &lg_info,
                      const std::multiset<Operation *> &op_set,
                      const std::set<Value, value_compare> &out_tensor_set) {
  auto users = v.getUsers();
  bool res = true;
  // bool has_outer_group_user = false;
  for (auto op : users) {
    if (std::find(lg_info.group_ops.begin(), lg_info.group_ops.end(), op) !=
        lg_info.group_ops.end()) {
      if (op_set.count(op) == 0) {
        return false;
      }
    } else {
      // has_outer_group_user = true;
    }
  }

  // if (has_outer_group_user) {
  //   res = out_tensor_set.find(v) != out_tensor_set.end();
  // }
  return res;
}


inline bool is_same_slice(const slice_pair_t &a, const slice_pair_t &b) {
  return (a.first == b.first) && (a.second == b.second);
}

bool is_same_slice_info(const slice_info_t &si0, const slice_info_t &si1) {
  if (si0.n.size() != si1.n.size() || si0.c.size() != si1.c.size() ||
      si0.h.size() != si1.h.size() || si0.d.size() != si1.d.size() ||
      si0.w.size() != si1.w.size()) {
    return false;
  }
  // check n
  for (size_t i = 0; i < si0.n.size(); ++i) {
    if (false == is_same_slice(si0.n[i], si1.n[i])) {
      return false;
    }
  }
  // check c
  for (size_t i = 0; i < si0.c.size(); ++i) {
    if (false == is_same_slice(si0.c[i], si1.c[i])) {
      return false;
    }
  }
  // check h
  for (size_t i = 0; i < si0.h.size(); ++i) {
    if (false == is_same_slice(si0.h[i], si1.h[i])) {
      return false;
    }
  }
  // check d
  for (size_t i = 0; i < si0.d.size(); ++i) {
    if (false == is_same_slice(si0.d[i], si1.d[i])) {
      return false;
    }
  }
  // check w
  for (size_t i = 0; i < si0.w.size(); ++i) {
    if (false == is_same_slice(si0.w[i], si1.w[i])) {
      return false;
    }
  }
  // for (auto it : llvm::zip(si0.n, si1.n)) {
  //   if (false == is_same_slice(std::get<0>(it), std::get<1>(it))) {
  //     return false;
  //   }
  // }
  // // check h
  // for (auto it : llvm::zip(si0.h, si1.h)) {
  //   if (false == is_same_slice(std::get<0>(it), std::get<1>(it))) {
  //     return false;
  //   }
  // }
  return true;
}

bool is_broadcast_binary(Operation *op, Value in) {
  if (!isa<tpu::AddOp, tpu::SubOp, tpu::MulOp, tpu::DivOp, tpu::MaxOp,
           tpu::MinOp>(op)) {
    return false;
  }
  auto other = in == op->getOperand(0) ? op->getOperand(1) : op->getOperand(0);
  auto in_shape = in.getType().cast<RankedTensorType>().getShape();
  auto other_shape = other.getType().cast<RankedTensorType>().getShape();
  if (in_shape.size() != other_shape.size()) {
    return false;
  }
  for (int i = 0; i < in_shape.size(); ++i) {
    if (in_shape[i] != other_shape[i] && in_shape[i] == 1) {
      return true;
    }
  }
  return false;
}

// if no transpose, split left matrix rows,
// if left/right are both transposed, split right matrix rows
bool is_matmul_right_tensor(Operation *op, Value v) {
  bool res = false;
  if (auto matmul_op = dyn_cast<tpu::MatMulOp>(op)) {
    bool left_trans = matmul_op.getLeftTranspose();
    bool right_trans = matmul_op.getRightTranspose();
    res = (v == matmul_op.getRight());
    if (left_trans && right_trans) {
      res = (v == matmul_op.getInput());
    }
  }
  return res;
}

bool is_attention_not_input_tensor(Operation *op, Value v) {
  bool res = false;
  if (auto attention_op = dyn_cast<tpu::AttentionOp>(op)) {
    res = (v != attention_op.getInput());
  }
  return res;
}

slice_info_t get_out_slice_info(const shape_secs_t &shape_secs, int64_t n,
                                int64_t c, int64_t h, int64_t d, int64_t w,
                                int64_t bitwidth) {
  slice_info_t slice_info;
  int64_t secs, idx, slice, step;
  // n slice info
  secs = shape_secs.nsecs;
  int64_t n_align = 32 / bitwidth;
  if (Arch::ALIGN_4N && n_align != 1) {
    step = align_up(ceiling_func(n, secs), n_align);
    for (int64_t i = 0; i < secs; ++i) {
      idx = i == 0 ? 0 : idx + slice;
      slice = (n - idx) > step ? step : (n - idx);
      slice_info.n.emplace_back(slice_pair_t(idx, slice));
    }
  } else {
    for (int64_t i = 0; i < secs; ++i) {
      step = n / secs + (n % secs > i ? 1 : 0);
      idx = n / secs * i + (n % secs > i ? i : n % secs);
      slice = (n - idx) > step ? step : (n - idx);
      // assert(idx < n);
      slice_info.n.emplace_back(slice_pair_t(idx, slice));
    }
  }
  // c slice info
  secs = shape_secs.csecs;
  int64_t c_per_npu = ceiling_func(c, Arch::NPU_NUM);
  for (int64_t i = 0; i < secs; ++i) {
    step = (c_per_npu / secs + (c_per_npu % secs > i)) * Arch::NPU_NUM;
    idx =
        (c_per_npu / secs * i + (c_per_npu % secs > i ? i : c_per_npu % secs)) *
        Arch::NPU_NUM;
    slice = (c - idx) > step ? step : (c - idx);
    // assert(idx < c);
    slice_info.c.emplace_back(slice_pair_t(idx, slice));
  }
  // h slice_info
  secs = shape_secs.hsecs;
  for (int64_t i = 0; i < secs; ++i) {
    step = h / secs + (h % secs > i ? 1 : 0);
    idx = h / secs * i + (h % secs > i ? i : h % secs);
    slice = (h - idx) > step ? step : (h - idx);
    // assert(idx < h);
    slice_info.h.emplace_back(slice_pair_t(idx, slice));
  }
  // d slice_info
  secs = shape_secs.dsecs;
  for (int64_t i = 0; i < secs; ++i) {
    step = d / secs + (d % secs > i ? 1 : 0);
    idx = d / secs * i + (d % secs > i ? i : d % secs);
    slice = (d - idx) > step ? step : (d - idx);
    // assert(idx < d);
    slice_info.d.emplace_back(slice_pair_t(idx, slice));
  }
  // w slice_info
  secs = shape_secs.wsecs;
  for (int64_t i = 0; i < secs; ++i) {
    step = w / secs + (w % secs > i ? 1 : 0);
    idx = w / secs * i + (w % secs > i ? i : w % secs);
    slice = (w - idx) > step ? step : (w - idx);
    // assert(idx < w);
    slice_info.w.emplace_back(slice_pair_t(idx, slice));
  }

  return slice_info;
}

bool get_backward_slice_info(slice_info_t &in_si, const slice_info_t &out_si,
                             Operation *op, Value in,
                             const shape_secs_t &shape_secs,
                             group_type_t group_type, bool &hold_in_lmem,
                             bool is_group_in) {
  int64_t n, c, d, h, w;
  module::getNCDHW(in, n, c, d, h, w, group_type);
  auto lg_op = cast<LocalGenInterface>(op);
  bool is_broadcast_tensor = is_broadcast_binary(op, in);
  bool is_right_matrix = is_matmul_right_tensor(op, in);
  bool is_no_input_attention = is_attention_not_input_tensor(op, in);

  int64_t idx = 0, slice = 0;
  if (shape_secs.nsecs == 1) {
    in_si.n.emplace_back(slice_pair_t(0, n));
  } else {
    for (auto &s : out_si.n) {
      auto ret = lg_op.BackwardN(idx, slice, s.first, s.second);
      if (is_broadcast_tensor && n == 1) {
        idx = 0;
        slice = 1;
      } else {
        if (failed(ret) || slice == 0) {
          llvm::errs() << "BackwardN fail, at op:"<<module::getName(op).str()<<"\n";
          return false;
        }
      }
      in_si.n.emplace_back(slice_pair_t(idx, slice));
    }
  }

  if (shape_secs.csecs == 1) {
    in_si.c.emplace_back(slice_pair_t(0, c));
  } else {
    for (auto &s : out_si.c) {
      auto ret = lg_op.BackwardC(idx, slice, s.first, s.second);
      if (is_broadcast_tensor && c == 1) {
        idx = 0;
        slice = 1;
      } else if (is_right_matrix) {
        idx = 0;
        slice = c;
        in_si.c.emplace_back(slice_pair_t(idx, slice));
        if (is_group_in) {
          hold_in_lmem = true;
          break;
        } else {
          hold_in_lmem = false;
          continue;
        }
      } else if (is_no_input_attention) {
        idx = 0;
        slice = c;
        in_si.c.emplace_back(slice_pair_t(idx, slice));
      } else {
        if (failed(ret) || slice == 0) {
          llvm::errs() << "BackwardC fail, at op:"<<module::getName(op).str()<<"\n";
          return false;
        }
      }
      in_si.c.emplace_back(slice_pair_t(idx, slice));
    }
  }

  int64_t pre_end_idx = 0;
  idx = slice = 0;
  if (shape_secs.dsecs == 1) {
    in_si.d.emplace_back(slice_pair_t(0, d));
  } else {
    for (int i = 0; i < out_si.d.size(); i++) {
      auto &s = out_si.d[i];
      auto ret = lg_op.BackwardD(idx, slice, s.first, s.second);
      if (is_broadcast_tensor && d == 1) {
        idx = 0;
        slice = 1;
      } else {
        bool end_reached = idx + slice == pre_end_idx;
        if (failed(ret) || slice == 0 || (idx == 0 && i > 0) || end_reached) {
          llvm::errs() << "BackwardD fail, at op:"<<module::getName(op).str()<<"\n";
          return false;
        }
      }
      pre_end_idx = idx + slice;
      in_si.d.emplace_back(slice_pair_t(idx, slice));
    }
  }

  pre_end_idx = 0;
  idx = slice = 0;
  if (shape_secs.hsecs == 1) {
    in_si.h.emplace_back(slice_pair_t(0, h));
  } else {
    for (int i = 0; i < out_si.h.size(); i++) {
      auto &s = out_si.h[i];
      auto ret = lg_op.BackwardH(idx, slice, s.first, s.second);
      if ((is_right_matrix || is_broadcast_tensor) && h == 1) {
        idx = 0;
        slice = 1;
      } else {
        bool end_reached = idx + slice == pre_end_idx;
        if (failed(ret) || slice == 0 || (idx == 0 && i > 0) || end_reached) {
          llvm::errs() <<"BackwardH fail, ret:"<<(failed(ret)?"failed":"success") << ", end_reached:"<<end_reached
                      <<", i:"<<i<<", idx:"<<idx<<", slice:"<<slice<<" at op:"<<module::getName(op).str()<< "\n";
          return false;
        }
      }
      pre_end_idx = idx + slice;
      in_si.h.emplace_back(slice_pair_t(idx, slice));
    }
  }

  pre_end_idx = 0;
  idx = slice = 0;
  if (shape_secs.wsecs == 1) {
    in_si.w.emplace_back(slice_pair_t(0, w));
  } else {
    for (int i = 0; i < out_si.w.size(); i++) {
      auto &s = out_si.w[i];
      auto ret = lg_op.BackwardW(idx, slice, s.first, s.second);
      if (is_broadcast_tensor && w == 1) {
        idx = 0;
        slice = 1;
      } else {
        bool end_reached = idx + slice == pre_end_idx;
        if (failed(ret) || slice == 0 || (idx == 0 && i > 0) || end_reached) {
          llvm::errs() << "BackwardW fail, at op:"<<module::getName(op).str()<<"\n";
          return false;
        }
      }
      pre_end_idx = idx + slice;
      in_si.w.emplace_back(slice_pair_t(idx, slice));
    }
  }
  return true;
}

bool check_hsecs(Value value, slice_info_t &si, group_type_t group_type) {
  assert(si.h.size() > 0);
  int64_t n, c, d, h, w;
  module::getNCDHW(value, n, c, d, h, w, group_type);
  int64_t total_h = 0;
  for (auto &it : si.h) {
    total_h += it.second;
  }
  if (total_h * 2 > h * 3) { // h increase 1.5 times
    return false;
  }
  return true;
}

static bool backward_update_slice(
    const LgInfo &lg_info, const shape_secs_t &shape_secs, const Value &out,
    std::list<Value> &tensor_branchs, TensorInfo &tensor_infos,
    std::multiset<Operation *> &op_set, const ValueSet &out_tensor_set) {

  // Don't backward when this out tensor is the input of the group
  if (std::find(lg_info.group_ins.begin(), lg_info.group_ins.end(), out) !=
      lg_info.group_ins.end()) {
    // return check_hsecs(out, tensor_infos[out].slice_info, lg_info.type);
    return true;
  }
  auto op = out.getDefiningOp();
  if (isa<tpu::Conv2DOp>(op) && module::isBM1684Family()) {
    auto conv_attr = dyn_cast<tpu::Conv2DOp>(op).parseParam();
    if (conv_attr.use_3ic_optimize) {
      return false;
    }
  }
  op_set.insert(op);
  auto mode = getRunMode(op);

  slice_info_t &out_si = tensor_infos[out].slice_info;
  auto &group_ins = lg_info.group_ins;

  for (auto in : op->getOperands()) {
    auto pre_op = in.getDefiningOp();
    if (pre_op != nullptr && isa<top::NoneOp>(pre_op)) {
      continue;
    }
    if (auto weight_op = dyn_cast_or_null<top::WeightOp>(pre_op)) {
      if (weight_op.getAllowSplit() == std::nullopt) {
        continue;
      }
      bool allow_split = true;
      auto weight_op_allow_split_attr = weight_op.getAllowSplitAttr();
      auto num_dims = weight_op_allow_split_attr.size();
      auto allow_split_array = module::getI64Array(weight_op_allow_split_attr);
      for (int i = 0; i < num_dims; ++i) {
        if (allow_split_array->at(i) == 0)
          allow_split = false;
      }
      if (allow_split == false) {
        continue;
      }
    }
    slice_info_t si;
    bool hold_in_lmem = false;
    bool is_group_in =
        std::find(group_ins.begin(), group_ins.end(), in) != group_ins.end();
    auto ret = get_backward_slice_info(si, out_si, op, in, shape_secs,
                                       lg_info.type, hold_in_lmem, is_group_in);
    if (pre_op && module::isDynWeight(in)) {
      auto shape = module::getShape(in);
      si.n.clear();
      si.n.emplace_back(std::pair(0, shape[0]));
      si.c.clear();
      si.c.emplace_back(std::pair(0, shape[1]));
      si.h.clear();
      si.h.emplace_back(std::pair(0, shape[2]));
      si.w.clear();
      si.w.emplace_back(std::pair(0, shape[3]));

      tensor_infos[in] = tensor_info_t(si);
      tensor_infos[in].hold_in_lmem = true;

      if (strip_back_judge(in, lg_info, op_set, out_tensor_set)) {
        tensor_branchs.push_back(in);
      }
      continue;
    }
    if (ret == false) {
      return false;
    }
    auto iter = tensor_infos.find(in);
    if (iter != tensor_infos.end()) {
      if (false == is_same_slice_info(si, iter->second.slice_info)) {
        if (module::isCV18xx() || mode == RunMode::TPU_DYNAMIC || !pre_op) {
          return false;
        }
        // only Conv2D allow differnece for now
        if (pre_op) {
          for (auto user : pre_op->getUsers()) {
            if (!(std::find(lg_info.group_ops.begin(), lg_info.group_ops.end(),
                            user) != lg_info.group_ops.end() &&
                  isa<tpu::Conv2DOp>(user) && module::isUniformQuantized(in)) ||
                lg_info.group_outs.size() != 1) {
              return false;
            }
          }
        }
        bool is_hw_overlap = true;
        for (int i = 0; i < shape_secs.hsecs; i++) {
          is_hw_overlap *=
              std::max(si.h[i].first, iter->second.slice_info.h[i].first) <
              std::min(si.h[i].first + si.h[i].second,
                       iter->second.slice_info.h[i].first +
                           iter->second.slice_info.h[i].second);
        }
        for (int i = 0; i < shape_secs.wsecs; i++) {
          is_hw_overlap *=
              std::max(si.w[i].first, iter->second.slice_info.w[i].first) <
              std::min(si.w[i].first + si.w[i].second,
                       iter->second.slice_info.w[i].first +
                           iter->second.slice_info.w[i].second);
        }
        if (is_hw_overlap) {
          slice_info_t si_both;
          si_both.n = si.n;
          si_both.c = si.c;
          si_both.d = si.d;
          for (int i = 0; i < shape_secs.hsecs; i++) {
            int64_t h_lowest =
                std::min(si.h[i].first, iter->second.slice_info.h[i].first);
            int64_t h_highest =
                std::max(si.h[i].first + si.h[i].second,
                         iter->second.slice_info.h[i].first +
                             iter->second.slice_info.h[i].second);
            si_both.h.push_back(
                std::pair<int64_t, int64_t>(h_lowest, h_highest - h_lowest));
          }
          for (int i = 0; i < shape_secs.wsecs; i++) {
            int64_t w_lowest =
                std::min(si.w[i].first, iter->second.slice_info.w[i].first);
            int64_t w_highest =
                std::max(si.w[i].first + si.w[i].second,
                         iter->second.slice_info.w[i].first +
                             iter->second.slice_info.w[i].second);
            si_both.w.push_back(
                std::pair<int64_t, int64_t>(w_lowest, w_highest - w_lowest));
          }
          tensor_infos[in] = tensor_info_t(si_both);
          tensor_infos[in].hold_in_lmem = hold_in_lmem;
        } else {
          return false;
        }
      }
    } else {
      tensor_infos[in] = tensor_info_t(si);
      tensor_infos[in].hold_in_lmem = hold_in_lmem;
    }
    if (strip_back_judge(in, lg_info, op_set, out_tensor_set)) {
      tensor_branchs.push_back(in);
    }
  }
  return true;
}


static bool backward_update_slice2(
    const LgInfo &lg_info, const shape_secs_t &shape_secs, const std::pair<Value, Operation*> &out,
    std::list<std::pair<Value, Operation*>> &tensor_branchs, TensorInfo &tensor_infos,
    std::multiset<Operation *> &op_set, const ValueSet &out_tensor_set) {

  // Don't backward when this out tensor is the input of the group
  if (std::find(lg_info.group_ins.begin(), lg_info.group_ins.end(), out.first) !=
      lg_info.group_ins.end()) {
    // return check_hsecs(out, tensor_infos[out].slice_info, lg_info.type);
    return true;
  }
  auto op = out.first.getDefiningOp();
  if (isa<tpu::Conv2DOp>(op) && module::isBM1684Family()) {
    auto conv_attr = dyn_cast<tpu::Conv2DOp>(op).parseParam();
    if (conv_attr.use_3ic_optimize) {
      return false; //todo ���䨪??D??��?�䨮??��?��??a?a2??����?
    }
  }
  auto mode = getRunMode(op);
  op_set.insert(op);

  slice_info_t &out_si = tensor_infos[out.first].slice_info;
  auto &group_ins = lg_info.group_ins;

  for (auto in : op->getOperands()) {
    auto pre_op = in.getDefiningOp();
    if (pre_op != nullptr && isa<top::NoneOp>(pre_op)) {
      continue;
    }
    if (auto weight_op = dyn_cast_or_null<top::WeightOp>(pre_op)) {
      if (weight_op.getAllowSplit() == std::nullopt) {
        continue;
      }
      bool allow_split = true;
      auto weight_op_allow_split_attr = weight_op.getAllowSplitAttr();
      auto num_dims = weight_op_allow_split_attr.size();
      auto allow_split_array = module::getI64Array(weight_op_allow_split_attr);
      for (int i = 0; i < num_dims; ++i) {
        if (allow_split_array->at(i) == 0)
          allow_split = false;
      }
      if (allow_split == false) {
        continue;
      }
    }
    slice_info_t si;
    bool hold_in_lmem = false;
    bool is_group_in =
        std::find(group_ins.begin(), group_ins.end(), in) != group_ins.end();
    // llvm::errs() << "xxx1:"<<module::getName(in)<<"\n";
    auto ret = get_backward_slice_info(si, out_si, op, in, shape_secs,
                                       lg_info.type, hold_in_lmem, is_group_in);
    if (is_value_weight(in)) {
      auto shape = module::getShape(in);
      si.n.clear();
      si.n.emplace_back(std::pair(0, shape[0]));
      si.c.clear();
      si.c.emplace_back(std::pair(0, shape[1]));
      si.h.clear();
      si.h.emplace_back(std::pair(0, shape[2]));
      si.w.clear();
      si.w.emplace_back(std::pair(0, shape[3]));

      tensor_infos[in] = tensor_info_t(op, si);
      llvm::errs() << "is_value_weight, in name:"<<module::getName(in)<<"\n";
      // tensor_infos[in].hold_in_lmem = true;

      // if (strip_back_judge(in, lg_info, op_set, out_tensor_set)) {
      tensor_branchs.push_back(std::make_pair(in, op));
      // }
      continue;
    }
    if (ret == false) {
      return false;
    }

    if (pre_op && isa<tpu::MaxPoolWithMaskOp>(pre_op)) {
      for (int j = 0; j < pre_op->getNumResults(); j++) {
        auto res = pre_op->getResult(j);
        if (res == in) {
          continue;
        }
        if (tensor_infos.find(res) != tensor_infos.end()) {
          tensor_infos[res] = tensor_info_t(op, si); //??��?????��?3???��D���䨪?tensor?����?��??��?��shape��?������?maxpool��?outputo��mask��D��?��??��
        }
      }
    }

    auto iter = tensor_infos.find(in);
    if (iter != tensor_infos.end()) {
      if (false == is_same_slice_info(si, iter->second.slice_info)) {
        if (module::isCV18xx() || mode == RunMode::TPU_DYNAMIC)
          return false;
        // only Conv2D allow differnece for now
        // if (pre_op) {
        //   for (auto user : pre_op->getUsers()) {
        //     if (isa<ReturnOp>(user)) {
        //       llvm::errs() << "skip ReturnOp\n";
        //       continue;
        //     }
        //     if (!(std::find(lg_info.group_ops.begin(), lg_info.group_ops.end(),
        //                     user) != lg_info.group_ops.end() &&
        //           isa<tpu::Conv2DOp>(user) && module::isUniformQuantized(in))||
        //            lg_info.group_outs.size() != 1 ) {
        //       llvm::errs() << "xxx1\n";
        //       return false;
        //     }
        //   }
        // }
        bool is_hw_overlap = true;
        for (int i = 0; i < shape_secs.hsecs; i++) {
          is_hw_overlap *=
              std::max(si.h[i].first, iter->second.slice_info.h[i].first) <
              std::min(si.h[i].first + si.h[i].second,
                       iter->second.slice_info.h[i].first +
                           iter->second.slice_info.h[i].second);
        }
        for (int i = 0; i < shape_secs.wsecs; i++) {
          is_hw_overlap *=
              std::max(si.w[i].first, iter->second.slice_info.w[i].first) <
              std::min(si.w[i].first + si.w[i].second,
                       iter->second.slice_info.w[i].first +
                           iter->second.slice_info.w[i].second);
        }
        if (is_hw_overlap) {
          slice_info_t si_both;
          si_both.n = si.n;
          si_both.c = si.c;
          si_both.d = si.d;
          for (int i = 0; i < shape_secs.hsecs; i++) {
            int64_t h_lowest =
                std::min(si.h[i].first, iter->second.slice_info.h[i].first);
            int64_t h_highest =
                std::max(si.h[i].first + si.h[i].second,
                         iter->second.slice_info.h[i].first +
                             iter->second.slice_info.h[i].second);
            si_both.h.push_back(
                std::pair<int64_t, int64_t>(h_lowest, h_highest - h_lowest));
          }
          for (int i = 0; i < shape_secs.wsecs; i++) {
            int64_t w_lowest =
                std::min(si.w[i].first, iter->second.slice_info.w[i].first);
            int64_t w_highest =
                std::max(si.w[i].first + si.w[i].second,
                         iter->second.slice_info.w[i].first +
                             iter->second.slice_info.w[i].second);
            si_both.w.push_back(
                std::pair<int64_t, int64_t>(w_lowest, w_highest - w_lowest));
          }
          auto tmp = tensor_info_t(op, si_both);
          auto slice_infos = tensor_infos[in].slice_infos;
          for (auto itr = slice_infos.begin(); itr != slice_infos.end(); ++itr) {
            tmp.add_slice_info(itr->first, itr->second);
          }
          tensor_infos[in] = tmp;
          tensor_infos[in].hold_in_lmem = hold_in_lmem;
          if (pre_op && isa<tpu::MaxPoolWithMaskOp>(pre_op)) {
            for (int j = 0; j < pre_op->getNumResults(); j++) {
              auto res = pre_op->getResult(j);
              if (res == in) {
                continue;
              }
              auto tmp = tensor_info_t(op, si_both);
              auto slice_infos = tensor_infos[res].slice_infos;
              for (auto itr = slice_infos.begin(); itr != slice_infos.end(); ++itr) {
                tmp.add_slice_info(itr->first, itr->second);
              }
              tensor_infos[res] = tmp;
            }
          }
        } else {
          llvm::errs() << "xxx2\n";
          return false;
        }
      } else {
        tensor_infos[in].add_slice_info(op, si);
      }
    } else {
      tensor_infos[in] = tensor_info_t(op, si);
      tensor_infos[in].hold_in_lmem = hold_in_lmem;
    }

    if (strip_back_judge2(in, lg_info, op_set, out_tensor_set)) {
      tensor_branchs.push_back(std::make_pair(in, op));
    }
  }
  return true;
}

int vec_index(const std::vector<Operation*>& group_ops, Operation* op) {
  int idx = 0;
  for (auto group_op : group_ops) {
    if (group_op == op) {
      return idx;
    }
    idx++;
  }
  assert(false);
  return 0;
}

static op_var_pos_info findVarBound(const std::vector<op_var_pos_info>& op_var_bound, std::pair<int, int> key) {
  for (int i = 0, size = op_var_bound.size(); i < size; i++) {
    if (op_var_bound[i].key == key) {
      return op_var_bound[i];
    }
  }
  return op_var_pos_info();
}

static int getTensorLmemBytes(Value& value, const slice_info_t& slice_info, const std::vector<int64_t>& ncdhw_idx, group_type_t type,
                      int* n = nullptr, int* c = nullptr, int* d = nullptr, int* h = nullptr, int* w = nullptr) {
    int tmp_n = slice_info.n[ncdhw_idx[0]].second;
    int tmp_c = slice_info.c[ncdhw_idx[1]].second;
    int tmp_d = slice_info.d[ncdhw_idx[2]].second;
    int tmp_h = slice_info.h[ncdhw_idx[3]].second;
    int tmp_w = slice_info.w[ncdhw_idx[4]].second;

    if (n)
      *n = tmp_n;
    if (c)
      *c = tmp_c;
    if (d)
      *d = tmp_d;
    if (h)
      *h = tmp_h;
    if (w)
      *w = tmp_w;
    return backend::Arch::get_tensor_lmem_bytes(value, tmp_n, tmp_c, tmp_h, tmp_d, tmp_w, type);
}

inline int64_t align_64(int64_t input)
{
    return (int64_t)((input + 63)/64*64);
}

int getOpLmemBytes(Operation* op, TensorInfo &tensor_infos, const std::vector<int64_t>& ncdhw_idx,
                  const LgInfo &lg_info, int& buffer_size, std::shared_ptr<dot_graph> dot_graph_log) {
  int in_n, in_c, in_d, in_h, in_w, out_n, out_c, out_d, out_h, out_w;
  auto ins = get_input_values(op);
  auto outs = get_output_values(op);
  auto op_name = replaceChars_for_dot(module::getName(op).str());

  // slice_info_t info = tensor_infos[ins[0]].slice_infos[op];
  slice_info_t info = tensor_infos[ins[0]].slice_info;
  int64_t in0_lmem_bytes = getTensorLmemBytes(ins[0], info, ncdhw_idx, lg_info.type, &in_n, &in_c, &in_d, &in_h, &in_w);
  auto shape = "n:" + std::to_string(in_n) +" c:" + std::to_string(in_c) +" d:" + std::to_string(in_d)
              +" h:" + std::to_string(in_h) +" w:" + std::to_string(in_w);
  dot_graph_log->add_node_label(op_name + "_ori", "in0_lmem_bytes:"+std::to_string(in0_lmem_bytes) + ", shape:" + shape);
  int64_t in_lmem_bytes = align_64(in0_lmem_bytes);
  llvm::errs() << "ncdhw, n:"<<in_n<<" c:"<<in_c<<" d:"<<in_d<<" h:"<<in_h<<" w:"<<in_w<< " in0_size:"<<in_lmem_bytes<<"\n";
  for (int i = 1; i < ins.size(); i++) {
    auto tmp_op = ins[i].getDefiningOp();
    if (tmp_op && isa<top::NoneOp>(tmp_op)) {
      continue;
    }

    if (!is_value_weight(ins[i])) {
      // info = tensor_infos[ins[i]].slice_infos[op];
      info = tensor_infos[ins[i]].slice_info;
      int64_t tmp = align_64(getTensorLmemBytes(ins[i], info, ncdhw_idx, lg_info.type));
      in_lmem_bytes += tmp;
      llvm::errs() << "tensor_in"<<i<< ", size:"<<tmp<<"\n";;
      dot_graph_log->add_node_label(op_name + "_ori", "tensor_in:"+std::to_string(tmp));
    } else {
      int64_t tmp = align_64(Arch::get_weight_lmem_bytes(ins[i], lg_info.type, false));
      in_lmem_bytes += tmp;
      llvm::errs() << "weight_in"<<i<< ", size:"<<tmp<<"\n";
      dot_graph_log->add_node_label(op_name + "_ori", "weight_in:"+std::to_string(tmp));
    }
  }

  info = tensor_infos[outs[0]].slice_info;
  int64_t out0_lmem_bytes =  getTensorLmemBytes(outs[0], info, ncdhw_idx, lg_info.type, &out_n, &out_c, &out_d, &out_h, &out_w);
  llvm::errs() << "out0_size:"<<out0_lmem_bytes<<"\n";;
  dot_graph_log->add_node_label(op_name + "_ori", "out0_lmem_bytes:"+std::to_string(out0_lmem_bytes));
  int64_t out_lmem_bytes = align_64(out0_lmem_bytes);
  for (int i = 1; i < outs.size(); i++) {
    info = tensor_infos[outs[i]].slice_info;
    int64_t tmp = align_64(getTensorLmemBytes(outs[i], info, ncdhw_idx, lg_info.type));
    out_lmem_bytes += tmp;
    llvm::errs() << "out"<<i<< ", size:"<<tmp<<"\n";
    dot_graph_log->add_node_label(op_name + "_ori", "out_lmem_bytes:"+std::to_string(tmp));
  }

  auto lg_op = cast<LocalGenInterface>(op);
  buffer_size = lg_op.getBufferSize(in0_lmem_bytes, out0_lmem_bytes, in_n,
                                    in_c, in_h, in_d, in_w, out_n, out_c,
                                    out_h, out_d, out_w, lg_info.type);
  dot_graph_log->add_node_label(op_name + "_ori", "buffer_size:"+std::to_string(buffer_size));
  return backend::Arch::LMEM_BYTES - in_lmem_bytes - out_lmem_bytes - align_64(buffer_size);
}


bool stripe_mine_max_slice(const LgInfo &lg_info,
                           const shape_secs_t &shape_secs,
                           TensorInfo &tensor_infos) {
  if (lg_info.group_ops.size() == 1 && false == LgPass::OPTIONS.group_by_cores) {
    return true;
  }
  tensor_infos.clear();

  int64_t n, c, d, h, w;
  int64_t max_nslice = 0, max_cslice = 0;
  int64_t max_dslice = 0, max_hslice = 0, max_wslice = 0;
  std::list<Value> tensor_branchs;
  std::multiset<Operation *> op_set;
  ValueSet out_tensor_set;
  slice_info_t si;
  for (auto out : lg_info.group_outs) {
    module::getNCDHW(out, n, c, d, h, w, lg_info.type);
    max_nslice = std::max(max_nslice, ceiling_func(n, shape_secs.nsecs));
    if (Arch::ALIGN_4N) {
      auto stype = module::getStorageType(out);
      int64_t align_n = 32 / stype.getIntOrFloatBitWidth();
      max_nslice = align_up(max_nslice, align_n);
    }
    max_dslice = ceiling_func(d, shape_secs.dsecs); // ?no max?
    max_hslice = ceiling_func(h, shape_secs.hsecs);
    max_wslice = ceiling_func(w, shape_secs.wsecs);
    max_cslice = align_up(ceiling_func(c, shape_secs.csecs), Arch::NPU_NUM);
    si.n.clear();
    si.h.clear();
    si.d.clear();
    si.w.clear();
    si.c.clear();
    si.n.emplace_back(slice_pair_t(0, max_nslice));
    si.h.emplace_back(slice_pair_t(0, max_hslice));
    si.d.emplace_back(slice_pair_t(0, max_dslice));
    si.w.emplace_back(slice_pair_t(0, max_wslice));
    si.c.emplace_back(slice_pair_t(0, max_cslice));
    tensor_infos[out] = tensor_info_t(si);

    out_tensor_set.insert(out);
    if (strip_back_judge(out, lg_info, op_set, out_tensor_set)) {
      tensor_branchs.push_back(out);
    }
  }

  bool ret = false;
  while (!tensor_branchs.empty()) {
    auto out_tensor = tensor_branchs.front();
    tensor_branchs.pop_front();
    ret = backward_update_slice(lg_info, shape_secs, out_tensor, tensor_branchs,
                                tensor_infos, op_set, out_tensor_set);
    if (!ret) {
      return false;
    }
  }

  //  if (check_tensor_slice(tensor_infos) == false) {
  //    return false;
  //  }

  return true;
}

bool stripe_mine_idx_slice(const LgInfo &lg_info,
                           const shape_secs_t &shape_secs,
                           TensorInfo &tensor_infos) {
  if (lg_info.group_ops.size() == 1  && false == LgPass::OPTIONS.group_by_cores) {
    return true;
  }
  tensor_infos.clear();

  int64_t n, c, d, h, w;
  std::list<Value> tensor_branchs;
  std::multiset<Operation *> op_set;
  std::set<Value, value_compare> out_tensor_set;
  for (auto out : lg_info.group_outs) {
    module::getNCDHW(out, n, c, d, h, w, lg_info.type);
    auto istype = module::getStorageType(lg_info.group_ins[0]);
    auto ostype = module::getStorageType(out);
    int64_t bitwidth = std::min(istype.getIntOrFloatBitWidth(),
                                ostype.getIntOrFloatBitWidth());
    auto si = get_out_slice_info(shape_secs, n, c, h, d, w, bitwidth);

    tensor_infos[out] = tensor_info_t(si);
    out_tensor_set.insert(out);
    if (strip_back_judge(out, lg_info, op_set, out_tensor_set)) {
      tensor_branchs.push_back(out);
    }
  }

  bool ret = false;
  while (!tensor_branchs.empty()) {
    auto out_tensor = tensor_branchs.front();
    tensor_branchs.pop_front();
    ret = backward_update_slice(lg_info, shape_secs, out_tensor, tensor_branchs,
                                tensor_infos, op_set, out_tensor_set);
    if (!ret) {
      return false;
    }
  }

  return true;
}


bool stripe_mine_idx_slice2(const LgInfo &lg_info,
                           const shape_secs_t &shape_secs,
                           TensorInfo &tensor_infos, Operation*& fail_op) {
  if (lg_info.group_ops.size() == 1) {
    return true;
  }
  tensor_infos.clear();

  int64_t n, c, d, h, w;
  std::list<std::pair<Value, Operation*>> tensor_branchs;
  std::multiset<Operation *> op_set;
  std::set<Value, value_compare> out_tensor_set;
  for (auto out : lg_info.group_outs) {
      module::getNCDHW(out, n, c, d, h, w, lg_info.type);
      auto istype = module::getStorageType(lg_info.group_ins[0]);
      auto ostype = module::getStorageType(out);
      int64_t bitwidth = std::min(istype.getIntOrFloatBitWidth(),
                                  ostype.getIntOrFloatBitWidth());
      auto si = get_out_slice_info(shape_secs, n, c, h, d, w, bitwidth);

      tensor_infos[out] = tensor_info_t(nullptr, si);
      out_tensor_set.insert(out);
      tensor_branchs.push_back(std::make_pair(out, nullptr));
  }

  bool ret = false;
  while (!tensor_branchs.empty()) {
    auto out_tensor = tensor_branchs.front();
    tensor_branchs.pop_front();
    ret = backward_update_slice2(lg_info, shape_secs, out_tensor, tensor_branchs,
                                tensor_infos, op_set, out_tensor_set);
    if (!ret) {
      fail_op = out_tensor.first.getDefiningOp();
      return false;
    }
  }

  return true;
}

bool backward_gen_ilp_var2(const LgInfo &lg_info,
                           const shape_secs_t &shape_secs,
                           TensorInfo &tensor_infos, std::shared_ptr<CycleCalculator> cycle_calculator_, ILPTimeStep& ilp_timeStep,
                           const std::vector<int64_t>& ncdhw_idx, int slice_idx,
                           std::vector<op_var_pos_info>& op_var_bound, Operation* returnOp, int64_t& load_bytes_for_next_ts,
                           std::shared_ptr<dot_graph> dot_graph_log, std::vector<std::pair<Value, int64_t>> tmp_value_size, Operation*& failOp,
                           bool l2m_en, bool last_slice, bool train, int max_ahead_or_delay_ts) {
  auto ops = lg_info.group_ops;
  assert (ops.size() > 1);
  std::string slice_name = llvm::formatv("slice_{0}_{1}_{2}_{3}_{4}",
            ncdhw_idx[0], ncdhw_idx[1], ncdhw_idx[2], ncdhw_idx[3], ncdhw_idx[4]);
  for (int cur_op_idx = ops.size() - 1; cur_op_idx >= 0; cur_op_idx--) {
    auto op = ops[cur_op_idx];
    auto op_name = replaceChars_for_dot(module::getName(op).str());
    auto var_pos_info = findVarBound(op_var_bound, std::make_pair(slice_idx, cur_op_idx));
    auto slice_pos_info = findVarBound(op_var_bound, std::make_pair(slice_idx, 0));
    llvm::errs() << "-------------------cur_op_idx: "<<cur_op_idx<< ", op: "<<op_name<<" ----\n";

    auto dot_node_name = op_name+"_slice"+std::to_string(slice_idx);
    dot_graph_log->add_node_label(dot_node_name, "op_idx:"+std::to_string(cur_op_idx));
    dot_graph_log->add_node_label(op_name + "_ori", "op_idx:"+std::to_string(cur_op_idx));
    dot_graph_log->add_node_label(dot_node_name, "ts_idx:"+std::to_string(var_pos_info.ts_id));
    dot_graph_log->add_node_label(op_name + "_ori", "ts_idx:"+std::to_string(var_pos_info.ts_id));

    int buffer_size = 0;
    int64_t mem_size_for_load = getOpLmemBytes(op, tensor_infos, ncdhw_idx, lg_info, buffer_size, dot_graph_log);
    dot_graph_log->add_node_label(op_name + "_ori", "mem_size_for_load:"+std::to_string(mem_size_for_load));
    if (mem_size_for_load - load_bytes_for_next_ts < 0) {
      failOp = op;
      llvm::errs() << "error, mem_size_for_load: "<<mem_size_for_load<< ", load_bytes_for_next_ts: "<<load_bytes_for_next_ts<<"\n";
      return false;
    }
    int bdc_cycle = cycle_calculator_->getLocalLayerCycle(op, tensor_infos, lg_info.type, true);//todo ?��?��1��?����?slice0��?��|???��?Y����?������
    llvm::errs() << "mem_size_for_load: "<<mem_size_for_load<< ", buffer_size: "<<buffer_size<<"\n";
    ilp_timeStep.addOpInfo(var_pos_info.ts_id, op, buffer_size, mem_size_for_load, bdc_cycle);
    dot_graph_log->add_node_label(op_name + "_ori",
      "buffer_size:"+std::to_string(buffer_size) + ", bdc_cycle:"+std::to_string(bdc_cycle));

    load_bytes_for_next_ts = 0;
    for (OpOperand &opd : op->getOpOperands()) {
      int opd_idx = opd.getOperandNumber();
      auto in = op->getOperand(opd_idx);
      auto inOp = in.getDefiningOp();
      if (inOp && isa<top::NoneOp>(inOp)) {
        continue;
      }
      int64_t lmem_bytes = 0;
      bool is_weight = false;
      if (!is_value_weight(in)) {
        lmem_bytes = getTensorLmemBytes(in, tensor_infos[in].slice_info, ncdhw_idx, lg_info.type);
      } else {
        lmem_bytes = align_64(Arch::get_weight_lmem_bytes(in, lg_info.type, false));
        is_weight = true;
      }
      std::string name = module::getName(in).str();
      auto itr = std::find(ops.begin(), ops.end(), inOp);//?�̨�����??D������?��?��??�騪?������?��?data��?��?value��??����?op?anull��?��?????��??����??��
      if (itr == ops.end()) { //D����a�䨮group��a??load��??��???������??
        tensor_info_t& info = tensor_infos[in];
        info.mode2 |= TIMESTEP2_LOAD;
        ilp_timeStep.addTensorSize(var_pos_info.ts_id, in, lmem_bytes);
        load_bytes_for_next_ts += lmem_bytes;

        int dma_cycle = 0;
        if (l2m_en && is_weight) {
          dma_cycle = cycle_calculator_->getGdmaCycle(in, info, lg_info.type)/4;
        } else {
          dma_cycle = cycle_calculator_->getGdmaCycle(in, info, lg_info.type);
        }
        ilp_timeStep.addTensorCycle(var_pos_info.ts_id, in, dma_cycle);
        std::vector<std::string> var_names;
        int ts_idx = var_pos_info.start_ts;
        bool small_tensor = false;
        for (auto it: tmp_value_size) {
          if (it.first == in) {
            small_tensor = true;
            break;
          }
        }
        for (; ts_idx < var_pos_info.ts_id; ts_idx++) {
          if (var_pos_info.ts_id - ts_idx > max_ahead_or_delay_ts) {
            continue;
          }
          std::string var_name;
          if (is_weight) {
            var_name = llvm::formatv("x_weight_{0}_atpos{1}_for{2}_load_{3}bytes_at_ts{4}_{5}",
                              name.c_str(), var_pos_info.ts_id, op_name.c_str(), lmem_bytes, ts_idx, slice_name.c_str()).str();
          } else {
            var_name = llvm::formatv("x_grp_input_{0}_atpos{1}_for{2}_load_{3}bytes_at_ts{4}_{5}",
                              name.c_str(), var_pos_info.ts_id, op_name.c_str(), lmem_bytes, ts_idx, slice_name.c_str()).str();
          }
          ilp_timeStep.addBinaryVar(ts_idx, slice_idx, -1, var_name, in, info, lmem_bytes);
          var_names.push_back(var_name);
          ilp_timeStep.addTimestepGdmaCycle(ts_idx, dma_cycle, var_name);
          ilp_timeStep.addTimestepMemUse(ts_idx, lmem_bytes, var_names);
          dot_graph_log->add_node_label(op_name + "_ori", "var_name:"+var_name);
        }

        if (small_tensor) { //1???������?��?D?����??����?��?��?������???��??��??����D��?������?can_merge��??����������??����o?
          int var_size = var_names.size();
          if (var_size > 3) {
            for (int i = 0; i < 2; i++) {
              // ilp_timeStep.setVarExpectValue(var_names[var_size -i -1], 0);
            }
          }
        }
        if (is_weight) {
          ilp_timeStep.addRowConstraint(var_pos_info.ts_id, in, var_names);
        } else {
          ilp_timeStep.addRowConstraint(-1, in, var_names);
        }
      } else {
        int producer_pos = slice_pos_info.ts_id + std::distance(ops.begin(), itr);
        if (producer_pos != var_pos_info.ts_id - 1) {
          load_bytes_for_next_ts += lmem_bytes;
        }
      }
    }

    std::vector<std::pair<int, MPVariable*>> coeff_var_items;
    for (int j = 0; j < op->getNumResults(); j++) {
      auto res = op->getResult(j);
      std::string name = module::getName(res).str();
      llvm::errs() << "process res name:"<<name<<"\n";
      int64_t lmem_bytes = getTensorLmemBytes(res, tensor_infos[res].slice_info, ncdhw_idx, lg_info.type);
      llvm::errs() << "addTensorSize,name:"<<name<<", lmem_bytes: "<<lmem_bytes<<"\n";
      assert(lmem_bytes > 0);
      ilp_timeStep.addTensorSize(var_pos_info.ts_id, res, lmem_bytes);
      std::map<int, Operation*> map_user_pos;
      bool have_grp_out = false;
      for (auto user: res.getUsers()) {
        auto itr = std::find(ops.begin(), ops.end(), user);
        if (itr != ops.end()) {
          int consumer_pos = slice_pos_info.ts_id + std::distance(ops.begin(), itr);
          map_user_pos[consumer_pos] = user;
        } else {
          have_grp_out = true;
        }
      }

      int pre_user_pos = var_pos_info.ts_id, dma_cycle;
      std::vector<std::string> all_store_varnames;
      std::string ada_var_name;
      MPVariable* x = nullptr;
      int idx = 0;
      bool first_user = true;
      if (map_user_pos.size() > 0) {
        std::map<MPVariable*, std::vector<std::pair<int, MPVariable*>>> map_x_var_items;
        for (auto it: map_user_pos) {
          llvm::errs() << "process "<<idx<<"th user\n";
          auto user_pos = it.first;
          auto user = it.second;
          auto user_name = module::getName(user).str();
          //?����??����|���訢?��?��?
          ada_var_name = llvm::formatv("ada_var_for_{0}_atpos{1}_for_user{1}", name.c_str(), var_pos_info.ts_id, idx);
          x = ilp_timeStep.solver->MakeIntVar(0, 1, ada_var_name);
          map_x_var_items[x] = std::vector<std::pair<int, MPVariable*>>();
          //op??��D1????����?��?user?��resnet2D2?��??����??��??��D2??������2??op�ꡧopo��?����|user??op2??������2??op��?2???��?????��??t
          if (user_pos - pre_user_pos >= 4) {
            //?����?store��?��?
            tensor_info_t& info = tensor_infos[res];
            info.mode2 |= TIMESTEP2_STORE_AND_LOAD;
            int full_slice_bytes = getTensorLmemBytes(res, info.slice_info, ncdhw_idx, lg_info.type);
            dma_cycle = cycle_calculator_->getGdmaCycle(res, info, lg_info.type); //nullptr, 0
            if (first_user) {
              ilp_timeStep.addTensorSize(var_pos_info.ts_id, res, full_slice_bytes);
              // ilp_timeStep.addTensorCycle(var_pos_info.ts_id, res, dma_cycle);
              llvm::errs() << "full_slice_bytes:"<<full_slice_bytes<< ", dma_cycle:"<<dma_cycle<<"\n";
            }
            std::vector<std::string> var_names;
            std::vector<std::pair<std::string, int>>  store_var_names;
            llvm::errs() << "define store_var, ts_idx from "<<pre_user_pos + 1<< " to "<<user_pos<<"\n";
            for (int ts_idx = pre_user_pos + 1; ts_idx < user_pos; ts_idx++) {
              std::string var_name = llvm::formatv("x_tensor_{0}_atpos{1}_store_{2}byte_at_ts{3}_{4}_for_consumer",
                                    name.c_str(), var_pos_info.ts_id, full_slice_bytes, ts_idx, slice_name.c_str());
              var_names.push_back(var_name);
              if (var_names.size() >= max_ahead_or_delay_ts) {
                break;
              }
            }
            for (int ts_idx = pre_user_pos + 1, offset = 0; ts_idx < user_pos; ts_idx++) {
              std::string var_name = llvm::formatv("x_tensor_{0}_atpos{1}_store_{2}byte_at_ts{3}_{4}_for_consumer",
                                name.c_str(), var_pos_info.ts_id, full_slice_bytes, ts_idx, slice_name.c_str());
              ilp_timeStep.addBinaryVar(ts_idx, slice_idx, 0, var_name, res, info, full_slice_bytes);
              ilp_timeStep.addTimestepGdmaCycle(ts_idx, dma_cycle, var_name);
              std::vector<std::string> var_names2;
              for (int n = offset++; n < var_names.size(); n++) {
                var_names2.push_back(var_names[n]);
              }
              store_var_names.push_back(std::make_pair(var_name, ts_idx));
              all_store_varnames.push_back(var_name);
              ilp_timeStep.addTimestepMemUse(ts_idx, lmem_bytes, var_names2);
              dot_graph_log->add_node_label(op_name + "_ori", "var_name:"+var_name);
              if (!first_user) {
                for (auto itr = map_x_var_items.begin(); itr != map_x_var_items.end(); ++itr) {
                  itr->second.push_back(std::make_pair(1, ilp_timeStep.getMPVarByName(var_name)));
                }
              }
              if (offset >= max_ahead_or_delay_ts) {
                break;
              }
            }

            //?����?load��?��?
            dma_cycle = cycle_calculator_->getGdmaCycle(res, info, lg_info.type, user, 1);
            ilp_timeStep.addTensorCycle(user_pos, res, dma_cycle);
            var_names.clear();
            std::vector<std::pair<std::string, int>>  load_var_names;
            llvm::errs() << "define load_var, ts_idx from "<<cur_op_idx + 3<< " to "<<user_pos<<"\n";
            for (int ts_idx = cur_op_idx + 3; ts_idx < user_pos; ts_idx++) {
              if (user_pos - ts_idx > max_ahead_or_delay_ts) {
                continue;
              }
              std::string var_name = llvm::formatv("x_producer_tensor_{0}_atpos{1}_for{2}_load_{3}bytes_at_ts{4}_{5}",
                          name.c_str(), user_pos, user_name.c_str(), full_slice_bytes, ts_idx, slice_name.c_str());
              ilp_timeStep.addBinaryVar(ts_idx, slice_idx, 1, var_name, res, info, full_slice_bytes);
              var_names.push_back(var_name);
              load_var_names.push_back(std::make_pair(var_name, ts_idx));
              ilp_timeStep.addTimestepGdmaCycle(ts_idx, dma_cycle, var_name);
              ilp_timeStep.addTimestepMemUse(ts_idx, full_slice_bytes, var_names);
              dot_graph_log->add_node_label(op_name + "_ori", "var_name:"+var_name);
            }

            std::set<int> ada_var_pos_set;
            for (auto load_var: load_var_names) {
              ada_var_pos_set.insert(load_var.second);
            }
            for (auto store_var: store_var_names) {
              ada_var_pos_set.insert(store_var.second);
            }
            for (auto pos: ada_var_pos_set) {
              ilp_timeStep.mem_contrains[pos].push_back(std::make_pair(full_slice_bytes, ada_var_name));
            }

            //?����?���訢?��?��?o��?��??store/load��?��?1??��
            coeff_var_items.clear();
            for (auto load_var: load_var_names) {
              coeff_var_items.push_back(std::make_pair(1, ilp_timeStep.getMPVarByName(load_var.first)));
            }
            for (auto store_var: store_var_names) {
              coeff_var_items.push_back(std::make_pair(-1, ilp_timeStep.getMPVarByName(store_var.first)));
            }
            ilp_timeStep.addConstraint(0, 0, coeff_var_items); //?����?sum(store_var) == sum(load_var)
            dot_graph_log->add_node_label(op_name + "_ori", "sum(store_var) == sum(load_var)");

            for (auto itr = map_x_var_items.begin(); itr != map_x_var_items.end(); ++itr) {
              itr->second.push_back(std::make_pair(-1, itr->first));
              ilp_timeStep.addConstraint(0, 0, itr->second);
              dot_graph_log->add_node_label(op_name + "_ori",  itr->first->name() + "ada_var == sum(next_ts_stores)");
            }

            coeff_var_items.clear();
            for (auto store_var: store_var_names) {
              coeff_var_items.push_back(std::make_pair(1, ilp_timeStep.getMPVarByName(store_var.first)));
            }
            coeff_var_items.push_back(std::make_pair(1, x));
            ilp_timeStep.addConstraint(1, 1, coeff_var_items); //?����?sum(store_var) + ada_var = 1
            dot_graph_log->add_node_label(op_name + "_ori", "sum(store_var) + ada_var = 1");

            coeff_var_items.clear();
            for (auto load_var: load_var_names) {
              coeff_var_items.push_back(std::make_pair(load_var.second, ilp_timeStep.getMPVarByName(load_var.first)));
            }
            for (auto store_var: store_var_names) {
              coeff_var_items.push_back(std::make_pair(-1*store_var.second, ilp_timeStep.getMPVarByName(store_var.first)));
            }
            coeff_var_items.push_back(std::make_pair(2, x));
            //?����?2*ada_var + sum(pos*load_var) - sum(pos*store_var) >= 2
            ilp_timeStep.addConstraint(2, MPSolver::infinity(), coeff_var_items);
            dot_graph_log->add_node_label(op_name + "_ori", "2*ada_var + sum(pos*load_var) - sum(pos*store_var) >= 2");
          } else {
            if (idx != map_user_pos.size() - 1) {
              ilp_timeStep.resideOpInValue(user, res);
            }
          }
          pre_user_pos = user_pos;
          idx++;
          first_user = false;
        }

        tensor_info_t info;
        info.mode2 = TIMESTEP2_STORE_ONLY_FREE;
        ts_var_t tmp;
        tmp.varName = ada_var_name;
        tmp.value = res;
        tmp.info = info;
        tmp.slice_idx = slice_idx;
        int stay_mem_free_ts_idx = map_user_pos.rbegin()->first;
        ilp_timeStep.timestep_table_[stay_mem_free_ts_idx].vec_op_infos[0].ada_var_for_free_mem.push_back(tmp);
        ilp_timeStep.addValueInfo(slice_idx, res, ada_var_name);

        ilp_var_info var_info;
        var_info.ilp_var = x; //todo ????��???2?D����a?
        var_info.tensor_info = info;
        var_info.slice_idx = slice_idx;
        var_info.ts_idx = stay_mem_free_ts_idx;
        ilp_timeStep.mapILPVarInfo[ada_var_name] = var_info;

        if (all_store_varnames.size() > 0) { //��Dstore��?��?��??3��?��|����3?��?store grp out
          if (have_grp_out) {//3?��?��Dstore grp out��?����???������1??store��?��??a1��?��?o��????��???3?��?store grp out��|����
            coeff_var_items.clear();
            for (auto store_var: all_store_varnames) {
              coeff_var_items.push_back(std::make_pair(1, ilp_timeStep.getMPVarByName(store_var)));
            }
            ilp_timeStep.addConstraint(1, 1, coeff_var_items);
            have_grp_out = false;
          } else {
            //3?��??Tstore��?grp out,D����a��?3?��?grp out��?������?
            ilp_timeStep.addNewOutIntoReturnOp(all_store_varnames, res);
          }
        }
      }

      if (have_grp_out) {
          //��???���????����??����2��������??��o?2��o����?3?��?����store��?��DD?����D����y?Y��?�����㨢��?ddr����?��?��?2???��?��
          slice_info_t& slice_info = tensor_infos[res].slice_info;
          tensor_info_t& info = tensor_infos[res];
          info.mode2 |= TIMESTEP2_STORE;
          int64_t lmem_bytes = getTensorLmemBytes(res, slice_info, ncdhw_idx, lg_info.type);
          int dma_cycle = cycle_calculator_->getGdmaCycle(res, info, lg_info.type);
          // ilp_timeStep.addTensorCycle(var_pos_info.ts_id, res, dma_cycle);
          std::vector<std::string> var_names;
          // ilp_timeStep.addTensorSize(var_pos_info.ts_id, res, lmem_bytes);
          std::string slice_name = llvm::formatv("slice_{0}_{1}_{2}_{3}_{4}",
                    ncdhw_idx[0], ncdhw_idx[1], ncdhw_idx[2], ncdhw_idx[3], ncdhw_idx[4]);
          int ts_idx = var_pos_info.ts_id + 1;
          int ts_idx2 = ts_idx;
          for (; ts_idx < var_pos_info.end_ts + 1; ts_idx++) {
            std::string var_name = llvm::formatv("x_tensor_{0}_atpos{1}_store_{2}byte_at_ts{3}_{4}",
                                                name.c_str(), var_pos_info.ts_id, lmem_bytes, ts_idx, slice_name.c_str());
            var_names.push_back(var_name);
            if (var_names.size() >= max_ahead_or_delay_ts) {
              break;
            }
          }
          ts_idx = ts_idx2;
          for (int offset = 0; ts_idx < var_pos_info.end_ts + 1; ts_idx++) {
            std::string var_name = llvm::formatv("x_tensor_{0}_atpos{1}_store_{2}byte_at_ts{3}_{4}",
                                                name.c_str(), var_pos_info.ts_id, lmem_bytes, ts_idx, slice_name.c_str());
            ilp_timeStep.addBinaryVar(ts_idx, slice_idx, -1, var_name, res, info, lmem_bytes);
            ilp_timeStep.addTimestepGdmaCycle(ts_idx, dma_cycle, var_name);
            dot_graph_log->add_node_label(op_name + "_ori", "var_name: "+var_name);
            std::vector<std::string> var_names2;
            for (int n = offset++; n < var_names.size(); n++) {
              var_names2.push_back(var_names[n]);
            }
            ilp_timeStep.addTimestepMemUse(ts_idx, lmem_bytes, var_names2);
            if (offset >= max_ahead_or_delay_ts) {
              break;
            }
          }
          if (var_names.size() > 0) {
            ilp_timeStep.addRowConstraint(-1, res, var_names);
          }
      }
    }
  }
  return true;
}

void get_max_slice_nchdw(const slice_info_t &slice_info, int64_t &max_nslice,
                         int64_t &max_cslice, int64_t &max_hslice,
                         int64_t &max_dslice, int64_t &max_wslice) {
  max_nslice = 0;
  max_cslice = 0;
  max_hslice = 0;
  max_dslice = 0;
  max_wslice = 0;
  for (auto &slice : slice_info.n) {
    max_nslice = std::max(max_nslice, slice.second);
  }
  for (auto &slice : slice_info.c) {
    max_cslice = std::max(max_cslice, slice.second);
  }
  for (auto &slice : slice_info.h) {
    max_hslice = std::max(max_hslice, slice.second);
  }
  for (auto &slice : slice_info.d) {
    max_dslice = std::max(max_dslice, slice.second);
  }
  for (auto &slice : slice_info.w) {
    max_wslice = std::max(max_wslice, slice.second);
  }
}

std::vector<slice_pair_t>
get_max_slice_nchdw_and_idx(const slice_info_t &slice_info, int64_t &max_nslice,
                            int64_t &max_cslice, int64_t &max_hslice,
                            int64_t &max_dslice, int64_t &max_wslice) {

  std::vector<slice_pair_t> slice_idx;
  max_nslice = 0;
  max_cslice = 0;
  max_hslice = 0;
  max_dslice = 0;
  max_wslice = 0;
  int n_idx = 0;
  int c_idx = 0;
  int h_idx = 0;
  int d_idx = 0;
  int w_idx = 0;
  for (int i = 0; i < slice_info.n.size(); i++) {
    if (slice_info.n[i].second > max_nslice) {
      max_nslice = slice_info.n[i].second;
      n_idx = slice_info.n[i].first;
    }
  }
  slice_idx.push_back({n_idx, max_nslice});

  for (int i = 0; i < slice_info.c.size(); i++) {
    if (slice_info.c[i].second > max_cslice) {
      max_cslice = slice_info.c[i].second;
      c_idx = slice_info.c[i].first;
    }
  }
  slice_idx.push_back({c_idx, max_cslice});

  for (int i = 0; i < slice_info.h.size(); i++) {
    if (slice_info.h[i].second > max_hslice) {
      max_hslice = slice_info.h[i].second;
      h_idx = slice_info.h[i].first;
    }
  }
  slice_idx.push_back({h_idx, max_hslice});

  for (int i = 0; i < slice_info.d.size(); i++) {
    if (slice_info.d[i].second > max_dslice) {
      max_dslice = slice_info.d[i].second;
      d_idx = slice_info.d[i].first;
    }
  }
  slice_idx.push_back({d_idx, max_dslice});

  for (int i = 0; i < slice_info.w.size(); i++) {
    if (slice_info.w[i].second > max_wslice) {
      max_wslice = slice_info.w[i].second;
      w_idx = slice_info.w[i].first;
    }
  }
  slice_idx.push_back({w_idx, max_wslice});

  return slice_idx;
}

int64_t get_buffer_size(Value v, tensor_info_t &ti,
                        group_type_t group_type, Operation* owner_op) {
  int64_t buf_size = 0;
  int64_t n, c, d, h, w;
  module::getNCDHW(v, n, c, d, h, w, group_type);
  bool allow_split = false;
  if (module::isWeight(v)) {
    auto weight_op = dyn_cast<top::WeightOp>(v.getDefiningOp());
    if (weight_op.getAllowSplit() != std::nullopt) {
      allow_split = true;
    }
  }
  if (module::isWeight(v) && allow_split == false) {
    if (group_type == GROUP_SMALL_C) {
      buf_size = Arch::get_tensor_lmem_bytes(v, n, c, d, h, w, ti.eu_align);
    } else {
      buf_size = Arch::get_weight_lmem_bytes(v, group_type, ti.eu_align);
    }
  } else if (module::isDynWeight(v)) { // TODO: need check
    buf_size = Arch::get_weight_lmem_bytes(v, group_type, ti.eu_align);
  } else {
    int64_t nslice, cslice, hslice, dslice, wslice;
    auto si = ti.slice_info;
    if (owner_op) {
      si = ti.slice_infos[owner_op];
    }
    get_max_slice_nchdw(si, nslice, cslice, hslice, dslice, wslice);
    buf_size = Arch::get_tensor_lmem_bytes(v, nslice, cslice, hslice, dslice,
                                           wslice, group_type, ti.eu_align);
  }
  return buf_size;
}

void set_fake_local_layer_param(Operation *op, int64_t nidx, int64_t nslice,
                                int64_t cidx, int64_t cslice, int64_t hidx,
                                int64_t hslice, int64_t didx, int64_t dslice,
                                int64_t widx, int64_t wslice) {
  auto ctx = op->getContext();
  auto builder = OpBuilder(ctx);
  int64_t group_type = 0;
  auto lg_attr = LayerGroupAttr::get(
      ctx, 0, 0, 0, 0, true, false, builder.getDenseI64ArrayAttr({}), builder.getDenseI64ArrayAttr({nidx}),
      builder.getDenseI64ArrayAttr({nslice}),
      builder.getDenseI64ArrayAttr({cidx}),
      builder.getDenseI64ArrayAttr({cslice}),
      builder.getDenseI64ArrayAttr({didx}),
      builder.getDenseI64ArrayAttr({dslice}),
      builder.getDenseI64ArrayAttr({hidx}),
      builder.getDenseI64ArrayAttr({hslice}),
      builder.getDenseI64ArrayAttr({widx}),
      builder.getDenseI64ArrayAttr({wslice}), 0, 0, 0, group_type);
  op->setAttr(LocalGenInterface::kLayerGroupAttrName, lg_attr);
}

// MatMul weight split case
// case1 : [4, 5, 6] * [4, 6, 7] = [4, 5, 7]  => batch = 4, M = 5, k = 6, N = 7
// case2 : [3, 4, 5, 6] * [3, 4, 6, 7] => batch = 12, M = 5, K = 6, N = 7
// case3 :matmul attrs: r_trans=true, hdim_is_batch=true
//=>[1, 16, 320, 64] * [1, 320, 320, 64] => [1, 16, 320, 64] * [1, 64, 320, 320] => batch =320 , M = 16, K = 64, N = 320
// other cases TODO
bool check_split_matmul(Operation *op) {
  if (!isa<tpu::MatMulOp>(op)) {
    return false;
  }
  auto matmulOp = dyn_cast<tpu::MatMulOp>(op);

  auto a_s = SmallVector<int64_t>(module::getShape(matmulOp.getInput()));
  auto b_s = SmallVector<int64_t>(module::getShape(matmulOp.getRight()));
  auto o_s = SmallVector<int64_t>(module::getShape(matmulOp.getOutput()));

  if (a_s.size() != b_s.size()) {
    return false;
  }

  // case 1
  if (a_s.size() == 3 && a_s[0] == b_s[0] && a_s[0] != 1 && a_s[2] == b_s[1]) {
    // if(a_s.size() == 3 && /*a_s[0] == b_s[0] && a_s[0] != 1 && */ a_s[2] ==
    // b_s[1]){
    return true;
  }

  // case 2
  if (a_s.size() == 4 && a_s[0] == b_s[0] && a_s[0] != 1 && a_s[1] == b_s[1] &&
      b_s[1] != 1 && a_s[3] == b_s[2]) {
    return true;
  }

  // case 3
  if (a_s.size() == 4 && a_s[0] == b_s[0]  && a_s[2] == b_s[2] &&
      a_s[3] == b_s[3]) {
    return true;
  }

  // other cases
  // TODO

  return false;
}

void set_weight_allow_split_attr(Operation *op) {
  auto ctx = op->getContext();
  auto builder = OpBuilder(ctx);
  if (isa<tpu::AddOp, tpu::SubOp, tpu::MulOp, tpu::DivOp, tpu::MinOp,
          tpu::MaxOp, tpu::CompareOp, tpu::ConcatOp, tpu::MatMulOp>(op) &&
      (module::isWeight(op->getOperand(0)) ||
       module::isWeight(op->getOperand(1)))) {

    if (isa<tpu::MatMulOp>(op) && !check_split_matmul(op)) {
      return;
    }
    top::WeightOp weight_op;
    if (module::isWeight(op->getOperand(0))) {
      weight_op = dyn_cast<top::WeightOp>(op->getOperand(0).getDefiningOp());
    } else if (module::isWeight(op->getOperand(1))) {
      weight_op = dyn_cast<top::WeightOp>(op->getOperand(1).getDefiningOp());
    }
    if (weight_op.getAllowSplit() != std::nullopt) {
      return;
    }
    auto out_shape = module::getShape(weight_op.getResult());
    std::vector<int64_t> AllowSplitVector(out_shape.size(), 1);
    weight_op.setAllowSplitAttr(builder.getI64ArrayAttr(AllowSplitVector));
  }
}

void delete_weight_allow_split_attr(Operation *op) {
  if (auto weight_op = dyn_cast<top::WeightOp>(op)) {
    if (weight_op.getAllowSplit() == std::nullopt) {
      return;
    }
    op->removeAttr("allow_split");
  }
}

void delete_fake_local_layer_param(Operation *op) {
  op->removeAttr(LocalGenInterface::kLayerGroupAttrName);
}

void generate_fake_global_addr(Operation *op) {
  int64_t offset = Arch::LMEM_BANK_BYTES;
  int64_t i = 0;
  auto ins = get_input_values(op);
  auto outs = get_output_values(op);
  for (auto in : ins) {
    module::setAddress(in, offset * i);
    ++i;
  }
  for (auto out : outs) {
    module::setAddress(out, offset * i);
    ++i;
  }
}

void delete_fake_global_addr(Operation *op) {

  auto ins = get_input_values(op);
  auto outs = get_output_values(op);
  for (auto in : ins) {
    auto type = in.getType().cast<RankedTensorType>();
    Builder builder(in.getContext());
    auto new_type =
        RankedTensorType::get(type.getShape(), type.getElementType());
    in.setType(new_type);
  }
  for (auto out : outs) {
    auto type = out.getType().cast<RankedTensorType>();
    Builder builder(out.getContext());
    auto new_type =
        RankedTensorType::get(type.getShape(), type.getElementType());
    out.setType(new_type);
  }
}

bool is_eu_align_cv18xx(Value opd) {
  auto op = *opd.user_begin();
  if (module::isWeight(opd)) {
    if (isa<tpu::LutOp>(op)) {
      if (opd == op->getOperand(1)) {
        return false;
      }
    } else if (isa<tpu::LutBF16Op>(op)) {
      if (opd == op->getOperand(1) || opd == op->getOperand(2)) {
        return false;
      }
    } else if (isa<tpu::ScaleLutOp>(op)) {
      if (opd == op->getOperand(1)) {
        return false;
      }
    } else if (isa<tpu::ScaleOp>(op)) {
      return false;
    } else if (auto castOp = dyn_cast<tpu::Conv2DOp>(op)) {
      auto attr = castOp.parseParam();
      if (opd == op->getOperand(1) && attr.is_dw) {
        // if is dw conv, filter need to align.
        return true;
      }
      if (module::isUniformQuantized(castOp.getOutput()) &&
          opd == op->getOperand(2) && !attr.is_dw && attr.groups > 1) {
        // if is int8 group conv, bias need to align.
        return true;
      }
      return false;
    } else if (auto castOp = dyn_cast<tpu::DeconvOp>(op)) {
      auto attr = castOp.parseParam();
      if (opd == op->getOperand(1) && attr.is_dw) {
        // if is dw conv, filter need to align.
        return true;
      }
      if (module::isUniformQuantized(castOp.getOutput()) &&
          opd == op->getOperand(2) && !attr.is_dw && attr.g > 1) {
        // if is int8 group conv, bias need to align.
        return true;
      }
      return false;
    } else if (isa<tpu::LayerNormOp>(op)) {
      return false;
    } else {
      return true; // prelu concat
    }
  }
  return true;
}

bool is_eu_align_bm168x(Value opd) {
  auto op = *opd.user_begin();

  if (module::isDynWeight(opd)) {
    return false;
  }

  if (module::isWeight(opd)) {
    if (isa<tpu::Conv2DOp, tpu::Conv3DOp, tpu::DeconvOp, tpu::GroupNormOp,
            tpu::LayerNormOp, tpu::PixelNormOp, tpu::InstanceNormOp>(op)) {
      if ((opd == op->getOperand(1) || opd == op->getOperand(2))) {
        return false;
      }
    } else if (isa<tpu::PReluOp, tpu::ScaleOp>(op)) {
      return false;
    } else if (module::isBM1688()) {
      if (isa<tpu::RequantIntAxisOp>(op)) {
        if ((opd == op->getOperand(1))) {
          return false;
        }
      }
    }
  }
  return true;
}

bool is_eu_align(Value opd) {
  // Eu align rule may be different in different platforms
  if (module::isCV18xx()) {
    return is_eu_align_cv18xx(opd);
  } else {
    return is_eu_align_bm168x(opd);
  }
}

bool is_value_weight(Value opd) {
  auto op = *opd.user_begin();
  if (isa<tpu::Conv2DOp, tpu::Conv3DOp, tpu::DeconvOp, tpu::GroupNormOp,
          tpu::LayerNormOp, tpu::PixelNormOp>(op)) {
    if ((opd == op->getOperand(1) || opd == op->getOperand(2))) {
      return true;
    }
  } else if (isa<tpu::PReluOp>(op)) {
    if (opd == op->getOperand(1)) {
      return true;
    }
  } else if (isa<tpu::ScaleOp>(op)) {
    if ((opd == op->getOperand(1) || opd == op->getOperand(2))) {
      return true;
    }
  } else if (module::isBM1688()) {
    if (isa<tpu::RequantIntAxisOp>(op)) {
      if ((opd == op->getOperand(1))) {
        return true;
      }
    }
  }
  return false;
}

bool need_bcast(Value opd) {
  if (opd.hasOneUse() == false) {
    return false;
  }
  auto use_op = *opd.user_begin();
  if (auto cast_op = dyn_cast<tpu::LutOp>(use_op)) {
    return opd == cast_op.getTable();
  } else if (auto cast_op = dyn_cast<tpu::LutBF16Op>(use_op)) {
    return opd == cast_op.getTable() || opd == cast_op.getMantissa();
  } else if (auto cast_op = dyn_cast<tpu::LRNOp>(use_op)) {
    return opd == cast_op.getTable() || opd == cast_op.getMantissa();
  } else if (auto cast_op = dyn_cast<tpu::LayerNormOp>(use_op)) {
    return module::isCV18xx() && module::isWeight(opd);
  } else {
    return false;
  }
}

int64_t use_3ic(Value opd) {
  for (auto use_op : opd.getUsers()) {
    if (auto cast_op = dyn_cast<tpu::Conv2DOp>(*use_op)) {
      if (opd == cast_op.getInput()) {
        return cast_op.getUse_3icOptimize();
      }
    }
  }
  return 0;
}

std::vector<Value> get_input_values(Operation *op) {
  auto value_vec = std::vector<Value>();
  for (auto in : op->getOperands()) {
    if (in.getType().isa<NoneType>()) {
      continue;
    }
    value_vec.push_back(in);
  }
  return std::move(value_vec);
}

std::vector<Value> get_output_values(Operation *op) {
  auto value_vec = std::vector<Value>();
  for (auto out : op->getResults()) {
    if (out.getType().isa<NoneType>()) {
      continue;
    }
    value_vec.push_back(out);
  }
  return std::move(value_vec);
}

} // namespace tpu
} // namespace tpu_mlir
