//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "ProfileCtx.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"

using namespace tpu_mlir::backend;
using namespace std;

namespace tpu_mlir {
namespace tpu {

template <typename T>
string array_to_string(const T *data, size_t len, const string &sep = " ",
                       const string &prefix = "[", const string &suffix = "]",
                       size_t max_len = 64) {
  if (len == 0)
    return prefix + suffix;
  std::ostringstream oss;
  oss << prefix;
  size_t real_len = len > max_len ? max_len : len;
  if (std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value) {
    for (size_t i = 0; i < real_len - 1; i++) {
      oss << std::setw(4) << data[i]
          << sep; // +: promote char to printable number
    }
    oss << std::setw(4)
        << data[real_len - 1]; // +: promote char to printable number
  } else {
    for (size_t i = 0; i < real_len - 1; i++) {
      oss << data[i] << sep;
    }
    oss << data[real_len - 1];
  }
  if (real_len != len) {
    oss << "...";
  }
  oss << suffix;
  return oss.str();
}

template string array_to_string<int64_t>(const int64_t *data, size_t len,
                                         const string &sep,
                                         const string &prefix,
                                         const string &suffix, size_t max_len);

ProfileCtx::ProfileCtx(AsmState::LocationMap *location, bool enable_profile) {
  opToLineCol = location;
  fake_tensor_id = -1;
  net_num = 0;
  cur_net_idx = 0;
  enable_profile_ = enable_profile;
}

int64_t ProfileCtx::get_tensor_id(Value value) {
  int64_t tensor_id = 0;
  auto src_op = value.getDefiningOp();
  if (src_op != nullptr) {
    if (isa<GroupOp>(src_op)) {
      auto groupOp = cast<GroupOp>(src_op);
      groupOp.walk([&](Operation *op) {
        if (isa<YieldOp>(op)) {
          int32_t pos = dyn_cast<OpResult>(value).getResultNumber();
          auto real_tensor = op->getOperand(pos);
          src_op = real_tensor.getDefiningOp();
        }
      });
    }
    auto it = opToLineCol->find(src_op);
    if (it != opToLineCol->end()) {
      tensor_id = it->second.first;
    }
  } else {
    tensor_id = (fake_tensor_id--);
  }
  return tensor_id;
}

void ProfileCtx::log_str(const char *fmt, ...) {
  if (!get_enable_profile()) {
    return;
  }
  FILE *fp = get_fp_profile();
  va_list params;
  va_start(params, fmt);
  vfprintf(fp, fmt, params);
  va_end(params);
}

bool isWeight(Value value) {
  if (auto op = dyn_cast_or_null<tpu::LoadOp>(value.getDefiningOp())) {
    return module::isWeight(op.getInput());
  }
  return module::isWeight(value);
}

void ProfileCtx::log_tensor(Value value, bool is_in, int64_t n_step,
                            int64_t h_step) {
  uint64_t gaddr = 0;
  size_t gsize = 0;
  uint32_t laddr = 0;
  uint32_t l2addr = 0;
  uint32_t n_slice = 0;
  uint32_t h_slice = 0;
  // uint32_t d_slice = 0;
  // uint32_t w_slice = 0;

  int64_t tensor_id = get_tensor_id(value);
  auto op = value.getDefiningOp();
  if (op != nullptr && op->hasAttr(LocalGenInterface::kLayerGroupAttrName)) {
    auto g_param = op->getAttr(LocalGenInterface::kLayerGroupAttrName)
                       .cast<tpu::LayerGroupAttr>();
    laddr = g_param.getOutAddr();
    auto n_slice_v = g_param.getNSlice();
    auto h_slice_v = g_param.getHSlice();
    if (!n_slice_v.empty()) {
      n_slice = n_slice_v.size() > n_step ? n_slice_v[n_step] : n_slice_v[0];
    }
    if (!h_slice_v.empty()) {
      h_slice = h_slice_v.size() > h_step ? h_slice_v[h_step] : h_slice_v[0];
    }
  } else {
    gaddr = module::getAddress(value);
  }

  auto dtype = BM168x::getDataType(value);
  auto shape_ = module::getShape(value);
  std::vector<int64_t> shape(shape_.size(), 1);
  for (size_t i = 0; i < shape.size(); ++i) {
    shape[i] = shape_[i];
  }

  auto shape_str = array_to_string(shape_.data(), shape_.size(), "x");

  log_str("[bmprofile] tensor_id=%d is_in=%d shape=%s dtype=%d is_const=%d "
          "gaddr=%lld gsize=%d loffset=%d nslice=%d hslice=%d l2addr=%d\n",
          tensor_id, is_in, shape_str.c_str(), dtype, isWeight(value), gaddr,
          gsize, laddr, n_slice, h_slice, l2addr);
}

void ProfileCtx::log_global_layer(Operation *op) {
  if (!get_enable_profile()) {
    return;
  }
  int64_t layer_id = 0;
  auto it = opToLineCol->find(op);
  if (it != opToLineCol->end()) {
    layer_id = it->second.first;
  }

  string layer_type = op->getName().getStringRef().str().substr(4);
  log_str(
      "\n[bmprofile] global_layer: layer_id=%d layer_type=%s layer_name=%s\n",
      layer_id, layer_type.c_str(), "");
  auto ins = get_input_values(op);
  auto outs = get_output_values(op);
  for (auto in : ins) {
    log_tensor(in, true);
  }
  for (auto out : outs) {
    log_tensor(out, false);
  }
}

void ProfileCtx::log_local_layer(Operation *op, int64_t n_step,
                                 int64_t h_step) {
  if (!get_enable_profile()) {
    return;
  }
  int64_t layer_id = 0;
  auto it = opToLineCol->find(op);
  if (it != opToLineCol->end()) {
    layer_id = it->second.first;
  }

  string layer_type = op->getName().getStringRef().str().substr(4);
  log_str("[bmprofile] local_layer: layer_id=%d layer_type=%s layer_name=%s\n",
          layer_id, layer_type.c_str(), "");
  auto ins = get_input_values(op);
  auto outs = get_output_values(op);
  for (auto in : ins) {
    log_tensor(in, true, n_step, h_step);
  }
  for (auto out : outs) {
    log_tensor(out, false, n_step, h_step);
  }
}

void ProfileCtx::set_profile_start(int subnet_id) {
  if (!get_enable_profile()) {
    return;
  }
  cur_net_idx = net_num++;
  std::string profile_path = "./net_";
  profile_path += std::to_string(cur_net_idx);
  profile_path += ".profile";

  FILE *fp = fopen(profile_path.c_str(), "w+");

  if (fp == nullptr) {
    llvm_unreachable("create profile file failed\n");
  }
  fp_profile.push_back(fp);
  BM168x::instance()->dl_enable_profile(get_enable_profile(), fp);

  fprintf(get_fp_profile(), "[bmprofile] is_mlir=1\n");
  fprintf(get_fp_profile(), "...Start Profile Log...\n");
  log_str("[bmprofile] start to run subnet_id=%d\n", subnet_id);
}

void ProfileCtx::set_profile_end(int subnet_id) {
  if (!get_enable_profile()) {
    return;
  }
  log_str("[bmprofile] end to run subnet_id=%d\n", subnet_id);
  if (fp_profile[cur_net_idx]) {
    fclose(fp_profile[cur_net_idx]);
    // disable profile to mitigate leaking, set "b_enable_profile" to false.
    BM168x::instance()->dl_enable_profile(false, fp_profile[cur_net_idx]);
  }
}

} // namespace tpu
} // namespace tpu_mlir
