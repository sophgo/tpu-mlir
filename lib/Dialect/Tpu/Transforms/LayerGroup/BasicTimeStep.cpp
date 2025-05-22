//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/TimeStepMethod.h"

namespace tpu_mlir {
namespace tpu {

using namespace tpu_mlir::tpu;
using namespace tpu_mlir::backend;

static inline void stream_tpu_field(const TpuTsField &field) {
  llvm::dbgs() << " [ ";
  for (int i = 0; i < field.size(); ++i) {
    if (i > 0)
      llvm::dbgs() << ", ";
    llvm::dbgs() << "C(\"" << field[i]->getName() << "\"), \""
                 << module::getName(field[i]) << "\"";
  }
  llvm::dbgs() << " ]";
}

static inline void stream_gdma_field(const GdmaTsField &field) {
  llvm::dbgs() << " [ ";
  for (int i = 0; i < field.size(); ++i) {
    auto mode = field[i].second.mode;
    auto hold_in_lmem = field[i].second.hold_in_lmem;
    auto modestr = "L";
    if (mode == TIMESTEP_STORE) {
      modestr = "S";
    }

    if (i > 0)
      llvm::dbgs() << ", ";
    std::string op_type = module::getOriValue(field[i].first)
                              .getDefiningOp()
                              ->getName()
                              .getStringRef()
                              .str();
    llvm::dbgs() << modestr << "(\"" << module::getName(field[i].first)
                 << "\", hold_in_lmem = " << hold_in_lmem << ")->" << op_type;
  }
  llvm::dbgs() << " ]";
}

BasicTimeStep::BasicTimeStep() {
  // options_ = options;
  swpipl_ = std::make_shared<SoftwarePipeline>();
  // timestep_method_ = std::make_shared<TimeStepMethod>(options);
  this->clear();
}

BasicTimeStep::BasicTimeStep(const LgOptions &options) {
  options_ = options;
  swpipl_ = std::make_shared<SoftwarePipeline>();
  timestep_method_ = std::make_shared<TimeStepMethod>(options);
  this->clear();
}

void BasicTimeStep::clear() {
  timestep_table_.clear();
  hold_coeff_.clear();
  lmem_buffer_.clear();
  lmem_occupy_ = 0;
  swpipl_stage_num_ = 1;
}

bool BasicTimeStep::assignTimeStep(const LgInfo &lg_info,
                                   const shape_secs_t &shape_secs,
                                   bool gen_idx) {
  clear();
  return timestep_method_->process(this, tensor_infos_, lg_info, shape_secs,
                                   gen_idx);
}

void BasicTimeStep::add_tpu0_ts_field(const TpuTsField &field) {
  TimestepRow row;
  row.tpu0_ts_field = field;
  timestep_table_.push_back(row);
  DEBUG_WITH_TYPE("timestep_assign", {
    llvm::dbgs() << "; action = add_tpu0_ts_field"
                 << "; ts = " << timestep_table_.size() - 1;

    stream_tpu_field(field);

    llvm::dbgs() << "\n";
  });
}

void BasicTimeStep::add_gdma0_ts_field(const GdmaTsField &field) {
  TimestepRow row;
  row.gdma0_ts_field = field;
  timestep_table_.push_back(row);

  DEBUG_WITH_TYPE("timestep_assign", {
    llvm::dbgs() << "; action = add_gdma0_ts_field"
                 << "; ts = " << timestep_table_.size() - 1;

    stream_gdma_field(field);

    llvm::dbgs() << "\n";
  });
}

void BasicTimeStep::add_tpu0_gdma0_ts_field(const TpuTsField &tpu_field,
                                            const GdmaTsField &gdma_field) {
  TimestepRow row;
  row.tpu0_ts_field = tpu_field;
  row.gdma0_ts_field = gdma_field;
  timestep_table_.push_back(row);

  DEBUG_WITH_TYPE("timestep_assign", {
    llvm::dbgs() << "; action = add_tpu0_gdma0_ts_field"
                 << "; ts = " << timestep_table_.size() - 1;

    stream_tpu_field(tpu_field);
    stream_gdma_field(gdma_field);

    llvm::dbgs() << "\n";
  });
}

void BasicTimeStep::update_gdma0_ts_field(int64_t ts,
                                          const GdmaTsField &field) {
  this->timestep_table_[ts].gdma0_ts_field.clear();
  this->timestep_table_[ts].gdma0_ts_field = field;

  DEBUG_WITH_TYPE("timestep_assign", {
    llvm::dbgs() << "; action = update_gdma0_ts_field"
                 << "; ts = " << ts;

    stream_gdma_field(field);

    llvm::dbgs() << "\n";
  });
}

void BasicTimeStep::show_timestep_table() {
  for (int i = 0; i < timestep_table_.size(); ++i) {
    llvm::dbgs() << "; ts = " << i << "; ";
    stream_tpu_field(timestep_table_[i].tpu0_ts_field);
    llvm::dbgs() << " || ";
    stream_gdma_field(timestep_table_[i].gdma0_ts_field);
    llvm::dbgs() << "\n";
  }
}

int64_t BasicTimeStep::get_layer_swpipl_stage(Operation *op) {
  return swpipl_stage_num_ == 1 ? 0 : 1;
}

int64_t BasicTimeStep::get_tensor_swpipl_stage(Value v) {
  if (swpipl_stage_num_ == 1)
    return 0;

  return swpipl_->get_tensor_swpipl_stage(v);
}

void BasicTimeStep::software_pipeline() {
  this->swpipl_stage_num_ =
      swpipl_->software_pipeline_schedule(this->timestep_table_);
  for (auto &row : timestep_table_) {
    for (auto &iter : row.gdma0_ts_field) {
      iter.second.stage = get_tensor_swpipl_stage(iter.first);
    }
  }
}

// lmem_buffer_debug_t
// start_ts, end_ts, size, name, op_type
// sort by following root
// 1. start_ts
// 2. end_ts
// 3. size
struct lmem_buffer_debug_t {
  int start_ts;
  int end_ts;
  size_t size;
  std::string name;
  std::string type;

  bool operator<(const lmem_buffer_debug_t &other) const {
    if (start_ts != other.start_ts)
      return start_ts < other.start_ts;
    if (end_ts != other.end_ts)
      return end_ts < other.end_ts;
    return size > other.size;
  }
};

void BasicTimeStep::debug_lmem_buffer() {
  llvm::dbgs() << "\n====================================\n";
  llvm::dbgs() << "== debug_lmem_buffer\n";
  mem_buffer_key_t key;
  mem_buffer_value_t value;
  size_t timestep_num = get_timestep_num();
  std::vector<std::vector<mem_buffer_key_t>> data(
      timestep_num, std::vector<mem_buffer_key_t>());
  std::vector<lmem_buffer_debug_t> lmem_buffer_vec;
  lmem_buffer_debug_t lmem_buffer_elt;
  for (auto &iter : lmem_buffer_) {
    key = iter.first;
    value = iter.second;
    if (module::isBM1684Family() && module::isWeight(key.value) &&
        llvm::any_of(key.value.getUsers(),
                     [](Operation *op) { return isa<tpu::LutOp>(op); })) {
      // 1684 LutOp use l2mem
      llvm::dbgs() << LOG_STEP("skip weight in l2mem for bm1684")
                   << LOG_KV("name", module::getName(key.value)) << "\n";
      continue;
    }
    bool first_step = true;
    lmem_buffer_elt.start_ts = value.start_ts;
    lmem_buffer_elt.end_ts = value.end_ts;
    lmem_buffer_elt.size = value.size;
    lmem_buffer_elt.name = module::getName(key.value).str();
    lmem_buffer_elt.type = key.lmem_type_str();
    lmem_buffer_vec.push_back(lmem_buffer_elt);
    for (int64_t ts = value.start_ts;
         ts != ((value.end_ts + 1) % timestep_num) || first_step;
         ts = (ts + 1) % timestep_num) {
      first_step = false;
      data[ts].push_back(key);
    }
  }

  for (size_t ts = 0; ts < timestep_num; ++ts) {
    llvm::dbgs() << "=== timestep = " << ts << "\n";
    llvm::dbgs() << "(name, size): ";
    int64_t total = 0;
    for (auto &iter : data[ts]) {
      value = lmem_buffer_[iter];
      total += value.size;
      llvm::dbgs() << "(" << module::getName(iter.value) << ", " << value.size
                   << "), ";
    }
    llvm::dbgs() << "\n"
                 << "total=" << total << "\n";
  }

  llvm::dbgs() << "====================================\n";

  std::sort(lmem_buffer_vec.begin(), lmem_buffer_vec.end());
  int cnt = 0;
  for (auto &iter : lmem_buffer_vec) {
    llvm::dbgs() << "=== lmem_buffer_vec = " << cnt++ << "(start_ts, end_ts): "
                 << "(" << iter.start_ts << ", " << iter.end_ts << ")"
                 << ", size: " << iter.size << ", name: " << iter.name
                 << ", type: " << iter.type << "\n";
  }
  llvm::dbgs() << "====================================\n";
}

void BasicTimeStep::show_timestep() {
  llvm::outs() << "============= show time step =============\n";
  size_t timestep_num = get_timestep_num();
  std::string s;
  llvm::raw_string_ostream ss(s);

  ValueIntMap value_ids;
  std::map<Operation *, int64_t> op_ids;
  int64_t idx = 0;
  for (size_t ts = 0; ts < timestep_num; ++ts) {
    auto &layer_field = getLayers(ts);
    for (auto op : layer_field) {
      if (op_ids.find(op) == op_ids.end()) {
        op_ids[op] = idx++;
      }

      for (auto in : op->getOperands()) {
        if (value_ids.find(in) == value_ids.end()) {
          value_ids[in] = idx++;
        }
      }
      for (auto out : get_output_values(op)) {
        if (value_ids.find(out) == value_ids.end()) {
          value_ids[out] = idx++;
        }
      }
    }
  }

  mem_buffer_key_t buffer_key;
  for (size_t ts = 0; ts < timestep_num; ++ts) {
    s.clear();
    ss << "=== timestep " << ts << ": \n";
    const auto &layer_field = getLayers(ts);
    for (auto op : layer_field) {
      ss << "layer " << op_ids[op] << "([";
      for (auto in : op->getOperands()) {
        if (in.getType().isa<NoneType>()) {
          continue;
        }
        buffer_key.value = in;
        if (dyn_cast_or_null<top::WeightOp>(in.getDefiningOp())) {
          buffer_key.type = LMEM_WEIGHT;
        } else {
          buffer_key.type = LMEM_ACTIVATION;
        }
        auto &buffer_value = get_lmem_buffer_value(buffer_key);
        ss << value_ids[in] << "(" << buffer_value.start_ts << ", "
           << buffer_value.end_ts << "), ";
      }
      ss << "] -> [";
      for (auto out : get_output_values(op)) {
        buffer_key.type = LMEM_ACTIVATION;
        buffer_key.value = out;
        auto &buffer_value = get_lmem_buffer_value(buffer_key);
        ss << value_ids[out] << "(" << buffer_value.start_ts << ", "
           << buffer_value.end_ts << "), ";
      }
      ss << "])\n";
    }

    const auto &tensor_field = getTensors(ts);
    ss << "tensor(start_ts, end_ts): ";
    for (auto &iter : tensor_field) {
      buffer_key.value = iter.first;
      if (dyn_cast_or_null<top::WeightOp>(iter.first.getDefiningOp())) {
        buffer_key.type = LMEM_WEIGHT;
      } else {
        buffer_key.type = LMEM_ACTIVATION;
      }
      auto &buffer_value = get_lmem_buffer_value(buffer_key);
      ss << value_ids[iter.first] << "(" << buffer_value.start_ts << ", "
         << buffer_value.end_ts << "), ";
    }
    ss << "\n";
    llvm::outs() << s;
  }
  llvm::outs() << "====================================\n";
}

void BasicTimeStep::gen_hold_coeff() {
  this->hold_coeff_.clear();

  Value v;
  for (size_t ts = 0; ts < this->get_timestep_num(); ++ts) {
    const GdmaTsField &gdma_field = this->timestep_table_[ts].gdma0_ts_field;
    for (size_t i = 0; i < gdma_field.size(); ++i) {
      v = gdma_field[i].first;
      auto &tensor_info = gdma_field[i].second;
      if (tensor_info.mode == TIMESTEP_LOAD) {
        if (auto src_op = dyn_cast_or_null<top::WeightOp>(v.getDefiningOp())) {
          bool allow_split = false;
          if (src_op.getAllowSplitAttr() != nullptr) {
            allow_split = true;
          }
          if (allow_split == false) {
            this->hold_coeff_[v] = ts;
          }
        }
        if (tensor_info.hold_in_lmem) {
          this->hold_coeff_[v] = ts;
        }
      }
    }
  }
}

// int64_t BasicTimeStep::get_start_timestep(Value v) {
//   for (auto &lmem_info : lmem_buffer_) {
//     if (lmem_info.first.value == v) {
//       return lmem_info.second.start_ts;
//     }
//   }
// }

void BasicTimeStep::gen_all_mem_buffer_ts() {
  // input: need_imm_buffers
  lmem_buffer_.clear();
  l2mem_buffer_.clear();

  mem_buffer_key_t lmem_key;
  mem_buffer_value_t lmem_value = {0};
  lmem_value.align_bytes = 32;

  for (int64_t stg = 0; stg < this->swpipl_stage_num_; ++stg) {
    // add for software pipeline
    // swpipl_stage_num_ always be 3 after software pipeline
    bool layer_timestep_valid =
        (swpipl_stage_num_ == 1) || (swpipl_stage_num_ > 1 && stg == 1);
    DEBUG_WITH_TYPE("lmem_buffer_assign", {
      llvm::dbgs() << "; action = lmem_buffer_assign"
                   << "; step = "
                   << "process_current_stage"
                   << "; stg = " << stg
                   << "; swpipl_stage_num_ = " << swpipl_stage_num_ << "\n";
    });
    for (size_t ts = 0; ts < get_timestep_num(); ++ts) {
      // process current timestep layers
      DEBUG_WITH_TYPE("lmem_buffer_assign", {
        llvm::dbgs() << "; action = lmem_buffer_assign"
                     << "; step =   "
                     << "process_current_timestep_layers"
                     << "; ts = " << ts << "\n";
      });
      const TpuTsField &cur_tpu_field = timestep_table_[ts].tpu0_ts_field;
      if (layer_timestep_valid) {
        for (auto op : cur_tpu_field) {
          // Results
          for (auto out : get_output_values(op)) {
            // Need some process for concat opt case
            lmem_key.value = out;
            lmem_key.type = LMEM_ACTIVATION;

            lmem_value.start_ts = ts;
            lmem_value.end_ts = -1;

            DEBUG_WITH_TYPE("lmem_buffer_assign", {
              llvm::dbgs() << "; action = lmem_buffer_assign"
                           << "; step =     "
                           << "initial_results_buffer"
                           << "; lmem_key = " << module::getName(lmem_key.value)
                           << "; lmem_type = " << lmem_key.lmem_type_str()
                           << "; lmem_start_ts = " << lmem_value.start_ts
                           << "; lmem_end_ts = " << lmem_value.end_ts << "\n";
            });

            lmem_buffer_[lmem_key] = lmem_value;
          }

          // Operands
          for (auto in : op->getOperands()) {
            if (in.getType().isa<NoneType>()) {
              continue;
            }
            if (module::isWeight(in)) {
              lmem_key.type = LMEM_WEIGHT;
            } else {
              lmem_key.type = LMEM_ACTIVATION;
            }
            lmem_key.value = in;

            if (lmem_buffer_.find(lmem_key) != lmem_buffer_.end()) {
              lmem_buffer_[lmem_key].end_ts = ts;
              DEBUG_WITH_TYPE("lmem_buffer_assign", {
                llvm::dbgs()
                    << "; action = lmem_buffer_assign"
                    << "; step =     "
                    << "update_operands_lmem_buffer"
                    << "; lmem_key = " << module::getName(lmem_key.value)
                    << "; lmem_type = " << lmem_key.lmem_type_str()
                    << "; lmem_start_ts = " << lmem_buffer_[lmem_key].start_ts
                    << "; timestep_mode = "
                    << get_tensor_mode_str(lmem_key.value)
                    << "; lmem_end_ts = " << lmem_buffer_[lmem_key].end_ts
                    << "\n";
              });
            } else {
              l2mem_buffer_[lmem_key].end_ts = ts;
              DEBUG_WITH_TYPE("lmem_buffer_assign", {
                llvm::dbgs()
                    << "; action = lmem_buffer_assign"
                    << "; step =     "
                    << "update_operands_l2mem_buffer"
                    << "; lmem_key = " << module::getName(lmem_key.value)
                    << "; lmem_type = " << lmem_key.lmem_type_str()
                    << "; lmem_start_ts = " << l2mem_buffer_[lmem_key].start_ts
                    << "; timestep_mode = "
                    << get_tensor_mode_str(lmem_key.value)
                    << "; lmem_end_ts = " << l2mem_buffer_[lmem_key].end_ts
                    << "\n";
              });
            }
          }

          // imm buffer
          lmem_key.op = op;
          lmem_key.type = LMEM_OPERATION;

          lmem_value.start_ts = ts;
          lmem_value.end_ts = ts;
          DEBUG_WITH_TYPE("lmem_buffer_assign", {
            llvm::dbgs() << "; action = lmem_buffer_assign"
                         << "; step =     "
                         << "update_imm_buffer"
                         << "; lmem_key = " << module::getName(lmem_key.value)
                         << "; lmem_type = " << lmem_key.lmem_type_str()
                         << "; lmem_start_ts = " << lmem_value.start_ts
                         << "; timestep_mode = "
                         << get_tensor_mode_str(lmem_key.value)
                         << "; lmem_end_ts = " << lmem_value.end_ts << "\n";
          });
          lmem_buffer_[lmem_key] = lmem_value;
        } // cur_tpu_field
      }
      // process current timestep tensors
      const GdmaTsField &cur_gdma_field = timestep_table_[ts].gdma0_ts_field;
      for (auto &tensor : cur_gdma_field) {
        auto &tensor_info = tensor.second;
        if (tensor_info.stage != stg) {
          continue;
        }

        if (tensor_info.mode == TIMESTEP_LOAD) {
          // Need some process for concat opt case
          if (module::isWeight(tensor.first)) {
            lmem_key.type = LMEM_WEIGHT;
          } else {
            lmem_key.type = LMEM_ACTIVATION;
          }
          lmem_key.value = tensor.first;

          lmem_value.start_ts = ts;
          lmem_value.end_ts = -1;

          if (module::isBM1684Family() && module::isWeight(tensor.first) &&
              llvm::any_of(tensor.first.getUsers(),
                           [](Operation *op) { return isa<tpu::LutOp>(op); })) {
            l2mem_buffer_[lmem_key] = lmem_value;
          } else {
            lmem_buffer_[lmem_key] = lmem_value;
            DEBUG_WITH_TYPE("lmem_buffer_assign", {
              llvm::dbgs() << "; action = lmem_buffer_assign"
                           << "; step =     "
                           << "update_load_buffer"
                           << "; lmem_key = " << module::getName(lmem_key.value)
                           << "; lmem_type = " << lmem_key.lmem_type_str()
                           << "; timestep_mode = " << tensor_info.mode_str()
                           << "; lmem_start_ts = " << lmem_value.start_ts
                           << "; lmem_end_ts = " << lmem_value.end_ts << "\n";
            });
          }
        } else if (tensor_info.mode == TIMESTEP_STORE) {
          lmem_key.value = tensor.first;
          lmem_key.type = LMEM_ACTIVATION;

          lmem_buffer_[lmem_key].end_ts = ts;
          DEBUG_WITH_TYPE("lmem_buffer_assign", {
            llvm::dbgs() << "; action = lmem_buffer_assign"
                         << "; step =     "
                         << "update_store_buffer"
                         << "; lmem_key = " << module::getName(lmem_key.value)
                         << "; lmem_type = " << lmem_key.lmem_type_str()
                         << "; timestep_mode = " << tensor_info.mode_str()
                         << "; lmem_start_ts = "
                         << lmem_buffer_[lmem_key].start_ts
                         << "; lmem_end_ts = " << lmem_buffer_[lmem_key].end_ts
                         << "\n";
          });
        }
      }
    }
  }
}

void BasicTimeStep::update_all_mem_buffer_size(const LgInfo &lg_info) {
  if (lmem_buffer_.empty()) {
    gen_all_mem_buffer_ts();
  }
  auto &tensor_infos = tensor_infos_;

  for (auto iter = lmem_buffer_.begin(); iter != lmem_buffer_.end(); ++iter) {
    if (iter->first.type == LMEM_OPERATION) {
      continue;
    }
    auto v = iter->first.value;
    auto &tensor_info = tensor_infos[v];
    iter->second.size = get_buffer_size(v, tensor_info, lg_info.type);
  }

  mem_buffer_key_t buffer_key0;
  mem_buffer_key_t buffer_key1;
  buffer_key0.type = LMEM_ACTIVATION;
  buffer_key1.type = LMEM_ACTIVATION;
  int64_t in_nslice, in_cslice, in_hslice, in_dslice, in_wslice;
  int64_t out_nslice, out_cslice, out_hslice, out_dslice, out_wslice;
  for (auto iter = lmem_buffer_.begin(); iter != lmem_buffer_.end(); ++iter) {
    if (iter->first.type == LMEM_OPERATION) {
      auto op = iter->first.op;
      auto in = op->getOperand(0);
      auto out = op->getResult(0);
      auto &in_ti = tensor_infos[in];
      auto &out_ti = tensor_infos[out];

      get_max_slice_nchdw(in_ti.slice_info, in_nslice, in_cslice, in_hslice,
                          in_dslice, in_wslice);
      get_max_slice_nchdw(out_ti.slice_info, out_nslice, out_cslice, out_hslice,
                          out_dslice, out_wslice);
      buffer_key0.value = in;
      buffer_key1.value = out;

      auto lg_op = cast<LocalGenInterface>(op);
      iter->second.size = lg_op.getBufferSize(
          lmem_buffer_[buffer_key0].size, lmem_buffer_[buffer_key1].size,
          in_nslice, in_cslice, in_hslice, in_dslice, in_wslice, out_nslice,
          out_cslice, out_hslice, out_dslice, out_wslice, lg_info.type);
    }
  }

  // erase mem_buffer whose size=0 because these ops don't need imm_buffer
  std::map<mem_buffer_key_t, mem_buffer_value_t>::iterator iter;
  for (iter = lmem_buffer_.begin(); iter != lmem_buffer_.end();) {
    iter = iter->second.size == 0 ? lmem_buffer_.erase(iter) : std::next(iter);
  }

  GROUP_DEBUG_WITH_TYPE("lmem_buffer", lg_info, [&]() { debug_lmem_buffer(); });
}

const mem_buffer_value_t &
BasicTimeStep::get_lmem_buffer_value(const mem_buffer_key_t &buffer_key) {
  //
  return lmem_buffer_[buffer_key];
}

const mem_buffer_value_t &
BasicTimeStep::get_l2mem_buffer_value(const mem_buffer_key_t &buffer_key) {
  //
  return l2mem_buffer_[buffer_key];
}

void BasicTimeStep::set_lmem_addr(const mem_buffer_key_t &buffer_key,
                                  int64_t lmem_addr) {
  lmem_buffer_[buffer_key].addr = lmem_addr;
}

int64_t BasicTimeStep::get_lmem_addr(const mem_buffer_key_t &buffer_key) {
  return lmem_buffer_[buffer_key].addr;
}
int64_t BasicTimeStep::get_lmem_size(const mem_buffer_key_t &buffer_key) {
  return lmem_buffer_[buffer_key].size;
}

MemBlock BasicTimeStep::find_buffer_locate(Value value, int64_t ts,
                                           const MemBuff &buffer,
                                           const MemBuff &l2buffer) {
  MemBlock lmem_locate(-1, -1);
  mem_buffer_key_t buffer_key;
  buffer_key.value = value;
  buffer_key.type = module::isWeight(value) ? LMEM_WEIGHT : LMEM_ACTIVATION;
  auto iter = buffer.find(buffer_key);
  if (iter == buffer.end()) {
    iter = l2buffer.find(buffer_key);
    assert(iter != buffer.end());
    lmem_locate.first = iter->second.addr;
    lmem_locate.second = iter->second.size;
  } else {
    auto &buffer_value = iter->second;
    int64_t start_ts = buffer_value.start_ts;
    int64_t end_ts = buffer_value.end_ts;
    if ((start_ts <= end_ts && ts >= start_ts && ts <= end_ts) ||
        (start_ts > end_ts && (ts >= start_ts || ts <= end_ts))) {
      lmem_locate.first = buffer_value.addr;
      lmem_locate.second = buffer_value.size;
    }
  }
  return lmem_locate;
}

MemBlock BasicTimeStep::get_lmem_locate(Value value, int64_t ts) {
  MemBlock buffer_locate =
      find_buffer_locate(value, ts, this->lmem_buffer_, this->l2mem_buffer_);
  if (buffer_locate.first == -1) {
    llvm_unreachable("cannot find local memory for this tensor.");
  }
  return buffer_locate;
}

bool BasicTimeStep::is_tensor_hold_in_lmem(Value v) {
  std::map<Value, int64_t, value_compare>::iterator iter =
      this->hold_coeff_.find(v);
  return (iter != this->hold_coeff_.end());
}

TensorInfo &BasicTimeStep::get_tensor_infos() { return tensor_infos_; }

std::string BasicTimeStep::get_tensor_mode_str(Value v) {
  return tensor_infos_[v].mode_str();
}

typedef struct {
  Value value;
  int64_t addr;
  int64_t size;
} eltwise_to_show_t;

void BasicTimeStep::show_lmem_buffer() {
  llvm::outs() << "====================================\n";
  llvm::outs() << "== show lmem buffer\n";
  mem_buffer_key_t key;
  mem_buffer_value_t value;
  size_t timestep_num = get_timestep_num();
  std::vector<std::vector<std::pair<int64_t, int64_t>>> data(
      timestep_num, std::vector<std::pair<int64_t, int64_t>>());
  for (auto &iter : lmem_buffer_) {
    key = iter.first;
    value = iter.second;
    bool first_step = true;
    for (int64_t ts = value.start_ts;
         ts != ((value.end_ts + 1) % timestep_num) || first_step;
         ts = (ts + 1) % timestep_num) {
      first_step = false;
      data[ts].push_back(std::make_pair(value.addr, value.size));
    }
  }

  for (size_t ts = 0; ts < timestep_num; ++ts) {
    llvm::outs() << "=== timestep = " << ts << "\n";
    llvm::outs() << "addr(end): ";
    int64_t total = 0;
    std::sort(data[ts].begin(), data[ts].end());
    for (auto &iter : data[ts]) {
      total += iter.second;
      llvm::outs() << "(" << iter.first << ", " << iter.first + iter.second
                   << "), ";
    }
    llvm::outs() << "total=" << total << "\n";
  }

  llvm::outs() << "====================================\n";
}

int64_t BasicTimeStep::get_tensor_range_end(const GdmaElt &tensor,
                                            int64_t cur_ts) {
  int64_t timestep_num = this->get_timestep_num();

  assert(this->swpipl_stage_num_ == 3);
  TIMESTEP_LD_ST gdma_type = tensor.second.mode;

  int64_t result = 0;
  bool find_flag = false;
  auto v = tensor.first;
  if (is_timestep_load(gdma_type)) {
    result = 0;
    for (int64_t ts = cur_ts; ts >= 0; --ts) {
      // layers
      auto &ts_layers = timestep_table_[ts].tpu0_ts_field;
      for (auto op : ts_layers) {
        auto ins = op->getOperands();
        find_flag = std::find(ins.begin(), ins.end(), v) != ins.end();
        if (find_flag) {
          result = ts + 1;
          break;
        }
      }
      if (find_flag) {
        break;
      }
    }
  } else {
    result = timestep_num - 1;
    for (int64_t ts = cur_ts; ts < timestep_num; ++ts) {
      // layers
      auto &ts_layers = timestep_table_[ts].tpu0_ts_field;
      for (auto op : ts_layers) {
        auto outs = op->getResults();
        find_flag = std::find(outs.begin(), outs.end(), v) != outs.end();
        if (find_flag) {
          result = std::min(result, ts - 1);
          break;
        }
      }
      if (find_flag) {
        break;
      }
    }
  }
  return result;
}

int64_t BasicTimeStep::get_tensor_life_time(Value v) {
  mem_buffer_key_t key = {LMEM_ACTIVATION, v, nullptr};
  if (module::isWeight(v)) {
    key.type = LMEM_WEIGHT;
  }
  auto &buffer_value = this->get_lmem_buffer_value(key);
  int64_t life_time = 0;
  int64_t timestep_num = this->get_timestep_num();
  int64_t start_ts = buffer_value.start_ts;
  int64_t end_ts = buffer_value.end_ts;
  if (end_ts >= start_ts) {
    life_time = end_ts - start_ts + 1;
  } else {
    life_time = timestep_num - start_ts + end_ts + 1;
  }
  return life_time;
}

void BasicTimeStep::cancel_tensor_hold_in_lmem(Value v) {
  auto iter = this->hold_coeff_.find(v);
  this->hold_coeff_.erase(iter);
}

void BasicTimeStep::reset_timestep(std::vector<TpuTsField> &ts_layers_v,
                                   std::vector<GdmaTsField> &ts_tensors_v,
                                   MemBuff &mem_buffer) {
  timestep_table_.clear();
  assert(ts_layers_v.size() == ts_tensors_v.size());
  for (size_t ts = 0; ts < ts_layers_v.size(); ++ts) {
    TimestepRow row;
    row.tpu0_ts_field = ts_layers_v[ts];
    row.gdma0_ts_field = ts_tensors_v[ts];
    timestep_table_.push_back(row);
  }

  lmem_buffer_ = mem_buffer;
  for (auto iter = lmem_buffer_.begin(); iter != lmem_buffer_.end(); iter++) {
    if (iter->first.type != LMEM_OPERATION) {
      if (hold_coeff_.find(iter->first.value) != hold_coeff_.end()) {
        hold_coeff_[iter->first.value] = iter->second.start_ts;
      }
      if (canceled_hold_coeff_.find(iter->first.value) !=
          canceled_hold_coeff_.end()) {
        canceled_hold_coeff_[iter->first.value] = iter->second.start_ts;
      }
    }
  }
}

bool BasicTimeStep::tensor_can_move(GdmaElt &ts_tensor, int64_t src_ts,
                                    int64_t dst_ts) {
  int64_t ts_num = timestep_table_.size();
  if (src_ts < 0 || src_ts >= ts_num || dst_ts < 0 || dst_ts >= ts_num) {
    return false;
  }

  int64_t start_ts = -1, end_ts = -1;
  MemBuff::iterator iter;
  for (iter = lmem_buffer_.begin(); iter != lmem_buffer_.end(); iter++) {
    if (iter->first.type != LMEM_OPERATION &&
        iter->first.value == ts_tensor.first) {
      start_ts = iter->second.start_ts;
      end_ts = iter->second.end_ts;
      break;
    }
  }

  // tensor loaded may be used not only one timestep
  // tensor can be stored between (begin timestep ,end timestep]
  auto dst_layers = ts_tensor.first.getUsers();
  if (is_timestep_load(ts_tensor.second.mode)) {
    assert(iter != lmem_buffer_.end());
    std::vector<Operation *> execute_layers;
    int64_t tmp = src_ts;
    while (tmp != dst_ts) {
      tmp += (src_ts > dst_ts ? -1 : 1);
      execute_layers.insert(execute_layers.end(),
                            timestep_table_[tmp].tpu0_ts_field.begin(),
                            timestep_table_[tmp].tpu0_ts_field.end());
    }
    for (auto op : dst_layers) {
      if (std::find(execute_layers.begin(), execute_layers.end(), op) !=
          execute_layers.end()) {
        return false;
      }
    }
  } else if (ts_tensor.second.mode == TIMESTEP_STORE) {
    assert(iter != lmem_buffer_.end());
    if (start_ts < end_ts && dst_ts <= start_ts) {
      return false;
    }
    if (start_ts > end_ts && src_ts > start_ts && dst_ts <= start_ts) {
      return false;
    }
    if (start_ts > end_ts && src_ts < start_ts && dst_ts >= start_ts) {
      return false;
    }
    auto src_layer = ts_tensor.first.getDefiningOp();
    int64_t ts = 0;
    for (ts = 0; ts < ts_num; ++ts) {
      if (std::find(timestep_table_[ts].tpu0_ts_field.begin(),
                    timestep_table_[ts].tpu0_ts_field.end(),
                    src_layer) != timestep_table_[ts].tpu0_ts_field.end()) {
        break;
      }
    }
    if (ts != ts_num && ts != start_ts && start_ts < end_ts && dst_ts <= ts) {
      return false;
    }
  }
  return true;
}

bool BasicTimeStep::layer_can_merge_backward(int64_t ts,
                                             bool consider_hold_in_coeff) {
  int64_t timestep_num = get_timestep_num();
  if (ts > timestep_num - 2) {
    return false;
  }

  const TpuTsField &ts_layers = timestep_table_[ts].tpu0_ts_field;
  const TpuTsField &next_ts_layers = timestep_table_[ts + 1].tpu0_ts_field;
  for (auto &tensor : timestep_table_[ts].gdma0_ts_field) {
    if (is_timestep_load(tensor.second.mode)) {
      auto dst_layers = tensor.first.getUsers();
      for (auto op : dst_layers) {
        // if there is hold coeff tensor, we regard this is valid to be merged
        // with back layer
        if ((ts == 0 || consider_hold_in_coeff ||
             !is_tensor_hold_in_lmem(tensor.first)) &&
            std::find(next_ts_layers.begin(), next_ts_layers.end(), op) !=
                next_ts_layers.end()) {
          return false;
        }
      }
    } else if (tensor.second.mode == TIMESTEP_STORE) {
      auto src_layer = tensor.first.getDefiningOp();
      if (std::find(next_ts_layers.begin(), next_ts_layers.end(), src_layer) !=
          next_ts_layers.end()) {
        return false;
      }
    }
  }

  for (auto &tensor : timestep_table_[ts + 1].gdma0_ts_field) {
    if (tensor.second.mode == TIMESTEP_STORE) {
      auto src_layer = tensor.first.getDefiningOp();
      if (std::find(ts_layers.begin(), ts_layers.end(), src_layer) !=
          ts_layers.end()) {
        return false;
      }
    } else if (is_timestep_load(tensor.second.mode)) {
      auto dst_layers = tensor.first.getUsers();
      for (auto op : dst_layers) {
        if (std::find(ts_layers.begin(), ts_layers.end(), op) !=
            ts_layers.end()) {
          return false;
        }
      }
    }
  }

  // if cur_tensor end_ts == ts_idx+1, do not merge to avoid inplace alloc_lmem
  ValueSet end_ts_tensors; // tensors that end_ts == ts + 1
  for (auto &elt : lmem_buffer_) {
    if (elt.first.type != LMEM_OPERATION && elt.second.end_ts == ts + 1) {
      end_ts_tensors.insert(elt.first.value);
    }
  }
  for (auto op : ts_layers) {
    auto inputs = get_input_values(op);
    for (auto in : inputs) {
      if (end_ts_tensors.count(in) != 0) {
        return false;
      }
    }
  }

  return true;
}

//==============================================
// functions for group overlap
//==============================================

// down_to_up overlap, update cur_time_step overlap info
void BasicTimeStep::insert_self_up_op(Value value) {
  self_up_overlap_ops_.insert(value);
}

// up_to_down overlap, update cur_time_step overlap info
void BasicTimeStep::insert_self_down_op(Value value) {
  self_down_overlap_ops_.insert(value);
}

// down_to_up overlap, update dst_time_step overlap info
void BasicTimeStep::insert_other_up_op(Value value, int64_t dst_ts) {
  if (other_up_overlap_ops_.find(dst_ts) == other_up_overlap_ops_.end()) {
    std::vector<Value> values;
    other_up_overlap_ops_.insert(std::make_pair(dst_ts, values));
  }
  other_up_overlap_ops_[dst_ts].push_back(value);
}

// up_to_group overlap, update dst_time_step overlap info
void BasicTimeStep::insert_other_down_op(Value value, int64_t dst_ts) {
  if (other_down_overlap_ops_.find(dst_ts) == other_down_overlap_ops_.end()) {
    std::vector<Value> values;
    other_down_overlap_ops_.insert(std::make_pair(dst_ts, values));
  }
  other_down_overlap_ops_[dst_ts].push_back(value);
}

} // namespace tpu
} // namespace tpu_mlir
