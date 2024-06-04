//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicNetIr.hpp"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynGdmaIrgen.hpp"

using namespace llvm;

using namespace tpu_mlir::backend;
using namespace std;
namespace tpu_mlir {
namespace tpu {
void SubnetIr::clear_all() {
  fw_ir_length = 0;
  net_input_tensor_id.clear();
  net_output_tensor_id.clear();
  ir_group_timestep_base_info.clear();
  ir_group_input_tensor_info.clear();
  ir_group_out_tensor_id_and_consumer_num.clear();
  ir_group_timestep_layer_param.clear();
  ir_group_timestep_tensor_gdma_param.clear();
  ir_group_extra_tensor_record.clear();
  // stage_param_vv.clear();
  // m_ir_buffer.clear();
  m_layer_groups_.clear();
  m_time_step_groups_.clear();
  for (auto iter : dynamic_layers_)
    delete iter.second;
  dynamic_layers_.clear();
}

int SubnetIr::get_dynamic_version() { return dynamic_version; }

bool SubnetIr::load_from_weightop(Value &v) {
  auto op = v.getDefiningOp();
  if (op != nullptr && isa<tpu::LoadOp>(op)) {
    return isa_and_nonnull<top::WeightOp>(op->getOperand(0).getDefiningOp());
  } else
    return false;
}

bool SubnetIr::loadOp_and_load_from_weightop(Operation *op) {
  return (op != nullptr && isa<tpu::LoadOp>(op) &&
          isa_and_nonnull<top::WeightOp>(op->getOperand(0).getDefiningOp()));
}

bool SubnetIr::is_eltwise_op(Operation *op) {
  bool res = false;
  // Todo
  if (isa<tpu::AddOp>(op) || isa<tpu::MulOp>(op) || isa<tpu::SubOp>(op) ||
      isa<tpu::MaxOp>(op) || isa<tpu::MinOp>(op)) {
    int input_nums = op->getNumOperands();
    vector<llvm::ArrayRef<int64_t>> shapes;
    int dims = module::getShape(op->getOperand(0)).size();
    for (int i = 0; i < input_nums; i++) {
      shapes.emplace_back(module::getShape(op->getOperand(i)));
    }

    for (int i = 0; i < input_nums - 1; i++) {
      for (int j = 0; j < dims; j++) {
        if (shapes[i][j] != shapes[i + 1][j]) {
          return false;
        }
      }
    }

    res = true;
  }
  return res;
}

bool SubnetIr::tensor_allow_slice_diff(const LgInfo &layer_group, Value &tensor,
                                       int &hslice_diff_flag) {
  //=============================
  // 1st, find the number of layer that tensor pointing to which is not allowed
  // h split conflict
  //     also find if pointed to layers that allow h split conflict
  //     Now layers that allow h split conflict contains: CONCAT
  bool allow = false;
  vector<Operation *> to_layers;
  for (auto user : tensor.getUsers()) {
    to_layers.push_back(user);
  }

  int non_split_num = 0;
  bool point_to_split_allow = false;
  for (auto &to_layer : to_layers) {
    if (std::find(layer_group.group_ops.begin(), layer_group.group_ops.end(),
                  to_layer) != layer_group.group_ops.end()) {
      if (isa<tpu::ConcatOp>(to_layer) &&
          to_layer == layer_group.group_ops[layer_group.group_ops.size() - 1]) {
        point_to_split_allow = true;
      } else {
        non_split_num++;
      }
    } else if (is_eltwise_op(to_layer)) {
      return false;
    }
  }

  //============================
  // 2nd, tensor point out of group

  if (non_split_num <= 1) {
    allow = point_to_split_allow;
  }

  return allow;
}

void SubnetIr::layer_data_back_dynamic_info(
    LgInfo &group_ops, Value tensor, list<Value> &tensor_branchs,
    map<Value, dynamic_tensor_info_t, value_cmp> &tensor_to_dynamic_info,
    std::multiset<Operation *> &layer_set,
    const set<Value, value_cmp> &out_tensor_set) {
  auto src_op = tensor.getDefiningOp();
  if (find(group_ops.group_ops.begin(), group_ops.group_ops.end(), src_op) ==
      group_ops.group_ops.end()) {
    return;
  }
  layer_set.insert(src_op);

  vector<Value> back_tensors;

  for (auto in : src_op->getOperands()) {
    if (!module::isWeight(in) && !load_from_weightop(in) &&
        !isa_and_nonnull<top::NoneOp>(in.getDefiningOp()))
      back_tensors.push_back(in);
  }

  int64_t h_slice, out_h_slice, h_in_idx;
  int64_t cur_h_slice;
  int64_t global_kh, global_stride_h, global_up_pad_h, global_down_pad_h;
  int64_t out_global_kh, out_global_stride_h, out_global_up_pad_h,
      out_global_down_pad_h;

  map<Value, dynamic_tensor_info_t, value_cmp>::iterator it =
      tensor_to_dynamic_info.find(tensor);
  dynamic_tensor_info_t out_tensor_dynamic_info;
  if (it != tensor_to_dynamic_info.end()) {
    out_tensor_dynamic_info = it->second;
  } else {
    llvm::errs() << "cannot find out tensor, when get dynamic tensor info\n";
    exit(-1);
  }
  out_h_slice = (int)out_tensor_dynamic_info.max_hslice;
  out_global_kh = (int)out_tensor_dynamic_info.global_kh;
  out_global_stride_h = (int)out_tensor_dynamic_info.global_stride_h;
  out_global_up_pad_h = (int)out_tensor_dynamic_info.global_up_pad_h;
  out_global_down_pad_h = (int)out_tensor_dynamic_info.global_down_pad_h;

  for (uint32_t i = 0; i < back_tensors.size(); ++i) {
    bool set_value = false;
    it = tensor_to_dynamic_info.find(back_tensors[i]);
    cur_h_slice = it->second.max_hslice;

    auto lg_op = cast<DynLocalGenInterface>(src_op);
    lg_op.DynBackwardH(h_in_idx, h_slice, 0, out_h_slice);
    lg_op.DynBackwardKh(global_kh, out_global_kh);
    lg_op.DynBackwardStrideH(global_stride_h, out_global_stride_h);
    lg_op.DynBackwardUpPadH(global_up_pad_h, out_global_up_pad_h);
    lg_op.DynBackwardDownPadH(global_down_pad_h, out_global_down_pad_h);

    int hslice_diff_flag = 0;
    if (cur_h_slice != 0) {
      if (cur_h_slice != h_slice) {
        if (tensor_allow_slice_diff(group_ops, back_tensors[i],
                                    hslice_diff_flag)) {
          set_value = h_slice > cur_h_slice;
        } else if (isa<tpu::StridedSliceOp>(src_op)) {
          set_value = h_slice > cur_h_slice;
        } else {
          llvm::errs() << "dynamic tensor information is conflicted\n";
          exit(-1);
        }
      }
    } else {
      set_value = true;
    }

    if (set_value) {
      it->second.max_hslice = h_slice;
      it->second.global_kh = global_kh;
      it->second.global_stride_h = global_stride_h;
      it->second.global_up_pad_h = global_up_pad_h;
      it->second.global_down_pad_h = global_down_pad_h;
    }

    if (strip_back_judge(back_tensors[i], group_ops, layer_set,
                         out_tensor_set)) {
      tensor_branchs.push_back(back_tensors[i]);
    }
  }
}

int SubnetIr::get_forward_output_height(Operation *op, int in_height) {
  int out_height = 0;
  auto lg_op = cast<DynLocalGenInterface>(op);
  out_height = lg_op.DynForwardHeight(in_height);
  return out_height;
}

int SubnetIr::get_group_global_pooling_kh(const LgInfo &layer_group,
                                          Value target_v, int global_kh,
                                          int global_up_pad_h,
                                          int global_down_pad_h) {
  Value tensor;
  Operation *op;
  list<Value> tensor_branchs;

  int left_global_kh = 1;
  int right_global_kh = global_kh - global_up_pad_h - global_down_pad_h;
  if (right_global_kh <= 0) {
    return global_kh;
  }
  int cur_global_kh = (right_global_kh + 1) / 2;
  int valid_global_pooling_kh = right_global_kh;

  map<Value, int, value_cmp> tensor_to_height;

  bool valid;

  while (cur_global_kh != left_global_kh && cur_global_kh != right_global_kh) {
    tensor_branchs.clear();
    tensor_branchs.push_back(target_v);

    tensor_to_height.clear();
    tensor_to_height.insert(make_pair(target_v, cur_global_kh));

    valid = true;
    while (!tensor_branchs.empty()) {
      tensor = tensor_branchs.front();
      tensor_branchs.pop_front();
      int tensor_height = tensor_to_height[tensor];
      int out_tensor_height = 0;
      vector<Operation *> to_layers;
      for (auto user : tensor.getUsers()) {
        to_layers.push_back(user);
      }
      for (u32 i = 0; i < to_layers.size(); ++i) {
        op = to_layers[i];
        if (find(layer_group.group_ops.begin(), layer_group.group_ops.end(),
                 op) == layer_group.group_ops.end())
          continue;

        out_tensor_height = get_forward_output_height(op, tensor_height);

        if (out_tensor_height == 0) {
          valid = false;
          break;
        }

        vector<Value> out_tensors = get_output_values(op);
        for (u32 j = 0; j < out_tensors.size(); ++j) {
          if (tensor_to_height.find(out_tensors[j]) == tensor_to_height.end()) {
            tensor_branchs.push_back(out_tensors[j]);
            tensor_to_height.insert(
                make_pair(out_tensors[j], out_tensor_height));
          }
        }
      }

      if (!valid)
        break;
    }

    if (valid) {
      valid_global_pooling_kh = cur_global_kh;
      right_global_kh = cur_global_kh;
      cur_global_kh = (left_global_kh + right_global_kh) / 2;
    } else {
      left_global_kh = cur_global_kh;
      cur_global_kh = (left_global_kh + right_global_kh + 1) / 2;
    }
  }

  return (valid_global_pooling_kh + global_up_pad_h + global_down_pad_h);
}

void SubnetIr::get_fw_input_tensor_info(
    LgInfo &group_ops, int hsecs,
    map<Value, dynamic_tensor_info_t, value_cmp> &tensor_to_dynamic_info) {
  // reset tensor_to_info
  tensor_to_dynamic_info.clear();
  dynamic_tensor_info_t tensor_info_tmp = {0, 1, 1, 0, 0};
  for (auto &it : group_ops.group_ops) {
    if (!loadOp_and_load_from_weightop(it)) {
      for (int i = 0; i < it->getNumOperands(); i++) {
        auto opd = it->getOperand(i);
        if (!module::isWeight(opd) && !load_from_weightop(opd) &&
            !isa_and_nonnull<top::NoneOp>(opd.getDefiningOp()) &&
            tensor_to_dynamic_info.find(opd) == tensor_to_dynamic_info.end()) {
          tensor_to_dynamic_info[opd] = tensor_info_tmp;
        }
      }

      for (int i = 0; i < it->getNumResults(); i++) {
        tensor_to_dynamic_info[it->getResult(i)] = tensor_info_tmp;
      }
    }
  }

  //----------------------
  // back deduction
  list<Value> tensor_branchs;
  int h_slice;

  std::multiset<Operation *> layer_set;
  set<Value, value_cmp> out_tensor_set;

  for (auto &it : group_ops.group_outs) {
    // base tensor process
    int64_t n, c, height, width;
    module::getNCHW(it, n, c, height, width);
    h_slice = (height + hsecs - 1) / hsecs;
    tensor_to_dynamic_info[it].max_hslice = (uint32_t)h_slice;

    out_tensor_set.insert(it);
    if (strip_back_judge(it, group_ops, layer_set, out_tensor_set)) {
      tensor_branchs.push_back(it);
    }

    // breadth-first search algorithm
    while (!tensor_branchs.empty()) {
      Value tensor = tensor_branchs.front();
      tensor_branchs.pop_front();
      // process back tensor
      layer_data_back_dynamic_info(group_ops, tensor, tensor_branchs,
                                   tensor_to_dynamic_info, layer_set,
                                   out_tensor_set);
    }
  }
}

uint32_t SubnetIr::get_tensor_group_consumer_num(Value v) {
  set<Operation *> group_set;
  for (auto user : v.getUsers()) {
    // double check
    if (!isa<tpu::LoadOp>(user)) {
      group_set.insert(user);
    }
  }

  return group_set.size();
}

// This function is for dynamic compile
void SubnetIr::generate_crop_layer_shape_tensor_record() {
  int count = 0;
  vector<uint32_t> id_consume_num;
  id_consume_num.clear();

  ir_group_extra_tensor_record.push_back(id_consume_num);
  fw_ir_length += (1 + count) * sizeof(uint32_t);
}

void SubnetIr::insert_produced_tensors(map<int, int> &tensor_to_consumer_num,
                                       int tensor_id) {
  if (tensor_to_consumer_num.find(tensor_id) == tensor_to_consumer_num.end()) {
    tensor_to_consumer_num[tensor_id] = 0;
  }
}

void SubnetIr::get_neuron_timestep_consumer(
    map<int, int> &tensor_to_consumer_num,
    shared_ptr<BasicTimeStep> time_step) {
  tensor_to_consumer_num.clear();
  int timestep_num = time_step->get_timestep_num();
  int swpipl_stage_num = time_step->get_swpipl_stage_num();

  for (int sw_i = 0; sw_i < swpipl_stage_num; ++sw_i) {
    for (int i = 0; i < timestep_num; ++i) {
      const TpuTsField &timestep_layers = time_step->getLayers(i);
      const GdmaTsField &timestep_tensors = time_step->getTensors(i);

      // about layer
      for (auto &op : timestep_layers) {
        // add for software pipeline
        if (time_step->get_layer_swpipl_stage(op) != sw_i)
          continue;

        // layer produced tensor
        vector<Value> out_tensors;
        for (auto out : op->getResults()) {
          if (module::isNone(out)) {
            continue;
          }
          out_tensors.push_back(out);
        }
        for (auto &out_tensor : out_tensors) {
          insert_produced_tensors(tensor_to_consumer_num,
                                  get_tensor_id(out_tensor));
        }

        // layer consumed neuron tensor
        vector<Value> in_tensors;
        for (auto in : op->getOperands()) {
          if (module::isNone(in)) {
            continue;
          }
          in_tensors.push_back(in);
        }
        for (auto &in_tensor : in_tensors) {
          if (!module::isWeight(in_tensor) && !load_from_weightop(in_tensor) &&
              !isa_and_nonnull<top::NoneOp>(in_tensor.getDefiningOp())) {
            tensor_to_consumer_num[get_tensor_id(in_tensor)] += 1;
          }
        }
      }

      // about tensor gdma
      for (auto tensor : timestep_tensors) {
        // check the pipe line
        if (time_step->get_tensor_swpipl_stage(tensor.first) != sw_i)
          continue;
        // store op
        if (isa_and_nonnull<tpu::StoreOp>(tensor.first.getDefiningOp())) {
          // gdma consumed tensor
          tensor_to_consumer_num[get_tensor_id(tensor.first)] =
              2; // ToDo: need to check
        } else if (!isa_and_nonnull<tpu::LoadOp>(
                       tensor.first.getDefiningOp())) {
          // load op
          // gdma produced tensor
          if (!module::isWeight(tensor.first)) {
            tensor_to_consumer_num[get_tensor_id(tensor.first)] = 1;
          }
        }
      }
    }
  }
}

void SubnetIr::global_layer_ir_generate_v2(Operation *op) {
  if (!dynamic_layers_.count(op)) {
    auto dynamic_instance = new dynamic_layer(op);
    dynamic_layers_.insert(make_pair(op, dynamic_instance));
  }
  fw_ir_length += sizeof(FW_LAYER_TYPE_T);
  fw_ir_length += dynamic_layers_[op]->get_global_ir_length();
}

// global layer ir generate
void SubnetIr::global_layer_ir_generate(Operation *op) {
  if (!dynamic_layers_.count(op)) {
    auto dynamic_instance = new dynamic_layer(op);
    dynamic_layers_.insert(make_pair(op, dynamic_instance));
  }

  ir_layer_info_t ir_layer_info;
  fw_ir_length += sizeof(FW_LAYER_TYPE_T);
  if (module::isBM1684Family()) {
    // Only BM1684 dose need the following information
    fw_ir_length += sizeof(DATA_SIZE_T);
    ir_layer_info.data_size = DSIZE_FP32; // default fp32
    fw_ir_length += sizeof(uint32_t);
    ir_layer_info.intensor_store_mode = 0;
    fw_ir_length += sizeof(uint32_t);
    ir_layer_info.outtensor_store_mode = 0;
  }
  // fw layer_parameter
  fw_ir_length += sizeof(uint32_t); // Layer parameter size
  fw_ir_length += dynamic_layers_[op]->get_global_ir_length(&ir_layer_info);

  // input and output global memory offset
  ir_layer_info.ir_tensor_info_v.clear();
  // input
  for (uint32_t i = 0; i < op->getNumOperands(); ++i) {
    if (module::isNone(op->getOperand(i)))
      continue;
    if (module::isGlobalBuffer(op->getOperand(i)))
      continue;
    fw_ir_length += push_back_layer_global_tensor(
        op->getOperand(i), ir_layer_info.ir_tensor_info_v, true);
  }
  // output
  for (uint32_t i = 0; i < op->getNumResults(); ++i) {
    if (module::isNone(op->getResult(i)))
      continue;
    fw_ir_length += push_back_layer_global_tensor(
        op->getResult(i), ir_layer_info.ir_tensor_info_v, false);
  }

  vector<ir_layer_info_t> ir_layer_info_t_v1;
  vector<vector<ir_layer_info_t>> ir_layer_info_t_v2;

  ir_layer_info_t_v1.push_back(ir_layer_info);
  ir_layer_info_t_v2.push_back(ir_layer_info_t_v1);
  ir_group_timestep_layer_param.push_back(ir_layer_info_t_v2);
}

void SubnetIr::local_layer_ir_generate_v2(Operation *op) {
  if (!dynamic_layers_.count(op)) {
    auto dynamic_instance = new dynamic_layer(op);
    dynamic_layers_.insert(make_pair(op, dynamic_instance));
  }
  fw_ir_length += sizeof(uint32_t);
  fw_ir_length += sizeof(FW_LAYER_TYPE_T);
  fw_ir_length += dynamic_layers_[op]->get_local_ir_length();
}

void SubnetIr::local_layer_ir_generate(Operation *op,
                                       vector<ir_layer_info_t> &layer_info_v1,
                                       uint8_t swpipl_stage_num) {
  if (!dynamic_layers_.count(op)) {
    auto dynamic_instance = new dynamic_layer(op);
    dynamic_layers_.insert(make_pair(op, dynamic_instance));
  }

  ir_layer_info_t ir_layer_info;

  ir_layer_info.layer_id = -1; // no use
  ir_layer_info.is_cpu_layer = false;

  ir_layer_info.swpipl_enable = swpipl_stage_num > 1;
  if (ir_layer_info.swpipl_enable)
    fw_ir_length += sizeof(uint32_t);
  // record base offset
  uint32_t base_layer_offset = fw_ir_length;

  fw_ir_length += sizeof(FW_LAYER_TYPE_T);

  if (module::isBM1684Family()) {
    // Only BM1684 need the following information
    fw_ir_length += sizeof(DATA_SIZE_T);
    ir_layer_info.data_size = DSIZE_FP32; // default fp32
    fw_ir_length += sizeof(u32);
    ir_layer_info.intensor_store_mode = STORE_MODE_1N;
    fw_ir_length += sizeof(u32);
    ir_layer_info.outtensor_store_mode = STORE_MODE_1N;
  }

  // Layer parameter
  fw_ir_length += sizeof(uint32_t); // Layer parameter size
  uint32_t pre_fw_ir_length = fw_ir_length;
  fw_ir_length += dynamic_layers_[op]->get_local_ir_length(&ir_layer_info);

  ir_layer_info.layer_fw_ir_length = fw_ir_length - pre_fw_ir_length;
  // add for software pipeline
  uint32_t layer_ir_length = fw_ir_length - base_layer_offset;
  uint32_t layer_stage = swpipl_stage_num == 1 ? 0 : 1;
  ir_layer_info.stage_and_ir_size =
      (layer_stage << 16) | (layer_ir_length & 0xffff);

  layer_info_v1.push_back(ir_layer_info);
}

void SubnetIr::gdma_tensor_ir_generate(
    Operation *op, vector<ir_tensor_gdma_info_t> &tensor_gdma_info_v1,
    group_type_t group_type, bool swpipl_enable, int stage,
    uint64_t local_addr) {
  ir_tensor_gdma_info_t ir_tensor_gdma_info;

  // add for software pipeline
  ir_tensor_gdma_info.swpipl_enable = swpipl_enable;
  if (ir_tensor_gdma_info.swpipl_enable)
    fw_ir_length += sizeof(uint32_t);
  // record base offset
  uint32_t base_ir_offset = fw_ir_length;

  int tensor_id = 0;
  fw_ir_length += sizeof(tensor_gdma_type_t);
  if (get_dynamic_version() >= 2) {
    fw_ir_length += sizeof(uint32_t); // Magic
  }

  // GDMA parameter size
  fw_ir_length += sizeof(uint32_t);
  auto opd = op->getOperand(0);
  tensor_id = get_tensor_id(opd);
  if (isa<tpu::LoadOp>(op)) {
    uint64_t global_addr = module::getAddress(opd);
    if (module::isBM1684Family() && BM1684::instance().isL2Load(op)) {
      fw_ir_length += static_ld_g2l2_irgen_ctrl(op, tensor_id, global_addr,
                                                local_addr, ir_tensor_gdma_info,
                                                get_dynamic_version() >= 2);
    } else if (isa_and_nonnull<top::WeightOp>(opd.getDefiningOp())) {
      fw_ir_length += static_ld_coeff_irgen_ctrl(
          op, tensor_id, global_addr, local_addr, ir_tensor_gdma_info,
          get_dynamic_version() >= 2);
    } else {
      fw_ir_length += static_ld_neuron_irgen_ctrl(
          op, tensor_id, global_addr, local_addr, ir_tensor_gdma_info,
          get_dynamic_version() >= 2);
    }
  } else if (isa<tpu::StoreOp>(op)) {
    uint64_t global_addr = module::getAddress(op->getResult(0));
    fw_ir_length += static_st_neuron_irgen_ctrl(op, tensor_id, global_addr,
                                                local_addr, ir_tensor_gdma_info,
                                                get_dynamic_version() >= 2);
  }

  // add for software pipeline
  uint32_t tensor_gdma_ir_length = fw_ir_length - base_ir_offset;

  uint32_t tensot_gdma_stage = stage;
  ir_tensor_gdma_info.stage_and_ir_size =
      (tensot_gdma_stage << 16) | (tensor_gdma_ir_length & 0xffff);
  tensor_gdma_info_v1.push_back(ir_tensor_gdma_info);
}

void SubnetIr::generate_group_time_step_ir(Operation *op) {
  auto in_shape = module::getShape(op->getOperand(0));
  int batch_num = in_shape.empty() ? 0 : in_shape[0];
  if (auto castOp = dyn_cast<tpu::GroupOp>(op)) {
    // local layer
    LgInfo sub_group;
    auto nsecs = castOp.getNsecs();
    auto hsecs = castOp.getHsecs();
    auto swpipl_stage_num = castOp.getSwpiplStageNum();
    auto &body = castOp.getBody().front();
    auto flow = module::getI64Array(castOp.getFlow());
    // 1. restore timestep_table from flow
    std::vector<std::vector<int64_t>> timestep_table;
    std::vector<int64_t> ts_row;
    int64_t max_id = 0;
    for (size_t i = 1; i < flow->size(); ++i) {
      if (flow->at(i) < 0) {
        timestep_table.push_back(ts_row);
        ts_row.clear();
        continue;
      }
      ts_row.push_back(flow->at(i));
      max_id = std::max(max_id, flow->at(i));
    }
    timestep_table.push_back(ts_row);
    uint32_t timestep_num = timestep_table.size();

    // 2. create a vector to map id to op
    for (int64_t id = 0; id < max_id;) {
      body.walk([&](Operation *op) {
        if (auto lgOp = dyn_cast<DynLocalGenInterface>(op)) {
          auto ginfo = lgOp.DynGetGroupInfo((int64_t)0, (int64_t)0);
          if (ginfo.id == id) {
            sub_group.group_ops.push_back(op);
            id++;
          }
        } else if (!isa<tpu::YieldOp>(op)) {
          op->dump();
          llvm_unreachable("such op need support dynamic local");
        }
      });
    }

    sub_group.update_group_io();
    m_layer_groups_.push_back(sub_group);

    // get max_nslice;
    uint32_t max_nslice = (batch_num + nsecs - 1) / nsecs;

    // get and push fw_timestep_base_info_t
    fw_timestep_base_info_t timestep_base_info = {
        .ts_num_and_split_tensor_num = (timestep_num << 16),
        .max_nslice_deprecated = 255, // 255 means invalid
        .input_tensor_num = (uint8_t)(sub_group.group_ins.size()),
        .output_tensor_num = (uint8_t)(sub_group.group_outs.size()),
        .flags =
            (uint8_t)((1 << 5) | (sub_group.type << 2) |
                      ((1 << 1) |
                       (hsecs >
                        1))), // group_type, using max_nslice, h_is_split or not
        .swpipl_stage_num = (uint8_t)(swpipl_stage_num),
        .max_nslice = max_nslice};
    ir_group_timestep_base_info.push_back(timestep_base_info);
    fw_ir_length += sizeof(fw_timestep_base_info_t);

    // get and push fw_input_tensor_info_t for each input tensor
    vector<fw_input_tensor_info_t> input_tensor_info_v;
    map<Value, dynamic_tensor_info_t, value_cmp> tensor_to_dynamic_info;
    if (hsecs == 1) {
      for (int i = 0; i < sub_group.group_ins.size(); i++) {
        fw_input_tensor_info_t input_tensor_info = {
            .tensor_id_and_max_hslice =
                ((uint32_t)(get_tensor_id(sub_group.group_ins[i])) << 16),
            .stride_h_and_kh = 0,
            .pad_h_top_and_bottom = 0,
            .min_pool_kh = 0};

        input_tensor_info_v.push_back(input_tensor_info);
        // because firmware only need tensor id actually, so the length of IR
        // info for firmware is sizeof(u32) NOTE: also in local mode
        fw_ir_length += sizeof(u32);
      }
    } else {
      tensor_to_dynamic_info.clear();
      get_fw_input_tensor_info(sub_group, hsecs, tensor_to_dynamic_info);

      for (int i = 0; i < sub_group.group_ins.size(); i++) {
        dynamic_tensor_info_t dynamic_tensor_info =
            tensor_to_dynamic_info[sub_group.group_ins[i]];
        uint32_t max_hslice = dynamic_tensor_info.max_hslice;
        uint32_t global_stride_h = dynamic_tensor_info.global_stride_h;
        uint32_t global_kh = dynamic_tensor_info.global_kh;
        uint32_t global_up_pad_h = dynamic_tensor_info.global_up_pad_h;
        uint32_t global_down_pad_h = dynamic_tensor_info.global_down_pad_h;

        fw_input_tensor_info_t input_tensor_info = {
            .tensor_id_and_max_hslice =
                ((uint32_t)(get_tensor_id(sub_group.group_ins[i])) << 16) |
                (max_hslice & 0xffff),
            .stride_h_and_kh = (global_stride_h << 16) | (global_kh & 0xffff),
            .pad_h_top_and_bottom =
                (global_up_pad_h << 16) | (global_down_pad_h & 0xffff),
            .min_pool_kh = (u32)get_group_global_pooling_kh(
                sub_group, sub_group.group_ins[i], global_kh, global_up_pad_h,
                global_down_pad_h)};

        input_tensor_info_v.push_back(input_tensor_info);
        fw_ir_length += sizeof(fw_input_tensor_info_t);
      }
    }
    ir_group_input_tensor_info.push_back(input_tensor_info_v);

    // get and push group_out_tensor_id_and_consumer_num
    vector<uint32_t> group_out_tensor_id_and_consumer_num;
    vector<Value> group_outs;
    /* note: double check because
        the get_tensor_group_consumer_num(sub_group.group_outs[i])
        maybe not right */
    for (auto v : op->getResults())
      group_outs.push_back(v);

    for (int i = 0; i < sub_group.group_outs.size(); i++) {
      // just for double check
      u32 group_consumer_num =
          get_tensor_group_consumer_num(sub_group.group_outs[i]);
      u32 another_consumer_num = get_tensor_group_consumer_num(group_outs[i]);
      group_consumer_num = (group_consumer_num >= another_consumer_num)
                               ? group_consumer_num
                               : another_consumer_num;
      uint32_t tensor_id_and_consumer_num =
          (((uint32_t)get_tensor_id(sub_group.group_outs[i])) << 16) |
          (group_consumer_num & 0xffff);

      group_out_tensor_id_and_consumer_num.push_back(
          tensor_id_and_consumer_num);
      fw_ir_length += sizeof(uint32_t);
    }
    ir_group_out_tensor_id_and_consumer_num.push_back(
        group_out_tensor_id_and_consumer_num);

    // ir info of each time step
    vector<ir_layer_info_t> layer_info_v1;
    vector<vector<ir_layer_info_t>> layer_info_v2;

    vector<ir_tensor_gdma_info_t> tensor_gdma_info_v1;
    vector<vector<ir_tensor_gdma_info_t>> tensor_gdma_info_v2;

    // rebuild the timestep for ir codegen
    auto time_step = std::make_shared<BasicTimeStep>();
    TpuTsField tpu_field;
    GdmaTsField gdma_field;

    for (uint32_t i = 0; i < timestep_num; i++) {
      // time step layers
      auto cur_op_ids = timestep_table[i];
      tpu_field.clear();
      gdma_field.clear();
      tensor_gdma_info_v1.clear();
      fw_ir_length += sizeof(uint32_t) * 2;
      layer_info_v1.clear();
      for (auto id : cur_op_ids) {
        if (isa<tpu::LoadOp, tpu::StoreOp>(sub_group.group_ops[id])) {
          auto lgOp = dyn_cast<DynLocalGenInterface>(sub_group.group_ops[id]);
          auto ginfo = lgOp.DynGetGroupInfo((int64_t)0, (int64_t)0);

          gdma_tensor_ir_generate(sub_group.group_ops[id], tensor_gdma_info_v1,
                                  sub_group.type, swpipl_stage_num > 1,
                                  ginfo.stage, ginfo.out_addr);

          if (isa<tpu::LoadOp>(sub_group.group_ops[id])) {
            tensor_info_t ti(TIMESTEP_LOAD);
            ti.stage = ginfo.stage;
            gdma_field.emplace_back(
                make_pair(sub_group.group_ops[id]->getOperand(0), ti));
          } else {
            tensor_info_t ti(TIMESTEP_STORE);
            ti.stage = ginfo.stage;
            gdma_field.emplace_back(
                make_pair(sub_group.group_ops[id]->getResult(0), ti));
          }
        } else {
          tpu_field.emplace_back(sub_group.group_ops[id]);
          if (get_dynamic_version() >= 2) {
            local_layer_ir_generate_v2(sub_group.group_ops[id]);
          } else {
            local_layer_ir_generate(sub_group.group_ops[id], layer_info_v1,
                                    swpipl_stage_num);
          }
        }
      }
      layer_info_v2.push_back(layer_info_v1);

      tensor_gdma_info_v2.push_back(tensor_gdma_info_v1);
      time_step->add_tpu0_gdma0_ts_field(tpu_field, gdma_field);
    }

    time_step->set_swpipl_stage_num(swpipl_stage_num);
    m_time_step_groups_.push_back(time_step);

    if (get_dynamic_version() < 2)
      ir_group_timestep_layer_param.push_back(layer_info_v2);
    ir_group_timestep_tensor_gdma_param.push_back(tensor_gdma_info_v2);
  } else if (auto castOp = dyn_cast<GlobalGenInterface>(op)) {
    if (module::isBM1684XFamily() || module::isBM1684Family() ||
        module::isBM1690Family()) {
      // global layer
      LgInfo sub_group;
      sub_group.group_ops.push_back(op);
      for (auto &&v : op->getOperands()) {
        if (!isa_and_nonnull<top::WeightOp, top::NoneOp, tpu::BufferOp>(
                v.getDefiningOp()))
          sub_group.group_ins.emplace_back(v);
      }
      for (auto &&v : op->getResults())
        sub_group.group_outs.emplace_back(v);
      m_layer_groups_.push_back(sub_group);
      // get and push fw_timestep_base_info_t
      fw_timestep_base_info_t timestep_base_info = {
          .ts_num_and_split_tensor_num = 1 << 16,
          // 252 is aligned to 4, because it may meet 4N
          .max_nslice_deprecated = 255, // 255 means invalid
          .input_tensor_num = (uint8_t)(sub_group.group_ins.size()),
          .output_tensor_num = (uint8_t)(sub_group.group_outs.size()),
          .flags =
              (uint8_t)((1 << 5) | (sub_group.type << 2) |
                        ((1 << 1) | 0)), // using max_nslice, h is not split
          .swpipl_stage_num = 1,
          .max_nslice = (uint32_t)batch_num};
      ir_group_timestep_base_info.push_back(timestep_base_info);
      fw_ir_length += sizeof(fw_timestep_base_info_t);

      // get and push fw_input_tensor_info_t for each input tensor
      vector<fw_input_tensor_info_t> input_tensor_info_v;
      for (int i = 0; i < sub_group.group_ins.size(); i++) {
        fw_input_tensor_info_t input_tensor_info = {
            .tensor_id_and_max_hslice =
                (uint32_t)(get_tensor_id(sub_group.group_ins[i])) << 16,
            .stride_h_and_kh = 0,
            .pad_h_top_and_bottom = 0,
            .min_pool_kh = 0};

        input_tensor_info_v.push_back(input_tensor_info);
        // because firmware only need tensor id actually, so the length of IR
        // info for firmware is sizeof(u32)
        fw_ir_length += sizeof(uint32_t);
      }
      ir_group_input_tensor_info.push_back(input_tensor_info_v);

      // get and push group_out_tensor_id_and_consumer_num
      vector<uint32_t> group_out_tensor_id_and_consumer_num;
      for (int i = 0; i < sub_group.group_outs.size(); i++) {
        uint32_t group_consumer_num =
            get_tensor_group_consumer_num(sub_group.group_outs[i]);
        uint32_t tensor_id_and_consumer_num =
            (((uint32_t)get_tensor_id(sub_group.group_outs[i])) << 16) |
            (group_consumer_num & 0xffff);
        group_out_tensor_id_and_consumer_num.push_back(
            tensor_id_and_consumer_num);
        fw_ir_length += sizeof(uint32_t);
      }
      ir_group_out_tensor_id_and_consumer_num.push_back(
          group_out_tensor_id_and_consumer_num);

      auto time_step = std::make_shared<BasicTimeStep>();
      TpuTsField tpu_field;
      tpu_field.emplace_back(op);
      time_step->add_tpu0_ts_field(tpu_field);
      m_time_step_groups_.push_back(time_step);

      vector<vector<ir_tensor_gdma_info_t>> tensor_gdma_dummy;
      ir_group_timestep_tensor_gdma_param.push_back(tensor_gdma_dummy);

      if (get_dynamic_version() >= 2) {
        global_layer_ir_generate_v2(op);
      } else {
        global_layer_ir_generate(op);
      }
    } else {
      llvm_unreachable("not support");
    }
  }
}

void SubnetIr::generate_compiler_ir(
    ModuleOp &s, func::CallOp &call,
    std::function<void(Operation *, SubnetIr *)> task) {
  std::vector<Value> inputs;
  std::vector<Value> outputs;
  module::getInputsOutputs(call, inputs, outputs);
  for (auto v : inputs) {
    net_input_tensor_id.push_back(get_tensor_id(v));
    fw_ir_length += sizeof(uint32_t);
  }

  for (auto v : outputs) {
    net_output_tensor_id.push_back(get_tensor_id(v));
    fw_ir_length += sizeof(uint32_t);
  }

  auto func = module::getFuncOp(s, call.getCallee());
  func.walk([&](Operation *op) { task(op, this); });
  // get layer group num
  fw_ir_length += sizeof(u32);
}

void *SubnetIr::write_local_layer_info_buffer_v2(
    void *p_ir_addr, Operation *op, FW_LAYER_TYPE_T fw_type,
    shared_ptr<BasicTimeStep> time_step) {
  if (!dynamic_layers_.count(op)) {
    auto dynamic_instance = new dynamic_layer(op);
    dynamic_layers_.insert(make_pair(op, dynamic_instance));
  }

  u32 *p_stage_ir_size = (u32 *)p_ir_addr;
  p_ir_addr = (u32 *)p_ir_addr + 1;

  *(FW_LAYER_TYPE_T *)p_ir_addr = static_cast<FW_LAYER_TYPE_T>(fw_type);
  p_ir_addr = (FW_LAYER_TYPE_T *)p_ir_addr + 1;

  map<int, int> tensor_to_consumer_num;
  get_neuron_timestep_consumer(tensor_to_consumer_num, time_step);

  size_t expect_len = dynamic_layers_[op]->get_local_ir_length();

  size_t wrote =
      dynamic_layers_[op]->write_local_ir(p_ir_addr, tensor_to_consumer_num);
  assert(expect_len == wrote);
  p_ir_addr = static_cast<char *>(p_ir_addr) + wrote;

  // Write ir size afterwise.
  u32 layer_stage = time_step->get_layer_swpipl_stage(op);
  *p_stage_ir_size = (layer_stage << 16) | (wrote + sizeof(FW_LAYER_TYPE_T));

  return p_ir_addr;
}

void *
SubnetIr::write_local_layer_info_buffer(void *p_ir_buf, Operation *op,
                                        ir_layer_info_t *p_ir_layer_info,
                                        shared_ptr<BasicTimeStep> time_step) {
  void *p_ir_addr = p_ir_buf;
  // add for software pipeline
  if (p_ir_layer_info->swpipl_enable) {
    // write software pipeline information
    *(u32 *)p_ir_addr = p_ir_layer_info->stage_and_ir_size;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }

  // write type
  FW_LAYER_TYPE_T fw_layer_type = p_ir_layer_info->fw_layer_type;
  *(FW_LAYER_TYPE_T *)p_ir_addr = fw_layer_type;
  p_ir_addr = (FW_LAYER_TYPE_T *)p_ir_addr + 1;

  if (module::isBM1684Family()) {
    // Only BM1684 need the following
    *(DATA_SIZE_T *)p_ir_addr = p_ir_layer_info->data_size;
    p_ir_addr = (DATA_SIZE_T *)p_ir_addr + 1;
    *(u32 *)p_ir_addr = (u32)(p_ir_layer_info->intensor_store_mode);
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(u32 *)p_ir_addr = (u32)(p_ir_layer_info->outtensor_store_mode);
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }

  map<int, int> tensor_to_consumer_num;
  get_neuron_timestep_consumer(tensor_to_consumer_num, time_step);

  // store consumer number for each local layer output
  const int output_num = op->getNumResults();
  SmallVector<int> out_tensors;
  for (int i = 0; i < output_num; ++i) {
    if (module::isNone(op->getResult(i)))
      continue;
    out_tensors.push_back(get_tensor_id(op->getResult(i)));
  }
  auto set_consumer_num = [&](int tensor_id) {
    uint32_t consumer_num = 1;
    if (tensor_to_consumer_num.find(tensor_id) !=
        tensor_to_consumer_num.end()) {
      consumer_num = (uint32_t)(tensor_to_consumer_num.find(tensor_id)->second);
    }
    // set consumer number
    for (auto &t_info : p_ir_layer_info->ir_tensor_info_v) {
      if (t_info.tensor_id == (uint32_t)(tensor_id)) {
        t_info.consumer_number = consumer_num;
      }
    }
  };
  for (int i = 0; i < out_tensors.size(); ++i) {
    set_consumer_num(out_tensors[i]);
    // TODO deal with ConcatOp
  }

  p_ir_addr =
      call_local_layer_ir_write(fw_layer_type, p_ir_addr, p_ir_layer_info);

  return p_ir_addr;
}

void *SubnetIr::write_tensor_gdma_info_buffer(
    void *p_ir_buf, ir_tensor_gdma_info_t *ir_tensor_gdma_info) {
  void *p_ir_addr = p_ir_buf;

  // add for software pipeline
  if (ir_tensor_gdma_info->swpipl_enable) {
    // write software pipeline information
    *(u32 *)p_ir_addr = ir_tensor_gdma_info->stage_and_ir_size;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }

  u64 pre_ir_addr = (u64)p_ir_addr;
  tensor_gdma_type_t fw_tensor_gdma_type =
      ir_tensor_gdma_info->fw_tensor_gdma_type;
  *(tensor_gdma_type_t *)p_ir_addr = fw_tensor_gdma_type;
  p_ir_addr = (tensor_gdma_type_t *)p_ir_addr + 1;

  if (get_dynamic_version() >= 2) {
    const uint32_t Magic = 0xf00ffff;
    *(u32 *)p_ir_addr = Magic;
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }

  switch (fw_tensor_gdma_type) {
  case LD_INPUT_NEURON:
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_gdma_ld_in_neuron_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(fw_gdma_ld_in_neuron_t *)p_ir_addr =
        ir_tensor_gdma_info->fw_tensor_gdma_param_u.fw_gdma_ld_in_neuron;
    p_ir_addr = (fw_gdma_ld_in_neuron_t *)p_ir_addr + 1;
    break;
  case ST_OUTPUT_NEURON:
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_gdma_st_out_neuron_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(fw_gdma_st_out_neuron_t *)p_ir_addr =
        ir_tensor_gdma_info->fw_tensor_gdma_param_u.fw_gdma_st_out_neuron;
    p_ir_addr = (fw_gdma_st_out_neuron_t *)p_ir_addr + 1;
    break;
  case LD_ITM_NEURON:
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_gdma_ld_itm_neuron_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(fw_gdma_ld_itm_neuron_t *)p_ir_addr =
        ir_tensor_gdma_info->fw_tensor_gdma_param_u.fw_gdma_ld_itm_neuron;
    p_ir_addr = (fw_gdma_ld_itm_neuron_t *)p_ir_addr + 1;
    break;
  case LD_ITM_EXTEND_NEURON:
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_gdma_ld_itm_extend_neuron_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(fw_gdma_ld_itm_extend_neuron_t *)p_ir_addr =
        ir_tensor_gdma_info->fw_tensor_gdma_param_u
            .fw_gdma_ld_itm_extend_neuron;
    p_ir_addr = (fw_gdma_ld_itm_extend_neuron_t *)p_ir_addr + 1;
    break;
  case ST_ITM_NEURON:
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_gdma_st_itm_neuron_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(fw_gdma_st_itm_neuron_t *)p_ir_addr =
        ir_tensor_gdma_info->fw_tensor_gdma_param_u.fw_gdma_st_itm_neuron;
    p_ir_addr = (fw_gdma_st_itm_neuron_t *)p_ir_addr + 1;
    break;
  case ST_ITM_EXTEND_NEURON:
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_gdma_st_itm_extend_neuron_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(fw_gdma_st_itm_extend_neuron_t *)p_ir_addr =
        ir_tensor_gdma_info->fw_tensor_gdma_param_u
            .fw_gdma_st_itm_extend_neuron;
    p_ir_addr = (fw_gdma_st_itm_extend_neuron_t *)p_ir_addr + 1;
    break;
  case LD_COEFF:
  case LD_COEFF_WINOGRAD:
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_gdma_coeff_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(fw_gdma_coeff_t *)p_ir_addr =
        ir_tensor_gdma_info->fw_tensor_gdma_param_u.fw_gdma_coeff;
    p_ir_addr = (fw_gdma_coeff_t *)p_ir_addr + 1;
    break;
  case LD_COEFF_NERUON:
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_gdma_coeff_neuron_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(fw_gdma_coeff_neuron_t *)p_ir_addr =
        ir_tensor_gdma_info->fw_tensor_gdma_param_u.fw_gdma_coeff_neuron;
    p_ir_addr = (fw_gdma_coeff_neuron_t *)p_ir_addr + 1;
    if (ir_tensor_gdma_info->fw_tensor_gdma_param_u.fw_gdma_coeff_neuron.n >
        0) {
      *(u32 *)p_ir_addr = (u32)(sizeof(fw_dynamic_output_info_t));
      p_ir_addr = (u32 *)p_ir_addr + 1;
      *(fw_dynamic_output_info_t *)p_ir_addr =
          ir_tensor_gdma_info->fw_shape_info;
      p_ir_addr = (fw_dynamic_output_info_t *)p_ir_addr + 1;
    }
    break;
  case MV_ITM_NEURON:
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_gdma_mv_itm_neuron_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(fw_gdma_mv_itm_neuron_t *)p_ir_addr =
        ir_tensor_gdma_info->fw_tensor_gdma_param_u.fw_gdma_mv_itm_neuron;
    p_ir_addr = (fw_gdma_mv_itm_neuron_t *)p_ir_addr + 1;
    break;
  case MV_ITM_EXTEND_NEURON:
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_gdma_mv_itm_extend_neuron_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(fw_gdma_mv_itm_extend_neuron_t *)p_ir_addr =
        ir_tensor_gdma_info->fw_tensor_gdma_param_u
            .fw_gdma_mv_itm_extend_neuron;
    p_ir_addr = (fw_gdma_mv_itm_extend_neuron_t *)p_ir_addr + 1;
    break;
  case MV_OUTPUT_NEURON:
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_gdma_mv_out_neuron_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(fw_gdma_mv_out_neuron_t *)p_ir_addr =
        ir_tensor_gdma_info->fw_tensor_gdma_param_u.fw_gdma_mv_out_neuron;
    p_ir_addr = (fw_gdma_mv_out_neuron_t *)p_ir_addr + 1;
    break;
  case LD_G2L2:
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_gdma_ld_g2l2_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(fw_gdma_ld_g2l2_t *)p_ir_addr =
        ir_tensor_gdma_info->fw_tensor_gdma_param_u.fw_gdma_ld_g2l2;
    p_ir_addr = (fw_gdma_ld_g2l2_t *)p_ir_addr + 1;
    break;
  case ST_OUTPUT_EXTEND_NEURON:
    *(u32 *)p_ir_addr = (u32)(sizeof(fw_gdma_st_out_extend_neuron_t));
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(fw_gdma_st_out_extend_neuron_t *)p_ir_addr =
        ir_tensor_gdma_info->fw_tensor_gdma_param_u
            .fw_gdma_st_out_extend_neuron;
    p_ir_addr = (fw_gdma_st_out_extend_neuron_t *)p_ir_addr + 1;
    break;
  default:
    exit(-1);
  }
  // check ir length
  u32 gdma_ir_length = ir_tensor_gdma_info->stage_and_ir_size & 0xffff;
  if ((u32)((u64)p_ir_addr - pre_ir_addr) != gdma_ir_length) {
    exit(-1);
  }
  return p_ir_addr;
}

void *SubnetIr::write_global_layer_info_buffer_v2(void *p_ir_addr,
                                                  Operation *op,
                                                  FW_LAYER_TYPE_T fw_type) {
  if (!dynamic_layers_.count(op)) {
    auto dynamic_instance = new dynamic_layer(op);
    dynamic_layers_.insert(make_pair(op, dynamic_instance));
  }
  *(FW_LAYER_TYPE_T *)p_ir_addr = static_cast<FW_LAYER_TYPE_T>(fw_type);
  p_ir_addr = (FW_LAYER_TYPE_T *)p_ir_addr + 1;
  size_t expect_len = dynamic_layers_[op]->get_global_ir_length();
  size_t wrote = dynamic_layers_[op]->write_global_ir(p_ir_addr);
  assert(expect_len == wrote);
  p_ir_addr = static_cast<char *>(p_ir_addr) + wrote;
  return p_ir_addr;
}

void *SubnetIr::write_global_layer_info_buffer(void *p_ir_buf,
                                               ir_layer_info_t *ir_layer_info) {
  void *p_ir_addr = p_ir_buf;
  FW_LAYER_TYPE_T fw_layer_type = ir_layer_info->fw_layer_type;

  *(FW_LAYER_TYPE_T *)p_ir_addr = fw_layer_type;
  p_ir_addr = (FW_LAYER_TYPE_T *)p_ir_addr + 1;

  if (module::isBM1684Family()) {
    // Only BM1684 need the following
    *(DATA_SIZE_T *)p_ir_addr = ir_layer_info->data_size;
    p_ir_addr = (DATA_SIZE_T *)p_ir_addr + 1;
    *(u32 *)p_ir_addr = (u32)(ir_layer_info->intensor_store_mode);
    p_ir_addr = (u32 *)p_ir_addr + 1;
    *(u32 *)p_ir_addr = (u32)(ir_layer_info->outtensor_store_mode);
    p_ir_addr = (u32 *)p_ir_addr + 1;
  }

  p_ir_addr =
      call_global_layer_ir_write(fw_layer_type, p_ir_addr, ir_layer_info);

  return p_ir_addr;
}

/* return writen ir length */
int SubnetIr::write_ir_to_buffer(void *ir_buffer,
                                 vector<unsigned int> &input_tensor_ids,
                                 vector<unsigned int> &output_tensor_ids,
                                 int group_start, int group_end) {

  cout << "======== write binary ir to buffer ========" << endl;
  void *p_ir_buf = ir_buffer;
  // write input and output of network
  for (auto &tensor_id : input_tensor_ids) {
    *(uint32_t *)p_ir_buf = tensor_id;
    p_ir_buf = (uint32_t *)p_ir_buf + 1;
  }
  for (auto &tensor_id : output_tensor_ids) {
    *(uint32_t *)p_ir_buf = tensor_id;
    p_ir_buf = (uint32_t *)p_ir_buf + 1;
  }

  // write group info
  *(uint32_t *)p_ir_buf = group_end - group_start + 1;
  p_ir_buf = (uint32_t *)p_ir_buf + 1;

  for (int group_idx = group_start; group_idx <= group_end; ++group_idx) {
    auto &layer_group = m_layer_groups_[group_idx];

    if (module::isBM1684Family()) {
      uint32_t crop_shape_tensor_num =
          ir_group_extra_tensor_record[group_idx].size();
      *(uint32_t *)p_ir_buf = crop_shape_tensor_num;
      p_ir_buf = (uint32_t *)p_ir_buf + 1;
      for (uint32_t i = 0; i < crop_shape_tensor_num; i++) {
        *(uint32_t *)p_ir_buf = ir_group_extra_tensor_record[group_idx][i];
        p_ir_buf = (uint32_t *)p_ir_buf + 1;
      }
    }

    // write base info
    fw_timestep_base_info_t fw_timestep_base_info =
        ir_group_timestep_base_info[group_idx];
    *(fw_timestep_base_info_t *)p_ir_buf = fw_timestep_base_info;
    p_ir_buf = (fw_timestep_base_info_t *)p_ir_buf + 1;

    uint32_t timestep_num =
        fw_timestep_base_info.ts_num_and_split_tensor_num >> 16;
    uint32_t group_input_num = (uint32_t)fw_timestep_base_info.input_tensor_num;
    uint32_t group_output_num =
        (uint32_t)fw_timestep_base_info.output_tensor_num;
    if (timestep_num > 1) {
      // local group
      // write input info
      int is_h_splited = fw_timestep_base_info.flags & 0x1;
      if (is_h_splited) {
        for (uint32_t i = 0; i < group_input_num; ++i) {
          *(fw_input_tensor_info_t *)p_ir_buf =
              ir_group_input_tensor_info[group_idx][i];
          p_ir_buf = (fw_input_tensor_info_t *)p_ir_buf + 1;
        }
      } else {
        for (uint32_t i = 0; i < group_input_num; ++i) {
          *(uint32_t *)p_ir_buf = ir_group_input_tensor_info[group_idx][i]
                                      .tensor_id_and_max_hslice >>
                                  16;
          p_ir_buf = (uint32_t *)p_ir_buf + 1;
        }
      }
      // write group out tensor id and consumer number
      for (uint32_t i = 0; i < group_output_num; ++i) {
        *(uint32_t *)p_ir_buf =
            ir_group_out_tensor_id_and_consumer_num[group_idx][i];
        p_ir_buf = (uint32_t *)p_ir_buf + 1;
      }

      // the following is the timestep info
      for (u32 ts_idx = 0; ts_idx < timestep_num; ++ts_idx) {
        // layer number
        const TpuTsField &timestep_layers =
            m_time_step_groups_[group_idx]->getLayers(ts_idx);
        const GdmaTsField &timestep_tensors =
            m_time_step_groups_[group_idx]->getTensors(ts_idx);
        *(u32 *)p_ir_buf = timestep_layers.size();
        p_ir_buf = (u32 *)p_ir_buf + 1;
        for (u32 i = 0; i < timestep_layers.size(); ++i) {
          if (get_dynamic_version() >= 2) {
            int64_t layer_type = -1;
            if (auto castOp =
                    dyn_cast<DynGlobalGenInterface>(timestep_layers[i])) {
              layer_type = castOp.get_fw_type_bm1684x();
            }
            assert(layer_type >= 0);
            p_ir_buf = write_local_layer_info_buffer_v2(
                p_ir_buf, timestep_layers[i], (FW_LAYER_TYPE_T)layer_type,
                m_time_step_groups_[group_idx]);
          } else {
            ir_layer_info_t ir_layer_info =
                ir_group_timestep_layer_param[group_idx][ts_idx][i];
            p_ir_buf = write_local_layer_info_buffer(
                p_ir_buf, timestep_layers[i], &ir_layer_info,
                m_time_step_groups_[group_idx]);
          }
        }

        // tensor gdma number
        u32 tensor_gdma_num = timestep_tensors.size();
        *(u32 *)p_ir_buf = tensor_gdma_num;
        p_ir_buf = (u32 *)p_ir_buf + 1;
        for (u32 i = 0; i < tensor_gdma_num; ++i) {
          ir_tensor_gdma_info_t ir_tensor_gdma_info =
              ir_group_timestep_tensor_gdma_param[group_idx][ts_idx][i];
          p_ir_buf =
              write_tensor_gdma_info_buffer(p_ir_buf, &ir_tensor_gdma_info);
        }
      }
    } else {
      // global layer
      // write input info
      for (uint32_t i = 0; i < group_input_num; ++i) {
        *(uint32_t *)p_ir_buf =
            ir_group_input_tensor_info[group_idx][i].tensor_id_and_max_hslice >>
            16;
        p_ir_buf = (uint32_t *)p_ir_buf + 1;
      }
      // write group out tensor id and consumer number
      for (uint32_t i = 0; i < group_output_num; ++i) {
        *(uint32_t *)p_ir_buf =
            ir_group_out_tensor_id_and_consumer_num[group_idx][i];
        p_ir_buf = (uint32_t *)p_ir_buf + 1;
      }
      const vector<Operation *> &layers = layer_group.group_ops;
      if (layers.size() > 0) {
        if (get_dynamic_version() >= 2) {
          int64_t layer_type = -1;
          if (auto castOp = dyn_cast<DynGlobalGenInterface>(layers[0])) {
            layer_type = castOp.get_fw_type_bm1684x();
          }
          assert(layer_type >= 0);
          p_ir_buf = write_global_layer_info_buffer_v2(
              p_ir_buf, layers[0], (FW_LAYER_TYPE_T)layer_type);
        } else {
          ir_layer_info_t ir_layer_info =
              ir_group_timestep_layer_param[group_idx][0][0];
          p_ir_buf = write_global_layer_info_buffer(p_ir_buf, &ir_layer_info);
        }
      }
    }
  }

  return (uint8_t *)p_ir_buf - (uint8_t *)ir_buffer;
}

void SubnetIr::write_binary_ir_to_buffer(std::unique_ptr<Context> &context) {
  uint32_t subnet_ir_offset =
      context->get_binary_ir().size() * sizeof(uint32_t);
  uint32_t fw_ir_length_in_word = (fw_ir_length + sizeof(uint32_t) - 1) /
                                  sizeof(uint32_t); // == subnet_ir_len;
  vector<uint32_t> &binary_ir_v = context->get_binary_ir();
  uint32_t size = binary_ir_v.size();
  binary_ir_v.resize(size + fw_ir_length_in_word);
  uint32_t *buffer = binary_ir_v.data() + size;

  int subnet_ir_len =
      write_ir_to_buffer((uint8_t *)buffer, net_input_tensor_id,
                         net_output_tensor_id, 0, m_layer_groups_.size() - 1);

  cout << "subnet_ir_len " << subnet_ir_len << " fw_ir_length " << fw_ir_length
       << endl;
  assert((uint32_t)subnet_ir_len <= fw_ir_length);
  set_ir_offset_len(subnet_ir_offset, subnet_ir_len);
  binary_ir_v.resize(size +
                     (subnet_ir_len + sizeof(uint32_t) - 1) / sizeof(uint32_t));
  uint32_t cur_net_ir_len = context->get_cur_net_ir_len();
  // context->set_cur_net_ir_len(cur_net_ir_len + subnet_ir_len);
  uint32_t aligned_net_ir_length = (subnet_ir_len + sizeof(uint32_t) - 1) /
                                   sizeof(uint32_t) * sizeof(uint32_t);
  context->set_cur_net_ir_len(cur_net_ir_len + aligned_net_ir_length);
  return;
}

bool SubnetIr::strip_back_judge(
    Value v, const LgInfo &lg_info, const std::multiset<Operation *> &op_set,
    const std::set<Value, value_cmp> &out_tensor_set) {
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

} // namespace tpu
} // namespace tpu_mlir
