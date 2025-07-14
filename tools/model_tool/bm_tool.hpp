//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"
#include "tpu_mlir/Builder/BM168x/bmodel.hpp"
#include "tpu_mlir/Builder/BM168x/bmodel_fbs.h"
#include <fstream>
#include <iostream>
#include <set>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace bmodel;
using namespace flatbuffers;
using namespace std;

#define FATAL(fmt, ...)                                                        \
  do {                                                                         \
    printf("[%s:%d] Error : " fmt "\n", __func__, __LINE__, ##__VA_ARGS__);    \
    exit(-1);                                                                  \
  } while (0)

static const char *type_name_array[] = {
    "float32", "float16", "int8",     "uint8", "int16", "uint16",
    "int32",   "uint32",  "bfloat16", "int4",  "uint4"};
static const int type_size_array[] = {4, 2, 1, 1, 2, 2, 4, 4};
static const int DATA_TYPE_NUM = sizeof(type_name_array) / sizeof(char *);

static const char *type_name(uint32_t data_type) {
  if (data_type >= DATA_TYPE_NUM) {
    return "unkown";
  }
  return type_name_array[data_type];
}

// print all model parameters by json format
void bm_print(const string &filename) {
  ModelCtx model_ctx(filename);
  if (!model_ctx) {
    FATAL("bmodel file[%s] is not correct", filename.c_str());
  }

  string json_text;
  Parser parser;
  parser.opts.output_default_scalars_in_json = true;
  if (true != parser.Parse(schema_text)) {
    FATAL("parse schema failed");
  }

  if (true != GenerateText(parser, model_ctx.data(), &json_text)) {
    FATAL("generate text failed");
  }
  cout << json_text << endl;
}

static string shape_str(const Shape *shape, bool n_dynamic = false,
                        bool h_w_dynamic = false) {
  auto size = shape->dim()->size();
  if (size == 0) {
    return "[ ]";
  }

  // h_w_dynamic and n_dynamic are used as warning flags in bmruntime now
  bool is_dynamic = n_dynamic || h_w_dynamic;
  string prefix = is_dynamic ? "max:" : "";

  stringstream ss;
  ss << "[";
  for (uint64_t index = 0; index < size; index++) {
    ss << prefix;
    ss << shape->dim()->Get(index);
    if ((index + 1) != size) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

static string tensor_str(const Tensor *tensor, bool is_output = false,
                         bool n_dynamic = false, bool h_w_dynamic = false,
                         int devid = -1) {
  stringstream ss;
  string prefix = is_output ? "output: " : "input: ";
  ss << prefix;
  if (devid >= 0) {
    ss << "devid(" << devid << ") ";
  }
  auto shape = tensor->shape()->Get(0);
  ss << tensor->name()->str() << ", "
     << shape_str(shape, n_dynamic, h_w_dynamic) << ", "
     << type_name(tensor->data_type()) << ", scale: " << tensor->scale()
     << ", zero_point: " << tensor->zero_point() << endl;
  return ss.str();
}

static void show(const NetParameter *parameter, bool dynamic = false) {
  auto input_tensors = parameter->input_tensor();
  auto output_tensors = parameter->output_tensor();

  for (uint32_t idx = 0; idx < input_tensors->size(); idx++) {
    cout << tensor_str(input_tensors->Get(idx), false, parameter->n_dynamic(),
                       parameter->h_w_dynamic());
  }

  for (uint32_t idx = 0; idx < output_tensors->size(); idx++) {
    cout << tensor_str(output_tensors->Get(idx), true);
  }
}

typedef std::pair<const bmodel::Tensor *, uint32_t> tensor_pair_t;

static void reorder(std::vector<tensor_pair_t> &pairs) {
  std::sort(pairs.begin(), pairs.end(),
            [](const tensor_pair_t &a, const tensor_pair_t &b) {
              if (a.second != b.second) {
                return a.second < b.second;
              }
              return a.first->index() <= b.first->index();
            });
}

// print brief model info
void bm_show(const string &filename, bool all) {
  ModelCtx model_ctx(filename);
  if (!model_ctx) {
    FATAL("file[%s] is not correct", filename.c_str());
  }
  auto model = model_ctx.model();
  cout << "bmodel version: " << model->type()->c_str() << "."
       << model->version()->c_str() << endl;
  cout << "chip: " << model->chip()->c_str();
  if (model->device_num() > 1) {
    cout << ",  device num: " << model->device_num();
  }
  cout << "\ncreate time: " << model->time()->c_str() << endl;
  // kernel_module info
  auto kernel_module = model->kernel_module();
  if (!kernel_module) {
    cout << "no kernel_module" << endl;
  } else {
    auto module_binary = kernel_module->binary();
    size_t module_size = module_binary->size();
    std::unique_ptr<uint8_t[]> binary(new uint8_t[module_size]);
    model_ctx.read_binary(module_binary, binary.get());
    cout << "kernel_module name: " << kernel_module->file_name()->c_str()
         << endl;
    cout << "kernel_module size: " << module_size << endl;
  }
  std::map<std::string, std::shared_ptr<vector<uint32_t>>> cascade_nets;
  for (uint32_t idx = 0; idx < model->net()->size(); idx++) {
    auto net = model->net()->Get(idx);
    auto cascade = net->cascade();
    if (cascade) {
      auto main_name = cascade->main_name()->str();
      if (!main_name.empty()) {
        auto it = cascade_nets.find(main_name);
        if (it != cascade_nets.end()) {
          it->second->push_back(idx);
        } else {
          auto net_idx = std::make_shared<vector<uint32_t>>();
          net_idx->push_back(idx);
          cascade_nets[main_name] = net_idx;
        }
        if (all == false) {
          continue;
        }
      }
    }
    auto parameter = net->parameter();
    if (parameter == NULL || parameter->size() == 0) {
      continue;
    }
    bool is_dynamic = (parameter->Get(0)->is_dynamic() != 0);
    string net_type = is_dynamic ? "dynamic" : "static";
    cout << "==========================================" << endl;
    cout << "net " << idx << ": [" << net->name()->c_str() << "]  " << net_type
         << endl;
    if (net->addr_mode()) {
      cout << "addr mode: " << net->addr_mode() << endl;
    }
    if (cascade) {
      cout << "device id: " << cascade->device_id() << endl;
    }
    for (uint32_t i = 0; i < parameter->size(); i++) {
      auto net_param = parameter->Get(i);
      auto subnet = net_param->sub_net();
      cout << "------------" << endl;
      cout << "stage " << i << ":" << endl;
      if (subnet != NULL && subnet->size() > 1) {
        cout << "subnet number: " << subnet->size() << endl;
      }
      show(parameter->Get(i), is_dynamic);
    }
  }
  for (auto &it : cascade_nets) {
    cout << "==========================================" << endl;
    cout << "net: [" << it.first << "]  cascade" << endl;
    // show inputs
    std::vector<tensor_pair_t> ins;
    std::vector<tensor_pair_t> outs;
    for (auto idx : *it.second) {
      auto net = model->net()->Get(idx);
      auto parameter = net->parameter()->Get(0);
      auto input_tensors = parameter->input_tensor();
      auto output_tensors = parameter->output_tensor();
      for (uint32_t idx = 0; idx < input_tensors->size(); idx++) {
        auto in = input_tensors->Get(idx);
        auto pair = tensor_pair_t(in, net->cascade()->device_id());
        if (in->hidden() == 1 || in->hidden() == 3) {
          ins.push_back(pair);
        } else if (in->hidden() == 2 || in->hidden() == 4) {
          outs.push_back(pair);
        }
      }
      for (uint32_t idx = 0; idx < output_tensors->size(); idx++) {
        auto out = output_tensors->Get(idx);
        auto pair = tensor_pair_t(out, net->cascade()->device_id());
        if (out->hidden() == 1 || out->hidden() == 3) {
          ins.push_back(pair);
        } else if (out->hidden() == 2 || out->hidden() == 4) {
          outs.push_back(pair);
        }
      }
    }
    reorder(ins);
    reorder(outs);
    for (size_t i = 0; i < ins.size(); i++) {
      cout << tensor_str(ins[i].first, false, false, false, ins[i].second);
    }
    for (size_t i = 0; i < outs.size(); i++) {
      cout << tensor_str(outs[i].first, true, false, false, outs[i].second);
    }
  }
  cout << std::endl;
  auto mem_info = model_ctx.get_bmodel_mem_info();
  cout << "device mem size: "
       << mem_info.coeff_mem_size + mem_info.neuron_mem_size +
              mem_info.bd_cmd_mem_size + mem_info.gdma_cmd_mem_size +
              mem_info.middle_buffer_size + mem_info.dynamic_ir_mem_size
       << " (weight: " << mem_info.coeff_mem_size << ", instruct: "
       << mem_info.bd_cmd_mem_size + mem_info.gdma_cmd_mem_size +
              mem_info.dynamic_ir_mem_size
       << ", runtime: "
       << mem_info.neuron_mem_size + mem_info.middle_buffer_size << ")"
       << std::endl;
  cout << "\t(coeff size: " << mem_info.coeff_mem_size
       << "; middle buffer size: " << mem_info.middle_buffer_size
       << "; neuron size: " << mem_info.neuron_mem_size << ")" << std::endl;
  cout << "host mem size: "
       << mem_info.host_coeff_mem_size + mem_info.host_neuron_mem_size
       << " (weight: " << mem_info.host_coeff_mem_size
       << ", runtime: " << mem_info.host_neuron_mem_size << ")" << std::endl;
}

// print chip of model
void bm_show_chip(const string &filename) {
  ModelCtx model_ctx(filename);
  if (!model_ctx) {
    FATAL("file[%s] is not correct", filename.c_str());
  }
  auto model = model_ctx.model();
  cout << model->chip()->c_str();
}

// print weight of model
void bm_show_weight(const string &filename) {
  ModelCtx model_ctx(filename);
  if (!model_ctx) {
    FATAL("file[%s] is not correct", filename.c_str());
  }
  auto model = model_ctx.model();
  auto num_net = model->net()->size();
  for (int i = 0; i < num_net; i++) {
    auto net = model->net()->Get(i);
    auto num_stage = model->net()->Get(i)->parameter()->size();
    for (int j = 0; j < num_stage; j++) {
      auto param = model->net()->Get(i)->parameter()->Get(j);
      auto coeff = param->coeff_mem();
      if (coeff == nullptr) {
        continue;
      }
      auto location = coeff->location();
      if (location == nullptr || location->size() == 0) {
        continue;
      }
      printf("net %d : \"%s\", stage:%d\n", i, net->name()->c_str(), j);
      cout << "-------------------------------" << endl;
      for (int k = 0; k < location->size(); k++) {
        auto info = location->Get(k);
        printf("%s : [0x%lx, 0x%lx)\n", info->name()->c_str(), info->offset(),
               info->offset() + info->size());
      }
      cout << "==========================================" << endl;
    }
  }
}

void bm_show_dynamic(const string &filename) {
  ModelCtx model_ctx(filename);
  if (!model_ctx) {
    FATAL("file[%s] is not correct", filename.c_str());
  }
  auto model = model_ctx.model();
  auto dyn_subnet_check = [&]() {
    auto subnet = model->net()->Get(0)->parameter()->Get(0)->sub_net();
    return subnet->Get(subnet->size() - 1)->is_dynamic();
  };
  if (model->net()->Get(0)->parameter()->Get(0)->is_dynamic() ||
      dyn_subnet_check()) {
    cout << "true";
  } else {
    cout << "false";
  }
}

// update binary data when copy one net to new flatbuffers
// it's a little complicated, using reflection of flatbuffers
static void update_table(Table *table, const StructDef *struct_def,
                         ModelGen &model_gen, ModelCtx &model_ctx,
                         bool skip_coeff = false, bool skip_cmd = false) {
  if (skip_coeff && struct_def->name == "CoeffMem") {
    return;
  }
  if (skip_cmd && struct_def->name == "CmdGroup") {
    return;
  }
  for (auto fd : struct_def->fields.vec) {
    if (false == table->CheckField(fd->value.offset)) {
      continue;
    }
    switch (fd->value.type.base_type) {
    case BASE_TYPE_STRUCT: {
      auto next_def = fd->value.type.struct_def;
      if (next_def->fixed) {
        if (next_def->name == "Binary") {
          auto binary = table->GetStruct<Binary *>(fd->value.offset);
          uint8_t *data = new uint8_t[binary->size()];
          model_ctx.read_binary(binary, data);
          auto new_binary = model_gen.WriteBinary(binary->size(), data);
          binary->mutate_start(new_binary.start());
          delete[] data;
        }
      } else {
        auto next_pointer = table->GetPointer<void *>(fd->value.offset);
        auto next_table = reinterpret_cast<Table *>(next_pointer);
        update_table(next_table, next_def, model_gen, model_ctx, skip_coeff,
                     skip_cmd);
      }
      break;
    }
    case BASE_TYPE_VECTOR: {
      auto pointer = table->GetPointer<void *>(fd->value.offset);
      auto type = fd->value.type.VectorType();
      if (type.base_type != BASE_TYPE_STRUCT) {
        break;
      }
      auto next_def = type.struct_def;
      if (next_def->fixed) {
        if (next_def->name == "Binary") {
          auto vector_pointer =
              reinterpret_cast<Vector<const Binary *> *>(pointer);
          for (uint32_t next_id = 0; next_id < vector_pointer->size();
               next_id++) {
            auto binary = vector_pointer->GetMutableObject(next_id);
            if (binary == nullptr || binary->size() == 0) {
              continue;
            }
            uint8_t *data = new uint8_t[binary->size()];
            model_ctx.read_binary(binary, data);
            auto new_binary = model_gen.WriteBinary(binary->size(), data);
            binary->mutate_start(new_binary.start());
            delete[] data;
          }
        }
        break;
      }
      auto vector_pointer = reinterpret_cast<Vector<Offset<void>> *>(pointer);
      for (uint32_t next_id = 0; next_id < vector_pointer->size(); next_id++) {
        auto next_pointer = vector_pointer->GetMutableObject(next_id);
        auto next_table = reinterpret_cast<Table *>(next_pointer);
        update_table(next_table, next_def, model_gen, model_ctx, skip_coeff,
                     skip_cmd);
      }
      break;
    }
    default: {
      break;
    }
    }
  }
}

// update whole model binary data
static void update_model(ModelGen &model_gen, ModelCtx &model_ctx,
                         bool skip_coeff = false) {
  Parser parser;
  parser.Parse(schema_text);
  auto buffer = model_gen.GetBufferPointer();
  auto root = GetMutableRoot<Table>(buffer);
  auto root_def = parser.root_struct_def_;
  update_table(root, root_def, model_gen, model_ctx, skip_coeff);
}

// update one net binary data
static void update_net(ModelGen &model_gen, ModelCtx &model_ctx,
                       uint32_t net_idx = 0, uint32_t sub_idx = 0,
                       bool skip_coeff = false, bool skip_cmd = false) {
  Parser parser;
  parser.Parse(schema_text);
  auto buffer = model_gen.GetBufferPointer();
  auto root_table = GetMutableRoot<Table>(buffer);
  auto root_def = parser.root_struct_def_;
  auto net_field = root_def->fields.Lookup("net");
  auto net_def = net_field->value.type.VectorType().struct_def;
  auto pointer = root_table->GetPointer<void *>(net_field->value.offset);
  auto net_pointer =
      reinterpret_cast<Vector<Offset<void>> *>(pointer)->GetMutableObject(
          net_idx);
  auto net_table = reinterpret_cast<Table *>(net_pointer);
  auto sub_net_field = net_def->fields.Lookup("parameter");
  auto sub_net_def = sub_net_field->value.type.VectorType().struct_def;
  auto sub_pointer = net_table->GetPointer<void *>(sub_net_field->value.offset);
  auto sub_net_pointer = reinterpret_cast<Vector<Offset<void>> *>(sub_pointer)
                             ->GetMutableObject(sub_idx);
  auto sub_net_table = reinterpret_cast<Table *>(sub_net_pointer);
  update_table(sub_net_table, sub_net_def, model_gen, model_ctx, skip_coeff,
               skip_cmd);
}

// bmodel combine coeff
struct addr_update_t {
  uint64_t addr;
  uint64_t size;
  int64_t offset;
};

uint64_t update_addr(const uint64_t coeff_limit, const int64_t ctx_offset,
                     uint64_t origin_addr, std::vector<addr_update_t> &addr_v) {
  if (origin_addr < coeff_limit) {
    // coeff addr
    for (int i = 0; i < addr_v.size(); ++i) {
      if (origin_addr >= addr_v[i].addr &&
          origin_addr < addr_v[i].addr + addr_v[i].size) {
        return origin_addr + addr_v[i].offset;
      }
    }
  } else {
    // neuron addr
    return origin_addr + ctx_offset;
  }
  return origin_addr;
}

inline void update_addr_1684x(uint32_t *cmd, uint64_t coeff_limit,
                              const int64_t ctx_offset,
                              std::vector<addr_update_t> &addr_v) {
  uint64_t addr = ((uint64_t)(cmd[1] & 0xff) << 32) | ((uint64_t)cmd[0]);
  uint64_t GLOBAL_MEM_START_ADDR = 0x100000000;
  if (addr >= GLOBAL_MEM_START_ADDR) {
    uint64_t fix_addr = update_addr(coeff_limit, ctx_offset, addr, addr_v);
    if (fix_addr != addr) {
      cmd[0] = fix_addr & 0xffffffff;
      cmd[1] = ((uint32_t)((fix_addr >> 32) & 0xff)) | (cmd[1] & 0xffffff00);
    }
  }
}

inline void update_addr_1688(uint32_t *cmd, uint64_t coeff_limit,
                             const int64_t ctx_offset,
                             std::vector<addr_update_t> &addr_v) {
  uint64_t addr = ((uint64_t)(cmd[1] & 0xff) << 32) | ((uint64_t)cmd[0]);
  if (((addr >> 39) & 0x1) &&
      (((addr >> 36) & 0x7) == 1)) { // tag == 1, coeff addr
    uint64_t fix_addr = addr & ((1ull << 36) - 1);
    for (int i = 0; i < addr_v.size(); ++i) {
      if (fix_addr >= addr_v[i].addr &&
          fix_addr < addr_v[i].addr + addr_v[i].size) {
        fix_addr += addr_v[i].offset;
        break;
      }
    }
    fix_addr |= (9ull << 36);
    if (fix_addr != addr) {
      cmd[0] = fix_addr & 0xffffffff;
      cmd[1] = ((uint32_t)((fix_addr >> 32) & 0xff)) | (cmd[1] & 0xffffff00);
    }
  }
}

void update_cmd(uint32_t *cmd, bool last_cmd, uint64_t coeff_limit,
                int64_t ctx_offset, std::vector<addr_update_t> &addr_v,
                std::string arch) {
  // cmd type: 0:DMA_tensor, 1:DMA_matrix, 2:DMA_masked_select, 3:DMA_general
  // 4:DMA_cw_trans, 5:DMA_nonzero, 6:DMA_sys, 7:DMA_gather, 8:DMA_scatter
  // 9:DMA_reverse 10:DMA_compress 11: DMA_decompress
  if ("BM1684X" == arch) {
    if (!last_cmd) {
      update_addr_1684x(cmd + 16, coeff_limit, ctx_offset, addr_v);
      update_addr_1684x(cmd + 18, coeff_limit, ctx_offset, addr_v);
      // fix index_tensor or mask_tensor addr
      int cmd_type = (cmd[1] & 0x0f);
      if (cmd_type == 2 || cmd_type == 7 || cmd_type == 8) {
        update_addr_1684x(cmd + 20, coeff_limit, ctx_offset, addr_v);
      }
    }
  } else if ("BM1688" == arch) {
    if (!last_cmd) {
      int cmd_type = (cmd[1] & 0x0f);
      if (cmd_type == 6)
        return;
      update_addr_1688(cmd + 16, coeff_limit, ctx_offset, addr_v);
      update_addr_1688(cmd + 18, coeff_limit, ctx_offset, addr_v);
      // fix index_tensor or mask_tensor addr
      if (cmd_type == 2 || cmd_type == 7 || cmd_type == 8 || cmd_type == 0xa ||
          cmd_type == 0xb) {
        update_addr_1688(cmd + 20, coeff_limit, ctx_offset, addr_v);
      }
    }
  } else {
    FATAL("Unkown BM TPU");
  }
}

static uint32_t get_gdma_cmd_len(const uint8_t *gdma_buffer,
                                 uint64_t start_offset, bool last_cmd,
                                 std::string arch) {
  uint32_t len = 96; // default: common gdma instrution size

  if ("BM1688" == arch || "BM1690" == arch || "MARS3" == arch ||
      "SGTPUV8" == arch || "SG2380" == arch) {
    uint32_t cmd_head[2] = {0};
    memcpy(cmd_head, gdma_buffer + start_offset, sizeof(cmd_head));
    uint32_t tsk_type = cmd_head[1] & 0xf;
    if (tsk_type == 0x6) { // DMA_sys
      len = 16;
    }
    // sys end
    if (last_cmd) {
      len = (start_offset + 16 - 1 + 128) / 128 - start_offset;
    }
  } else if ("BM1684X" == arch) {
    // sys end
    if (last_cmd) {
      len = (start_offset + 16 - 1 + 128) / 128 - start_offset;
    }
  } else {
    FATAL("Unkown BM TPU");
  }
  return len;
}

void update_cmd_group(
    ModelGen &model_gen, ModelCtx *model_ctx,
    const std::vector<std::unique_ptr<bmodel::CmdGroupT>> &cmd_group,
    std::vector<addr_update_t> &addr_update_v, uint64_t coeff_limit,
    int64_t ctx_offset) {
  uint32_t gdam_total_cmd_byte = 0;
  if (cmd_group.size() == 0) {
    return;
  }
  for (uint32_t i = 0; i < cmd_group.size(); i++) {
    gdam_total_cmd_byte += cmd_group[i]->gdma_cmd_byte;
  }
  if (gdam_total_cmd_byte > 0) {
    for (uint32_t group_idx = 0; group_idx < cmd_group.size(); group_idx++) {
      uint64_t gdma_offset = 0;
      // copy bdc command, do not change
      if (cmd_group[group_idx]->bdc_num > 0) {
        uint8_t *data = new uint8_t[cmd_group[group_idx]->binary_bdc->size()];
        model_ctx->read_binary(cmd_group[group_idx]->binary_bdc.get(), data);
        auto new_binary = model_gen.WriteBinary(
            cmd_group[group_idx]->binary_bdc->size(), data);
        cmd_group[group_idx]->binary_bdc->mutate_start(new_binary.start());
        delete[] data;
      }
      if (0 == cmd_group[group_idx]->gdma_num) {
        continue;
      }
      // update gdma command, include coeff & neuron addr
      Binary new_binary;
      uint8_t *gdma_buffer =
          new uint8_t[cmd_group[group_idx]->binary_gdma->size()];
      model_ctx->read_binary(cmd_group[group_idx]->binary_gdma.get(),
                             gdma_buffer);
      for (uint32_t cmd_idx = 0; cmd_idx < cmd_group[group_idx]->gdma_num - 1;
           cmd_idx++) {
        uint32_t gdma_size = get_gdma_cmd_len(
            gdma_buffer, gdma_offset, false, model_ctx->model()->chip()->str());
        update_cmd((uint32_t *)(gdma_buffer + gdma_offset), false, coeff_limit,
                   ctx_offset, addr_update_v,
                   model_ctx->model()->chip()->str());
        gdma_offset += gdma_size;
      }
      new_binary = model_gen.WriteBinary(
          cmd_group[group_idx]->binary_gdma->size(), gdma_buffer);
      cmd_group[group_idx]->binary_gdma->mutate_start(new_binary.start());
      cmd_group[group_idx]->binary_gdma->mutate_size(new_binary.size());
      delete[] gdma_buffer;
    }
  }
}

void update_static_cmd(ModelGen &model_gen, ModelCtx *model_ctx,
                       const NetParameterT *param,
                       std::vector<addr_update_t> &addr_update_v,
                       uint64_t coeff_limit, int64_t ctx_offset) {
  const auto core_num = param->core_num != 0 ? param->core_num : 1;
  auto &cmd_group = param->cmd_group;
  if (param->sub_net.size() > 0) {
    for (auto &subnet : param->sub_net) {
      auto &core_commands = subnet->core_commands;
      if (core_commands.size() > 0) {
        for (uint32_t core_idx = 0; core_idx < core_num; core_idx++) {
          auto &cmd_group = core_commands[core_idx]->gdma_tiu_commands;
          update_cmd_group(model_gen, model_ctx, cmd_group, addr_update_v,
                           coeff_limit, ctx_offset);
        }
      } else {
        auto &cmd_group = subnet->cmd_group;
        update_cmd_group(model_gen, model_ctx, cmd_group, addr_update_v,
                         coeff_limit, ctx_offset);
      }
    }
  }
  update_cmd_group(model_gen, model_ctx, cmd_group, addr_update_v, coeff_limit,
                   ctx_offset);
}

void update_tensor(const NetParameterT *param, const int64_t ctx_offset) {
  for (auto &tensor : param->input_tensor) {
    tensor->device_addr += ctx_offset;
  }
  for (auto &tensor : param->output_tensor) {
    tensor->device_addr += ctx_offset;
  }
  for (auto &subnet : param->sub_net) {
    for (auto &tensor : subnet->input_tensor) {
      tensor->device_addr += ctx_offset;
    }
    for (auto &tensor : subnet->output_tensor) {
      tensor->device_addr += ctx_offset;
    }
  }
}

struct location_t {
  string name;
  uint64_t offset;
  uint64_t size;
};

uint64_t coeff_combine(uint8_t *base_buffer, uint64_t base_size,
                       std::vector<location_t> *location_vector,
                       uint8_t *coeff_buffer, uint64_t coeff_size,
                       const Vector<Offset<bmodel::Location>> *coeff_locations,
                       bool is_first,
                       std::vector<addr_update_t> *addr_update_v) {
  uint64_t buffer_offset = 0;
  addr_update_v->clear();
  flatbuffers::FlatBufferBuilder fbb;
  if (is_first) {
    assert(base_size == 0);
    // copy first coeff to buffer.
    memcpy(base_buffer, coeff_buffer, coeff_size);
    // copy first location to location_vector.
    for (int i = 0; i < coeff_locations->size(); ++i) {
      auto location = coeff_locations->Get(i);
      location_t loc;
      if (location->offset() >= coeff_size) {
        continue;
      }
      loc.name.assign(location->name()->str());
      loc.offset = location->offset();
      loc.size = location->size();
      location_vector->push_back(loc);
    }
    return coeff_size;
  }
  // check & copy coeff, update location_vector, update addr_update_v
  buffer_offset += base_size;
  auto location_num = location_vector->size();
  for (int i = 0; i < coeff_locations->size(); ++i) {
    auto location = coeff_locations->Get(i);
    if (location->offset() >= coeff_size) {
      continue;
    }
    bool is_same = false;
    for (int j = 0; j < location_num; ++j) {
      auto location_base = location_vector->at(j);
      if (location_base.name != location->name()->str()) {
        continue;
      }
      if (location_base.size != location->size()) {
        continue;
      }
      if (memcmp(base_buffer + location_base.offset,
                 coeff_buffer + location->offset(), location->size()) == 0) {
        addr_update_t addr_update;
        addr_update.addr = location->offset();
        addr_update.size = location->size();
        // TODO: coeff base addr
        addr_update.offset = location_base.offset - location->offset();
        if (addr_update.offset != 0) {
          addr_update_v->push_back(addr_update);
        }
        is_same = true;
        break;
      }
    }
    if (is_same) {
      continue;
    }
    addr_update_t addr_update;
    addr_update.addr = location->offset();
    addr_update.size = location->size();
    addr_update.offset = buffer_offset - location->offset();
    addr_update_v->push_back(addr_update);
    location_t loc;
    loc.name = location->name()->str();
    loc.offset = buffer_offset;
    loc.size = location->size();
    location_vector->push_back(loc);
    memcpy(base_buffer + buffer_offset, coeff_buffer + location->offset(),
           location->size());
    buffer_offset += location->size();
  }
  return buffer_offset;
}

// extract multi-net bmodel to multi one-net bmodels
void bm_extract(const string &filename) {
  ModelCtx model_ctx(filename);
  if (!model_ctx) {
    FATAL("file[%s] is not correct", filename.c_str());
  }
  auto model = model_ctx.model();
  for (uint32_t net_idx = 0; net_idx < model->net()->size(); net_idx++) {
    auto net = model->net()->Get(net_idx);
    string net_name = net->name()->str();
    if (net->parameter() == NULL || net->parameter()->size() == 0) {
      continue;
    }
    for (uint32_t idx = 0; idx < net->parameter()->size(); idx++) {
      ModelGen model_gen(model_ctx.header().binary_size);
      auto &builder = model_gen.Builder();
      auto parameter = net->parameter()->Get(idx);
      auto netT = parameter->UnPack();
      auto net_offset = NetParameter::Pack(builder, netT);
      delete netT;
      model_gen.AddChip(model->chip()->str());
      model_gen.AddNet(net_name, net_offset);
      model_gen.Finish();
      update_model(model_gen, model_ctx);
      ostringstream filename;
      filename << "bm_net" << net_idx << "_stage" << idx << ".bmodel";
      cout << "Generate file [" << filename.str() << "] ......" << endl;
      model_gen.Save(filename.str());
    }
  }
  cout << "Success: all files have been generated!" << endl;
}

// combine bmodels
typedef struct {
  uint32_t net_idx;
  uint32_t stage_idx;
  char *input;
  size_t input_size;
  char *output;
  size_t output_size;
} NET_INDEX_T;

typedef struct {
  shared_ptr<ModelCtx> model_ctx;
  ifstream input_f;
  ifstream output_f;
  vector<shared_ptr<NET_INDEX_T>> net_index_v;
} MODEL_CTX_T;

static shared_ptr<ofstream> g_input_ref;
static shared_ptr<ofstream> g_output_ref;

static size_t tensor_bytes(const Vector<Offset<Tensor>> *tensor) {
  size_t size = 0;
  for (uint32_t idx = 0; idx < tensor->size(); idx++) {
    auto type = tensor->Get(idx)->data_type();
    if (type >= DATA_TYPE_NUM) {
      FATAL("unknown data type[%u]", type);
    }
    size_t lsize = type_size_array[type];
    auto shape = tensor->Get(idx)->shape()->Get(0)->dim();
    for (uint32_t i = 0; i < shape->size(); i++) {
      lsize *= shape->Get(i);
    }
    size += lsize;
  }
  return size;
}

static void read_input_output_ref(const NetParameter *param, ifstream &fin_ref,
                                  ifstream &fout_ref, NET_INDEX_T *net_idx) {
  net_idx->input_size = tensor_bytes(param->input_tensor());
  net_idx->output_size = tensor_bytes(param->output_tensor());
  net_idx->input = new char[net_idx->input_size];
  net_idx->output = new char[net_idx->output_size];
  fin_ref.read(net_idx->input, net_idx->input_size);
  fout_ref.read(net_idx->output, net_idx->output_size);
}

static bool write_input_output_ref(vector<shared_ptr<MODEL_CTX_T>> &model_vec,
                                   uint32_t net_idx, uint32_t stage_idx) {
  for (auto &model_info : model_vec) {
    for (auto &net_index : model_info->net_index_v) {
      if (net_index->net_idx == net_idx && net_index->stage_idx == stage_idx) {
        g_input_ref->write(net_index->input, net_index->input_size);
        g_output_ref->write(net_index->output, net_index->output_size);
        delete[] net_index->input;
        delete[] net_index->output;
        return true;
      }
    }
  }
  return false;
}

static void write_input_output_ref(vector<shared_ptr<MODEL_CTX_T>> &model_vec) {
  for (int net_idx = 0; net_idx < 256; net_idx++) {
    for (int stage_idx = 0; true; stage_idx++) {
      bool ret = write_input_output_ref(model_vec, net_idx, stage_idx);
      if (ret == true) {
        continue;
      } else if (stage_idx == 0) {
        return;
      } else {
        break;
      }
    }
  }
}

static bool add_kernel_module(ModelGen &model_gen,
                              shared_ptr<ModelCtx> &model_ctx) {
  auto km = model_ctx->model()->kernel_module();
  if (km) {
    auto binary = km->binary();
    uint8_t *data = new uint8_t[binary->size()];
    model_ctx->read_binary(binary, data);
    auto new_binary = model_gen.WriteBinary(binary->size(), data);
    auto filename = km->file_name()->str();
    model_gen.AddKernelModule(filename, new_binary);
    delete[] data;
    return true;
  }
  return false;
}

static void combine_bmodels(ModelGen &model_gen,
                            vector<shared_ptr<MODEL_CTX_T>> &model_vec,
                            bool is_dir = false) {
  model_gen.AddChip(model_vec[0]->model_ctx->model()->chip()->str());
  auto &builder = model_gen.Builder();
  bool kernel_load = false;
  uint32_t device_num = 0;
  for (uint32_t model_idx = 0; model_idx < model_vec.size(); model_idx++) {
    auto &model_info = model_vec[model_idx];
    auto model = model_info->model_ctx->model();
    if (model->device_num() > device_num) {
      device_num = model->device_num();
    }
    if (kernel_load == false) {
      kernel_load = add_kernel_module(model_gen, model_info->model_ctx);
    }
    for (uint32_t net_idx = 0; net_idx < model->net()->size(); net_idx++) {
      auto net = model->net()->Get(net_idx);
      if (net->parameter() == NULL || net->parameter()->size() == 0) {
        continue;
      }
      auto net_name = net->name()->str();
      auto cascade = net->cascade();
      if (cascade) {
        // no more stage
        assert(net->parameter()->size() == 1);
      }
      auto addr_mode = net->addr_mode();
      for (uint32_t idx = 0; idx < net->parameter()->size(); idx++) {
        shared_ptr<NET_INDEX_T> net_idx(new NET_INDEX_T);
        if (is_dir) {
          read_input_output_ref(net->parameter()->Get(idx), model_info->input_f,
                                model_info->output_f, net_idx.get());
        }
        auto netT = net->parameter()->Get(idx)->UnPack();
        auto net_offset = NetParameter::Pack(builder, netT);
        model_gen.AddNet(net_name, net_offset, &net_idx->net_idx,
                         &net_idx->stage_idx, cascade, addr_mode);
        delete netT;
        model_info->net_index_v.push_back(net_idx);
      }
    }
  }
  model_gen.AddNumDevice(device_num);
  model_gen.Finish();
  for (uint32_t idx = 0; idx < model_vec.size(); idx++) {
    auto &model_info = model_vec[idx];
    for (auto &net_index : model_info->net_index_v) {
      update_net(model_gen, *model_info->model_ctx, net_index->net_idx,
                 net_index->stage_idx);
    }
  }
  if (is_dir) {
    write_input_output_ref(model_vec);
  }
}

static void combine_bmodels_coeff(ModelGen &model_gen,
                                  vector<shared_ptr<MODEL_CTX_T>> &model_vec,
                                  bool is_dir = false) {
  model_gen.AddChip(model_vec[0]->model_ctx->model()->chip()->str());
  auto &builder = model_gen.Builder();
  uint32_t device_num = 0;

  std::vector<uint8_t> base_buffer;
  std::vector<uint8_t> dynamic_buffer;
  uint64_t base_size = 0, dynamic_base_size = 0;
  std::vector<location_t> loc_v;
  std::vector<std::vector<addr_update_t>> addr_update_v(model_vec.size());
  uint64_t coeff_addr;
  // first loop: generate combined coeff & location
  for (uint32_t model_idx = 0; model_idx < model_vec.size(); model_idx++) {
    auto &model_info = model_vec[model_idx];
    auto model = model_info->model_ctx->model();
    auto param = model->net()->Get(0)->parameter()->Get(0);
    // check model is signal net, signal stage. or combined model
    assert(model->net()->size() == 1 &&
               model->net()->Get(0)->parameter()->size() == 1 ||
           model->bmodel_type() == 1);

    uint64_t start = param->coeff_mem()->binary_coeff()->start();
    uint64_t size = param->coeff_mem()->binary_coeff()->size();
    for (uint32_t net_idx = 0; net_idx < model->net()->size(); net_idx++) {
      auto net = model->net()->Get(net_idx);
      for (uint32_t idx = 0; idx < net->parameter()->size(); idx++) {
        auto parameter = net->parameter()->Get(idx);
        if (model->bmodel_type() == 1) {
          // use first net/stage to combine
          assert(start == parameter->coeff_mem()->binary_coeff()->start());
          assert(size == parameter->coeff_mem()->binary_coeff()->size());
        }
        assert(parameter->is_dynamic() == 0);
        coeff_addr = parameter->coeff_mem()->address();
      }
    }
    // combine coeff
    auto coeff = param->coeff_mem()->binary_coeff();
    base_buffer.resize(base_size + param->dynamic_combined_coeff_offset());
    std::unique_ptr<uint8_t[]> new_buffer(new uint8_t[coeff->size()]);
    model_info->model_ctx->read_binary(coeff, new_buffer.get());
    if (param->coeff_mem()->location() == NULL) {
      assert(0);
    }
    base_size = coeff_combine(
        base_buffer.data(), base_size, &loc_v, new_buffer.get(),
        param->dynamic_combined_coeff_offset(), param->coeff_mem()->location(),
        model_idx == 0, &(addr_update_v[model_idx]));
    // combine dynamic coeff
    auto dynamic_coeff_size =
        coeff->size() - param->dynamic_combined_coeff_offset();
    dynamic_buffer.resize(dynamic_base_size + dynamic_coeff_size);
    memcpy(dynamic_buffer.data() + dynamic_base_size,
           new_buffer.get() + param->dynamic_combined_coeff_offset(),
           dynamic_coeff_size);
    dynamic_base_size += dynamic_coeff_size;
    // update addr_v
    if (model->chip()->str() == "BM1684X") {
      // BM1684X save real address. addr = base_addr + offset;
      auto cur_coeff_addr = param->coeff_mem()->address();
      for (int i = 0; i < addr_update_v[model_idx].size(); ++i) {
        addr_update_v[model_idx][i].addr += cur_coeff_addr;
      }
    }
  }
  base_buffer.resize(base_size + dynamic_base_size);
  memcpy(base_buffer.data() + base_size, dynamic_buffer.data(),
         dynamic_base_size);
  // dynamic coeff location
  for (uint32_t model_idx = 0; model_idx < model_vec.size(); model_idx++) {
    auto &model_info = model_vec[model_idx];
    auto model = model_info->model_ctx->model();
    auto param = model->net()->Get(0)->parameter()->Get(0);
    for (int i = 0; i < param->coeff_mem()->location()->size(); ++i) {
      auto location = param->coeff_mem()->location()->Get(i);
      location_t loc;
      if (location->offset() < param->dynamic_combined_coeff_offset()) {
        continue;
      }
      loc.name.assign(location->name()->str());
      loc.offset = location->offset() - param->dynamic_combined_coeff_offset() +
                   base_size;
      loc.size = location->size();
      loc_v.push_back(loc);
    }
  }
  uint64_t dynamic_offset = base_size;
  base_size += dynamic_base_size;

  // second loop: update net param, update instruction
  Binary new_binary;
  new_binary = model_gen.WriteBinary(base_size, base_buffer.data());
  std::vector<uint8_t> crc32(bmodel::SHA256_LEN);
  bmodel::CalcSha256(base_buffer.data(), base_size, crc32.data());
  for (uint32_t model_idx = 0; model_idx < model_vec.size(); model_idx++) {
    auto &model_info = model_vec[model_idx];
    auto model = model_info->model_ctx->model();
    auto p = model->net()->Get(0)->parameter()->Get(0);
    auto dynamic_size = p->coeff_mem()->binary_coeff()->size() -
                        p->dynamic_combined_coeff_offset();
    auto dynamic_changed = dynamic_offset - p->dynamic_combined_coeff_offset();
    for (uint32_t net_idx = 0; net_idx < model->net()->size(); net_idx++) {
      auto net = model->net()->Get(net_idx);
      if (net->parameter() == NULL || net->parameter()->size() == 0) {
        continue;
      }
      auto net_name = net->name()->str();
      auto cascade = net->cascade();
      if (cascade) {
        // no more stage
        assert(net->parameter()->size() == 1);
      }
      auto addr_mode = net->addr_mode();
      for (uint32_t idx = 0; idx < net->parameter()->size(); idx++) {
        shared_ptr<NET_INDEX_T> netidx(new NET_INDEX_T);
        if (is_dir) {
          read_input_output_ref(net->parameter()->Get(idx), model_info->input_f,
                                model_info->output_f, netidx.get());
        }
        auto param = net->parameter()->Get(idx)->UnPack();
        auto cur_coeff_addr = param->coeff_mem->address;
        // update coeff mem
        param->coeff_mem->binary_coeff->mutate_start(new_binary.start());
        param->coeff_mem->binary_coeff->mutate_size(new_binary.size());
        param->coeff_mem->check_code = crc32;
        int location_size = param->coeff_mem->location.size();
        int i = 0;
        for (; i < location_size; ++i) {
          param->coeff_mem->location[i]->name = loc_v[i].name;
          param->coeff_mem->location[i]->offset = loc_v[i].offset;
          param->coeff_mem->location[i]->size = loc_v[i].size;
        }
        for (; i < loc_v.size(); ++i) {
          std::unique_ptr<LocationT> loc(new LocationT);
          loc->name = loc_v[i].name;
          loc->offset = loc_v[i].offset;
          loc->size = loc_v[i].size;
          param->coeff_mem->location.push_back(std::move(loc));
        }
        param->coeff_mem->address = coeff_addr;
        if (cur_coeff_addr != coeff_addr) {
          FATAL("Failed to combine coeff. Coeff base addr is not same.");
        }
        int64_t ctx_offset =
            coeff_addr + base_size -
            (param->io_size > 0 ? param->io_addr : param->ctx_addr);
        ctx_offset = ctx_offset > 0 ? ctx_offset : 0;
        // update static cmd
        uint64_t coeff_limit =
            param->io_size > 0 ? param->io_addr : param->ctx_addr;
        update_static_cmd(model_gen, model_info->model_ctx.get(), param,
                          addr_update_v[model_idx], coeff_limit, ctx_offset);
        if (param->io_size > 0) {
          param->io_addr += ctx_offset;
        }
        param->ctx_addr += ctx_offset;
        // update tensor
        update_tensor(param, ctx_offset);
        // update dynamic coeff offset
        if (model->bmodel_type() == 1) {
          param->dynamic_combined_coeff_offset += dynamic_changed;
        } else {
          param->dynamic_combined_coeff_offset = dynamic_offset;
        }
        auto net_offset = NetParameter::Pack(builder, param);
        model_gen.AddNet(net_name, net_offset, &netidx->net_idx,
                         &netidx->stage_idx, cascade, addr_mode);
        delete param;
        model_info->net_index_v.push_back(netidx);
      }
      dynamic_offset += dynamic_size;
    }
  }
  addr_update_v.clear();
  loc_v.clear();

  auto kernel_module = model_vec[0]->model_ctx->model()->kernel_module();
  if (kernel_module) {
    auto module_binary = kernel_module->binary();
    auto module_name = kernel_module->file_name()->str();
    std::unique_ptr<uint8_t[]> binary(new uint8_t[module_binary->size()]);
    model_vec[0]->model_ctx->read_binary(module_binary, binary.get());
    auto module_tmp =
        model_gen.WriteBinary(module_binary->size(), binary.get());
    model_gen.AddKernelModule(module_name, module_tmp);
  }
  auto cpuop_module = model_vec[0]->model_ctx->model()->cpuop_module();
  if (cpuop_module) {
    auto module_binary = cpuop_module->binary();
    auto module_name = cpuop_module->file_name()->str();
    std::unique_ptr<uint8_t[]> binary(new uint8_t[module_binary->size()]);
    model_vec[0]->model_ctx->read_binary(module_binary, binary.get());
    auto module_tmp =
        model_gen.WriteBinary(module_binary->size(), binary.get());
    model_gen.AddCpuModule(module_name, module_tmp);
  }
  model_gen.AddNumDevice(device_num);
  model_gen.AddBmodelType(1);
  model_gen.Finish();
  for (uint32_t idx = 0; idx < model_vec.size(); idx++) {
    auto &model_info = model_vec[idx];
    for (auto &net_index : model_info->net_index_v) {
      update_net(model_gen, *model_info->model_ctx, net_index->net_idx,
                 net_index->stage_idx, true, true);
    }
  }
  if (is_dir) {
    write_input_output_ref(model_vec);
  }
}

static bool make_directory(const char *dirname) {
  if (dirname == NULL || dirname[0] == '\0') {
    return false;
  }
  string dname = dirname;
  if (dname.back() != '/' && dname.back() != '\\') {
    dname += '/';
  }
  if (dname.length() >= 256) {
    return false;
  }
  char tmpDirPath[256] = {0};
  for (uint32_t i = 0; i < dname.length(); ++i) {
    tmpDirPath[i] = dname[i];
    if (tmpDirPath[i] == '\\' || tmpDirPath[i] == '/') {
      if (access(tmpDirPath, 0) != 0) {
        int32_t ret = mkdir(tmpDirPath, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (ret != 0) {
          return false;
        }
      }
    }
  }
  return true;
}

static void prepare_output(string &path, bool is_dir = false) {
  if (is_dir) {
    if (path.empty()) {
      path = "bm_combined";
    }
    if (false == make_directory(path.c_str())) {
      FATAL("mkdir[%s] failed", path.c_str());
    }
    g_input_ref = make_shared<ofstream>(path + "/input_ref_data.dat",
                                        ios::trunc | ios::binary);
    g_output_ref = make_shared<ofstream>(path + "/output_ref_data.dat",
                                         ios::trunc | ios::binary);
    path += "/compilation.bmodel";
  } else {
    if (path.empty()) {
      path = "bm_combined.bmodel";
    }
  }
}

// combine bmodels
void bm_combine_bmodels(int argc, char **argv, bool is_dir = false,
                        bool combine_coeff = false) {
  vector<shared_ptr<MODEL_CTX_T>> model_vec;

  string ofile = "";
  for (int index = 2; index < argc; index++) {
    string param = argv[index];
    if (param == "-o") {
      index++;
      if (index >= argc) {
        FATAL("there is no output filename");
      }
      ofile = argv[index];
      continue;
    }
    shared_ptr<MODEL_CTX_T> model_info(new MODEL_CTX_T);
    if (is_dir == false) {
      model_info->model_ctx = make_shared<ModelCtx>(param);
    } else {
      model_info->model_ctx =
          make_shared<ModelCtx>(param + "/compilation.bmodel");
      model_info->input_f.open(param + "/input_ref_data.dat");
      model_info->output_f.open(param + "/output_ref_data.dat");
      if (model_info->input_f.fail() || model_info->output_f.fail()) {
        FATAL("dir[%s] is not correct", param.c_str());
      }
    }
    if (model_info->model_ctx == NULL || !(*model_info->model_ctx)) {
      FATAL("file[%s] is not correct", param.c_str());
    }
    model_vec.push_back(model_info);
  }
  prepare_output(ofile, is_dir);
  ModelGen model_gen;
  if (combine_coeff) {
    combine_bmodels_coeff(model_gen, model_vec, is_dir);
  } else {
    combine_bmodels(model_gen, model_vec, is_dir);
  }
  model_gen.Save(ofile);
  cout << "Success: combined to [" << ofile << "]." << endl;
}

// read binary from bmodel
static uint64_t str2ull(const char *str) {
  string ull_str(str);
  if (ull_str.empty()) {
    return 0;
  }
  if (ull_str.compare(0, 2, "0x") == 0 || ull_str.compare(0, 2, "0X") == 0) {
    return strtoull(ull_str.c_str(), 0, 16);
  } else {
    return strtoull(ull_str.c_str(), 0, 10);
  }
}

void bm_dump_binary(int argc, char **argv) {
  if (argc != 6) {
    FATAL("--dump parameter error.");
  }
  ModelCtx model(argv[2]);
  if (!model) {
    FATAL("file[%s] is not correct", argv[2]);
  }
  uint64_t start = str2ull(argv[3]);
  uint64_t size = str2ull(argv[4]);
  if (size == 0 || (start + size) > model.header().binary_size) {
    FATAL("start[0x%lx] size[0x%lx] is not supproted\n", start, size);
  }

  ofstream ofile(argv[5], ios::out | ios::trunc | ios::binary);
  if (!ofile) {
    FATAL("save file[%s] failed\n", argv[5]);
  }
  uint8_t *data = new uint8_t[size];
  Binary binary(start, size);
  model.read_binary(&binary, data);
  ofile.write((char *)data, size);
  ofile.close();
  delete[] data;
  printf("save file[%s] success\n", argv[5]);
}
