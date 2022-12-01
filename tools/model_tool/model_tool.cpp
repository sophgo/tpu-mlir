//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <fstream>
#include <unistd.h>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <sys/stat.h>
#include "tpu_mlir/Builder/BM168x/bmodel.hpp"
#include "flatbuffers/idl.h"
#include "flatbuffers/util.h"
#include "tpu_mlir/Builder/BM168x/bmodel_fbs.h"

using namespace bmodel;
using namespace flatbuffers;
using namespace std;

#define FATAL(fmt, ...)                                                     \
  do {                                                                      \
    printf("[%s:%d] Error : " fmt "\n", __func__, __LINE__, ##__VA_ARGS__); \
    exit(-1);                                                               \
  } while (0)

// show usage of tool
static void usage(void)
{
  cout << "Usage:" << endl;
  cout << "  model_tool" << endl
       << "    --info model_file : show brief model info" << endl
       << "    --print model_file : show detailed model info" << endl
       << "    --extract model_file : extract one multi-net bmodel to multi one-net bmodels" << endl
       << "    --combine file1 .. fileN -o new_file: combine bmodels to one bmodel by filepath" << endl
       << "    --combine_dir dir1 .. dirN -o new_dir: combine bmodels to one bmodel by directory path" << endl
       << "    --dump model_file start_offset byte_size out_file: dump binary data to file from bmodel" << endl
       << endl;
}

// print all model parameters by json format
static void print(const string &filename)
{
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

static const char *type_name_array[] = {"float32", "float16", "int8", "uint8", "int16", "uint16", "int32", "uint32"};
static const int  type_size_array[]  = {4, 2, 1, 1, 2, 2, 4, 4};
static const int DATA_TYPE_NUM = sizeof(type_name_array) / sizeof(char *);

static const char *type_name(uint32_t data_type)
{
  if (data_type >= DATA_TYPE_NUM) {
    return "unkown";
  }
  return type_name_array[data_type];
}

static string shape_str(const Shape *shape, bool n_dynamic = false,
                        bool h_w_dynamic = false)
{
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
                         bool n_dynamic = false, bool h_w_dynamic = false)
{
  stringstream ss;
  string prefix = is_output ? "output: " : "input: ";
  ss << prefix;
  auto shape = tensor->shape()->Get(0);
  ss << tensor->name()->str() << ", " << shape_str(shape, n_dynamic, h_w_dynamic) << ", "
     << type_name(tensor->data_type()) << ", scale: " << tensor->scale() << endl;
  return ss.str();
}

static void show(const NetParameter *parameter, bool dynamic = false)
{
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

// print brief model info
static void show(const string &filename)
{
  ModelCtx model_ctx(filename);
  if (!model_ctx) {
    FATAL("file[%s] is not correct", filename.c_str());
  }
  auto model = model_ctx.model();
  cout << "bmodel version: " << model->type()->c_str() << "." << model->version()->c_str() << endl;
  cout << "chip: " << model->chip()->c_str() << endl;
  cout << "create time: " << model->time()->c_str() << endl;

  for (uint32_t idx = 0; idx < model->net()->size(); idx++) {
    auto net = model->net()->Get(idx);
    auto parameter = net->parameter();
    if (parameter == NULL || parameter->size() == 0) {
      continue;
    }
    bool is_dynamic = (parameter->Get(0)->is_dynamic() != 0);
    string net_type = is_dynamic ? "dynamic" : "static";
    cout << "==========================================" << endl;
    cout << "net " << idx << ": [" << net->name()->c_str() << "]  " << net_type << endl;
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
    auto mem_info = model_ctx.get_bmodel_mem_info();
    cout << std::endl;
    cout << "device mem size: "<<
            mem_info.coeff_mem_size +
            mem_info.neuron_mem_size +
            mem_info.bd_cmd_mem_size +
            mem_info.gdma_cmd_mem_size +
            mem_info.middle_buffer_size +
            mem_info.dynamic_ir_mem_size
         << " (coeff: "<<mem_info.coeff_mem_size
         << ", instruct: "<<mem_info.bd_cmd_mem_size + mem_info.gdma_cmd_mem_size+mem_info.dynamic_ir_mem_size
         << ", runtime: "<<mem_info.neuron_mem_size + mem_info.middle_buffer_size
         << ")"<< std::endl;
    cout << "host mem size: "<<
            mem_info.host_coeff_mem_size +
            mem_info.host_neuron_mem_size
         << " (coeff: "<<mem_info.host_coeff_mem_size
         << ", runtime: "<<mem_info.host_neuron_mem_size
         << ")"<< std::endl;
  }
}

// update binary data when copy one net to new flatbuffers
// it's a little complicated, using reflection of flatbuffers
static void update_table(Table *table, const StructDef *struct_def, ModelGen &model_gen,
                         ModelCtx &model_ctx)
{
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
          update_table(next_table, next_def, model_gen, model_ctx);
        }
        break;
      }
      case BASE_TYPE_VECTOR: {
        auto pointer = table->GetPointer<void *>(fd->value.offset);
        auto vector_pointer = reinterpret_cast<Vector<Offset<void>> *>(pointer);
        auto type = fd->value.type.VectorType();
        if (type.base_type != BASE_TYPE_STRUCT) {
          break;
        }
        auto next_def = type.struct_def;
        if (next_def->fixed) {
          if (next_def->name == "Binary") {
            for (uint32_t next_id = 0; next_id < vector_pointer->size(); next_id++) {
              auto next_pointer = vector_pointer->GetMutableObject(next_id);
              auto binary = reinterpret_cast<Binary *>(next_pointer);
              uint8_t *data = new uint8_t[binary->size()];
              model_ctx.read_binary(binary, data);
              auto new_binary = model_gen.WriteBinary(binary->size(), data);
              binary->mutate_start(new_binary.start());
            }
          }
          break;
        }
        for (uint32_t next_id = 0; next_id < vector_pointer->size(); next_id++) {
          auto next_pointer = vector_pointer->GetMutableObject(next_id);
          auto next_table = reinterpret_cast<Table *>(next_pointer);
          update_table(next_table, next_def, model_gen, model_ctx);
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
static void update_model(ModelGen &model_gen, ModelCtx &model_ctx)
{
  Parser parser;
  parser.Parse(schema_text);
  auto buffer = model_gen.GetBufferPointer();
  auto root = GetMutableRoot<Table>(buffer);
  auto root_def = parser.root_struct_def_;
  update_table(root, root_def, model_gen, model_ctx);
}

// update one net binary data
static void update_net(ModelGen &model_gen, ModelCtx &model_ctx, uint32_t net_idx = 0,
                       uint32_t sub_idx = 0)
{
  Parser parser;
  parser.Parse(schema_text);
  auto buffer = model_gen.GetBufferPointer();
  auto root_table = GetMutableRoot<Table>(buffer);
  auto root_def = parser.root_struct_def_;
  auto net_field = root_def->fields.Lookup("net");
  auto net_def = net_field->value.type.VectorType().struct_def;
  auto pointer = root_table->GetPointer<void *>(net_field->value.offset);
  auto net_pointer = reinterpret_cast<Vector<Offset<void>> *>(pointer)->GetMutableObject(net_idx);
  auto net_table = reinterpret_cast<Table *>(net_pointer);
  auto sub_net_field = net_def->fields.Lookup("parameter");
  auto sub_net_def = sub_net_field->value.type.VectorType().struct_def;
  auto sub_pointer = net_table->GetPointer<void *>(sub_net_field->value.offset);
  auto sub_net_pointer =
      reinterpret_cast<Vector<Offset<void>> *>(sub_pointer)->GetMutableObject(sub_idx);
  auto sub_net_table = reinterpret_cast<Table *>(sub_net_pointer);
  update_table(sub_net_table, sub_net_def, model_gen, model_ctx);
}

// extract multi-net bmodel to multi one-net bmodels
static void extract(const string &filename)
{
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

static size_t tensor_bytes(const Vector<Offset<Tensor>> * tensor)
{
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
                                  ifstream &fout_ref, NET_INDEX_T *net_idx)
{
  net_idx->input_size = tensor_bytes(param->input_tensor());
  net_idx->output_size = tensor_bytes(param->output_tensor());
  net_idx->input = new char[net_idx->input_size];
  net_idx->output = new char[net_idx->output_size];
  fin_ref.read(net_idx->input, net_idx->input_size);
  fout_ref.read(net_idx->output, net_idx->output_size);
}

static bool write_input_output_ref(vector<shared_ptr<MODEL_CTX_T>>& model_vec, uint32_t net_idx, uint32_t stage_idx)
{
  for (auto &model_info : model_vec) {
    for (auto &net_index : model_info->net_index_v) {
      if (net_index->net_idx == net_idx && net_index->stage_idx == stage_idx) {
        g_input_ref->write(net_index->input, net_index->input_size);
        g_output_ref->write(net_index->output, net_index->output_size);
        delete [] net_index->input;
        delete [] net_index->output;
        return true;
      }
    }
  }
  return false;
}

static void write_input_output_ref(vector<shared_ptr<MODEL_CTX_T>>& model_vec)
{
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

static void combine_bmodels(ModelGen &model_gen, vector<shared_ptr<MODEL_CTX_T>>& model_vec, bool is_dir = false)
{
  model_gen.AddChip(model_vec[0]->model_ctx->model()->chip()->str());
  auto &builder = model_gen.Builder();
  for (uint32_t model_idx = 0; model_idx < model_vec.size(); model_idx++) {
    auto &model_info = model_vec[model_idx];
    auto model = model_info->model_ctx->model();
    for (uint32_t net_idx = 0; net_idx < model->net()->size(); net_idx++) {
      auto net = model->net()->Get(net_idx);
      if (net->parameter() == NULL || net->parameter()->size() == 0) {
        continue;
      }
      auto net_name = net->name()->str();
      for (uint32_t idx = 0; idx < net->parameter()->size(); idx++) {
        shared_ptr<NET_INDEX_T> net_idx(new NET_INDEX_T);
        if (is_dir) {
          read_input_output_ref(net->parameter()->Get(idx), model_info->input_f,
                                model_info->output_f, net_idx.get());
        }
        auto netT = net->parameter()->Get(idx)->UnPack();
        auto net_offset = NetParameter::Pack(builder, netT);
        model_gen.AddNet(net_name, net_offset, &net_idx->net_idx, &net_idx->stage_idx);
        delete netT;
        model_info->net_index_v.push_back(net_idx);
      }
    }
  }
  model_gen.Finish();
  for (uint32_t idx = 0; idx < model_vec.size(); idx++) {
    auto &model_info = model_vec[idx];
    for (auto &net_index : model_info->net_index_v) {
      update_net(model_gen, *model_info->model_ctx, net_index->net_idx, net_index->stage_idx);
    }
  }
  if (is_dir) {
    write_input_output_ref(model_vec);
  }
}

static bool make_directory(const char *dirname)
{
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

static void prepare_output(string& path, bool is_dir = false)
{
  if (is_dir) {
    if (path.empty()) {
      path = "bm_combined";
    }
    if (false == make_directory(path.c_str())) {
      FATAL("mkdir[%s] failed", path.c_str());
    }
    g_input_ref = make_shared<ofstream>(path+"/input_ref_data.dat", ios::trunc | ios::binary);
    g_output_ref = make_shared<ofstream>(path+"/output_ref_data.dat", ios::trunc | ios::binary);
    path += "/compilation.bmodel";
  } else {
    if (path.empty()) {
      path = "bm_combined.bmodel";
    }
  }
}

// combine bmodels
static void combine_bmodels(int argc, char **argv, bool is_dir = false)
{
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
      model_info->model_ctx = make_shared<ModelCtx>(param + "/compilation.bmodel");
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
  combine_bmodels(model_gen, model_vec, is_dir);
  model_gen.Save(ofile);
  cout << "Success: combined to [" << ofile << "]." << endl;
}

// read binary from bmodel
static uint64_t str2ull(const char * str)
{
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

static void dump_binary(int argc, char **argv)
{
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

int main(int argc, char **argv)
{
  if (argc < 3) {
    usage();
    exit(-1);
  }

  string cmd = argv[1];
  if (cmd == "--print") {
    print(argv[2]);
  } else if (cmd == "--info") {
    show(argv[2]);
  } else if (cmd == "--extract") {
    extract(argv[2]);
  } else if (cmd == "--combine") {
    combine_bmodels(argc, argv);
  } else if (cmd == "--combine_dir") {
    combine_bmodels(argc, argv, true);
  } else if (cmd == "--dump") {
    dump_binary(argc, argv);
  } else {
    usage();
    exit(-1);
  }

  return 0;
}
