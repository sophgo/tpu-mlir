//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <fstream>
#include <iostream>
#include <string>
#include "bm_tool.hpp"
#include "cv_tool.hpp"

using namespace std;

// show usage of tool
static void usage(void) {
  cout << "Usage:" << endl;
  // clang-format off
  cout << "  model_tool" << endl
       << "    [bmodel]:" << endl
       << "      --info model_file : show brief model info, --all will show all hidden info" << endl
       << "      --chip model_file : show chip of model" << endl
       << "      --dynamic model_file : true or false" << endl
       << "      --print model_file : show detailed model info" << endl
       << "      --extract model_file : extract one multi-net bmodel to multi one-net bmodels" << endl
       << "      --combine file1 .. fileN -o new_file: combine bmodels to one bmodel by filepath" << endl
       << "      --combine_dir dir1 .. dirN -o new_dir: combine bmodels to one bmodel by directory path" << endl
       << "      --dump model_file start_offset byte_size out_file: dump binary data to file from bmodel" << endl
       << "      --kernel_dump model_file -o kernel_file_name : dump kernel_module file" << endl
       << "      --kernel_update model_file kernel_name : add/update kernel_module file" << endl
       << endl
       << "    [cvimodel]:" << endl
       << "      --info model_file : show model info" << endl
       << "      --chip model_file : show chip of model" << endl
       << "      --extract model_file : extract cmdbuf and weight" << endl
       << endl;
  // clang-format on
}

static inline bool isCv18xx(const string &filename) {
  string extension = filename.substr(filename.find_last_of("."));
  return extension == ".cvimodel";
}

static void print(const string &filename) {
  if (isCv18xx(filename)) {
    cout << "cv18xx not supported!" << endl;
  } else {
    bm_print(filename);
  }
}

static void show(const string &filename, bool all = false) {
  if (isCv18xx(filename)) {
    cvtool::Model model(filename);
    model.dump();
  } else {
    bm_show(filename, all);
  }
}

static void show_chip(const string &filename) {
  if (isCv18xx(filename)) {
    cvtool::Model model(filename);
    model.show_chip();
  } else {
    bm_show_chip(filename);
  }
}

static void show_dynamic(const string &filename) {
  if (isCv18xx(filename)) {
    cout << "cv18xx not supported!" << endl;
  } else {
    bm_show_dynamic(filename);
  }
}

static void extract(const string &filename) {
  if (isCv18xx(filename)) {
    cvtool::Model model(filename);
    model.extract();
  } else {
    bm_extract(filename);
  }
}

static void combine(int argc, char **argv) {
  if (isCv18xx(argv[2])) {
    std::vector<std::shared_ptr<cvtool::Model>> models;
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
      auto model = std::make_shared<cvtool::Model>(param);
      models.emplace_back(model);
    }
    if (models.empty()) {
      cout << "No input file!\n" << endl;
      return;
    }
    models[0]->merge(models, ofile);
  } else {
    bm_combine_bmodels(argc, argv);
  }
}

static void update_kernel(ModelGen &model_gen, shared_ptr<MODEL_CTX_T>& model_info, uint8_t* module_binary, size_t binary_size, string module_name)
{
  model_gen.AddChip(model_info->model_ctx->model()->chip()->str());
  model_gen.AddNumDevice(model_info->model_ctx->model()->device_num());
  auto &builder = model_gen.Builder();
  auto model = model_info->model_ctx->model();
  for (uint32_t net_idx = 0; net_idx < model->net()->size(); net_idx++) {
    auto net = model->net()->Get(net_idx);
    if (net->parameter() == NULL || net->parameter()->size() == 0) {
      continue;
    }
    auto net_name = net->name()->str();
    for (uint32_t idx = 0; idx < net->parameter()->size(); idx++) {
      shared_ptr<NET_INDEX_T> net_idx(new NET_INDEX_T);
      auto netT = net->parameter()->Get(idx)->UnPack();
      auto net_offset = NetParameter::Pack(builder, netT);
      auto cascade = net->cascade();
      if (cascade) {
        // no more stage
        assert(net->parameter()->size() == 1);
      }
      model_gen.AddNet(net_name, net_offset, &net_idx->net_idx, &net_idx->stage_idx, cascade);
      delete netT;
      model_info->net_index_v.push_back(net_idx);
    }
  }
  auto kernel_module = model_gen.WriteBinary(binary_size, module_binary);
  model_gen.AddKernelModule(module_name, kernel_module);
  model_gen.Finish();
  for (auto &net_index : model_info->net_index_v) {
    update_net(model_gen, *model_info->model_ctx, net_index->net_idx, net_index->stage_idx);
  }
}

static void update_kernel_module(int argc, char **argv) {
  // tpu_model --kernel_add xx.bmodel xx.so
  if (argc != 4) {
    FATAL("--update_kernel parameter error.");
  }
  string model_path = argv[2];
  string kernel_path = argv[3];
  ifstream f_kernel(kernel_path, ios::in | ios::binary);
  if (!f_kernel) {
    FATAL("module name [%s] is not correct", kernel_path.c_str());
  }
  shared_ptr<MODEL_CTX_T> model_info(new MODEL_CTX_T);
  model_info->model_ctx = make_shared<ModelCtx>(model_path);
  if (model_info->model_ctx == NULL || !(*model_info->model_ctx)) {
    FATAL("file[%s] is not correct", model_path.c_str());
  }

  f_kernel.seekg(0, f_kernel.end);
  int binary_size = f_kernel.tellg();
  f_kernel.seekg(0, f_kernel.beg);
  shared_ptr<char> module_binary(new char[binary_size]);
  f_kernel.read(module_binary.get(), binary_size);
  string module_name = kernel_path.substr(kernel_path.find_last_of('/')+1);

  ModelGen model_gen;
  update_kernel(model_gen, model_info, (uint8_t*)module_binary.get(), binary_size, module_name);
  model_gen.Save(model_path);
  cout << "Success: update to [" << module_name << "]." << endl;

  f_kernel.close();
}

static void dump_kernel_module(int argc, char **argv) {
  // tpu_model --kernel_dump xx.bmodel -o xx.so
  if (argc != 3 && argc != 5) {
    FATAL("--dump_kernel parameter error.");
  }
  ModelCtx model_ctx(argv[2]);
  if (!model_ctx) {
    FATAL("file[%s] is not correct", argv[2]);
  }
  auto kernel_module = model_ctx.model()->kernel_module();
  if (kernel_module) {
    auto module_binary = kernel_module->binary();
    string module_name = kernel_module->file_name()->str();
    size_t binary_size = module_binary->size();
    string save_name = module_name;
    if(argc == 5) {
      if (strcmp(argv[3], "-o") == 0) {
        save_name = argv[4];
        if (save_name.length()<3 || save_name.compare(save_name.length()-3, 3, ".so")) {
          save_name = save_name.compare(save_name.length()-1, 1, "/") ? save_name + "/" + module_name : save_name + module_name;
        }
      } else {
        FATAL("--dump_kernel parameter error.");
      }
    }
    ofstream ofile(save_name, ios::out | ios::trunc | ios::binary);
    if (!ofile) {
      FATAL("save file[%s] failed\n", save_name.c_str());
    }
    std::unique_ptr<uint8_t> binary(new uint8_t[binary_size]);
    model_ctx.read_binary(module_binary, binary.get());
    ofile.write((char*)binary.get(), binary_size);
    cout << "Success: dump kernel_module to [" << save_name << "]." << endl;
    ofile.close();
  } else {
    FATAL("no kernel_module found.");
  }
}

static void combine(int argc, char **argv, bool is_dir) {
  if (isCv18xx(argv[2])) {
    cout << "cv18xx not supported!" << endl;
  } else {
    bm_combine_bmodels(argc, argv, is_dir);
  }
}

static void dump(int argc, char **argv) {
  if (isCv18xx(argv[2])) {
    cout << "cv18xx not supported!" << endl;
  } else {
    bm_dump_binary(argc, argv);
  }
}

#ifndef MLIR_VERSION
#define MLIR_VERSION "version unknown"
#endif

int main(int argc, char **argv) {
  if (argc < 2) {
    usage();
    exit(-1);
  }

  string cmd = argv[1];
  if (cmd == "--print") {
    print(argv[2]);
  } else if (cmd == "--info") {
    bool all = false;
    if (argc == 4 && string(argv[3]) == "--all") {
      all = true;
    }
    show(argv[2], all);
  } else if (cmd == "--chip") {
    show_chip(argv[2]);
  } else if (cmd == "--is_dynamic") {
    show_dynamic(argv[2]);
  } else if (cmd == "--extract") {
    extract(argv[2]);
  } else if (cmd == "--combine") {
    combine(argc, argv);
  } else if (cmd == "--combine_dir") {
    combine(argc, argv, true);
  } else if (cmd == "--dump") {
    dump(argc, argv);
  } else if (cmd == "--version") {
    printf("%s\n", MLIR_VERSION);
  } else if (cmd == "--kernel_dump") {
    dump_kernel_module(argc, argv);
  } else if (cmd == "--kernel_update") {
    update_kernel_module(argc, argv);
  } else {
    usage();
    exit(-1);
  }

  return 0;
}
