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

int main(int argc, char **argv) {
  if (argc < 3) {
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
  } else {
    usage();
    exit(-1);
  }

  return 0;
}
