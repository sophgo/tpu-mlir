#pragma once
#include "cnpy.h"

void load_npy(std::string file_path, char *data) {
  cnpy::NpyArray nparray = cnpy::npy_load(file_path);
  memcpy(data, nparray.data<char>(), nparray.num_bytes());
  printf("read %s success\n", file_path.c_str());
}

void load_npz(std::string file_path, std::string key, char *data) {
  cnpy::NpyArray nparray = cnpy::npz_load(file_path, key);
  memcpy(data, nparray.data<char>(), nparray.num_bytes());
  printf("read %s success\n", file_path.c_str());
}
