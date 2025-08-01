#ifndef __NPZ_HELPER_H__
#define __NPZ_HELPER_H__
#include "cnpy.h"
#include "host_test_utils.h"
 
int get_npy_dtype(cnpy::NpyArray &arr) {
  if (arr.type == 'f') {
    if (arr.word_size == 4) {
      return DT_FP32;
    } else if (arr.word_size == 2) {
      return DT_FP16;
    }
  } else if (arr.type == 'u' || arr.type == 'i') {
    if (arr.word_size == 1) {
      return arr.type == 'u' ? DT_UINT8 : DT_INT8;
    } else if (arr.word_size == 2) {
      return arr.type == 'u' ? DT_UINT16 : DT_INT16;
    } else if (arr.word_size == 4) {
      return arr.type == 'u' ? DT_UINT32 : DT_INT32;
    }
  }
  printf("Error! npy's data type not support!\n");
  return 0;
}
 
void load_npy_array(cnpy::NpyArray &arr, char *data, size_t data_size,
                    size_t ele_size, int dtype) {
  if (ele_size != arr.num_vals) {
    printf("Error! npy shape not equal to data!\n");
    exit(1);
  }
  auto npy_dtype = get_npy_dtype(arr);
  convert_data_type(data, arr.data<void>(), ele_size, dtype, npy_dtype);
}
 
void load_npy(std::string file_path, char *data, size_t data_size,
              size_t ele_size, int dtype) {
  std::ifstream infile(file_path, std::ios::binary);
  if (!infile.good()) {
    printf("Error! Can't open data file %s!\n", file_path.c_str());
    exit(1);
  }
  cnpy::NpyArray arr = cnpy::npy_load(file_path);
  load_npy_array(arr, data, data_size, ele_size, dtype);
  printf("load %s success\n", file_path.c_str());
}
 
void load_npz(std::string file_path, char *data, size_t data_size,
              size_t ele_size, int dtype, std::string arr_name) {
  std::ifstream infile(file_path, std::ios::binary);
  if (!infile.good()) {
    printf("Error! Can't open data file %s!\n", file_path.c_str());
    exit(1);
  }
  cnpy::npz_t npz_file = cnpy::npz_load(file_path);
  cnpy::NpyArray arr = npz_file[arr_name];
  load_npy_array(arr, data, data_size, ele_size, dtype);
  printf("load npz file:%s array:%s success\n", file_path.c_str(),
         arr_name.c_str());
}
 
void npz_add(cnpy::npz_t &npz, std::string key, char *data, dim4 shape,
             int dtype) {
  size_t element_size = shape.n * shape.c * shape.h * shape.w;
  std::vector<float> float_val;
  convert_src_to_float(float_val, data, element_size, dtype);
 
  std::vector<size_t> shape_vec = {
      static_cast<size_t>(shape.n), static_cast<size_t>(shape.c),
      static_cast<size_t>(shape.h), static_cast<size_t>(shape.w)};
  cnpy::npz_add_array<float>(npz, key, float_val.data(), shape_vec);
}
 
void npz_save(std::string file_path, std::string file_name, std::string ext,
              cnpy::npz_t &npz) {
  std::stringstream out_path;
  out_path << file_path << "/" << file_name << ext << ".npz";
  cnpy::npz_save_all(out_path.str(), npz);
  printf("save %s success\n", out_path.str().c_str());
}
 
void npz_add(cnpy::npz_t &npz, const char *key, mem_t &mem) {
  npz_add(npz, key, mem.host_mem, mem.shape, mem.dtype);
}
 
void npz_in_add(cnpy::npz_t &npz, mem_t &mem) {
  static int count = 0;
  std::string key = std::to_string(count++);
  npz_add(npz, key.c_str(), mem);
}
void npz_out_add(cnpy::npz_t &npz, mem_t &mem) {
  static int count = 0;
  std::string key = std::to_string(count++);
  npz_add(npz, key.c_str(), mem);
}
void npz_ref_add(cnpy::npz_t &npz, mem_t &mem) {
  static int count = 0;
  std::string key = std::to_string(count++);
  npz_add(npz, key.c_str(), mem);
}
 
void npz_save(cnpy::npz_t &npz, const char *ext) {
  const char *data_env = getenv("PPL_DATA_PATH");
  if (!data_env) {
    printf("Please set env PPL_DATA_PATH to data dir");
    assert(0);
  }
  std::string data_dir(data_env);
  const char *file_name_env = getenv("PPL_FILE_NAME");
  if (!file_name_env) {
    printf("Please set env PPL_FILE_NAME to pl file name");
    assert(0);
  }
  std::string file_name(file_name_env);
  npz_save(data_dir, file_name, ext, npz);
}
 
void npz_save_input(cnpy::npz_t &npz) { npz_save(npz, "_input"); }
 
void npz_save_output(cnpy::npz_t &npz) { npz_save(npz, "_tar"); }
void npz_save_ref(cnpy::npz_t &npz) { npz_save(npz, "_ref"); }
 
#endif
