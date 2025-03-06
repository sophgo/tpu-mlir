#ifndef __NPZ_HELPER_H__
#define __NPZ_HELPER_H__
#include "tpu_defs.h"
#include "cnpy.h"

void load_npy(std::string file_path, char *data, int dtype, int file_dtype) {
  cnpy::NpyArray nparray = cnpy::npy_load(file_path);
  if (dtype != file_dtype) {
    char *tmp_data = nparray.data<char>();
    std::vector<float> numbers;
    for (size_t i = 0; i < nparray.num_vals; ++i) {
      float value = 0.0l;
      if (file_dtype == DT_FP32)
        value = reinterpret_cast<float *>(tmp_data)[i];
      else if (file_dtype == DT_INT8)
        value = reinterpret_cast<char *>(tmp_data)[i];
      else if (file_dtype == DT_UINT8)
        value = reinterpret_cast<unsigned char *>(tmp_data)[i];
      else if (file_dtype == DT_INT32)
        value = reinterpret_cast<int *>(tmp_data)[i];
      else if (file_dtype == DT_UINT32)
        value = reinterpret_cast<u32 *>(tmp_data)[i];
      else if (file_dtype == DT_UINT16)
        value = reinterpret_cast<u16 *>(tmp_data)[i];
      else if (file_dtype == DT_INT16)
        value = reinterpret_cast<short *>(tmp_data)[i];
      else if (file_dtype == DT_FP16)
        value = fp16_to_fp32(reinterpret_cast<fp16 *>(tmp_data)[i]).fval;
      else if (file_dtype == DT_BFP16)
        value = bf16_to_fp32(reinterpret_cast<bf16 *>(tmp_data)[i]).fval;
      else {
        printf("dtype not supported:%d\n", file_dtype);
        assert(0);
      }
      // printf("   %f   \n", value);
      numbers.emplace_back(value);
    }

    transport(data, numbers, DtypeSize(dtype), nparray.num_bytes(), dtype);
  } else {
    memcpy(data, nparray.data<char>(), nparray.num_bytes());
  }

  printf("read %s success\n", file_path.c_str());
}

void load_npz(std::string file_path, std::string key, char *data, int dtype,
              int file_dtype) {
  cnpy::NpyArray nparray = cnpy::npz_load(file_path, key);
  if (dtype != file_dtype) {
    char *tmp_data = nparray.data<char>();
    std::vector<float> numbers;
    for (size_t i = 0; i < nparray.num_vals; ++i) {
      float value = 0.0l;
      if (file_dtype == DT_FP32)
        value = reinterpret_cast<float *>(tmp_data)[i];
      else if (file_dtype == DT_INT8)
        value = reinterpret_cast<char *>(tmp_data)[i];
      else if (file_dtype == DT_UINT8)
        value = reinterpret_cast<unsigned char *>(tmp_data)[i];
      else if (file_dtype == DT_INT32)
        value = reinterpret_cast<int *>(tmp_data)[i];
      else if (file_dtype == DT_UINT32)
        value = reinterpret_cast<u32 *>(tmp_data)[i];
      else if (file_dtype == DT_UINT16)
        value = reinterpret_cast<u16 *>(tmp_data)[i];
      else if (file_dtype == DT_INT16)
        value = reinterpret_cast<short *>(tmp_data)[i];
      else if (file_dtype == DT_FP16)
        value = fp16_to_fp32(reinterpret_cast<fp16 *>(tmp_data)[i]).fval;
      else if (file_dtype == DT_BFP16)
        value = bf16_to_fp32(reinterpret_cast<bf16 *>(tmp_data)[i]).fval;
      else {
        printf("dtype not supported:%d\n", file_dtype);
        assert(0);
      }
      // printf("   %f   \n", value);
      numbers.emplace_back(value);
    }

    transport(data, numbers, DtypeSize(dtype), nparray.num_bytes(), dtype);
  } else {
    memcpy(data, nparray.data<char>(), nparray.num_bytes());
  }

  printf("read %s success\n", file_path.c_str());
}

void npz_add(cnpy::npz_t &npz, std::string key, char *data, dim4 *shape,
             int dtype) {
  size_t element_size = shape->n * shape->c * shape->h * shape->w;
  std::vector<float> float_val;
  for (size_t i = 0; i < element_size; ++i) {
    float value = 0.0l;
    if (dtype == DT_FP32)
      value = reinterpret_cast<float *>(data)[i];
    else if (dtype == DT_INT8)
      value = reinterpret_cast<char *>(data)[i];
    else if (dtype == DT_UINT8)
      value = reinterpret_cast<unsigned char *>(data)[i];
    else if (dtype == DT_INT32)
      value = reinterpret_cast<int *>(data)[i];
    else if (dtype == DT_UINT32)
      value = reinterpret_cast<u32 *>(data)[i];
    else if (dtype == DT_UINT16)
      value = reinterpret_cast<u16 *>(data)[i];
    else if (dtype == DT_INT16)
      value = reinterpret_cast<short *>(data)[i];
    else if (dtype == DT_FP16)
      value = fp16_to_fp32(reinterpret_cast<fp16 *>(data)[i]).fval;
    else if (dtype == DT_BFP16)
      value = bf16_to_fp32(reinterpret_cast<bf16 *>(data)[i]).fval;
    else {
      printf("dtype not supported:%d\n", dtype);
      assert(0);
    }
    float_val.emplace_back(value);
  }
  std::vector<size_t> shape_vec = {
      static_cast<size_t>(shape->n), static_cast<size_t>(shape->c),
      static_cast<size_t>(shape->h), static_cast<size_t>(shape->w)};
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
  npz_add(npz, key, mem.host_mem, &mem.shape, mem.dtype);
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

#endif
