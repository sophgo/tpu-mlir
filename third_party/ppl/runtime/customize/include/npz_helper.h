#pragma once
#include "host_test_utils.h"

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
