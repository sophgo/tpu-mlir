#pragma once
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/WinoGrad.h"

/**
 *     float *new_weight = new float[input_c * output_c * 4 * 4 / groups];
    memset((char*)new_weight, 0, input_c * output_c * 4 * 4 / groups *
 sizeof(float));
    winograd_weight_transform_subfunc((char*)weight_trans,(char*)weight,
 input_c, output_c, groups, winograd_flag);
 *
*/

// (oc, ic, h, w) -> (ic/4, oc, h*w, 4)
void tensor_1IC_to_4IC_transpose(int oc, int ic, int hw, char *src, char *dst) {
  for (int oc_idx = 0; oc_idx < oc; oc_idx++) {
    for (int ic_idx = 0; ic_idx < ic; ic_idx++) {
      for (int hw_idx = 0; hw_idx < hw; hw_idx++) {
        dst[ic_idx / 4 * oc * hw * 4 + oc_idx * hw * 4 + hw_idx * 4 +
            ic_idx % 4] = src[oc_idx * ic * hw + ic_idx * hw + hw_idx];
      }
    }
  }
}

/**currently only support int8*/

template <typename T>
void winograd_weight_transform_subfunc(T *old_weight,T *new_weight,
                                       int input_c, int output_c, int groups,
                                       int winograd_flag) {
  int NPU_NUM = 64;
  int ic = input_c / groups;
  int oc = output_c / groups;
  for (int i_g = 0; i_g < groups; ++i_g) {
    if (winograd_flag == 1 && sizeof(T) == 1) {
      tensor_1IC_to_4IC_transpose(
          oc, ic, 16, (char *)old_weight + i_g * oc * ic * 16,
          (char *)new_weight + i_g * oc * align_up(ic, 4) * 16);
    } else if (winograd_flag == 1 && sizeof(T) == 4) {
      for (int i_c = 0; i_c < ic; ++i_c)
        for (int i_n = 0; i_n < oc; ++i_n) {
          memcpy(&new_weight[i_g * oc * ic * 4 * 4 + i_c * oc * 4 * 4 +
                             i_n * 4 * 4],
                 &old_weight[i_g * oc * ic * 4 * 4 + i_n * ic * 4 * 4 +
                             i_c * 4 * 4],
                 sizeof(T) * 4 * 4);
        }
    } else if (winograd_flag == 2 && sizeof(T) == 1) {
      int dim[4];
      dim[0] = groups * align_up(ic, 4) / 4;
      int Min = oc < NPU_NUM ? oc : NPU_NUM;
      int Cnt = ceiling_func(oc, NPU_NUM);
      dim[1] = Min * Cnt;
      dim[2] = 4 * 4;
      dim[3] = 4;
      char *ptr_tmp = new char[dim[0] * dim[1] * dim[2] * dim[3] * 4];

      memset(ptr_tmp, 0, dim[0] * dim[1] * dim[2] * dim[3] * 4);
      tensor_1IC_to_4IC_transpose(
          oc, ic, 16, (char *)old_weight + i_g * oc * ic * 16,
          (char *)ptr_tmp + i_g * oc * align_up(ic, 4) * 16);

      int BatchSize = 4 * 4 * 4;
      int IC4CEIL = align_up(ic, 4) / 4;
      int ocidx = 0;

      // Conversion of (ic/4, oc, 64) ==> (1, Min, ic/4 * Cnt * 64)
      for (int npu_idx = 0; npu_idx < Min; ++npu_idx) {
        for (int i_c = 0; i_c < IC4CEIL; ++i_c) {
          for (int o_c = 0; o_c < Cnt; ++o_c) {
            ocidx = o_c * Min + npu_idx;
            if (ocidx < oc)
              memcpy(&new_weight[i_g * Min * Cnt * IC4CEIL * BatchSize +
                                 npu_idx * IC4CEIL * Cnt * BatchSize +
                                 (i_c * Cnt + o_c) * BatchSize],
                     &ptr_tmp[i_g * oc * IC4CEIL * BatchSize +
                              i_c * oc * BatchSize +
                              (o_c * Min + npu_idx) * BatchSize],
                     sizeof(T) * BatchSize);
            else
              memset(&new_weight[i_g * Min * Cnt * IC4CEIL * BatchSize +
                                 npu_idx * IC4CEIL * Cnt * BatchSize +
                                 (i_c * Cnt + o_c) * BatchSize],
                     0, sizeof(T) * BatchSize);
          }
        }
      }

      delete[] ptr_tmp;
    } else if (winograd_flag == 2 && sizeof(T) == 4) { // fp32 usewino=2
      int occupy_npu_nm =
          (output_c / groups) < NPU_NUM ? (output_c / groups) : NPU_NUM;
      int oc_per_npu = ceiling_func(output_c / groups, NPU_NUM);
      for (int npu_idx = 0; npu_idx < occupy_npu_nm; ++npu_idx) {
        for (int i_c = 0; i_c < ic; ++i_c) {
          for (int o_c = 0; o_c < oc_per_npu; ++o_c) {
            if (o_c * occupy_npu_nm + npu_idx < oc)
              memcpy(
                  &new_weight[i_g * occupy_npu_nm * oc_per_npu * ic * 4 * 4 +
                              i_c * oc_per_npu * occupy_npu_nm * 4 * 4 +
                              (o_c * occupy_npu_nm + npu_idx) * 4 * 4],
                  &old_weight[i_g * oc * ic * 4 * 4 +
                              ((o_c * occupy_npu_nm) + npu_idx) * ic * 4 * 4 +
                              i_c * 4 * 4],
                  sizeof(T) * 4 * 4);
            else
              memset(&new_weight[i_g * occupy_npu_nm * oc_per_npu * ic * 4 * 4 +
                                 i_c * oc_per_npu * occupy_npu_nm * 4 * 4 +
                                 (o_c * occupy_npu_nm + npu_idx) * 4 * 4],
                     0, sizeof(T) * 4 * 4);
          }
        }
      }
    }
  }
}

template <typename T>
void winograd_bias_transform_subfunc(T *old_bias, T *new_bias, int output_c,
                                     int groups) {
  int NPU_NUM = 64;
  int occupy_npu_nm =
      (output_c / groups) < NPU_NUM ? (output_c / groups) : NPU_NUM;
  int oc_per_npu = ceiling_func(output_c / groups, NPU_NUM);
  for (int i_g = 0; i_g < groups; ++i_g) {
    for (int npu_idx = 0; npu_idx < occupy_npu_nm; ++npu_idx) {
      for (int o_c = 0; o_c < oc_per_npu; ++o_c) {
        if (o_c * occupy_npu_nm + npu_idx < output_c)
          new_bias[i_g * occupy_npu_nm * oc_per_npu + npu_idx * oc_per_npu +
                     o_c] = old_bias[i_g * occupy_npu_nm * oc_per_npu +
                                 o_c * occupy_npu_nm + npu_idx];
        else
          new_bias[i_g * occupy_npu_nm * oc_per_npu + npu_idx * oc_per_npu +
                     o_c] = 0;
      }
    }
  }
}
