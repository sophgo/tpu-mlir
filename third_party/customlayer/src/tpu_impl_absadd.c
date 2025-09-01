#include "tpu_impl_custom_ops.h"


static void pipeline_move_(unsigned long long *array, int num) {
  for (int i = num - 1; i > 0; i--) {
    array[i] = array[i - 1];
  }
}

void tpu_impl_absadd_global(global_addr_t input_global_addr,
                            global_addr_t output_global_addr, const int *shape,
                            float b_val, data_type_t dtype) {
  // Step 1. Lmem address allocation
  // 2 bank for work0, 2 bank for work1, 1 bank for buffer
  // 2 bank for input, 2 bank for output
  // ***Note: gdma and bdc operations should prevent bank conflicts to avoid performance degradation
  // tpu_local_mem_size_per_npu = 262144bit = 256KB
  // tpu_bank_num() = 16;
  int bank_size = tpu_local_mem_size_per_npu() / tpu_bank_num(); //16KB
  // in0 = 0 ~ 16KB
  // in1 = 16KB ~ 32KB
  local_addr_t in_local_addr[2] = {0, bank_size};
  // out0 = 32KB ~ 48KB
  // out1 = 48KB ~ 64KB
  local_addr_t out_local_addr[2] = {2 * bank_size, 3 * bank_size};
  // buffer = 64KB ~ 256KB
  local_addr_t buffer_addr = 4 * bank_size;


  // Step 2. Data Splitting
  unsigned long long length = 1;
  for (int i = 0; i < 4; i++) {
    length *= (unsigned long long)shape[i];
  }

  int npu_num = tpu_npu_num(); //64
  int eu_num = tpu_eu_num(dtype); //fp32 : fp32_eu_num(=16) , fp16 : fp16_eum  = fp32_eu_num >> 2 , int8 : fp16_eum >> 2

  /// continus 128byte can get better peformance
  int tensor_w = MAX(DIV_UP(MIN(length , bank_size / tpu_data_type_size(dtype)), npu_num),
                     DIV_UP(128, eu_num * tpu_data_type_size(dtype))); // The number of data items transferred to the i-th NPU each time.
  // unsigned long long slice = tensor_w * npu_num; // Max slice length
  unsigned long long slice = MIN(MIN(length, (unsigned long long)npu_num * tensor_w), bank_size / tpu_data_type_size(dtype)); // Maximum slice length
  int max_rows_per_time = (bank_size) / (tensor_w * tpu_data_type_size(dtype));// An NPU divides into 16 banks, each 16KB. Now, in_local[0] is allocated 16KB. This step calculates how many data blocks of size tensor_w can fit into this space.
  int rows = DIV_UP(length, slice); // How many slices can be cut in total?
  int rows_secs = DIV_UP(rows, max_rows_per_time); // The input data's length must allow splitting into at least rows_secs blocks, each with size [max_rows_per_time, slice].
  // at least loop two times to overlap all bdc time
  int rows_slice = DIV_UP(rows, MAX(rows_secs, 2)); // The input data's length must be able to be split into at least MAX(rows_secs, 2) blocks (total of MAX(rows_secs, 2) computations), each block's size equals the data in [rows_slice, slice], with each block corresponding to 64 NPU.

  unsigned long long cur_idx[3] = {0}, cur_rows[3] = {0}, cur_cols[3] = {0};
  int stage_idx = 0, draning_idx = 0;

  // Step3. Parallel Pipeline
  // cur_idx[2] stores processed data length, obtained from previous operation's update via pipeline_move_(cur_idx, 3);
  while (cur_idx[2] < length) {
    tpu_parallel_start(); // Make the BDC and GDMA instructions between tpu_parallel_start() and tpu_parallel_end() perform CMD ID division to eliminate instruction dependencies and enable parallel execution.
    // update load info
    if (draning_idx < 1) {
      unsigned long long cur_len = MIN(length - cur_idx[0], rows_slice * slice); // Here calculate the number of the current slice requiring further processing
      cur_cols[0] = MIN(cur_len, slice); // Calculates the number of columns for the current slice data when forming a matrix
      cur_rows[0] = MAX(1, cur_len / cur_cols[0]); // This calculates the number of rows for the current slice when assembled into a matrix. The formed matrix shape is [cur_cols[0], cur_rows[0]].
    }
    // The code's flow is roughly as follows:
    //---0--1--2----(-2)-------> time
    //  L0 C0 S0
    //     L1 C1 S1
    //        L2 C2 S2
    //           L3 C3 S3
    //              L4 C4 S4
    // store output
    if (stage_idx > 1) {
      tpu_gdma_matrix_L2S(output_global_addr +
                              cur_idx[2] * tpu_data_type_size(dtype),
                          out_local_addr[stage_idx & 0x1], cur_rows[2],
                          cur_cols[2], tensor_w, cur_cols[2], dtype);
    }

    // load input
    if (draning_idx < 1) {
      tpu_gdma_matrix_S2L(
          in_local_addr[stage_idx & 0x1],
          input_global_addr + cur_idx[0] * tpu_data_type_size(dtype),
          cur_rows[0], cur_cols[0], tensor_w, cur_cols[0], dtype);
    }

    // compute
    if (stage_idx > 0 && draning_idx < 2) {
      int cur_shape[4] = {cur_rows[1], DIV_UP(cur_cols[1], tensor_w), 1,
                          tensor_w};
      tpu_impl_absadd_local(in_local_addr[(stage_idx - 1) & 0x1],
                            out_local_addr[(stage_idx - 1) & 0x1], buffer_addr,
                            cur_shape, b_val, dtype);
    }

    tpu_parallel_end();// This step performs cmd id merge and tpu poll synchronization, re-establishing instruction dependencies. It is after the first round of pipelining ends, requiring state updates to proceed to the next round.
    pipeline_move_(cur_idx, 3);  // [The data blocks processed last time, the data blocks processed last time,] cur_idx[0] will be updated in line 94 to the total number of data blocks processed so far.
    pipeline_move_(cur_cols, 3); // [current processed slice's column count, last processed slice's column count,]
    pipeline_move_(cur_rows, 3); // [current slice line count, previous slice line count,]
    if (draning_idx < 1) {
      cur_idx[0] += cur_cols[0] * cur_rows[0]; // Calculate the current processed data length and update to cur_idx[0]
      if (cur_idx[0] >= length) {
        draning_idx++; // Update draning_idx; the data has been moved here, waiting for calculation. After storing, end the flow and break.
      }
    } else {
      draning_idx++; // Update state machine; subsequent flow will be established based on these new states.
    }
    stage_idx++; // Update the state machine; establish the flow based on these new states.
  }
}

int get_absadd_local_bfsz(const int *shape, data_type_t dtype) {
  dim4 _shape = {.n = shape[0], .c = shape[1], .h = shape[2], .w = shape[3]};
  return tpu_get_local_size(&_shape, dtype, 0, true);
}

void tpu_impl_absadd_local(local_addr_t input_addr, local_addr_t output_addr,
                           local_addr_t buffer_addr, const int *shape,
                           float b_val, data_type_t dtype) {
  dim4 _shape = {.n = shape[0], .c = shape[1], .h = shape[2], .w = shape[3]};
  tpu_bdc_abs(buffer_addr, input_addr, &_shape, NULL, NULL, dtype);
  scalar_t C = {.f32 = b_val};
  tpu_bdc_fp_add_C(output_addr, buffer_addr,
                   tpu_fp_cast(C, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
                   &_shape, NULL, NULL, dtype);
}
