#include "tpu_impl_custom_ops.h"


static void pipeline_move_(unsigned long long *array, int num) {
  for (int i = num - 1; i > 0; i--) {
    array[i] = array[i - 1];
  }
}

void tpu_impl_absadd_global(global_addr_t input_global_addr,
                            global_addr_t output_global_addr, const int *shape,
                            float b_val, data_type_t dtype) {
  // Step1. Lmem addr分配
  // 2 bank for work0, 2 bank for work1, 1 bank for buffer
  // 2 bank for input, 2 bank for output
  // ***Note: gdma 和 bdc 操作需要防止bank冲突以免导致性能下降
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


  // Step2. 数据切分
  unsigned long long length = 1;
  for (int i = 0; i < 4; i++) {
    length *= (unsigned long long)shape[i];
  }

  int npu_num = tpu_npu_num(); //64
  int eu_num = tpu_eu_num(dtype); //fp32 : fp32_eu_num(=16) , fp16 : fp16_eum  = fp32_eu_num >> 2 , int8 : fp16_eum >> 2

  /// continus 128byte can get better peformance
  int tensor_w = MAX(DIV_UP(MIN(length , bank_size / tpu_data_type_size(dtype)), npu_num),
                     DIV_UP(128, eu_num * tpu_data_type_size(dtype))); //每次搬运到第i个npu上的数据个数
  // unsigned long long slice = tensor_w * npu_num; // 切片的最大长度
  unsigned long long slice = MIN(MIN(length, (unsigned long long)npu_num * tensor_w), bank_size / tpu_data_type_size(dtype)); //切片的最大长度
  int max_rows_per_time = (bank_size) / (tensor_w * tpu_data_type_size(dtype));//一个npu会划分16个bank,每个bank16KB,现在划分给in_local[0]=16KB,这步就是求这片空间可以存放多少数据量为tensor_w的数据块
  int rows = DIV_UP(length, slice); //总共可以切出多少片slice
  int rows_secs = DIV_UP(rows, max_rows_per_time); //length长度的输入数据至少能切成rows_secs块，每一块大小等于[max_rows_per_time,slice]的数据
  // at least loop two times to overlap all bdc time
  int rows_slice = DIV_UP(rows, MAX(rows_secs, 2)); //length长度的输入数据至少能切成MAX(rows_secs, 2)块（一共计算MAX(rows_secs, 2)次），每次计算的一块大小等于（[rows_slice,slice]）的数据，一块对应64个npu;

  unsigned long long cur_idx[3] = {0}, cur_rows[3] = {0}, cur_cols[3] = {0};
  int stage_idx = 0, draning_idx = 0;

  // Step3. 并行流水
  // cur_idx[2] 用来存储已经处理完的数据长度，它的数据来自上步操作后更新得到 pipeline_move_(cur_idx, 3);
  while (cur_idx[2] < length) {
    tpu_parallel_start(); //让夹在tpu_parallel_start()和 tpu_parallel_end()之间的bdc 和 gdma 指令做cmd id divide，去除指令依赖，并行执行指令
    // update load info
    if (draning_idx < 1) {
      unsigned long long cur_len = MIN(length - cur_idx[0], rows_slice * slice); // 这里计算当前切片有多少数量需要进一步处理
      cur_cols[0] = MIN(cur_len, slice); //这里计算当前切片数据拼成矩阵有多少列
      cur_rows[0] = MAX(1, cur_len / cur_cols[0]); //这里计算当前切片拼成矩阵有多少行 ，这里拼成矩阵形状为[cur_cols[0],cur_rows[0]]
    }
    //整个代码的流水大致如下：
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

    tpu_parallel_end();//这一步是在做cmd id merge 和 tpu poll同步，并重新建立指令依赖关系，就是第一轮流水结束，需要更新状态，以便进行下一轮流水
    pipeline_move_(cur_idx, 3);  // [上次已经处理的数据块，上次已经处理的数据块，] cur_idx[0] 会在第94行更新成当前已经处理完的数据块总数
    pipeline_move_(cur_cols, 3); // [当前处理的切片的列数，上次处理的切片的列数，]
    pipeline_move_(cur_rows, 3); // [当前处理的切片的行数，上次处理的切片的行数，]
    if (draning_idx < 1) {
      cur_idx[0] += cur_cols[0] * cur_rows[0]; // 计算当前已经处理的数据长度，并更新到cur_idx[0]
      if (cur_idx[0] >= length) {
        draning_idx++; //更新draning_idx, 这里已经搬完数据了，等待计算，store完后就可以结束流水break出去了
      }
    } else {
      draning_idx++; //更新状态机，后续根据这些新状态建立流水
    }
    stage_idx++; //更新状态机，后续根据这些新状态建立流水
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
