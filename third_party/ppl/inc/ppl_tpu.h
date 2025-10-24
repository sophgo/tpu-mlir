//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022  Technologies Inc.  All rights reserved.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "ppl_types.h"

#define LANE_NUM ppl::lane_num()
#define NPU_NUM LANE_NUM
#define EU_BYTES ppl::eu_bytes()

namespace ppl {
int lane_num();

int eu_bytes();

void enable_pipeline();

void set_core_num(int num);

int get_core_num();

int get_core_index();

void set_group_num(int num);

void set_block_num(int num);

int get_group_num();

int get_block_num();

int get_group_index();

int get_block_index();

void tpu_poll();

void cancel_tpu_init();

void cancel_tpu_poll();

void parallel_start();
void parallel_end();

void sync_engine(int sync_type, bool all_core);

void sync_core() { sync_engine(SYNC_ALL, false); }

void sync_all() { sync_engine(SYNC_ALL, true); }

void sync() { sync_all(); }

void sync_sdma(bool all_core = false) { sync_engine(SDMA_SYNC, all_core); }
void sync_hau() { sync_engine(HAU_SYNC, false); }
void sync_tiu_dma(bool all_core = false) {
  sync_engine(TIU_DMA_SYNC, all_core);
}

void msg_send(int msg_idx, int wait_cnt, bool is_dma);

void msg_wait(int msg_idx, int send_cnt, bool is_dma);

void fence();

void lane_mask(int mask, bool long_valid);

void vset(int ew, int lmul, int v_len);

template <typename DataType> int get_eu_num() {
  if constexpr (std::is_same_v<DataType, int4> ||
                std::is_same_v<DataType, fp4> ||
                std::is_same_v<DataType, uint4>) {
    return 2 * EU_BYTES;
  } else {
    return EU_BYTES / sizeof(DataType);
  }
}

int chip_num();

int chip_id();

int *chip_map();

/************************************************************************************
 */
/************************************************************************************
 */
/*        deprecated */
/**************************************************************************************/
/**************************************************************************************/

void hau_poll() { sync_engine(HAU_SYNC, false); }

} // namespace ppl
