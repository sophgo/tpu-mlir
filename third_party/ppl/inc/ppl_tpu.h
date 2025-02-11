//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
//===----------------------------------------------------------------------===//


#pragma once

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

void tpu_sync_core();

void sync();

void hau_poll();

void tpu_poll();

void cancel_tpu_init();

void cancel_tpu_poll();

void msg_send(int msg_idx, int wait_cnt, bool is_dma);

void msg_wait(int msg_idx, int send_cnt, bool is_dma);

void fence();

void lane_mask(int mask, bool long_valid);

void vset(int ew, int lmul, int v_len);

template <typename DataType> int get_eu_num() {
  if constexpr (std::is_same_v<DataType, int4>) {
    return 2 * EU_BYTES;
  } else {
    return EU_BYTES / sizeof(DataType);
  }
}


} // namespace ppl
