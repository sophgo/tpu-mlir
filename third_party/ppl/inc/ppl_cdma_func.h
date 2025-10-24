//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022  Technologies Inc.  All rights reserved.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "ppl_defs.h"
#include "ppl_types.h"

namespace ppl {
namespace cdma {

template <typename DataType>
void write(gtensor<DataType> &dst, const gtensor<DataType> &src, int src_chipid,
           int dst_chipid, bool is_fill_const, int const_val,
           bool stride_enable, bool nchw_copy, int opcode = 0, int msg_en = 0,
           int msg_id = 0, int wcnt = 0);

template <typename DataType>
void write(gtensor<DataType> &dst, const gtensor<DataType> &src) {
  write(dst, src, 0, 0, false, 0, false, false);
}

template <typename DataType>
void write(gtensor<DataType> &dst, const gtensor<DataType> &src,
           int dst_chipid) {
  write(dst, src, 0, dst_chipid, false, 0, false, false);
}

template <typename DataType>
void write(gtensor<DataType> &dst, const gtensor<DataType> &src, int src_chipid,
           int dst_chipid) {
  write(dst, src, src_chipid, dst_chipid, false, 0, false, false);
}

template <typename DataType>
void read(gtensor<DataType> &dst, gtensor<DataType> &src, int src_chipid,
          int dst_chipid, int opcode, int stride_enable);

template <typename DataType>
void read(gtensor<DataType> &dst, gtensor<DataType> &src) {
  read(dst, src, 0, 0, 0, 0);
}

template <typename DataType>
void read(gtensor<DataType> &dst, gtensor<DataType> &src, int dst_chipid) {
  read(dst, src, 0, dst_chipid, 0, 0);
}

template <typename DataType>
void read(gtensor<DataType> &dst, gtensor<DataType> &src, int src_chipid,
          int dst_chipid) {
  read(dst, src, src_chipid, dst_chipid, 0, 0);
}

template <typename DataType>
void send(gtensor<DataType> &src, int src_chipid, int dst_chipid, int opcode);

template <typename DataType> void send(gtensor<DataType> &src) {
  send(src, 0, 0, 0);
}

template <typename DataType> void send(gtensor<DataType> &src, int dst_chipid) {
  send(src, 0, dst_chipid, 0);
}

template <typename DataType>
void send(gtensor<DataType> &src, int src_chipid, int dst_chipid) {
  send(src, src_chipid, dst_chipid, 0);
}

template <typename DataType>
void lossy_compress(gtensor<DataType> &src, int src_chipid, int dst_chipid,
                    int opcode);

template <typename DataType> void lossy_compress(gtensor<DataType> &src) {
  lossy_compress(src, 0, 0, 0);
}

template <typename DataType>
void lossy_compress(gtensor<DataType> &src, int dst_chipid) {
  lossy_compress(src, 0, dst_chipid, 0);
}

template <typename DataType>
void lossy_compress(gtensor<DataType> &src, int src_chipid, int dst_chipid) {
  lossy_compress(src, src_chipid, dst_chipid, 0);
}

template <typename DataType>
void lossy_decompress(gtensor<DataType> &src, int src_chipid, int dst_chipid,
                      int opcode);

template <typename DataType> void lossy_decompress(gtensor<DataType> &src) {
  lossy_decompress(src, 0, 0, 0);
}

template <typename DataType>
void lossy_decompress(gtensor<DataType> &src, int dst_chipid) {
  lossy_decompress(src, 0, dst_chipid, 0);
}

template <typename DataType>
void lossy_decompress(gtensor<DataType> &src, int src_chipid, int dst_chipid) {
  lossy_decompress(src, src_chipid, dst_chipid, 0);
}

template <typename DataType0, typename DataType1>
void recv(gtensor<DataType0> &dst, DataType1 src, int src_chipid,
          int dst_chipid, int opcode);


template <typename DataType>
void recv(gtensor<DataType> &dst, int src_chipid) {
  int cur_chipid = ppl::chip_id();
  recv(dst, 0, src_chipid, cur_chipid, 0);
}

template <typename DataType>
void recv(gtensor<DataType> &dst, int src_chipid, int dst_chipid) {
  recv(dst, 0, src_chipid, dst_chipid, 0);
}

template <typename DataType>
void recv(gtensor<DataType> &dst, gtensor<DataType> &src, int src_chipid,
          int dst_chipid) {
  recv(dst, src, src_chipid, dst_chipid, 0);
}

template <typename DataType>
void remote_msgsend(gtensor<DataType> &index_list, int chip_id,
                    int index_list_num, int msg_id, int wcnt);

template <typename DataType1, typename DataType2>
void scatter(gtensor<DataType1> &src, gtensor<DataType2> &index_list,
             int chip_id, int index_list_num, int length_shift, int opcode = 0,
             int msg_en = 0, int msg_id = 0, int wcnt = 0);

void initialize();

int get_port(int self, int peer, int direction);

void send_msg(int port, int msg_id, int wait_cnt);

void wait_msg(int port, int msg_id, int send_cnt);

void tx_send_msg(int port, int msg_id, int wait_cnt);

void tx_wait_msg(int port, int msg_id, int send_cnt);

void rx_send_msg(int port, int msg_id, int wait_cnt);

void rx_wait_msg(int port, int msg_id, int send_cnt);

void poll();

void port_poll(int port);

void port_init(int port);

void nop(int port);


} // namespace cdma
} // namespace ppl
