#include "tpu_mlir/Support/TPUNnvlcUtil.h"
#include "tpu_mlir/Backend/BM168x/BM1688.h"
#include <algorithm>
#include <iostream>
#include <memory.h>
using namespace tpu_mlir::backend;

namespace tpu_mlir {

static inline uint8_t bit_val(void *buf, int32_t byte_idx, int32_t bit_idx) {
  return (((uint8_t *)buf)[byte_idx] >> bit_idx) & 0x1;
}

static uint8_t sign_to_unsign(uint8_t val) {
  uint8_t sign = (val >> 7) & 0x1;
  uint8_t abs_val = abs(((int8_t)val));
  return ((abs_val << 1) - sign);
}

static int8_t unsign_to_sign(uint8_t val) {
  uint8_t sign = val & 0x1;
  int32_t abs_val = (((int)val) + 1) >> 1;
  return (uint8_t)((sign == 1) ? (-abs_val) : abs_val);
}

void BitStream::write(void *src, int32_t bit_len) {
  for (int32_t pos = 0; pos < bit_len; pos++) {
    int32_t src_byte = pos / 8;
    int32_t src_bit = pos % 8;
    int32_t dst_byte = (this->bit_pos + pos) / 8;
    int32_t dst_bit = (this->bit_pos + pos) % 8;
    this->stream[dst_byte] |= (bit_val(src, src_byte, src_bit) << dst_bit);
  }
  this->bit_pos += bit_len;
  assert(this->bit_pos <= this->buf_size * 8);
}

void BitStream::read(void *dst, int32_t bit_len) {
  memset(dst, 0, (bit_len + 7) >> 3);
  for (int32_t pos = 0; pos < bit_len; pos++) {
    int32_t dst_byte = pos / 8;
    int32_t dst_bit = pos % 8;
    int32_t src_byte = (this->bit_pos + pos) / 8;
    int32_t src_bit = (this->bit_pos + pos) % 8;
    ((uint8_t *)dst)[dst_byte] |=
        (bit_val(this->stream, src_byte, src_bit) << dst_bit);
  }
  this->bit_pos += bit_len;
  assert(this->bit_pos <= this->buf_size * 8);
}
void BitStream::read(void *dst, int32_t bit_len, int32_t bit_pos) {
  assert(bit_pos + bit_len <= this->bit_pos);
  memset(dst, 0, (bit_len + 7) >> 3);
  for (int32_t pos = 0; pos < bit_len; pos++) {
    int32_t dst_byte = pos / 8;
    int32_t dst_bit = pos % 8;
    int32_t src_byte = (bit_pos + pos) / 8;
    int32_t src_bit = (bit_pos + pos) % 8;
    ((uint8_t *)dst)[dst_byte] |=
        (bit_val(this->stream, src_byte, src_bit) << dst_bit);
  }
}
uint8_t CenterShift::transform(uint8_t val, uint8_t bias, bool zero_guard) {
  if (val == 0 && zero_guard)
    return 0;

  int16_t shift_data_i = val - bias;
  uint8_t range = (bias <= 128) ? bias : 255 - bias;
  if (bias <= 128) {
    return (val >= (range << 1)) ? val
                                 : sign_to_unsign(shift_data_i) + zero_guard;
  } else {
    return (val < (bias - range)) ? (range + bias - val + zero_guard)
                                  : (sign_to_unsign(shift_data_i) + zero_guard);
  }
}

uint8_t CenterShift::inverse(uint8_t val, uint8_t bias, uint8_t zero_guard) {
  if (val == 0 && zero_guard)
    return 0;
  uint8_t uval = val - zero_guard;
  uint8_t range = (bias <= 128) ? bias : 255 - bias;
  if (bias <= 128) {
    return (val >= (range << 1)) ? val : unsign_to_sign(uval) + bias;
  } else {
    return (uval > (range << 1)) ? (range + bias - val + zero_guard)
                                 : unsign_to_sign(uval) + bias;
  }
}

int8_t TwoSideCircularShift::transform(int8_t val, uint8_t bias0,
                                       uint8_t bias1) {
  if (val == 0)
    return 0;

  uint8_t sign = (val < 0) ? true : false;
  int32_t abs_val = abs(val);
  abs_val -= (sign) ? bias1 : bias0;
  abs_val += (abs_val <= 0) ? (127 + sign) : 0;
  int8_t shift_val = (sign) ? -abs_val : abs_val;
  return sign_to_unsign(shift_val);
}

int8_t TwoSideCircularShift::inverse(uint8_t uval, uint8_t bias0,
                                     uint8_t bias1) {
  int8_t val = unsign_to_sign(uval);
  if (val == 0)
    return 0;

  uint8_t sign = (val < 0) ? true : false;
  uint32_t abs_val = (uint32_t)abs(val);
  abs_val += (sign) ? bias1 : bias0;
  int32_t abs_val_minus = abs_val - (127 + sign);
  uint8_t abs_val_lsb =
      ((abs_val_minus <= 0) ? (uint8_t)abs_val : (uint8_t)abs_val_minus) & 0xFF;
  return (sign) ? -abs_val_lsb : abs_val_lsb;
}

int32_t GREncoder::estimate_order_k(uint8_t *blk_in, bool zero_guard) {
  int32_t best_k = 0;
  int32_t best_bs_size = 0x7FFFFFFF;

  for (int32_t k = 0; k <= (int)MAX_ORDER_K; k++) {
    uint8_t remain_field_size = k << 4;
    int32_t unary_field_len = 0;
    for (int32_t i = 0; i < 16; i++) {
      uint8_t quotient = blk_in[i] >> k;
      unary_field_len += (quotient + 1);
    }
    int32_t znum_bit = (zero_guard && k > 0) ? 4 : 0;
    int32_t blk_size = (unary_field_len <= MAX_UNARY_FIELD_SIZE)
                           ? remain_field_size + unary_field_len + znum_bit
                           : 255;
    if (blk_size < best_bs_size) {
      best_k = k;
      best_bs_size = blk_size;
    }
  }
  best_k = (best_bs_size > 128) ? -1 : best_k;
  return best_k;
}

uint8_t GREncoder::block_encode(BitStream &stream, uint8_t *blk_in,
                                int32_t order_k, bool zero_guard) {
  // uncompressed mode
  if (order_k == -1) {
    stream.write(blk_in, 128);
    return 128;
  }
  // remain field
  uint8_t remain_field[16] = {0};
  uint8_t unary_field[8] = {0};
  uint8_t sym_end_pos[16] = {0};
  uint8_t unary_field_len = 0;
  int32_t sym_end_pos_accum = -1;
  // bit plane encode for remain field
  for (int32_t k = 0; k < order_k; k++) {
    uint8_t bit_plane0 = 0, bit_plane1 = 0;
    for (int32_t i = 0; i < 8; i++) {
      bit_plane0 |= (bit_val(blk_in, i, k) << i);
      bit_plane1 |= (bit_val(blk_in, i + 8, k) << i);
    }
    remain_field[k << 1] = bit_plane0;
    remain_field[(k << 1) + 1] = bit_plane1;
  }
  stream.write(remain_field, order_k << 4);

  if (zero_guard && order_k > 0) {
    int32_t zero_num = 0;
    for (int32_t i = 0; i < 16; i++) {
      if (blk_in[i] == 0)
        zero_num++;
    }
    if (zero_num >= 16)
      return 0;

    stream.write((uint8_t *)&zero_num, 4);
  }
  // unary encode for unary field
  for (int32_t i = 0; i < 16; i++) {
    int32_t quotient = blk_in[i] >> order_k;
    sym_end_pos_accum += (quotient + 1);
    sym_end_pos[i] = sym_end_pos_accum;
    int32_t byte_idx = sym_end_pos[i] / 8;
    int32_t bit_idx = sym_end_pos[i] % 8;
    unary_field[byte_idx] |= (1 << (bit_idx));
  }
  unary_field_len = sym_end_pos[15] + 1;

  assert(unary_field_len <= MAX_UNARY_FIELD_SIZE);

  uint8_t ulen = (unary_field_len - 16) & 0x1F;
  stream.write(unary_field, unary_field_len);
  return ulen;
}

int32_t Int8VlcEncoder::encode(uint8_t *ibuf, int32_t isz, uint8_t *obuf) {
  auto blk_num = calc_blk_num(isz, this->blk_len);
  auto kmap_size = calc_kmap_sz(blk_num);

  BitStream kmap_strm(obuf, kmap_size, true);
  BitStream payload_strm(obuf + kmap_size, blk_num << 4, true);

  TwoSideCircularShift remapping;
  for (int32_t idx = 0, pos = 0; idx < blk_num; idx++, pos += 16) {
    uint8_t blk_data[16] = {0};
    int32_t in_num = std::min(isz - pos, 16);
    memcpy(blk_data, &ibuf[idx << 4], sizeof(uint8_t) * in_num);
    if (signedness) {
      for (int32_t i = 0; i < 16; i++) {
        blk_data[i] = remapping.transform((int8_t)blk_data[i], bias0, bias1);
      }
    }
    int32_t k = estimate_order_k(blk_data, false);
    uint8_t ulen = block_encode(payload_strm, blk_data, k, false);
    uint8_t k_info = (k == -1) ? 0xE0 : (k << 5) + ulen;
    kmap_strm.write(&k_info, 8);
  }
  int32_t blk_bs_size = div_up(((payload_strm.pos() + 7) >> 3), 16) << 4;
  return blk_bs_size;
}

int32_t Int16VlcEncoder::encode(uint8_t *ibuf, int32_t isz, uint8_t *obuf) {
  auto blk_num = calc_blk_num(isz, this->blk_len);
  auto kmap_size = calc_kmap_sz(blk_num);

  BitStream kmap_strm(obuf, kmap_size, true);
  BitStream payload_strm(obuf + kmap_size, blk_num << 5, true);

  TwoSideCircularShift remapping;
  for (int32_t idx = 0, pos = 0; idx < blk_num; idx++, pos += 32) {
    uint8_t high[16] = {0};
    uint8_t low[16] = {0};
    uint8_t hbuf[16] = {0};
    int32_t in_num = std::min(isz - pos, 32) >> 1;
    auto ptr = (uint16_t *)(ibuf + pos);
    for (int32_t i = 0; i < in_num; i++) {
      high[i] = (uint8_t)((ptr[i] >> 8) & 0xFF);
      hbuf[i] = high[i];
      low[i] = (uint8_t)(ptr[i] & 0xFF);
      if (signedness) {
        high[i] = remapping.transform((int8_t)high[i], bias0, bias1);
      }
    }

    int32_t k = estimate_order_k(high, false);
    uint8_t ulen = block_encode(payload_strm, high, k, false);
    uint8_t k_info = (k == -1) ? 0xE0 : (k << 5) + ulen;
    kmap_strm.write(&k_info, 8);

    for (int32_t i = 0; i < 16; i++) {
      payload_strm.write(&low[i], 8);
    }
  }

  int32_t blk_bs_size = div_up(((payload_strm.pos() + 7) >> 3), 16) << 4;
  return blk_bs_size;
}

int32_t Float16VlcEncoder::encode(uint8_t *ibuf, int32_t isz, uint8_t *obuf) {
  auto blk_num = calc_blk_num(isz, this->blk_len);
  auto kmap_size = calc_kmap_sz(blk_num);
  // block encode
  BitStream kmap_strm(obuf, kmap_size, true);
  BitStream payload_strm(obuf + kmap_size, blk_num << 5, true);

  CenterShift remapping;
  for (int32_t idx = 0, pos = 0; idx < blk_num; idx++, pos += 32) {
    uint8_t exp[16] = {0};
    uint8_t frac[16] = {0};
    uint8_t exp_buf[16] = {0};
    int32_t in_num = std::min(isz - pos, 32) >> 1;
    auto ptr = (uint16_t *)(ibuf + pos);
    for (int32_t i = 0; i < 16; i++) {
      exp[i] = i < in_num ? (uint8_t)((ptr[i] >> 7) & 0xFF) : 0;
      frac[i] =
          i < in_num ? (uint8_t)(((ptr[i] >> 15) << 7) | (ptr[i] & 0x7F)) : 0;
      if (is_fp16 && zero_guard) {
        exp[i] = (exp[i] >> 3) == 0 ? 0 : exp[i];
      }
      exp_buf[i] = exp[i];
      exp[i] = remapping.transform(exp[i], bias0, zero_guard);
    }

    int32_t k = estimate_order_k(exp, zero_guard);
    uint8_t ulen = block_encode(payload_strm, exp, k, zero_guard);
    uint8_t k_info = (k == -1) ? 0xE0 : (k << 5) + ulen;
    kmap_strm.write(&k_info, 8);
    for (int32_t i = 0; i < 16; i++) {
      if (exp[i] != 0 || !zero_guard) {
        payload_strm.write(&frac[i], 8);
      }
    }
  }
  int32_t blk_bs_size = div_up(((payload_strm.pos() + 7) >> 3), 16) << 4;
  return blk_bs_size;
}

GREncoder *create_encoder(mlir::Type dtype, uint8_t bias0, uint8_t bias1,
                          bool is_signed, bool zero_guard) {
  bool is_float16 = dtype.isF16() || dtype.isBF16();
  bool is_int16 = dtype.isInteger(16);

  if (is_float16) {
    // assert(bias0 != 0);
    return new Float16VlcEncoder(bias0, (dtype.isF16()), zero_guard);
  } else if (is_int16) {
    return new Int16VlcEncoder(bias0, bias1, is_signed);
  } else {
    return new Int8VlcEncoder(bias0, bias1, is_signed);
  }
}

std::tuple<bool, uint8_t *> nnvlc_encode(uint8_t *ibuf, int32_t isz,
                                         mlir::Type dtype, uint8_t bias0,
                                         uint8_t bias1, bool is_signed,
                                         bool zero_guard, int32_t &osz) {
  auto encoder = create_encoder(dtype, bias0, bias1, is_signed, zero_guard);
  int32_t max_buf_sz = encoder->max_enc_size(isz, encoder->blk_len);
  max_buf_sz += sizeof(EncodeHeader);
  uint8_t *obuf = new (std::nothrow) uint8_t[max_buf_sz];
  memset(obuf, 0, max_buf_sz);
  assert(obuf);

  int32_t enc_sz = encoder->encode(ibuf, isz, obuf + sizeof(EncodeHeader));
  auto blk_num = div_up(isz, encoder->blk_len);
  auto kmap_size = div_up(blk_num, 16) << 4;
  delete encoder;

  EncodeHeader header{};
  header.blk_enc_size = enc_sz;
  memcpy(obuf, &header, sizeof(header));

  osz = sizeof(header) + enc_sz + kmap_size;
  bool do_compress = true;
  if (osz * 1.0 / isz > 1) {
    do_compress = false;
  }
  return std::make_tuple(do_compress, obuf);
  // return obuf;
}

int get_bytesize(mlir::Type dtype) {
  int bytesize = 4;
  if (dtype.isInteger(8) || dtype.isInteger(4)) {
    bytesize = 1;
  } else if (dtype.isInteger(16) || dtype.isF16() || dtype.isBF16()) {
    bytesize = 2;
  }
  return bytesize;
}

int tpu_compress_normal_max_bytes(shape_t shape, mlir::Type dtype) {
  int size = shape.n * shape.c * shape.h * shape.w * get_bytesize(dtype);
  int blk_len =
      (dtype.isInteger(16) || dtype.isF16() || dtype.isBF16()) ? 32 : 16;
  int blk_num = (size + blk_len - 1) / blk_len;
  int kmap_sz = ((blk_num + 15) / 16) << 4;
  return kmap_sz + (blk_num * blk_len);
}

int tpu_compress_RACU_max_meta_bytes(shape_t shape) {
  int lane_num = div_up(shape.c, Arch::NPU_NUM);
  return shape.n * lane_num * shape.h * 4;
}

int tpu_compress_RACU_max_racu_bytes(shape_t shape, mlir::Type dtype) {
  int lane_num = div_up(shape.c, Arch::NPU_NUM);
  shape_t gcw_shape = {1, (int32_t)Arch::NPU_NUM, 1, shape.w};
  int gcw = tpu_compress_normal_max_bytes(gcw_shape, dtype);
  return shape.n * lane_num * shape.h * gcw;
}

shape_t tpu_compress_RACU_racu_stride(shape_t shape, mlir::Type dtype) {
  int lane_num = div_up(shape.c, Arch::NPU_NUM);
  shape_t gcw_shape = {1, std::min((int32_t)Arch::NPU_NUM, shape.c), 1,
                       shape.w};
  int gcw = tpu_compress_normal_max_bytes(gcw_shape, dtype);
  //[n,lane_num,h,gcw]
  shape_t stride;
  stride.w = 1;
  stride.h = gcw;
  stride.c = shape.h * stride.h;
  stride.n = lane_num * stride.c;
  return stride;
}

shape_t tpu_compress_RACU_meta_stride(shape_t shape) {
  int lane_num = div_up(shape.c, Arch::NPU_NUM);
  //[n,lane_num,h,1]
  shape_t stride;
  stride.h = 1;
  stride.w = 1;
  stride.c = shape.h;
  stride.n = lane_num * shape.h;
  return stride;
}
} // namespace tpu_mlir
