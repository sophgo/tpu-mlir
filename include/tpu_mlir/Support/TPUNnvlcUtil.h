#ifndef TPU_NNVLC_UTIL_H_
#define TPU_NNVLC_UTIL_H_

#include <assert.h>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <memory.h>
#include "mlir/IR/Types.h"

namespace tpu_mlir {

#define MAX_UNARY_FIELD_SIZE 47
#define MAX_ORDER_K 5

typedef struct {
  int32_t n;
  int32_t c;
  int32_t h;
  int32_t w;
} shape_t;

inline int32_t div_up(int32_t v, int32_t align) {
  return (v + align - 1) / align;
}

inline int32_t align_up(int32_t v, int32_t align) {
  return div_up(v, align) * align;
}

typedef struct {
  uint32_t blk_enc_size : 24;
  uint8_t reserved1[13];
} EncodeHeader;

class BitStream {
public:
  BitStream(uint8_t *buf, int32_t buf_size, bool write_mode = false)
      : stream(buf), bit_pos(0), buf_size(buf_size) {
    if (write_mode)
      memset(buf, 0, buf_size);
  }

  void write(void *src, int32_t bit_len);
  void read(void *dst, int32_t bit_len);
  void read(void *dst, int32_t bit_len, int32_t bit_pos);
  int32_t pos() { return this->bit_pos; }

private:
  uint8_t *stream;
  int32_t bit_pos;
  int32_t buf_size;
};

class CenterShift {
public:
  uint8_t transform(uint8_t val, uint8_t bias, bool zero_guard);
  uint8_t inverse(uint8_t val, uint8_t bias, uint8_t zero_guard);
};

class TwoSideCircularShift {
public:
  int8_t transform(int8_t val, uint8_t bias0, uint8_t bias1);
  int8_t inverse(uint8_t uval, uint8_t bias0, uint8_t bias1);
};

class GREncoder {
public:
  GREncoder(int32_t blk_len, uint8_t bias0, uint8_t bias1, bool signedness)
      : blk_len(blk_len), bias0(bias0), bias1(bias1), signedness(signedness) {}
  GREncoder() = delete;
  virtual ~GREncoder() {}
  virtual int32_t encode(uint8_t *ibuf, int32_t isz, uint8_t *obuf) = 0;

  static int32_t max_enc_size(int32_t size, int32_t blk_len) {
    auto blk_num = calc_blk_num(size, blk_len);
    auto kmap_sz = calc_kmap_sz(blk_num);
    return kmap_sz + (blk_num * blk_len);
  }

  int32_t blk_len;
  uint8_t bias0;
  uint8_t bias1;
  bool signedness;

protected:
  static int32_t calc_blk_num(int32_t size, int32_t blk_len) {
    return div_up(size, blk_len);
  }
  static int32_t calc_kmap_sz(int32_t blk_num) {
    return div_up(blk_num, 16) << 4;
  }

  int32_t estimate_order_k(uint8_t *blk_in, bool zero_guard);

  uint8_t block_encode(BitStream &stream, uint8_t *blk_in, int32_t order_k,
                       bool zero_guard);
};

class Int8VlcEncoder : public GREncoder {
public:
  Int8VlcEncoder(uint8_t bias0, uint8_t bias1, bool signedness)
      : GREncoder(16, bias0, bias1, signedness) {}

  int32_t encode(uint8_t *ibuf, int32_t isz, uint8_t *obuf);
};

class Int16VlcEncoder : public GREncoder {
public:
  Int16VlcEncoder(uint8_t bias0, uint8_t bias1, bool signedness)
      : GREncoder(32, bias0, bias1, signedness) {}

  int32_t encode(uint8_t *ibuf, int32_t isz, uint8_t *obuf);
};

class Float16VlcEncoder : public GREncoder {
public:
  Float16VlcEncoder(uint8_t bias0, bool is_fp16, bool zero_guard)
      : GREncoder(32, bias0, 0, false), is_fp16(is_fp16),
        zero_guard(zero_guard) {}

  int32_t encode(uint8_t *ibuf, int32_t isz, uint8_t *obuf);

private:
  bool is_fp16;
  bool zero_guard;
};

GREncoder *create_encoder(mlir::Type dtype, uint8_t bias0, uint8_t bias1,
                          bool is_signed, bool zero_guard);

std::tuple<bool, uint8_t *> nnvlc_encode(uint8_t *ibuf, int32_t isz,
                                         mlir::Type dtype, uint8_t bias0,
                                         uint8_t bias1, bool is_signed,
                                         bool zero_guard, int32_t &osz);

/* nnvlc 2.0, random access compress/decompress*/
int get_bytesize(mlir::Type dtype);

int tpu_compress_normal_max_bytes(shape_t shape, mlir::Type dtype);

int tpu_compress_RACU_max_meta_bytes(shape_t shape);

int tpu_compress_RACU_max_racu_bytes(shape_t shape, mlir::Type dtype);

shape_t tpu_compress_RACU_racu_stride(shape_t shape, mlir::Type dtype);

shape_t tpu_compress_RACU_meta_stride(shape_t shape);

} // namespace tpu_mlir

#endif // TPU_NNVLC_UTIL_H_
