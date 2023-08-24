//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once


#include <llvm/Support/Debug.h>
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"

namespace tpu_mlir {
namespace backend {
typedef struct {
  int32_t n;
  int32_t c;
  int32_t h;
  int32_t w;
  int32_t n_pos;
  int32_t c_pos;
  int32_t h_pos;
  uint64_t input_offset;
  uint64_t output_offset;
} EltwiseTile;

class TgEltwiseKernel {
public:
  TgEltwiseKernel(::TypeID typeID) : typeID(typeID) {}
  TypeID typeID;


  void init(uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c,
    int32_t h, int32_t w, bool do_relu,
    bool do_early_stride, int32_t stride_h,
    int32_t stride_w, int32_t rshift,
    const int32_t *multipliers,
    const int32_t *coeffs);

  void init(uint32_t layer_id,
    gaddr_t ga_inputs[], gaddr_t ga_output,
    int32_t operand_num, int32_t n, int32_t c,
    int32_t h, int32_t w, bool do_relu,
    bool do_early_stride, int32_t stride_h,
    int32_t stride_w,
    const float *coeffs);

  void selectTilePolicy();
  void schedule();

protected:
  virtual void compute(int32_t step_idx) = 0;

  void allocLmem(
      cvk_tl_shape_t &input_shape,
      cvk_tl_shape_t &output_shape);
  void deallocLmem();
  void doTileForStrideCase();
  void doTileForNormalCase();
  void load(int32_t step_idx);
  void store(int32_t step_idx);

  gaddr_t *ga_inputs;
  gaddr_t ga_output;

  cvk_tl_t *tl_input[2];
  cvk_tl_t *tl_output[2];
  cvk_tl_t *tl_output_h[2];

  int32_t operand_num;
  int32_t n;
  int32_t c;
  int32_t h;
  int32_t w;
  bool do_early_stride;
  int32_t stride_h;
  int32_t stride_w;
  int32_t rshift;
  const int32_t *multipliers;
  const int32_t *coeffs;
  const float *coeffs_float;
  int32_t layer_id;
  bool do_relu;
  cvk_fmt_t fmt;
  uint32_t elementSize;

  int32_t input_flip = 0;
  int32_t output_flip = 0;

  std::vector<EltwiseTile> tiles;
};

class TgInt8EltwiseAddKernel : public TgEltwiseKernel {
public:
  TgInt8EltwiseAddKernel()
      : TgEltwiseKernel(TypeID::get<TgInt8EltwiseAddKernel>()) {}

protected:
  void compute(int32_t step_idx);
private:
  void symmetric_compute(const int opd_idx,
                            cvk_tl_t &input,
                            cvk_tl_t &output,
                            cvk_tl_t &output_high);
};

class TgInt8EltwiseMaxKernel : public TgEltwiseKernel {
public:
  TgInt8EltwiseMaxKernel()
      : TgEltwiseKernel(TypeID::get<TgInt8EltwiseMaxKernel>()) {}

protected:
  void compute(int32_t step_idx);
};

class TgInt8EltwiseMinKernel : public TgEltwiseKernel {
public:
  TgInt8EltwiseMinKernel()
      : TgEltwiseKernel(TypeID::get<TgInt8EltwiseMinKernel>()) {}

protected:
  void compute(int32_t step_idx);
};

class TgInt8EltwiseMulKernel : public TgEltwiseKernel {
public:
  TgInt8EltwiseMulKernel()
      : TgEltwiseKernel(TypeID::get<TgInt8EltwiseMulKernel>()) {}

protected:
  void compute(int32_t step_idx);
};

class TgBf16EltwiseAddKernel : public TgEltwiseKernel {
public:
  TgBf16EltwiseAddKernel()
      : TgEltwiseKernel(TypeID::get<TgBf16EltwiseAddKernel>()) {}

protected:
  void compute(int32_t step_idx);
};

class TgBf16EltwiseMaxKernel : public TgEltwiseKernel {
public:
  TgBf16EltwiseMaxKernel()
      : TgEltwiseKernel(TypeID::get<TgBf16EltwiseMaxKernel>()) {}

protected:
  void compute(int32_t step_idx);
};

class TgBf16EltwiseMinKernel : public TgEltwiseKernel {
public:
  TgBf16EltwiseMinKernel()
      : TgEltwiseKernel(TypeID::get<TgBf16EltwiseMinKernel>()) {}

protected:
  void compute(int32_t step_idx);
};

class TgBf16EltwiseMulKernel : public TgEltwiseKernel {
public:
  TgBf16EltwiseMulKernel()
      : TgEltwiseKernel(TypeID::get<TgBf16EltwiseMulKernel>()) {}

protected:
  void compute(int32_t step_idx);
};

class TgBf16EltwiseMinMaxKernel : public TgEltwiseKernel {
public:
  TgBf16EltwiseMinMaxKernel()
      : TgEltwiseKernel(TypeID::get<TgBf16EltwiseMinMaxKernel>()) {}

  static bool classof(const TgEltwiseKernel *eltwiseKernel) {
    return eltwiseKernel->typeID ==
           TypeID::get<TgBf16EltwiseMinMaxKernel>();
  }

protected:
  void compute(int32_t step_idx);
};

class TgEltwiseAbsKernel : public TgEltwiseKernel {
public:
  TgEltwiseAbsKernel() : TgEltwiseKernel(TypeID::get<TgEltwiseAbsKernel>()) {}

protected:
  void compute(int32_t step_idx);
};
} // namespace backend
} // namespace tpu_mlir
