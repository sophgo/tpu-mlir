#pragma once
#include "ppl_defs.h"
#include "ppl_types.h"

namespace ppl {

namespace ez {

template <typename DTYPE> struct tensor2 {
public:
  // explicit tensor2(dim4& _shape, tensor_mode_t mode, long long address = -1);
  // explicit tensor2(dim4& _shape, tensor_mode_t mode, dtype *address =
  // nullptr);
  tensor2(const tensor2 &);
  template <typename DTYPE2> tensor2<DTYPE2> &view();
  template <typename DTYPE2> tensor2<DTYPE2> &view(const dim4 &_shape);
  tensor2<DTYPE> &sub_view(const dim4 &shape, const dim4 &offset);
  tensor2<DTYPE> &permute(const dim4 &permute_order);
  tensor2<DTYPE> &reshape(const dim4 &shape);
  template <typename DTYPE2>
  tensor2<DTYPE2> &cast(rounding_mode_t round_mode = RM_HALF_TO_EVEN);
  dim4 &shape();
  dim4 &block_shape();
  data_type_t dtype() { return convert_dtype<DTYPE>(); }

private:
  tensor2();
  DTYPE *data;
};

template <typename dtype>
tensor2<dtype> &make_blob(const dim4 &shape, dtype *address);
template <typename dtype>
tensor2<dtype> &make_blob(const dim4 &shape, const dim4 &stride,
                          dtype *address);
template <typename dtype>
tensor2<dtype> &empty(const dim4 &block_shape, const dim4 &shape,
                      tensor_mode_t mode = LOCAL);

template <typename DataType> char *to_string(tensor2<DataType> &src);

namespace tiu {

#define ARITH(op)                                                              \
  template <typename dtype>                                                    \
  tensor2<dtype> &op(tensor2<dtype> &a, tensor2<dtype> &b);                    \
  template <typename dtype, typename dtype2>                                   \
  tensor2<dtype> &op(tensor2<dtype> &a, dtype2 b);                             \
  template <typename dtype, typename dtype2>                                   \
  tensor2<dtype> &op(dtype2 a, tensor2<dtype> &b);

ARITH(add)
ARITH(sub)
ARITH(mul)
ARITH(div)
ARITH(max)
ARITH(min)

#define ARITH_INT(op)                                                          \
  template <typename dtype>                                                    \
  tensor2<dtype> &op(tensor2<dtype> &a, tensor2<dtype> &b, int8_t shift,       \
                     rounding_mode_t round_mode, bool saturation);             \
  template <typename dtype, typename dtype1>                                   \
  tensor2<dtype> &op(tensor2<dtype> &a, dtype1 b, int8_t shift,                \
                     rounding_mode_t round_mode, bool saturation);             \
  template <typename dtype, typename dtype1>                                   \
  tensor2<dtype> &op(dtype1 a, tensor2<dtype> &b, int8_t shift,                \
                     rounding_mode_t round_mode, bool saturation);

ARITH_INT(add)
ARITH_INT(sub)
ARITH_INT(mul)

template <typename dtype, typename dtype2>
tensor2<dtype> &full(const dim4 &block_shape, const dim4 &shape, dtype2 value,
                     tensor_mode_t mode = LOCAL);
template <typename dtype>
tensor2<dtype> &zeros(const dim4 &block_shape, const dim4 &shape,
                      tensor_mode_t mode = LOCAL) {
  return full<dtype>(block_shape, shape, 0, mode);
}
template <typename dtype, typename dtype2>
void fill(tensor2<dtype> &dst, dtype2 value);

template <typename DataTyp1, typename DataType2, typename DataType3 = DataTyp1>
tensor2<DataType3> &mm2_nn(tensor2<DataTyp1> &left, tensor2<DataType2> &right);
template <typename DataTyp1, typename DataType2, typename DataType3 = DataTyp1>
tensor2<DataType3> &mm2_nt(tensor2<DataTyp1> &left, tensor2<DataType2> &right);
template <typename DataTyp1, typename DataType2, typename DataType3 = DataTyp1>
tensor2<DataType3> &mm2_tt(tensor2<DataTyp1> &left, tensor2<DataType2> &right);
template <typename DataTyp1, typename DataType2, typename DataType3 = DataTyp1>
tensor2<DataType3> &mm2a_nn(tensor2<DataTyp1> &left, tensor2<DataType2> &right);
template <typename DataTyp1, typename DataType2, typename DataType3 = DataTyp1>
tensor2<DataType3> &mm2a_nt(tensor2<DataTyp1> &left, tensor2<DataType2> &right);
template <typename DataTyp1, typename DataType2, typename DataType3 = DataTyp1>
tensor2<DataType3> &mm2a_tt(tensor2<DataTyp1> &left, tensor2<DataType2> &right);

template <typename dtype> tensor2<dtype> &exp_base(tensor2<dtype> &src);
template <typename dtype>
tensor2<dtype> &round(tensor2<dtype> &src, rounding_mode_t round_mode);
template <typename dtype> tensor2<dtype> &floor(tensor2<dtype> &src) {
  return round(src, RM_DOWN);
}
template <typename dtype2, typename dtype>
tensor2<dtype2> &cast(tensor2<dtype> &src,
                      rounding_mode_t round_mode = RM_HALF_TO_EVEN);
template <typename dtype> tensor2<dtype> &clone(tensor2<dtype> &src);
template <typename dtype>
tensor2<dtype> &permute(tensor2<dtype> &src, const dim4 &permute_order);
template <typename dtype>
tensor2<dtype> &pool_max(tensor2<dtype> &src, pool_param kernel, pool_param pad,
                         pool_param insert);
template <typename dtype>
tensor2<dtype> &pool_max(tensor2<dtype> &src, pool_param kernel,
                         pool_param pad) {
  pool_param inst_param = pool::param::insert(0, 1, 1, 0, 0);
  return pool_max(src, kernel, pad, inst_param);
}
template <typename dtype, typename dtype2 = dtype>
tensor2<dtype> &pool_avg(tensor2<dtype> &src, pool_param kernel, pool_param pad,
                         pool_param insert, float scale, int rshift);
template <typename dtype, typename dtype2 = dtype>
tensor2<dtype> &pool_avg(tensor2<dtype> &src, pool_param kernel, pool_param pad,
                         float scale, int rshift) {
  pool_param inst_param = pool::param::insert(0, 1, 1, 0, 0);
  return pool_avg(src, kernel, pad, inst_param, scale, rshift);
}
template <typename dtype, typename dtype2 = dtype>
tensor2<dtype> &pool_avg(tensor2<dtype> &src, pool_param kernel,
                         pool_param pad) {
  pool_param insert = pool::param::insert(0, 1, 1, 0, 0);
  return pool_avg(src, kernel, pad, insert, 1.0f, 0);
}
template <typename dtype>
tensor2<dtype> concat(tensor2<dtype> &src1, tensor2<dtype> &src2, int axis);

} // namespace tiu

namespace dma {
template <typename dtype>
tensor2<dtype> &load(tensor2<dtype> &src, const dim4 &block_shape);

template <typename dtype> void store(tensor2<dtype> &dst, tensor2<dtype> &src);

} // namespace dma
} // namespace ez
} // namespace ppl

#define OP_RELOAD(op, name)                                                    \
  template <typename dtype>                                                    \
  ppl::ez::tensor2<dtype> &operator op(ppl::ez::tensor2<dtype> &a,             \
                                       ppl::ez::tensor2<dtype> &b) {           \
    return ppl::ez::tiu::name(a, b);                                           \
  }                                                                            \
  template <typename dtype, typename dtype2>                                   \
  ppl::ez::tensor2<dtype> &operator op(ppl::ez::tensor2<dtype> &a, dtype2 b) { \
    return ppl::ez::tiu::name(a, b);                                           \
  }                                                                            \
  template <typename dtype, typename dtype2>                                   \
  ppl::ez::tensor2<dtype> &operator op(dtype2 a, ppl::ez::tensor2<dtype> &b) { \
    return ppl::ez::tiu::name(a, b);                                           \
  }

OP_RELOAD(+, add)
OP_RELOAD(-, sub)
OP_RELOAD(*, mul)
OP_RELOAD(/, div)

// *************************************************************
// wrapper function
// *************************************************************

namespace ppl {
namespace ez {
namespace tiu {

template <typename dtype> tensor2<dtype> &exp(tensor2<dtype> &src) {
  fp32 min_C = 0;
  if constexpr (std::is_same<dtype, fp32>::value) {
    min_C = -3.40282e35f;
  } else if (std::is_same<dtype, fp16>::value) {
    min_C = -45403.f;
  } else {
    min_C = -3.40282e35f;
  }

  auto maxc_tensor = max(src, min_C);
  auto minc_tensor = empty<dtype>(src.block_shape(), src.shape());
  if (std::is_same<dtype, fp16>::value) {
    minc_tensor = min(src, 45403.f);
  } else {
    minc_tensor = src;
  }

  auto fp_mulc_tensor = minc_tensor * 1.4426950f;
  auto fp_floor_tensor = floor(fp_mulc_tensor);
  auto fp_mulc_tensor2 = fp_floor_tensor * 0.69314718f;
  auto fp_sub = maxc_tensor - fp_mulc_tensor2;
  // auto out = empty<dtype>(src.block_shape(), src.shape());
  if constexpr (std::is_same<dtype, fp32>::value) {
    auto cast_out = cast<int16>(fp_floor_tensor, RM_HALF_AWAY_FROM_ZERO);
    auto minc_tensor = min(cast_out, (int16)127);
    auto maxc_tensor2 = max(minc_tensor, (int16)-127);
    auto maxc_tensor2_i32 = cast<int32>(maxc_tensor2);
    tensor2<int32> add_intc_tensor =
        add(maxc_tensor2_i32, (int16)127, 23, RM_HALF_AWAY_FROM_ZERO, true);
    auto exp_out = exp_base(fp_sub);
    auto cast_intc_tensor = add_intc_tensor.view<fp32>();
    auto out = exp_out * cast_intc_tensor;
    return out;
  } else if constexpr (std::is_same<dtype, fp16>::value) {
    auto cast_out = cast<int8>(fp_floor_tensor, RM_HALF_AWAY_FROM_ZERO);
    auto minc_tensor = min(cast_out, (int16)15);
    auto maxc_tensor2 = max(minc_tensor, (int16)-15);
    auto maxc_tensor2_i16 = cast<int16>(maxc_tensor2);
    tensor2<int16> add_intc_tensor =
        add(maxc_tensor2_i16, (int16)15, 10, RM_HALF_AWAY_FROM_ZERO, true);
    auto exp_out = exp_base(fp_sub);
    auto cast_intc_tensor = add_intc_tensor.view<fp16>();
    auto out = exp_out * cast_intc_tensor;
    return out;
  } else if constexpr (std::is_same<dtype, bf16>::value) {
    auto cast_out = cast<int16>(fp_floor_tensor, RM_HALF_AWAY_FROM_ZERO);
    auto minc_tensor = min(cast_out, (int16)127);
    auto maxc_tensor2 = max(minc_tensor, (int16)-127);
    tensor2<int16> add_intc_tensor =
        add(maxc_tensor2, (int16)127, 7, RM_HALF_AWAY_FROM_ZERO, true);
    auto exp_out = exp_base(fp_sub);
    auto cast_intc_tensor = add_intc_tensor.view<bf16>();
    auto out = exp_out * cast_intc_tensor;
    return out;
  }
}

template <int REDUCE_MODE, typename INTYPE, typename OUTTYPE>
tensor2<OUTTYPE> &reduce_base(tensor2<INTYPE> &src, int axis) {
  if (axis != 3) {
    static_assert(true, "only support reduce on axis 3 now");
  }
  if (src.shape().h != 1) {
    static_assert(true, "only support reduce on h=1 now");
  }
  int n = src.shape().n;
  int c = src.shape().c;
  int w = src.shape().w;

  int eu_num = get_eu_num<INTYPE>();
  int opti_w = eu_num;

  int align_w = align(w, eu_num);
  int slice = align_w / eu_num;
  if (align_w > w) {
    dim4 fill_offset = {0, 0, 0, w};
    dim4 fill_shape = {n, c, 1, align_w - w};
    auto fill_tensor = src.sub_view(fill_shape, fill_offset);
    fill(fill_tensor, 0);
  }
  dim4 in_reduce_h = {n, c, slice, opti_w};

  pool_param kernel_param0 = pool::param::kernel(slice, 1, 1, 1);
  pool_param pad_param = pool::param::padding(0, 0, 0, 0, 0);
  pool_param kernel_param1 = pool::param::kernel(1, opti_w, 1, 1);
  if constexpr (REDUCE_MODE == 0) {
    auto reduce_h_tensor =
        tiu::pool_max(src.reshape(in_reduce_h), kernel_param0, pad_param);
    return tiu::pool_max(reduce_h_tensor, kernel_param1, pad_param);
  } else if constexpr (REDUCE_MODE == 1) {
    auto reduce_h_tensor = tiu::pool_avg<OUTTYPE>(src.reshape(in_reduce_h),
                                                  kernel_param0, pad_param);
    return tiu::pool_avg<OUTTYPE>(reduce_h_tensor, kernel_param1, pad_param);
  } else {
    static_assert(true, "only support reduce max and sum now");
  }
}

template <typename dtype>
tensor2<dtype> &reduce_max(tensor2<dtype> &src, int axis) {
  return reduce_base<0, dtype, dtype>(src, axis);
}
template <typename dtype, typename dtype2 = dtype>
tensor2<dtype2> &reduce_sum(tensor2<dtype> &src, int axis) {
  return reduce_base<1, dtype, dtype2>(src, axis);
}

} // namespace tiu
} // namespace ez
} // namespace ppl
