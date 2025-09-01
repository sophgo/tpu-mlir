#include "ppl.h"  // PPL 代码依赖的头文件
using namespace ppl;
#include "ppl_wrapper_func.h"
#ifdef __bm1690__
#define CORE_NUM 8
#elif __bm1688__
#define CORE_NUM 2
#else
#define CORE_NUM 1
#endif
#define DTYPE fp32
template <typename T>
void interp_(T *ptr_output, T *ptr_input, const int g_core_num, const int N, const int C, const int H_in, const int W_in, const int H_out, const int W_out, const int block_h, const int block_w, const int align_corners) {
  ppl::set_core_num(g_core_num);
  int core_num = ppl::get_core_num();
  int core_idx = ppl::get_core_index();
  if (core_idx >= core_num) {
    return;
  }
  int c_per_core = div_up(C, core_num);
  int c_start = core_idx * c_per_core;
  int c_end = min(c_start + c_per_core, C);
  int block_c = LANE_NUM;

  int start = 0;
  int step  = 1;
  int W_end = W_out;
  int H_end = H_out;
  double scale_h =static_cast<double>(H_in - align_corners) / (H_out - align_corners);
  double scale_w =static_cast<double>(W_in - align_corners) / (W_out - align_corners);
  double scalar_c = 1;
  double const_c = 0.5;
  double scalar_min = 0;
  double scalar_h_max = H_in-1;
  double scalar_w_max = W_in-1;

  int count_w = 0;
  dim4 global_input_shape = {N, C, H_in, W_in};
  dim4 global_output_shape = {N, C, H_out, W_out};
  auto in_gtensor = gtensor<T>(global_input_shape, GLOBAL, ptr_input);
  auto res_gtensor = gtensor<T>(global_output_shape, GLOBAL, ptr_output);


  for(auto idx_w = 0; idx_w < W_out; idx_w += block_w){
    int cur_w = min(block_w, W_out - idx_w);
    dim4 shape_local_real_w = {1, 1, 1, cur_w};
    dim4 shape_local_block_w = {1, 1, 1, block_w};
    auto shape_W = make_tensor<int32>(shape_local_block_w,shape_local_real_w);
    tiu::arange_broadcast(shape_W, idx_w, step, cur_w);
    auto shape_W_fp32 = make_tensor<fp32>(shape_local_block_w,shape_local_real_w);
    tiu::cast(shape_W_fp32,shape_W);
    if(align_corners){
      tiu::fmul(shape_W_fp32,shape_W_fp32,scale_w);
    }else{
      tiu::fadd(shape_W_fp32,shape_W_fp32,const_c);
      tiu::fmul(shape_W_fp32,shape_W_fp32,scale_w);
      tiu::fsub(shape_W_fp32,shape_W_fp32,const_c);
    }
    tiu::fmax(shape_W_fp32, shape_W_fp32, scalar_min);
    tiu::fmin(shape_W_fp32, shape_W_fp32, scalar_w_max);
    auto wi = make_tensor<fp32>(shape_local_block_w,shape_local_real_w);
    auto wi_1 = make_tensor<fp32>(shape_local_block_w,shape_local_real_w);
    tiu::floor(wi, shape_W_fp32);
    tiu::fadd(wi_1, wi, scalar_c);
    tiu::fmax(wi_1, wi_1, scalar_min);
    tiu::fmin(wi_1, wi_1, scalar_w_max);
    auto weight_a = tensor<fp32>(shape_local_block_w);
    tiu::fsub(weight_a,shape_W_fp32,wi);
    auto weight_1_a = tensor<fp32>(shape_local_block_w);
    tiu::fsub(weight_1_a,1,weight_a);
    dim4 wi_block_shape = {1, block_w, 1, 2};
    dim4 wi_real_shape = {1, cur_w, 1, 2};
    auto wi_tensor = make_tensor<fp32>(wi_block_shape,wi_real_shape,TPU_COMPACT);
    tiu::fill(wi_tensor, 0);
    dim4 wi_trans_block_shape = {1, block_w, 1, 1};
    dim4 wi_trans_real_shape = {1, cur_w, 1, 1};
    auto wi_trans_tensor = make_tensor<fp32>(wi_trans_block_shape,wi_trans_real_shape,TPU_COMPACT);
    dma::transpose_cw(wi_trans_tensor.view(wi_trans_real_shape),wi.view(shape_local_real_w));
    dim4 wi_tensor_stride;
    get_stride<fp32>(&wi_tensor_stride, &wi_trans_real_shape, TPU_COMPACT);
    wi_tensor_stride.c = 2;
    wi_tensor_stride.w = 2;
    tiu::move(wi_tensor.view(wi_trans_real_shape,wi_tensor_stride),wi_trans_tensor.view(wi_trans_real_shape));

    dim4 wi_1_block_shape = {1, block_w, 1, 2};
    dim4 wi_1_real_shape = {1, cur_w, 1, 2};
    auto wi_1_tensor = make_tensor<fp32>(wi_1_block_shape,wi_1_real_shape,TPU_COMPACT);
    dma::fill(wi_1_tensor, 0);
    dim4 wi_1_trans_block_shape = {1, block_w, 1, 1};
    dim4 wi_1_trans_real_shape = {1, cur_w, 1, 1};
    // //dma指令
    auto wi_1_trans_tensor = make_tensor<fp32>(wi_1_trans_block_shape,wi_1_trans_real_shape,TPU_COMPACT);
    dma::transpose_cw(wi_1_trans_tensor.view(wi_1_trans_real_shape),wi_1.view(shape_local_real_w));
    dim4 wi_1_tensor_stride;
    get_stride<fp32>(&wi_1_tensor_stride, &wi_1_trans_real_shape, TPU_COMPACT);
    wi_1_tensor_stride.c = 2;
    wi_1_tensor_stride.w = 2;
    tiu::move(wi_1_tensor.view(wi_1_trans_real_shape,wi_1_tensor_stride),wi_1_trans_tensor.view(wi_1_trans_real_shape));

    int count_h = 0;
    for (auto idx_h = 0; idx_h < H_out; idx_h += block_h) {
      int cur_h = min(block_h, H_out - idx_h);
      dim4 shape_local_real_H = {1, 1, 1, cur_h};
      dim4 shape_local_block_H = {1, 1, 1, block_h};
      auto shape_H  = make_tensor<int32>(shape_local_block_H, shape_local_real_H);
      tiu::arange_broadcast(shape_H, idx_h, step, cur_h);
      auto shape_H_fp32 = make_tensor<fp32>(shape_local_block_H, shape_local_real_H);
      tiu::cast(shape_H_fp32,shape_H);
      if(align_corners){
        tiu::fmul(shape_H_fp32,shape_H_fp32,scale_h);
      }else{
        tiu::fadd(shape_H_fp32,shape_H_fp32,const_c);
        tiu::fmul(shape_H_fp32,shape_H_fp32,scale_h);
        tiu::fsub(shape_H_fp32,shape_H_fp32,const_c);
      }
      tiu::fmax(shape_H_fp32, shape_H_fp32, scalar_min);
      tiu::fmin(shape_H_fp32, shape_H_fp32, scalar_h_max);
      auto hi = make_tensor<fp32>(shape_local_block_H, shape_local_real_H);
      auto hi_1 = make_tensor<fp32>(shape_local_block_H, shape_local_real_H);
      tiu::floor(hi, shape_H_fp32);
      tiu::fadd(hi_1, hi, scalar_c);
      tiu::fmax(hi_1, hi_1, scalar_min);
      tiu::fmin(hi_1, hi_1, scalar_h_max);
      auto weight_b_re = make_tensor<fp32>(shape_local_block_H, shape_local_real_H);
      tiu::fsub(weight_b_re,shape_H_fp32,hi);
      dim4 shape_h_blcok_reshape = {1, 1, block_h, 1};
      dim4 shape_h_real_reshape = {1, 1, cur_h, 1};
      auto weight_b = weight_b_re.view(shape_h_real_reshape);
      auto weight_1_b = make_tensor<fp32>(shape_h_blcok_reshape, shape_h_real_reshape);
      tiu::fsub(weight_1_b,1,weight_b);
      dim4 hi_block_shape = {1, 1, 1, block_h*2};
      dim4 hi_real_shape = {1, 1, 1, cur_h*2};
      auto hi_tensor = make_tensor<fp32>(hi_block_shape, hi_real_shape);
      tiu::fill(hi_tensor, 0);
      dim4 hi_tensor_stride;
      get_stride<fp32>(&hi_tensor_stride, &shape_local_real_H, TPU_ALIGN);
      hi_tensor_stride.w = 2;
      dim4 hi_offset = {0,0,0,1};
      auto hi_tensor_offset = hi_tensor.sub_view(hi_real_shape,hi_offset);
      tiu::move(hi_tensor_offset.view(shape_local_real_H,hi_tensor_stride),hi);
      dim4 hi_reshape = {1, 1, cur_h, 2};
      auto hi_tensor_reshape = hi_tensor.view(hi_reshape);
      dim4 hi_1_block_shape = {1, 1, 1, block_h*2};
      dim4 hi_1_real_shape = {1, 1, 1, cur_h*2};
      auto hi_1_tensor = make_tensor<fp32>(hi_1_block_shape, hi_1_real_shape);
      tiu::fill(hi_1_tensor, 0);
      dim4 hi_1_tensor_stride;
      get_stride<fp32>(&hi_1_tensor_stride, &shape_local_real_H, TPU_ALIGN);
      hi_1_tensor_stride.w = 2;
      dim4 hi_1_offset = {0,0,0,1};
      auto hi_1_tensor_offset = hi_1_tensor.sub_view(hi_1_real_shape,hi_1_offset);
      tiu::move(hi_1_tensor_offset.view(shape_local_real_H,hi_1_tensor_stride),hi_1);
      dim4 hi_1_reshape = {1, 1, cur_h, 2};
      auto hi_1_tensor_reshape = hi_1_tensor.view(hi_1_reshape);

      dim4 hi_bcast_real_shape = {1, LANE_NUM, cur_h, 2};
      dim4 hi_bcast_block_shape = {1, LANE_NUM, block_h, 2};
      auto hi_bcast_tensor = make_tensor<fp32>(hi_bcast_block_shape, hi_bcast_real_shape);
      tiu::broadcast(hi_bcast_tensor,hi_tensor_reshape);
      auto hi_1_bcast_tensor = make_tensor<fp32>(hi_bcast_block_shape, hi_bcast_real_shape);
      tiu::broadcast(hi_1_bcast_tensor,hi_1_tensor_reshape);
      dim4 hi_c_bc_stride;
      get_stride<fp32>(&hi_c_bc_stride, &hi_bcast_real_shape, TPU_ALIGN);
      hi_c_bc_stride.c = 0;

      dim4 index_out_real_shape = {1, cur_w, cur_h, 2};
      dim4 index_out_block_shape = {1, block_w, block_h, 2};
      auto hi_bcast_all = hi_bcast_tensor.view(index_out_real_shape, hi_c_bc_stride);
      auto hi_1_bcast_all = hi_1_bcast_tensor.view(index_out_real_shape, hi_c_bc_stride);

      auto real_k = cur_w * cur_h;
      auto block_k = block_w * block_h;
      int real_align_k = div_up(real_k, LANE_NUM);
      int block_align_k = div_up(block_k, LANE_NUM);
      dim4 local_index_trans_block_shape = {1, 2, block_h, block_w};
      dim4 local_index_trans_cpy_block_shape = {1, 2, block_align_k, LANE_NUM};
      dim4 local_index_trans_cpy_trans_block_shape = {1, LANE_NUM, block_align_k, 2};
      dim4 local_index_trans_real_shape = {1, 2, cur_h, cur_w};
      dim4 local_index_trans_cpy_real_shape = {1, 2, real_align_k, LANE_NUM};
      dim4 local_index_trans_cpy_trans_real_shape = {1, LANE_NUM, real_align_k, 2};
      dim4 real_index_shape = {1, cur_w*cur_h, 1, 2};
      // index1
      auto local_index1 = tensor<fp32>(index_out_block_shape,TPU_ALIGN);
      tiu::fadd(local_index1.view(index_out_real_shape),wi_tensor, hi_bcast_all);
      auto local_index1_align = tensor<fp32>(index_out_block_shape,TPU_ALIGN);
      tiu::move(local_index1_align.view(index_out_real_shape), local_index1.view(index_out_real_shape));
      auto local_index1_trans_tensor = tensor<fp32>(local_index_trans_block_shape,TPU_ROW_ALIGN);
      tiu::transpose_cw(local_index1_trans_tensor.view(local_index_trans_real_shape),local_index1_align.view(index_out_real_shape));
      auto local_index1_trans_tensor_copy = tensor<fp32>(local_index_trans_cpy_block_shape,TPU_ALIGN);
      tiu::move(local_index1_trans_tensor_copy.view(local_index_trans_real_shape), local_index1_trans_tensor.view(local_index_trans_real_shape));
      auto local_index1_new = local_index1_trans_tensor_copy.view(local_index_trans_cpy_real_shape);
      auto local_index1_trans_cpy_trans_tensor = tensor<fp32>(local_index_trans_cpy_trans_block_shape,TPU_ROW_ALIGN);
      tiu::transpose_cw(local_index1_trans_cpy_trans_tensor.view(local_index_trans_cpy_trans_real_shape),local_index1_new);
      // index2
      auto local_index2 = tensor<fp32>(index_out_block_shape,TPU_COMPACT);
      tiu::fadd(local_index2.view(index_out_real_shape),wi_tensor, hi_1_bcast_all);
      auto local_index2_align = tensor<fp32>(index_out_block_shape,TPU_ALIGN);
      tiu::move(local_index2_align.view(index_out_real_shape), local_index2.view(index_out_real_shape));
      auto local_index2_trans_tensor = tensor<fp32>(local_index_trans_block_shape,TPU_ROW_ALIGN);
      tiu::transpose_cw(local_index2_trans_tensor.view(local_index_trans_real_shape),local_index2_align.view(index_out_real_shape));
      auto local_index2_trans_tensor_copy = tensor<fp32>(local_index_trans_cpy_block_shape,TPU_ALIGN);
      tiu::move(local_index2_trans_tensor_copy.view(local_index_trans_real_shape), local_index2_trans_tensor.view(local_index_trans_real_shape));
      auto local_index2_new = local_index2_trans_tensor_copy.view(local_index_trans_cpy_real_shape);
      auto local_index2_trans_cpy_trans_tensor = tensor<fp32>(local_index_trans_cpy_trans_block_shape,TPU_ROW_ALIGN);
      tiu::transpose_cw(local_index2_trans_cpy_trans_tensor.view(local_index_trans_cpy_trans_real_shape),local_index2_new);
      // index3
      auto local_index3 = tensor<fp32>(index_out_block_shape,TPU_COMPACT);
      tiu::fadd(local_index3.view(index_out_real_shape),wi_1_tensor, hi_bcast_all);
      auto local_index3_align = tensor<fp32>(index_out_block_shape,TPU_ALIGN);
      tiu::move(local_index3_align.view(index_out_real_shape), local_index3.view(index_out_real_shape));
      auto local_index3_trans_tensor = tensor<fp32>(local_index_trans_block_shape,TPU_ROW_ALIGN);
      tiu::transpose_cw(local_index3_trans_tensor.view(local_index_trans_real_shape),local_index3_align.view(index_out_real_shape));
      auto local_index3_trans_tensor_copy = tensor<fp32>(local_index_trans_cpy_block_shape,TPU_ALIGN);
      tiu::move(local_index3_trans_tensor_copy.view(local_index_trans_real_shape), local_index3_trans_tensor.view(local_index_trans_real_shape));
      auto local_index3_new = local_index3_trans_tensor_copy.view(local_index_trans_cpy_real_shape);
      auto local_index3_trans_cpy_trans_tensor = tensor<fp32>(local_index_trans_cpy_trans_block_shape,TPU_ROW_ALIGN);
      tiu::transpose_cw(local_index3_trans_cpy_trans_tensor.view(local_index_trans_cpy_trans_real_shape),local_index3_new);
      // index4
      auto local_index4 = tensor<fp32>(index_out_block_shape,TPU_COMPACT);
      tiu::fadd(local_index4.view(index_out_real_shape),wi_1_tensor, hi_1_bcast_all);
      auto local_index4_align = tensor<fp32>(index_out_block_shape,TPU_ALIGN);
      tiu::move(local_index4_align.view(index_out_real_shape), local_index4.view(index_out_real_shape));
      auto local_index4_trans_tensor = tensor<fp32>(local_index_trans_block_shape,TPU_ROW_ALIGN);
      tiu::transpose_cw(local_index4_trans_tensor.view(local_index_trans_real_shape),local_index4_align.view(index_out_real_shape));
      auto local_index4_trans_tensor_copy = tensor<fp32>(local_index_trans_cpy_block_shape,TPU_ALIGN);
      tiu::move(local_index4_trans_tensor_copy.view(local_index_trans_real_shape), local_index4_trans_tensor.view(local_index_trans_real_shape));
      auto local_index4_new = local_index4_trans_tensor_copy.view(local_index_trans_cpy_real_shape);
      auto local_index4_trans_cpy_trans_tensor = tensor<fp32>(local_index_trans_cpy_trans_block_shape,TPU_ROW_ALIGN);
      tiu::transpose_cw(local_index4_trans_cpy_trans_tensor.view(local_index_trans_cpy_trans_real_shape),local_index4_new);

      auto index1_uint16 = tensor<uint16>(local_index_trans_cpy_trans_block_shape, TPU_COMPACT);
      auto index2_uint16 = tensor<uint16>(local_index_trans_cpy_trans_block_shape, TPU_COMPACT);
      auto index3_uint16 = tensor<uint16>(local_index_trans_cpy_trans_block_shape, TPU_COMPACT);
      auto index4_uint16 = tensor<uint16>(local_index_trans_cpy_trans_block_shape, TPU_COMPACT);
      tiu::cast(index1_uint16.view(real_index_shape),local_index1_trans_cpy_trans_tensor.view(real_index_shape));
      tiu::cast(index2_uint16.view(real_index_shape),local_index2_trans_cpy_trans_tensor.view(real_index_shape));
      tiu::cast(index3_uint16.view(real_index_shape),local_index3_trans_cpy_trans_tensor.view(real_index_shape));
      tiu::cast(index4_uint16.view(real_index_shape),local_index4_trans_cpy_trans_tensor.view(real_index_shape));

      int index_h_dif = (int)floor(count_h * block_h * scale_h);
      int index_w_dif = (int)floor(count_w * block_w * scale_w);
      if(count_h){
        index_h_dif = (int)floor(count_h * block_h * scale_h - 1);
        if(index_h_dif < 0){
          index_h_dif = 0;
        }
      }
      if(count_w){
        index_w_dif = (int)floor(count_w * block_w * scale_w - 1);
        if(index_w_dif < 0){
          index_w_dif = 0;
        }
      }
      if(!align_corners){
        index_h_dif = (int)floor((count_h * block_h + const_c) * scale_h - const_c);
        if(index_h_dif < 0){
          index_h_dif = 0;
        }
        index_w_dif = (int)floor((count_w * block_w + const_c) * scale_w - const_c);
        if(index_w_dif < 0){
          index_w_dif = 0;
        }
      }
      dim4 h_stride;
      get_stride<int16>(&h_stride, &real_index_shape, TPU_COMPACT);
      h_stride.w = 2;
      dim4 w_stride;
      get_stride<int16>(&w_stride, &real_index_shape, TPU_COMPACT);
      w_stride.w = 2;
      dim4 offset = {0,0,0,1};
      dim4 real_index_stride_shape = {1, cur_h * cur_w, 1, 1};

      auto index1_int16 = index1_uint16.view<int16>(real_index_shape);
      auto index1_h_int16_offset = index1_int16.sub_view(real_index_shape,offset);
      auto index1_w_tensor = index1_int16.view(real_index_stride_shape, w_stride);
      auto index1_h_tensor = index1_h_int16_offset.view(real_index_stride_shape, h_stride);
      tiu::sub(index1_h_tensor, index1_h_tensor, index_h_dif, 0, RM_DOWN, false);
      tiu::sub(index1_w_tensor, index1_w_tensor, index_w_dif, 0, RM_DOWN, false);

      auto index2_int16 = index2_uint16.view<int16>(real_index_shape);
      auto index2_h_int16_offset = index2_int16.sub_view(real_index_shape,offset);
      auto index2_w_tensor = index2_int16.view(real_index_stride_shape, w_stride);
      auto index2_h_tensor = index2_h_int16_offset.view(real_index_stride_shape, h_stride);
      tiu::sub(index2_h_tensor, index2_h_tensor, index_h_dif, 0, RM_DOWN, false);
      tiu::sub(index2_w_tensor, index2_w_tensor, index_w_dif, 0, RM_DOWN, false);

      auto index3_int16 = index3_uint16.view<int16>(real_index_shape);
      auto index3_h_int16_offset = index3_int16.sub_view(real_index_shape,offset);
      auto index3_w_tensor = index3_int16.view(real_index_stride_shape, w_stride);
      auto index3_h_tensor = index3_h_int16_offset.view(real_index_stride_shape, h_stride);
      tiu::sub(index3_h_tensor, index3_h_tensor, index_h_dif, 0, RM_DOWN, false);
      tiu::sub(index3_w_tensor, index3_w_tensor, index_w_dif, 0, RM_DOWN, false);

      auto index4_int16 = index4_uint16.view<int16>(real_index_shape);
      auto index4_h_int16_offset = index4_int16.sub_view(real_index_shape,offset);
      auto index4_w_tensor = index4_int16.view(real_index_stride_shape, w_stride);
      auto index4_h_tensor = index4_h_int16_offset.view(real_index_stride_shape, h_stride);
      tiu::sub(index4_h_tensor, index4_h_tensor, index_h_dif, 0, RM_DOWN, false);
      tiu::sub(index4_w_tensor, index4_w_tensor, index_w_dif, 0, RM_DOWN, false);

      int weight_block_h = block_h;
      int weight_block_w = block_w;
      int cur_weight_h = min(weight_block_h, cur_h);
      int cur_weight_w = min(weight_block_w, cur_w);
      dim4 real_weight_block_shape = {1, 1, weight_block_h, weight_block_w};
      dim4 real_weight_real_shape = {1, 1, cur_weight_h, cur_weight_w};
      auto cur_weight1_tensor = tensor<DTYPE>(real_weight_block_shape);
      auto cur_weight2_tensor = tensor<DTYPE>(real_weight_block_shape);
      auto cur_weight3_tensor = tensor<DTYPE>(real_weight_block_shape);
      auto cur_weight4_tensor = tensor<DTYPE>(real_weight_block_shape);
      tiu::fmul(cur_weight1_tensor.view(real_weight_real_shape), weight_a,weight_b);
      tiu::fmul(cur_weight2_tensor.view(real_weight_real_shape), weight_a,weight_1_b);
      tiu::fmul(cur_weight3_tensor.view(real_weight_real_shape), weight_1_a,weight_b);
      tiu::fmul(cur_weight4_tensor.view(real_weight_real_shape), weight_1_a,weight_1_b);

      for (auto idx_c = 0; idx_c < C; idx_c += block_c){
        int c = min(block_c, C - idx_c);
        int idx_input_h =  floor(idx_h * scale_h);
        int idx_input_w =  floor(idx_w * scale_w);
        if(count_h){
          idx_input_h =  floor(idx_h * scale_h - 1);
          if(idx_input_h < 0){
            idx_input_h = 0;
          }
        }
        if(count_w){
          idx_input_w =  floor(idx_w * scale_w - 1);
          if(idx_input_w < 0){
            idx_input_w = 0;
          }
        }
        if(!align_corners){
          idx_input_h = floor((idx_h + const_c) * scale_h - const_c);
          if(idx_input_h < 0){
            idx_input_h = 0;
          }
          idx_input_w = floor((idx_w + const_c) * scale_w - const_c);
          if(idx_input_w < 0){
            idx_input_w = 0;
          }
        }
        int input_block_h = (int)(block_h * scale_h + 5);
        int input_block_w = (int)(block_w * scale_w + 5);

        int cur_input_h = min(input_block_h, H_in - idx_input_h);
        int cur_input_w = min(input_block_w, W_in - idx_input_w);

        dim4 real_input_block_shape = {N, block_c, input_block_h, input_block_w};
        dim4 real_input_real_shape = {N, c, cur_input_h, cur_input_w};

        dim4 input_offset = {0, idx_c, idx_input_h, idx_input_w};  // 当前需要计算的数据在 ddr 上的偏移
        auto cur_in_tensor = make_tensor<T>(real_input_block_shape, real_input_real_shape);
        dma::load(cur_in_tensor.view(real_input_real_shape), in_gtensor.sub_view(real_input_real_shape, input_offset));

        dim4 real_output_block_shape = {N, block_c, block_h, block_w};
        int output_real_block_h = min(block_h, cur_h);
        int output_real_block_w = min(block_w, cur_w);
        dim4 output_real_shape = {N, c, output_real_block_h, output_real_block_w};
        auto output1_tensor = make_tensor<T>(real_output_block_shape, output_real_shape);
        auto output2_tensor = make_tensor<T>(real_output_block_shape, output_real_shape);
        dim4 out_offset = {0, idx_c, idx_h, idx_w};
        if constexpr (std::is_same_v<T, bf16>) {
          auto cur_weight1_tensor_T = tensor<T>(real_weight_block_shape);
          auto cur_weight2_tensor_T = tensor<T>(real_weight_block_shape);
          auto cur_weight3_tensor_T = tensor<T>(real_weight_block_shape);
          auto cur_weight4_tensor_T = tensor<T>(real_weight_block_shape);
          tiu::cast(cur_weight1_tensor_T.view(real_weight_real_shape),cur_weight1_tensor.view(real_weight_real_shape));
          tiu::cast(cur_weight2_tensor_T.view(real_weight_real_shape),cur_weight2_tensor.view(real_weight_real_shape));
          tiu::cast(cur_weight3_tensor_T.view(real_weight_real_shape),cur_weight3_tensor.view(real_weight_real_shape));
          tiu::cast(cur_weight4_tensor_T.view(real_weight_real_shape),cur_weight4_tensor.view(real_weight_real_shape));
          tiu::gather_hw(output1_tensor.view(output_real_shape), cur_in_tensor.view(real_input_real_shape), index1_uint16.view<uint16>(real_index_shape));
          tiu::fmul(output1_tensor.view(output_real_shape), output1_tensor.view(output_real_shape), cur_weight4_tensor_T);
          tiu::gather_hw(output2_tensor.view(output_real_shape), cur_in_tensor.view(real_input_real_shape), index2_uint16.view<uint16>(real_index_shape));
          tiu::fmul(output2_tensor.view(output_real_shape), output2_tensor.view(output_real_shape), cur_weight3_tensor_T);
          tiu::fadd(output1_tensor.view(output_real_shape), output1_tensor.view(output_real_shape), output2_tensor.view(output_real_shape));
          tiu::gather_hw(output2_tensor.view(output_real_shape), cur_in_tensor.view(real_input_real_shape), index3_uint16.view<uint16>(real_index_shape));
          tiu::fmul(output2_tensor.view(output_real_shape), output2_tensor.view(output_real_shape), cur_weight2_tensor_T);
          tiu::fadd(output1_tensor.view(output_real_shape), output1_tensor.view(output_real_shape), output2_tensor.view(output_real_shape));
          tiu::gather_hw(output2_tensor.view(output_real_shape), cur_in_tensor.view(real_input_real_shape), index4_uint16.view<uint16>(real_index_shape));
          tiu::fmul(output2_tensor.view(output_real_shape), output2_tensor.view(output_real_shape), cur_weight1_tensor_T);
          tiu::fadd(output1_tensor.view(output_real_shape), output1_tensor.view(output_real_shape), output2_tensor.view(output_real_shape));
          dma::store(res_gtensor.sub_view(output_real_shape, out_offset), output1_tensor.view(output_real_shape));
        } else if constexpr (std::is_same_v<T, fp16>) {
          auto cur_weight1_tensor_T = tensor<T>(real_weight_block_shape);
          auto cur_weight2_tensor_T = tensor<T>(real_weight_block_shape);
          auto cur_weight3_tensor_T = tensor<T>(real_weight_block_shape);
          auto cur_weight4_tensor_T = tensor<T>(real_weight_block_shape);
          tiu::cast(cur_weight1_tensor_T.view(real_weight_real_shape),cur_weight1_tensor.view(real_weight_real_shape));
          tiu::cast(cur_weight2_tensor_T.view(real_weight_real_shape),cur_weight2_tensor.view(real_weight_real_shape));
          tiu::cast(cur_weight3_tensor_T.view(real_weight_real_shape),cur_weight3_tensor.view(real_weight_real_shape));
          tiu::cast(cur_weight4_tensor_T.view(real_weight_real_shape),cur_weight4_tensor.view(real_weight_real_shape));
          tiu::gather_hw(output1_tensor.view(output_real_shape), cur_in_tensor.view(real_input_real_shape), index1_uint16.view<uint16>(real_index_shape));
          tiu::fmul(output1_tensor.view(output_real_shape), output1_tensor.view(output_real_shape), cur_weight4_tensor_T);
          tiu::gather_hw(output2_tensor.view(output_real_shape), cur_in_tensor.view(real_input_real_shape), index2_uint16.view<uint16>(real_index_shape));
          tiu::fmul(output2_tensor.view(output_real_shape), output2_tensor.view(output_real_shape), cur_weight3_tensor_T);
          tiu::fadd(output1_tensor.view(output_real_shape), output1_tensor.view(output_real_shape), output2_tensor.view(output_real_shape));
          tiu::gather_hw(output2_tensor.view(output_real_shape), cur_in_tensor.view(real_input_real_shape), index3_uint16.view<uint16>(real_index_shape));
          tiu::fmul(output2_tensor.view(output_real_shape), output2_tensor.view(output_real_shape), cur_weight2_tensor_T);
          tiu::fadd(output1_tensor.view(output_real_shape), output1_tensor.view(output_real_shape), output2_tensor.view(output_real_shape));
          tiu::gather_hw(output2_tensor.view(output_real_shape), cur_in_tensor.view(real_input_real_shape), index4_uint16.view<uint16>(real_index_shape));
          tiu::fmul(output2_tensor.view(output_real_shape), output2_tensor.view(output_real_shape), cur_weight1_tensor_T);
          tiu::fadd(output1_tensor.view(output_real_shape), output1_tensor.view(output_real_shape), output2_tensor.view(output_real_shape));
          dma::store(res_gtensor.sub_view(output_real_shape, out_offset), output1_tensor.view(output_real_shape));
        } else {
          tiu::gather_hw(output1_tensor.view(output_real_shape), cur_in_tensor.view(real_input_real_shape), index1_uint16.view<uint16>(real_index_shape));
          tiu::fmul(output1_tensor.view(output_real_shape), output1_tensor.view(output_real_shape), cur_weight4_tensor);
          tiu::gather_hw(output2_tensor.view(output_real_shape), cur_in_tensor.view(real_input_real_shape), index2_uint16.view<uint16>(real_index_shape));
          tiu::fmul(output2_tensor.view(output_real_shape), output2_tensor.view(output_real_shape), cur_weight3_tensor);
          tiu::fadd(output1_tensor.view(output_real_shape), output1_tensor.view(output_real_shape), output2_tensor.view(output_real_shape));
          tiu::gather_hw(output2_tensor.view(output_real_shape), cur_in_tensor.view(real_input_real_shape), index3_uint16.view<uint16>(real_index_shape));
          tiu::fmul(output2_tensor.view(output_real_shape), output2_tensor.view(output_real_shape), cur_weight2_tensor);
          tiu::fadd(output1_tensor.view(output_real_shape), output1_tensor.view(output_real_shape), output2_tensor.view(output_real_shape));
          tiu::gather_hw(output2_tensor.view(output_real_shape), cur_in_tensor.view(real_input_real_shape), index4_uint16.view<uint16>(real_index_shape));
          tiu::fmul(output2_tensor.view(output_real_shape), output2_tensor.view(output_real_shape), cur_weight1_tensor);
          tiu::fadd(output1_tensor.view(output_real_shape), output1_tensor.view(output_real_shape), output2_tensor.view(output_real_shape));
          dma::store(res_gtensor.sub_view(output_real_shape, out_offset), output1_tensor.view(output_real_shape));
        }
      }
      count_h++;
    }
    count_w++;
  }
}
__KERNEL__ void interp_linear_bf16(bf16 *ptr_output, bf16 *ptr_input, const int g_core_num, const int N, const int C, const int H_in, const int W_in, const int H_out, const int W_out, const int block_h, const int block_w, const int align_corners) {
            interp_<bf16>(ptr_output, ptr_input,  g_core_num, N,  C,  H_in,  W_in,  H_out,  W_out,  block_h, block_w, align_corners);
}
__KERNEL__ void interp_linear_fp16(fp16 *ptr_output, fp16 *ptr_input, const int g_core_num, const int N, const int C, const int H_in, const int W_in, const int H_out, const int W_out, const int block_h, const int block_w, const int align_corners) {
            interp_<fp16>(ptr_output, ptr_input,  g_core_num, N,  C,  H_in,  W_in,  H_out,  W_out,  block_h, block_w, align_corners);
}
__KERNEL__ void interp_linear_fp32(fp32 *ptr_output, fp32 *ptr_input, const int g_core_num, const int N, const int C, const int H_in, const int W_in, const int H_out, const int W_out, const int block_h, const int block_w, const int align_corners) {
            interp_<fp32>(ptr_output, ptr_input,  g_core_num, N,  C,  H_in,  W_in,  H_out,  W_out,  block_h, block_w, align_corners);
}
