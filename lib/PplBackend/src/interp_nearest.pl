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

void interp_(T *ptr_output, T *ptr_input, DTYPE *ptr_index, const int g_core_num, const int N, const int C, const int H_in, const int W_in, const int H_out, const int W_out, const int block_h, const int align_corners) {

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

  int H_num = H_end -start;
  dim4 shape_local_H = {1, 1, 1, H_num};
  auto shape_H = tensor<int32>(shape_local_H);
  tiu::arange_broadcast(shape_H, start, step, H_num);
  auto shape_H_fp32 = tensor<fp32>(shape_local_H);
  tiu::cast(shape_H_fp32,shape_H);

  tiu::fmul(shape_H_fp32,shape_H_fp32,scale_h);


  auto hi = tensor<fp32>(shape_local_H);
  auto hi_1 = tensor<fp32>(shape_local_H);
  if(align_corners){
    tiu::round(hi, shape_H_fp32, RM_HALF_AWAY_FROM_ZERO);
  }else{
    tiu::floor(hi, shape_H_fp32);
  }
  // tiu::round(hi, shape_H_fp32, RM_HALF_AWAY_FROM_ZERO);
  tiu::fadd(hi_1, hi, scalar_c);
  tiu::fmax(hi_1, hi_1, scalar_min);
  tiu::fmin(hi_1, hi_1, scalar_h_max);



  int W_num = W_end -start;
  dim4 shape_local_W = {1, 1, 1, W_num};
  auto shape_W = tensor<int32>(shape_local_W);
  tiu::arange_broadcast(shape_W, start, step, W_num);
  auto shape_W_fp32 = tensor<fp32>(shape_local_W);
  tiu::cast(shape_W_fp32,shape_W);

  tiu::fmul(shape_W_fp32,shape_W_fp32,scale_w);


  auto wi = tensor<fp32>(shape_local_W);
  auto wi_1 = tensor<fp32>(shape_local_W);
  if(align_corners){
    tiu::round(wi, shape_W_fp32, RM_HALF_AWAY_FROM_ZERO);
  }else{
    tiu::floor(wi, shape_W_fp32);
  }
  // tiu::round(wi, shape_W_fp32, RM_HALF_AWAY_FROM_ZERO);
  tiu::fadd(wi_1, wi, scalar_c);
  tiu::fmax(wi_1, wi_1, scalar_min);
  tiu::fmin(wi_1, wi_1, scalar_w_max);


  dim4 hi_shape = {1, H_num, 1, 2};
  auto hi_tensor = tensor<fp32>(hi_shape,TPU_COMPACT);
  dma::fill(hi_tensor, 0);
  dim4 hi_trans_shape = {1, H_num, 1, 1};
  // //dma指令
  auto hi_trans_tensor = tensor<fp32>(hi_trans_shape,TPU_COMPACT);
  dma::transpose_cw(hi_trans_tensor,hi);
  dim4 hi_tensor_stride;
  get_stride<fp32>(&hi_tensor_stride, &hi_trans_shape, TPU_COMPACT);
  hi_tensor_stride.c = 2;
  hi_tensor_stride.w = 2;
  dim4 hi_offset = {0,0,0,1};
  auto hi_tensor_offset = hi_tensor.sub_view(hi_shape,hi_offset);
  tiu::move(hi_tensor_offset.view(hi_trans_shape,hi_tensor_stride),hi_trans_tensor);

  dim4 hi_1_shape = {1, H_num, 1, 2};
  auto hi_1_tensor = tensor<fp32>(hi_1_shape,TPU_COMPACT);
  dma::fill(hi_1_tensor, 0);
  dim4 hi_1_trans_shape = {1, H_num, 1, 1};
  // //dma指令
  auto hi_1_trans_tensor = tensor<fp32>(hi_1_trans_shape,TPU_COMPACT);
  dma::transpose_cw(hi_1_trans_tensor,hi_1);
  dim4 hi_1_tensor_stride;
  get_stride<fp32>(&hi_1_tensor_stride, &hi_1_trans_shape, TPU_COMPACT);
  hi_1_tensor_stride.c = 2;
  hi_1_tensor_stride.w = 2;
  dim4 hi_1_offset = {0,0,0,1};
  auto hi_1_tensor_offset = hi_1_tensor.sub_view(hi_1_shape,hi_1_offset);
  tiu::move(hi_1_tensor_offset.view(hi_1_trans_shape,hi_1_tensor_stride),hi_1_trans_tensor);


  dim4 wi_shape = {1, 1, 1, W_num*2};
  auto wi_tensor = tensor<fp32>(wi_shape);
  dma::fill(wi_tensor, 0);
  dim4 wi_tensor_stride;
  get_stride<fp32>(&wi_tensor_stride, &shape_local_W, TPU_ALIGN);
  wi_tensor_stride.w = 2;
  tiu::move(wi_tensor.view(shape_local_W,wi_tensor_stride),wi);
  dim4 wi_reshape = {1, 1, W_num, 2};
  auto wi_tensor_reshape = wi_tensor.view(wi_reshape);


  dim4 wi_1_shape = {1, 1, 1, W_num*2};
  auto wi_1_tensor = tensor<fp32>(wi_1_shape);
  dma::fill(wi_1_tensor, 0);
  dim4 wi_1_tensor_stride;
  get_stride<fp32>(&wi_1_tensor_stride, &shape_local_W, TPU_ALIGN);
  wi_1_tensor_stride.w = 2;
  tiu::move(wi_1_tensor.view(shape_local_W,wi_1_tensor_stride),wi_1);
  dim4 wi_1_reshape = {1, 1, W_num, 2};
  auto wi_1_tensor_reshape = wi_1_tensor.view(wi_1_reshape);


  dim4 wi_bcast_shape = {1, LANE_NUM, W_num, 2};
  auto wi_bcast_tensor = tensor<fp32>(wi_bcast_shape);
  tiu::broadcast(wi_bcast_tensor,wi_tensor_reshape);
  auto wi_1_bcast_tensor = tensor<fp32>(wi_bcast_shape);
  tiu::broadcast(wi_1_bcast_tensor,wi_1_tensor_reshape);

  dim4 wi_c_bc_stride;
  get_stride<fp32>(&wi_c_bc_stride, &wi_bcast_shape, TPU_ALIGN);
  wi_c_bc_stride.c = 0;


  dim4 buffer_shape = {1, H_num*4, W_num, 2};
  auto index_gtensor = gtensor<fp32>(buffer_shape, GLOBAL, ptr_index);
  dim4 index_out_shape = {1, H_num, W_num, 2};
  dim4 index1_offset = {0, 0, 0, 0};
  auto index1_gtensor = index_gtensor.sub_view(index_out_shape,index1_offset);


  auto wi_bcast_all = wi_bcast_tensor.view(index_out_shape, wi_c_bc_stride);
  auto wi_1_bcast_all = wi_1_bcast_tensor.view(index_out_shape, wi_c_bc_stride);
  dim4 index_out_stride;
  get_stride<fp32>(&index_out_stride, &index_out_shape, TPU_COMPACT);

  for (auto idx_h = 0; idx_h < W_out; idx_h += block_h) {
    int cur_h = min(block_h, W_out - idx_h);
    dim4 h_offset = {0, 0, idx_h, 0};
    dim4 real_index_shape = {1, 1, cur_h, 2};
    dim4 block_out_shape = {1, H_out, block_h, 2};
    dim4 real_out_shape = {1,H_out, cur_h,2};

    auto local_index1 = tensor<fp32>(block_out_shape,TPU_COMPACT);
    tiu::fadd(local_index1.view(real_out_shape),hi_tensor, wi_bcast_all.sub_view(real_out_shape,h_offset));
    auto index1_gtensor_cur = index1_gtensor.sub_view(real_out_shape,h_offset);
    dma::store(index1_gtensor_cur.view(real_out_shape,index_out_stride),local_index1.view(real_out_shape));
  }

  dim4 global_input_shape = {N, C, H_in, W_in};
  dim4 global_output_shape = {N, C, H_out, W_out};
  dim4 global_index_shape = {1, H_out*W_out, 1, 2};
  auto index1_gtensor_reshape = index1_gtensor.view(global_index_shape);


  auto in_gtensor = gtensor<T>(global_input_shape, GLOBAL, ptr_input);   // 使用tensor封装global memory上的数据
  auto res_gtensor = gtensor<T>(global_output_shape, GLOBAL, ptr_output);

  const int block_index = block_h * W_out;
  int count = 0;
  int max_index = H_out * W_out;


  for (auto idx_index = 0; idx_index < max_index; idx_index += block_index) {
    int cur_index = min(block_index, max_index - idx_index);
    dim4 real_index_block_shape = {1, block_index, 1, 2};
    dim4 real_index_real_shape = {1, cur_index, 1, 2};
    dim4 index_offset = {0, idx_index, 0, 0};
    auto index1 = tensor<DTYPE>(real_index_block_shape, TPU_COMPACT);
    dma::load_compact(index1.view(real_index_real_shape), index1_gtensor_reshape.sub_view(real_index_real_shape, index_offset));//应该把index都放进local里面了
    auto index1_uint16 = tensor<uint16>(real_index_block_shape, TPU_COMPACT);
    tiu::cast(index1_uint16.view(real_index_real_shape),index1.view(real_index_real_shape));



    dim4 w_stride;
    get_stride<int16>(&w_stride, &real_index_real_shape, TPU_COMPACT);
    w_stride.w = 2;
    int index_dif = (int)floor(count * block_h * scale_h);
    if(count){
      index_dif = (int)floor(count * block_h * scale_h - 1);
    }
    dim4 real_index_stride_shape = {1, cur_index, 1, 1};
    dim4 offset = {0,0,0,1};
    auto index1_int16 = index1_uint16.view<int16>(real_index_real_shape);
    auto index1_int16_offset = index1_int16.sub_view(real_index_real_shape,offset);
    auto index1_tensor = index1_int16_offset.view(real_index_stride_shape, w_stride);
    tiu::sub(index1_tensor, index1_tensor, index_dif, 0, RM_DOWN, false);

    for (auto idx_c = 0; idx_c < C; idx_c += block_c){
      int c = min(block_c, C - idx_c);
      int idx_h =  floor(idx_index / W_out * scale_h);
      if(count){
        idx_h =  floor(idx_index / W_out * scale_h - 1);
      }
      int input_block_h = (int)(block_h * scale_h + 5);
      int cur_h = min(input_block_h, H_in - idx_h);
      dim4 real_input_block_shape = {N, block_c, input_block_h, W_in};
      dim4 real_input_real_shape = {N, c, cur_h, W_in};
      dim4 input_offset = {0, idx_c, idx_h, 0};  // 当前需要计算的数据在 ddr 上的偏移
      auto cur_in_tensor = make_tensor<T>(real_input_block_shape, real_input_real_shape);
      dma::load(cur_in_tensor.view(real_input_real_shape), in_gtensor.sub_view(real_input_real_shape, input_offset));

      dim4 real_output_block_shape = {N, block_c, block_h, W_out};
      int output_real_block_h = min(block_h, cur_index/W_out);
      dim4 output_real_shape = {N, c, output_real_block_h, W_out};
      int idx_output_h = idx_index/W_out;
      auto output1_tensor = make_tensor<T>(real_output_block_shape, output_real_shape);
      auto output2_tensor = make_tensor<T>(real_output_block_shape, output_real_shape);
      dim4 out_offset = {0, idx_c, idx_output_h, 0};

      if constexpr (std::is_same_v<T, bf16>) {
        tiu::gather_hw(output1_tensor.view(output_real_shape), cur_in_tensor.view(real_input_real_shape), index1_uint16.view<uint16>(real_index_real_shape));
        dma::store(res_gtensor.sub_view(output_real_shape, out_offset), output1_tensor.view(output_real_shape));
      } else if constexpr (std::is_same_v<T, fp16>) {
        tiu::gather_hw(output1_tensor.view(output_real_shape), cur_in_tensor.view(real_input_real_shape), index1_uint16.view<uint16>(real_index_real_shape));
        dma::store(res_gtensor.sub_view(output_real_shape, out_offset), output1_tensor.view(output_real_shape));
      } else {
        tiu::gather_hw(output1_tensor.view(output_real_shape), cur_in_tensor.view(real_input_real_shape), index1_uint16.view<uint16>(real_index_real_shape));
        dma::store(res_gtensor.sub_view(output_real_shape, out_offset), output1_tensor.view(output_real_shape));
      }
    }
    count++;
  }
}

__KERNEL__ void interp_nearest_bf16(bf16 *ptr_output, bf16 *ptr_input, DTYPE *ptr_index, const int g_core_num, const int N, const int C, const int H_in, const int W_in, const int H_out, const int W_out, const int block_h, const int align_corners) {
            interp_<bf16>(ptr_output, ptr_input, ptr_index,  g_core_num, N,  C,  H_in,  W_in,  H_out,  W_out,  block_h, align_corners);
}

__KERNEL__ void interp_nearest_fp16(fp16 *ptr_output, fp16 *ptr_input, DTYPE *ptr_index, const int g_core_num, const int N, const int C, const int H_in, const int W_in, const int H_out, const int W_out, const int block_h, const int align_corners) {
            interp_<fp16>(ptr_output, ptr_input, ptr_index,  g_core_num, N,  C,  H_in,  W_in,  H_out,  W_out,  block_h, align_corners);
}

__KERNEL__ void interp_nearest_fp32(fp32 *ptr_output, fp32 *ptr_input, DTYPE *ptr_index, const int g_core_num, const int N, const int C, const int H_in, const int W_in, const int H_out, const int W_out, const int block_h, const int align_corners) {
            interp_<fp32>(ptr_output, ptr_input, ptr_index,  g_core_num, N,  C,  H_in,  W_in,  H_out,  W_out,  block_h, align_corners);
}
