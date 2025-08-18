#include "ppl.h"
#include "ppl_wrapper_func.h"
using namespace ppl;



template <typename T>
void selective_scan(T *ptr_res, T *ptr_C, T *ptr_deltaA,
                                          T *ptr_deltaB_u, T *ptr_u, T *ptr_D, int Batch, int KC_dim, int L) {



  const int block_c = 64;
  const int block_l = 49;
  const int block_batch = 32;

  dim4 shape_res = {L, KC_dim*2, 1, Batch};
  dim4 shape_C = {L, KC_dim*2, 1, Batch};
  dim4 shape_deltaA = {1, KC_dim*2, L, Batch};
  dim4 shape_deltaB_u = {1, KC_dim*2, L, Batch};
  dim4 shape_us = {L, KC_dim*2, 1, Batch};
  dim4 shape_Ds = {1, KC_dim*2, 1, 1};

  auto res_gtensor = gtensor<T>(shape_res, GLOBAL, ptr_res);

  auto C_gtensor = gtensor<T>(shape_C, GLOBAL, ptr_C);
  auto deltaA_gtensor = gtensor<T>(shape_deltaA, GLOBAL, ptr_deltaA);
  auto deltaB_u_gtensor = gtensor<T>(shape_deltaB_u, GLOBAL, ptr_deltaB_u);
  auto u_gtensor = gtensor<T>(shape_us, GLOBAL, ptr_u);
  auto D_gtensor = gtensor<T>(shape_Ds, GLOBAL, ptr_D);

  dim4 shape_delta_block = {1, block_c, block_l, block_batch};
  dim4 shape_C_block = {block_l, block_c, 1, block_batch};
  dim4 shape_slice = {1, block_c, 1, block_batch};
  dim4 shape_D_slice = {1, block_c, 1, 1};

  // auto block_deltaA_up = tensor<T>(shape_delta_block);
  // auto block_deltaB_u_up = tensor<T>(shape_delta_block);
  // auto block_C_up = tensor<T>(shape_C_block);
  // auto block_deltaA_down = tensor<T>(shape_delta_block);
  // auto block_deltaB_u_down = tensor<T>(shape_delta_block);
  // auto block_C_down = tensor<T>(shape_C_block);


  // auto x_up_slice = tensor<T>(shape_slice);
  // auto x_down_slice = tensor<T>(shape_slice);
  // auto x_up_buffer = tensor<T>(shape_slice);
  // auto x_down_buffer = tensor<T>(shape_slice);
  // auto y_up_tensor = tensor<T>(shape_slice);
  // auto y_down_tensor = tensor<T>(shape_slice);

  auto D_up_slice = tensor<T>(shape_D_slice);
  auto D_down_slice = tensor<T>(shape_D_slice);

  for (int index_w = 0; index_w < Batch; index_w += block_batch) {

    int real_batch = min(block_batch, Batch - index_w);
    dim4 real_shape_delta_block = {1, block_c, block_l, real_batch};
    dim4 real_shape_C_block = {block_l, block_c, 1, real_batch};

        auto block_deltaA_up = make_tensor<T>(shape_delta_block, real_shape_delta_block);
        auto block_deltaB_u_up = make_tensor<T>(shape_delta_block, real_shape_delta_block);
        auto block_C_up = make_tensor<T>(shape_C_block, real_shape_C_block);
        auto block_u_up = make_tensor<T>(shape_C_block, real_shape_C_block);
        auto block_C_down = make_tensor<T>(shape_C_block, real_shape_C_block);
        auto block_deltaA_down = make_tensor<T>(shape_delta_block, real_shape_delta_block);
        auto block_deltaB_u_down = make_tensor<T>(shape_delta_block, real_shape_delta_block);
        auto block_u_down = make_tensor<T>(shape_C_block, real_shape_C_block);

    dim4 real_shape_slice = {1, block_c, 1, real_batch};
        auto x_up_slice = make_tensor<T>(shape_slice, real_shape_slice);
        auto x_down_slice = make_tensor<T>(shape_slice, real_shape_slice);
        auto x_up_buffer = make_tensor<T>(shape_slice, real_shape_slice);
        auto x_down_buffer = make_tensor<T>(shape_slice, real_shape_slice);
        auto y_up_tensor = make_tensor<T>(shape_slice, real_shape_slice);
        auto y_down_tensor = make_tensor<T>(shape_slice, real_shape_slice);


    for (int index_c = 0; index_c < KC_dim; index_c += block_c) {
      dim4 D_up_offset = {0, index_c, 0, 0};
      dim4 D_down_offset = {0, KC_dim + index_c, 0, 0};

      dma::load(D_up_slice,
                    D_gtensor.sub_view(shape_D_slice, D_up_offset));
      dma::load(D_down_slice,
                    D_gtensor.sub_view(shape_D_slice, D_down_offset));

      for (int index_l = 0; index_l < L; index_l += block_l) {


        if (index_l > 0) {
          tiu::fadd(x_up_slice, x_up_buffer, 0.0);
          tiu::fadd(x_down_slice, x_down_buffer, 0.0);
        }

        int L_block_offset_down = L - index_l - block_l;

        dim4 delta_up_offset = {0, index_c, index_l, index_w};
        dim4 delta_down_offset = {0, index_c + KC_dim, L_block_offset_down, index_w};

        dim4 c_up_offset = {index_l, index_c, 0, index_w};
        dim4 c_down_offset = {L_block_offset_down, index_c + KC_dim, 0, index_w};



        dma::load(block_deltaA_up,
                    deltaA_gtensor.sub_view(real_shape_delta_block, delta_up_offset));
        dma::load(block_deltaB_u_up,
                    deltaB_u_gtensor.sub_view(real_shape_delta_block, delta_up_offset));
        dma::load(block_C_up,
                    C_gtensor.sub_view(real_shape_C_block, c_up_offset));
        dma::load(block_u_up,
                    u_gtensor.sub_view(real_shape_C_block, c_up_offset));

        dma::load(block_deltaA_down,
                    deltaA_gtensor.sub_view(real_shape_delta_block, delta_down_offset));
        dma::load(block_deltaB_u_down,
                    deltaB_u_gtensor.sub_view(real_shape_delta_block, delta_down_offset));
        dma::load(block_C_down,
                    C_gtensor.sub_view(real_shape_C_block, c_down_offset));
        dma::load(block_u_down,
                    u_gtensor.sub_view(real_shape_C_block, c_down_offset));


        if (index_l > 0) {
          dma::move(x_up_slice, x_up_buffer);
          dma::move(x_down_slice, x_down_buffer);
        }
        else{
          tiu::zero(x_up_slice);
          tiu::zero(x_down_slice);
        }


        for(int _l = 0 ; _l < block_l ; _l++) {


          int L_slice_offset_down = block_l - _l -1;
          dim4 delta_up_slice_offset = {0, 0, _l, 0};
          dim4 c_up_slice_offset = {_l, 0, 0, 0};
          dim4 res_up_offset = {index_l + _l,index_c, 0, 0};

          dim4 delta_down_slice_offset = {0, 0, L_slice_offset_down, 0};
          dim4 c_down_slice_offset = {L_slice_offset_down, 0, 0, 0};
          dim4 res_down_offset = {L - index_l - _l - 1,index_c +KC_dim , 0, 0};

          // auto deltaA_up_block_slice = make_tensor<T>(shape_slice, real_shape_slice);
          // auto deltaBu_up_block_slice = make_tensor<T>(shape_slice, real_shape_slice);
          // auto C_block_up_slice = make_tensor<T>(shape_slice, real_shape_slice);
          // auto deltaA_down_block_slice = make_tensor<T>(shape_slice, real_shape_slice);
          // auto deltaBu_down_block_slice = make_tensor<T>(shape_slice, real_shape_slice);
          // auto C_block_down_slice = make_tensor<T>(shape_slice, real_shape_slice);

          // auto deltaA_up_block_slice = block_deltaA_up.sub_view(real_shape_slice, delta_up_slice_offset);
          // auto deltaBu_up_block_slice = block_deltaB_u_up.sub_view(real_shape_slice, delta_up_slice_offset);
          // auto C_block_up_slice = block_C_up.sub_view(real_shape_slice, c_up_slice_offset);

          // auto deltaA_down_block_slice = block_deltaA_down.sub_view(real_shape_slice, delta_down_slice_offset);
          // auto deltaBu_down_block_slice = block_deltaB_u_down.sub_view(real_shape_slice, delta_down_slice_offset);
          // auto C_block_down_slice = block_C_down.sub_view(real_shape_slice, c_down_slice_offset);


            dim4 slice_stride;
            get_stride<T>(&slice_stride, &real_shape_slice, TPU_ALIGN);
            auto deltaA_up_block_slice_tmp =
                block_deltaA_up.sub_view(real_shape_slice, delta_up_slice_offset);
            auto deltaA_up_block_slice =
                deltaA_up_block_slice_tmp.view(real_shape_slice, slice_stride);
            auto deltaBu_up_block_slice_tmp =
                block_deltaB_u_up.sub_view(real_shape_slice, delta_up_slice_offset);
            auto deltaBu_up_block_slice =
                deltaBu_up_block_slice_tmp.view(real_shape_slice, slice_stride);
            auto C_block_up_slice_tmp =
                block_C_up.sub_view(real_shape_slice, c_up_slice_offset);
            auto C_block_up_slice =
                C_block_up_slice_tmp.view(real_shape_slice, slice_stride);

            auto u_block_up_slice_tmp =
                block_u_up.sub_view(real_shape_slice, c_up_slice_offset);
            auto u_block_up_slice =
                u_block_up_slice_tmp.view(real_shape_slice, slice_stride);


            auto deltaA_down_block_slice_tmp =
                block_deltaA_down.sub_view(real_shape_slice, delta_down_slice_offset);
            auto deltaA_down_block_slice =
                deltaA_down_block_slice_tmp.view(real_shape_slice, slice_stride);
            auto deltaBu_down_block_slice_tmp =
                block_deltaB_u_down.sub_view(real_shape_slice, delta_down_slice_offset);
            auto deltaBu_down_block_slice =
                deltaBu_down_block_slice_tmp.view(real_shape_slice, slice_stride);
            auto C_block_down_slice_tmp =
                block_C_down.sub_view(real_shape_slice, c_down_slice_offset);
            auto C_block_down_slice =
                C_block_down_slice_tmp.view(real_shape_slice, slice_stride);
            auto u_block_down_slice_tmp =
                block_u_down.sub_view(real_shape_slice, c_down_slice_offset);
            auto u_block_down_slice =
                u_block_down_slice_tmp.view(real_shape_slice, slice_stride);



          tiu::fmul(x_up_slice, x_up_slice, deltaA_up_block_slice);
          tiu::fadd(x_up_slice, x_up_slice, deltaBu_up_block_slice);
          tiu::fmul(y_up_tensor, x_up_slice, C_block_up_slice);
          tiu::fmul(u_block_up_slice, u_block_up_slice, D_up_slice);
          tiu::fadd(y_up_tensor, y_up_tensor, u_block_up_slice);

          tiu::fmul(x_down_slice, x_down_slice, deltaA_down_block_slice);
          tiu::fadd(x_down_slice, x_down_slice, deltaBu_down_block_slice);
          tiu::fmul(y_down_tensor, x_down_slice, C_block_down_slice);
          tiu::fmul(u_block_down_slice, u_block_down_slice, D_down_slice);
          tiu::fadd(y_down_tensor, y_down_tensor, u_block_down_slice);


          if (_l == block_l -1){
            dma::move(x_up_buffer, x_up_slice);
            dma::move(x_down_buffer, x_down_slice);
          }

          dma::store(res_gtensor.sub_view(real_shape_slice, res_up_offset), y_up_tensor);
          dma::store(res_gtensor.sub_view(real_shape_slice, res_down_offset), y_down_tensor);
        }
      }
      }
    }
  }




__KERNEL__ void selective_scan_f32(float *ptr_res, float *ptr_C, float *ptr_deltaA,
                                          float *ptr_deltaB_u, float *ptr_u, float *ptr_D, int Batch, int KC_dim, int L) {
    selective_scan(ptr_res, ptr_C, ptr_deltaA, ptr_deltaB_u, ptr_u, ptr_D, Batch, KC_dim, L);
}
__KERNEL__ void selective_scan_fp16(fp16 *ptr_res, fp16 *ptr_C, fp16 *ptr_deltaA,
                                          fp16 *ptr_deltaB_u, fp16 *ptr_u, fp16 *ptr_D, int Batch, int KC_dim, int L) {
    selective_scan(ptr_res, ptr_C, ptr_deltaA, ptr_deltaB_u, ptr_u, ptr_D, Batch, KC_dim, L);
}
__KERNEL__ void selective_scan_bf16(bf16 *ptr_res, bf16 *ptr_C, bf16 *ptr_deltaA,
                                          bf16 *ptr_deltaB_u, bf16 *ptr_u, bf16 *ptr_D, int Batch, int KC_dim, int L) {
    selective_scan(ptr_res, ptr_C, ptr_deltaA, ptr_deltaB_u, ptr_u, ptr_D, Batch, KC_dim, L);
}

__TEST__ void test_selective_scan() {

  int Batch = 16;
  int KC_dim = 1536;
  int L = 3136;


  dim4 shape_res = {L, KC_dim*2, 1, Batch};
  dim4 shape_C = {L, KC_dim*2, 1, Batch};
  dim4 shape_deltaA = {1, KC_dim*2, L, Batch};
  dim4 shape_deltaB_u = {1, KC_dim*2, L, Batch};
  dim4 shape_us = {L, KC_dim*2, 1, Batch};
  dim4 shape_Ds = {1, KC_dim*2, 1, 1};

  auto ptr_res = malloc<fp16>(&shape_res);
  auto ptr_C = rand<fp16>(&shape_C, -1.0, 1.5);
  auto ptr_deltaA = rand<fp16>(&shape_deltaA, -1.0, 1.5);
  auto ptr_deltaB_u = rand<fp16>(&shape_deltaB_u , -1.0, 1.5);

  auto ptr_u = rand<fp16>(&shape_us , -1.0, 1.5);
  auto ptr_D = rand<fp16>(&shape_Ds , -1.0, 1.5);




  selective_scan_fp16(ptr_res, ptr_C, ptr_deltaA,
                                ptr_deltaB_u ,ptr_u , ptr_D,
                                Batch, KC_dim, L);
}

