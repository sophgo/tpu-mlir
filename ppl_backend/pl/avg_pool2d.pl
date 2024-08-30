#include "ppl.h"

using namespace ppl;
__KERNEL__ void avg_pool_2d_bf16(bf16 *ptr_rst, bf16 *ptr_inp, int N, int C,
                                 const int kh, const int kw, const int stride_h,
                                 const int stride_w, const int pad_h_t, const int pad_h_b,
                                 const int pad_w_l, const int pad_w_r, const int H, const int W,
                                 const int OH, const int OW,
                                 const int block_c, const int block_oh) {

  const int dilation_h = 1;
  const int dilation_w = 1;

  int effective_kh = kh + (kh - 1) * (dilation_h - 1);
  int effective_kw = kw + (kw - 1) * (dilation_w - 1);
  int new_C = N * C;
  dim4 inp_shape = {1, new_C, H, W};
  dim4 rst_shape = {1, new_C, OH, OW};

  dim2 kernel = {kh, kw};

  dim2 stride = {stride_h, stride_w};
  dim2 dilation = {dilation_h, dilation_w};

  auto inp_gtensor = gtensor<bf16>(inp_shape, GLOBAL, ptr_inp);
  auto out_gtensor = gtensor<bf16>(rst_shape, GLOBAL, ptr_rst);

  int block_ih = (block_oh - 1) * stride_h + effective_kh;
  dim4 inp_local_mem_shape = {1, block_c, block_ih, W};
  dim4 out_local_mem_shape = {1, block_c, block_oh, OW};

  auto inp = tensor<bf16>(inp_local_mem_shape);
  auto res = tensor<bf16>(out_local_mem_shape);


  for (int cidx = 0; cidx < new_C; cidx += block_c) {
    int real_c = min(block_c, new_C - cidx);
    for (int ohidx = 0; ohidx < OH; ohidx += block_oh) {
      int real_oh = min(block_oh, OH - ohidx);
      ppl::enable_pipeline();

      int start_h = ohidx * stride_h - pad_h_t;
      int end_h = (ohidx + block_oh - 1) * stride_h - pad_h_t + effective_kh;

      start_h = max(start_h, 0);
      end_h = min(end_h, H);
      int load_h = end_h - start_h;

      dim4 inp_local_real_shape = {1, real_c, load_h, W};
      dim4 out_local_real_shape = {1, real_c, real_oh, OW};

      dim4 inp_goffset = {0, cidx, start_h, 0};
      dim4 out_goffset = {0, cidx, ohidx, 0};

      int sub_pad_top = 0;
      int sub_pad_bottom = 0;
      if (ohidx == 0) {
        sub_pad_top = pad_h_t;
      }
      if (ohidx >= OH - block_oh) {
        sub_pad_bottom = pad_h_b;
      }
      padding_t pad = {sub_pad_top, sub_pad_bottom, pad_w_l, pad_w_r};
      auto inp_sub = inp.view(inp_local_real_shape);
      auto res_sub = res.view(out_local_real_shape);

      dma::load(inp_sub, inp_gtensor.sub_view(inp_local_real_shape, inp_goffset));
      tiu::fpool_avg(res_sub, inp_sub, &kernel, &pad, &stride, &dilation, 1 / (float)(kh * kw));
      dma::store(out_gtensor.sub_view(out_local_real_shape, out_goffset), res_sub);
    }
  }
}


// __TEST__ void avg_pool2d_main() {
//   int N = 64;
//   int C = 8;
//   int H = 40;
//   int W = 30;

//   int OH = 19;
//   int OW = 14;

//   int kh = 3, kw = 3;
//   int stride_h = 2, stride_w = 2;
//   int pad_h = 0, pad_w = 0;

//   int block_c = 64;
//   int block_oh = 8;

//   dim4 res_shape = {N, C, OH, OW};
//   dim4 inp_shape = {N, C, H, W};
//   bf16 *res = rand<bf16>(&res_shape);
//   bf16 *inp = rand<bf16>(&inp_shape, -1.f, 1.f);

//   avg_pool_2d_bf16(res, inp, N, C, kh, kw, stride_h, stride_w, pad_h, pad_w, H,
//                    W, OH, OW, block_c, block_oh);
// }

