#include "ppl.h"
using namespace ppl;

__KERNEL__ void avg_pool_2d_local(int32 rst_addr, int32 inp_addr, int kh,
                                  int kw, int stride_h, int stride_w,
                                  int pad_h_t, int pad_h_b, int pad_w_l,
                                  int pad_w_r, const int block_n,
                                  const int block_c, const int block_oh,
                                  const int block_iw, const int block_ow,
                                  const int block_ih) {
  const int dilation_h = 1;
  const int dilation_w = 1;

  int effective_kh = kh + (kh - 1) * (dilation_h - 1);
  int effective_kw = kw + (kw - 1) * (dilation_w - 1);

  dim2 kernel = {kh, kw};

  dim2 stride = {stride_h, stride_w};
  dim2 dilation = {dilation_h, dilation_w};

  dim4 inp_local_mem_shape = {block_n, block_c, block_ih, block_iw};
  dim4 out_local_mem_shape = {block_n, block_c, block_oh, block_ow};

  auto inp = tensor<bf16>(inp_local_mem_shape, TPU_ALIGN, inp_addr);
  auto res = tensor<bf16>(out_local_mem_shape, TPU_ALIGN, rst_addr);

  padding_t pad = {pad_h_t, pad_h_b, pad_w_l, pad_w_r};
  tiu::fpool_avg(res, inp, &kernel, &pad, &stride, &dilation,
                 1 / (float)(kh * kw));
}