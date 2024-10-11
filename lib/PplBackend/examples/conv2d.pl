#include "ppl.h"

using namespace ppl;

__KERNEL__ void fconv2d(fp16 *ptr_res, fp16 *ptr_in, fp16 *ptr_w, fp32 *ptr_b,
                        int N, int IC, int H, int W, int OC, bool has_bias,
                        bool has_relu, const int kh, const int kw, const int sh,
                        const int sw, const int dh, const int dw,
                        const int pad_t, const int pad_b, const int pad_l,
                        const int pad_r, const int block_n, const int block_oc,
                        const int block_oh, const int block_ic,
                        const int block_ow) {
                            // int kh = 3;
  // int kw = 3;
  // int stride_h = 1;
  // int stride_w = 1;
  // int pad_t = 1;
  // int pad_b = 1;
  // int pad_l = 1;
  // int pad_r = 1;
  // oh  = (ih + pad_t + pad_b - dilation_h  * (kh - 1) - 1) / stride_h +1
  int kh_ext = dh * (kh - 1) + 1;
  int kw_ext = dw * (kw - 1) + 1;
  int ih_ext = H + pad_t + pad_b;
  int iw_ext = W + pad_l + pad_r;
  int oh = (ih_ext - kh_ext) / sh + 1; // 计算输出特征图的高度
  int ow = (iw_ext - kw_ext) / sw + 1; // 计算输出特征图的宽度

  int nic = get_nic<fp16>();
  int ws_w = nic;
  int ws_h = div_up(IC, nic) * kh * kw;

  padding_t pad = {pad_t, pad_b, pad_l, pad_r};
  dim2 stride = {sh, sw};
  dim2 dilation = {dh, dw};
  dim2 kernel = {kh, kw};

  int block_ih = (block_oh - 1) * sh + kh_ext;
  int block_iw = (block_ow - 1) * sw + kw_ext;

  dim4 global_in_shape = {N, IC, H, W};
  dim4 global_weight_shape = {1, OC, ws_h, ws_w};
  dim4 global_bias_shape = {1, OC, 1, 1};
  dim4 global_out_shape = {N, OC, oh, ow};

  auto in_gt = gtensor<fp16>(global_in_shape, GLOBAL, ptr_in);
  auto w_gt = gtensor<fp16>(global_weight_shape, GLOBAL, ptr_w);
  auto b_gt = gtensor<fp32>(global_bias_shape, GLOBAL, ptr_b);
  auto o_gt = gtensor<fp16>(global_out_shape, GLOBAL, ptr_res);

  int oh_secs = div_up(oh, block_oh);
  int ow_secs = div_up(ow, block_ow);
  int ic_secs = div_up(IC, block_ic);
  int oc_secs = div_up(OC, block_oc);
  int n_secs = div_up(N, block_n);

  // split order n-oc-oh-ic-ow
  // calc order ic-oh-ow-n-oc

  int oc_num = ic_secs * oh_secs * ow_secs * n_secs;
  int n_num = ic_secs * oh_secs * ow_secs;
  int ow_num = ic_secs * oh_secs;
  int oh_num = ic_secs;

  dim4 bias_block_shape = {1, block_oc, 1, 1};
  dim4 weight_block_shape = {1, block_oc, block_ic / nic * kh * kw, nic};
  dim4 input_block_shape = {block_n, block_ic, block_ih, block_iw};
  dim4 output_block_shape = {block_n, block_oc, block_oh, block_ow};

  auto bias_tensor = tensor<fp32>(bias_block_shape);
  bool is_ic_split = ic_secs > 1;
  bool is_oc_split = oc_secs > 1;
  for (int count = 0; count < n_secs * ow_secs * oh_secs * ic_secs * oc_secs;
       ++count) {
    ppl::enable_pipeline();
    int remain = count;
    int oc_count = remain / oc_num;
    remain %= oc_num;
    int n_count = remain / n_num;
    remain %= n_num;
    int ow_count = remain / ow_num;
    remain %= ow_num;
    int oh_count = remain / oh_num;
    remain %= oh_num;
    int ic_count = remain;

    // calc ic/oh/oc/n start and size
    int idx_ic = ic_count * block_ic;
    int idx_oh = oh_count * block_oh;
    int idx_ow = ow_count * block_ow;
    int idx_oc = oc_count * block_oc;
    int idx_n = n_count * block_n;

    int curr_ic = min(block_ic, IC - idx_ic);
    int curr_oh = min(block_oh, oh - idx_oh);
    int curr_ow = min(block_ow, ow - idx_ow);
    int curr_oc = min(block_oc, OC - idx_oc);
    int curr_n = min(block_n, N - idx_n);

    // calc input h
    int ih_start = idx_oh * sh - pad_t;
    int ih_end = (idx_oh + curr_oh - 1) * sh + kh_ext - pad_t;
    int slice_pad_t = ih_start < 0 ? -ih_start : 0;
    int slice_pad_b = ih_end > H ? ih_end - H : 0;
    ih_start = max(0, ih_start);
    ih_end = min(H, ih_end);
    int curr_ih = ih_end - ih_start;

    // calc input w
    int iw_start = idx_ow * sw - pad_l;
    int iw_end = (idx_ow + curr_ow - 1) * sw + kw_ext - pad_l;
    int slice_pad_l = iw_start < 0 ? -iw_start : 0;
    int slice_pad_r = iw_end > W ? iw_end - W : 0;
    iw_start = max(0, iw_start);
    iw_end = min(W, iw_end);
    int curr_iw = iw_end - iw_start;

    padding_t slice_pad = {slice_pad_t, slice_pad_b, slice_pad_l, slice_pad_r};
    // print("n=%d, oc=%d, ic=%d, oh=%d, ih=%d, ow=%d, iw=%d, idx_n=%d, "
    //       "idx_oc=%d, "
    //       "idx_ic=%d, idx_oh=%d, idx_ih=%d, idx_ow=%d, idx_iw=%d\n",
    //       curr_n, curr_oc, curr_ic, curr_oh, curr_ih, curr_ow, curr_iw, idx_n,
    //       idx_oc, idx_ic, idx_oh, ih_start, idx_ow, iw_start);

    data_type_t out32 =
        (is_ic_split && ic_count != ic_secs - 1) ? DT_FP32 : DT_FP16;
    bool result_add = is_ic_split && ic_count != 0;
    bool do_bias = ic_count == 0 && has_bias;

    // load bias
    dim4 bias_real_shape = {1, curr_oc, 1, 1};
    dim4 bias_offset = {0, idx_oc, 0, 0};
    auto bias = bias_tensor.view(bias_real_shape);
    if (count % oc_num < 2 && has_bias) {
      dma::load_compact(bias, b_gt.sub_view(bias_real_shape, bias_offset));
    }
    // print("bias:%s\n", to_string(bias));
    // load weight
    dim4 weight_real_shape = {1, curr_oc, div_up(curr_ic, nic) * kh * kw, nic};
    auto weight = make_tensor<fp16>(weight_block_shape, weight_real_shape);
    // print("weight:%s\n", to_string(weight));

    dim4 weight_offset = {0, idx_oc, (idx_ic / nic) * kh * kw, 0};
    // FIX BUG
    // if (is_ic_split) {
    //   if ((ic_count == 0 || ic_count == 1)) {
    //     dma::load(weight, w_gt.sub_view(weight_real_shape, weight_offset));
    //   }
    // } else {
    //   if (count % oc_num == 0 || count % oc_num == 1) {
    //     dma::load(weight, w_gt.sub_view(weight_real_shape, weight_offset));
    //   }
    // }
    // if ((is_ic_split && (ic_count == 0 || ic_count == 1)) ||
    //     !is_ic_split && (count % oc_num == 0 || count % oc_num == 1)) {
    //       dma::load(weight, w_gt.sub_view(weight_real_shape, weight_offset));
    // }
    // if ((is_ic_split && (ic_count == 0 || ic_count == 1))) {
    //   dma::load(weight, w_gt.sub_view(weight_real_shape, weight_offset));
    // }
    // if (!is_ic_split && (count % oc_num == 0 || count % oc_num == 1)) {
    //   dma::load(weight, w_gt.sub_view(weight_real_shape, weight_offset));
    // }
    bool need_load_weight = is_ic_split || (!is_ic_split && count % oc_num < 2);
    if (need_load_weight) {
      dma::load(weight, w_gt.sub_view(weight_real_shape, weight_offset));
    }

    // load input
    dim4 input_real_shape = {curr_n, curr_ic, curr_ih, curr_iw};
    dim4 input_offset = {idx_n, idx_ic, ih_start, iw_start};
    auto input = make_tensor<fp16>(input_block_shape, input_real_shape);
    // print("input:%s\n", to_string(input));
    dma::load(input, in_gt.sub_view(input_real_shape, input_offset));

    dim4 output_real_shape = {curr_n, curr_oc, curr_oh, curr_ow};
    auto output = make_tensor<fp32>(output_block_shape, output_real_shape);
    auto out_f16 = make_tensor<fp16>(output_block_shape, output_real_shape);
    // print("output:%s\n", to_string(output));

    // print("do_bias:%d\n", do_bias);
    if (is_ic_split) {
      tiu::fconv(output, input, weight, bias, curr_oc, &kernel, &stride,
                 &dilation, &slice_pad, nullptr, false, result_add, out32,
                 do_bias, false, false);
    } else {
      tiu::fconv(out_f16, input, weight, bias, curr_oc, &kernel, &stride,
                 &dilation, &slice_pad, nullptr, false, result_add, out32,
                 do_bias, false, false);
    }
    if (is_ic_split && out32 != DT_FP32) {
      tiu::move(out_f16, output.view<fp16>());
    }
    if (has_relu) {
      tiu::fmax(out_f16, out_f16, 0);
    }

    // ifOp can't contain 2 stage,so make they condition difference
    if (out32 != DT_FP32) {
      // dma::store(o_gt.sub_view(output_real_shape, o_offset), output);
      dim4 o_offset = {idx_n, idx_oc, idx_oh, idx_ow};
      dma::store(o_gt.sub_view(output_real_shape, o_offset), out_f16);
    }
  }
}

// __TEST__ void fconv2d_main() {
//   int N = 1;
//   int IC = 32;
//   int H = 128;
//   int W = 64;
//   int OC = 128;
//   int kh = 3;
//   int kw = 3;

//   int oh = 128;
//   int ow = 64;
//   int nic = get_nic<fp16>();
//   dim4 res_shape = {N, OC, oh, ow};
//   auto res = ppl::malloc<fp16>(&res_shape);

//   dim4 in_shape = {N, IC, H, W};
//   auto in = ppl::malloc<fp16>(&in_shape);
//   ppl::rand(in, &in_shape, -1.f, 1.f);

//   dim4 w_shape = {1, OC, div_up(IC, nic) * kh * kw, nic};
//   auto w = ppl::malloc<fp16>(&w_shape);
//   ppl::rand(w, &w_shape, -1.f, 1.f);

//   dim4 b_shape = {1, OC, 1, 1};
//   auto b = ppl::malloc<fp32>(&b_shape);
//   ppl::rand(b, &b_shape, -1.f, 1.f);

//   fconv2d(res, in, w, b, N, IC, H, W, OC, kh, kw, 1, 1, 1, 1, 1, 1, 1, 1);
// }
