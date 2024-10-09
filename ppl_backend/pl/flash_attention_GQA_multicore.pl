#include "ppl.h"
#include "ppl_wrapper_func.h"
using namespace ppl;

// bm1684
// only support fp16/bf16
// #ifdef __DECODE
// const int dmax = 128;
// const int block_m = 1;
// const int block_k = 160;
// const int block_h = 16;
// #else
// const int dmax = 128;
// const int block_m = 128;
// const int block_k = 64;
// const int block_h = 16;
// #endif

// bm1688
// #define __DECODE
// #ifdef __DECODE
//   // for bm1688
//   const int dmax = 128;
//   const int block_m = 32;
//   const int block_k = 352;
//   const int block_h = 8;
// #else
//   // for bm1688
//   const int dmax = 128;
//   const int block_m = 224;
//   const int block_k = 96;
//   // const int block_k = 80; // bf16
//   const int block_h = 8;
// #endif
template <typename T>
void flash_attention_gqa(T *ptr_out, T *ptr_q, T *ptr_k, T *ptr_v, T *ptr_musk,
                         int b, int qm, int kvm, int d, int q_head, int kv_head,
                         float sqrt_d, int has_musk, const int dmax,
                         const int block_m, const int block_k,
                         const int block_h) {
  // ppl::set_core_num(1);
  int head_rep = q_head / kv_head;

  int core_num = get_core_num();
  int core_index = get_core_index();
  if (core_index >= core_num)
    return;

  int kv_head_per_core = div_up(kv_head, core_num);
  int q_head_per_core = kv_head_per_core * head_rep;

  int q_head_start = core_index * q_head_per_core;
  int q_head_end = min(q_head_start + q_head_per_core, q_head);

  assert(d <= dmax);
  int cur_d = min(d, dmax);
  int block_k_iter = max(min(block_k, kvm / 2), 1);
  int block_h_iter = block_h / head_rep * head_rep;

  dim4 q_shape = {block_h, block_m, 1, dmax};
  dim4 kv_shape = {block_h, block_k, 1, dmax};
  dim4 qk_shape = {block_h, block_m, 1, block_k};
  dim4 musk_shape = {1, block_m, 1, block_k};

  dim4 mi_shape = {block_h, block_m, 1, 1};
  dim4 li_shape = {block_h, block_m, 1, 1};
  dim4 acc_shape = {block_h, block_m, 1, dmax};

  dim4 qo_global_shape = {b, qm, q_head, cur_d};
  auto q_global_tensor = gtensor<T>(qo_global_shape, GLOBAL, ptr_q);
  int order[4] = {2, 1, 0, 3};
  auto out_global_tensor = gtensor<T>(qo_global_shape, GLOBAL, ptr_out);

  dim4 kv_global_shape = {b, kvm, kv_head, cur_d};
  auto k_global_tensor = gtensor<T>(kv_global_shape, GLOBAL, ptr_k);
  auto v_global_tensor = gtensor<T>(kv_global_shape, GLOBAL, ptr_v);

  dim4 musk_global_shape = {b, qm, 1, kvm};
  auto musk_global_tensor = gtensor<T>(musk_global_shape, GLOBAL, ptr_musk);

  for (int _b = 0; _b < b; _b += 1) {
    dim4 q_sub_shape = {1, qm, q_head, cur_d};
    dim4 q_sub_reshape = {qm, q_head, 1, cur_d};
    dim4 kv_sub_shape = {1, kvm, kv_head, cur_d};
    dim4 kv_sub_reshape = {kvm, kv_head, 1, cur_d};
    dim4 sub_offset = {_b, 0, 0, 0};
    auto q_sub_global =
        q_global_tensor.sub_view(q_sub_shape, sub_offset).view(q_sub_reshape);
    auto k_sub_global =
        k_global_tensor.sub_view(kv_sub_shape, sub_offset).view(kv_sub_reshape);
    auto v_sub_global =
        v_global_tensor.sub_view(kv_sub_shape, sub_offset).view(kv_sub_reshape);
    auto out_sub_global =
        out_global_tensor.sub_view(q_sub_shape, sub_offset).view(q_sub_reshape);
    for (int _h = q_head_start; _h < q_head_end; _h += block_h_iter) {
      int real_q_h = min(block_h_iter, q_head - _h);
      real_q_h = min(real_q_h, block_h);
      int real_kv_h = real_q_h / head_rep;
      for (int _m = 0; _m < qm; _m += block_m) {
        int real_m = min(block_m, qm - _m);
        dim4 qi_real_local_shape = {real_q_h, real_m, 1, cur_d};
        dim4 qi_real_global_shape = {real_m, real_q_h, 1, cur_d};
        dim4 qi_offset = {_m, _h, 0, 0};
        auto qi_tensor = make_tensor<T>(q_shape, qi_real_local_shape);
        dma::load_transpose_nc(
            qi_tensor, q_sub_global.sub_view(qi_real_global_shape, qi_offset));
        auto qi_tensor_scale = make_tensor<T>(q_shape, qi_real_local_shape);
        tiu::fmul(qi_tensor_scale, qi_tensor, sqrt_d);

        dim4 mi_real_shape = {real_q_h, real_m, 1, 1};
        dim4 li_real_shape = {real_q_h, real_m, 1, 1};
        dim4 acc_real_shape = {real_q_h, real_m, 1, cur_d};
        auto mi_sub_tensor = make_tensor<T>(mi_shape, mi_real_shape);
        auto li_sub_tensor = make_tensor<T>(li_shape, li_real_shape);
        auto acc_sub_tensor = make_tensor<T>(acc_shape, acc_real_shape);
        tiu::fill(mi_sub_tensor, -15000);
        tiu::zero(li_sub_tensor);
        tiu::zero(acc_sub_tensor);
        for (int _k = 0; _k < kvm; _k += block_k_iter) {
          ppl::enable_pipeline();
          int real_k = min(block_k_iter, kvm - _k);
          dim4 kvi_real_local_shape = {real_kv_h, real_k, 1, cur_d};
          dim4 kvi_real_global_shape = {real_k, real_kv_h, 1, cur_d};
          dim4 kvi_offset = {_k, _h / head_rep, 0, 0};
          dim4 qk_real_shape = {real_q_h, real_m, 1, real_k};
          dim4 musk_real_shape = {1, real_m, 1, real_k};
          dim4 musk_offset = {_b, _m, 0, _k};

          auto ki_tensor = make_tensor<T>(kv_shape, kvi_real_local_shape);
          auto vi_tensor = make_tensor<T>(kv_shape, kvi_real_local_shape);
          dma::load_transpose_nc(
              ki_tensor,
              k_sub_global.sub_view(kvi_real_global_shape, kvi_offset));
          dma::load_transpose_nc(
              vi_tensor,
              v_sub_global.sub_view(kvi_real_global_shape, kvi_offset));
          auto musk_tensor = make_tensor<T>(musk_shape, musk_real_shape);
          if (has_musk) {
            dma::load(musk_tensor, musk_global_tensor.sub_view(musk_real_shape,
                                                               musk_offset));
          }
          dim4 qk_batch_shape = {1, real_m, 1, real_k};
          dim4 qi_batch_shape = {1, real_m, 1, cur_d};
          dim4 ki_batch_shape = {1, real_k, 1, cur_d};
          auto qk_sub_tensor = make_tensor<T>(qk_shape, qk_real_shape);
          for (int i = 0; i < real_q_h; i++) {
            dim4 batch_q_offset = {i, 0, 0, 0};
            dim4 batch_k_offset = {i / head_rep, 0, 0, 0};
            auto qk_tensor_batch =
                qk_sub_tensor.sub_view(qk_batch_shape, batch_q_offset);
            auto qi_tensor_batch =
                qi_tensor_scale.sub_view(qi_batch_shape, batch_q_offset);
            auto ki_tensor_batch =
                ki_tensor.sub_view(ki_batch_shape, batch_k_offset);

            tiu::fmm2(qk_tensor_batch, qi_tensor_batch, ki_tensor_batch, false,
                      true, false);
          }
          if (has_musk) {
            // broadcast add (bc dim n)
            tiu::fadd(qk_sub_tensor, qk_sub_tensor, musk_tensor);
          }

          auto max_out = make_tensor<T>(mi_shape, mi_real_shape);
          auto mi_new_tensor = make_tensor<T>(mi_shape, mi_real_shape);
          quick_pooling(max_out, qk_sub_tensor, &qk_shape, &qk_real_shape,
                        -15000, 0);

          tiu::fmax(mi_new_tensor, mi_sub_tensor, max_out);

          auto alpha = make_tensor<T>(mi_shape, mi_real_shape);
          auto sub_out = make_tensor<T>(mi_shape, mi_real_shape);
          auto li_tmp_tensor = make_tensor<T>(li_shape, li_real_shape);
          tiu::fsub(sub_out, mi_sub_tensor, mi_new_tensor);
          tiu::move(mi_sub_tensor, mi_new_tensor);
          exp_no_overflow(alpha, sub_out, &mi_shape, &mi_real_shape);
          // broadcast mul (w)
          tiu::fmul(acc_sub_tensor, acc_sub_tensor, alpha);

          tiu::fmul(li_tmp_tensor, li_sub_tensor, alpha);

          auto sub_out1 = make_tensor<T>(qk_shape, qk_real_shape);
          // broadcast sub (w)
          tiu::fsub(sub_out1, qk_sub_tensor, mi_new_tensor);

          auto p_T = make_tensor<T>(qk_shape, qk_real_shape);
          auto sum = make_tensor<T>(li_shape, li_real_shape);
          exp_no_overflow(p_T, sub_out1, &qk_shape, &qk_real_shape);

          quick_pooling(sum, p_T, &qk_shape, &qk_real_shape, 0, 1);
          tiu::fadd(li_sub_tensor, li_tmp_tensor, sum);

          tensor<T> pv_tensor = make_tensor<T>(acc_shape, acc_real_shape);
          dim4 pv_batch_shape = {1, real_m, 1, cur_d};
          dim4 p_batch_shape = {1, real_m, 1, real_k};
          dim4 vi_batch_shape = {1, real_k, 1, cur_d};
          for (int i = 0; i < real_q_h; i++) {
            dim4 batch_p_offset = {i, 0, 0, 0};
            dim4 batch_v_offset = {i / head_rep, 0, 0, 0};
            auto pv_tensor_batch =
                pv_tensor.sub_view(pv_batch_shape, batch_p_offset);
            auto p_tensor_batch = p_T.sub_view(p_batch_shape, batch_p_offset);
            auto vi_tensor_batch =
                vi_tensor.sub_view(vi_batch_shape, batch_v_offset);

            tiu::fmm2(pv_tensor_batch, p_tensor_batch, vi_tensor_batch);
          }
          tiu::fadd(acc_sub_tensor, acc_sub_tensor, pv_tensor);
        }

        auto li_tensor_fp32 = make_tensor<fp32>(li_shape, li_real_shape);
        auto div_li_tensor = make_tensor<fp32>(li_shape, li_real_shape);
        auto div_li_tensor_T = make_tensor<T>(li_shape, li_real_shape);
        auto qkvo_tensor = make_tensor<T>(acc_shape, acc_real_shape);
        tiu::cast(li_tensor_fp32, li_sub_tensor);
        tiu::fdiv(div_li_tensor, 1.0f, li_tensor_fp32, 3);
        tiu::cast(div_li_tensor_T, div_li_tensor);
        // broadcast mul (w)
        tiu::fmul(qkvo_tensor, acc_sub_tensor, div_li_tensor_T);

        dim4 qkv_offset = {_m, _h, 0, 0};
        dma::store_transpose_nc(
            out_sub_global.sub_view(qi_real_global_shape, qkv_offset),
            qkvo_tensor);
      }
    }
  }
}

__KERNEL__ void flash_attention_gqa_f16(fp16 *ptr_out, fp16 *ptr_q, fp16 *ptr_k,
                                        fp16 *ptr_v, fp16 *ptr_musk, int b,
                                        int qm, int kvm, int d, int q_head,
                                        int kv_head, float sqrt_d, int has_musk,
                                        const int dmax, const int block_m,
                                        const int block_k, const int block_h) {
  flash_attention_gqa<fp16>(ptr_out, ptr_q, ptr_k, ptr_v, ptr_musk, b, qm, kvm,
                            d, q_head, kv_head, sqrt_d, has_musk, dmax, block_m,
                            block_k, block_h);
}

__KERNEL__ void flash_attention_gqa_bf16(
    bf16 *ptr_out, bf16 *ptr_q, bf16 *ptr_k, bf16 *ptr_v, bf16 *ptr_musk, int b,
    int qm, int kvm, int d, int q_head, int kv_head, float sqrt_d, int has_musk,
    const int dmax, const int block_m, const int block_k, const int block_h) {
  flash_attention_gqa<bf16>(ptr_out, ptr_q, ptr_k, ptr_v, ptr_musk, b, qm, kvm,
                            d, q_head, kv_head, sqrt_d, has_musk, dmax, block_m,
                            block_k, block_h);
}
