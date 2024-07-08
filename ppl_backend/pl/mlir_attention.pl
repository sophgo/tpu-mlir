#include "ppl.h"
#include "ppl_wrapper_func.h"
using namespace ppl;

template <typename d>
void pooling2(tensor<d> &out_tensor, tensor<d> &in_tensor,
              tensor<d> &tmp_tensor, int n, int c, int w, int mode) {
  int eu_num = get_eu_num<d>();
  int opti_w = eu_num;
  int align_w = align(w, eu_num);
  int slice = align_w / eu_num;
  int h = 1;
  if (align_w > w) {
    dim4 in_shape = {n, c, 1, w};
    dim4 fill_offset = {0, 0, 0, w};
    dim4 fill_shape = {n, c, 1, align_w - w};
    auto fill_tensor =
        in_tensor.view(in_shape).sub_view(fill_shape, fill_offset);
    tiu::fill(fill_tensor, mode == 0 ? -15000 : 0);
    // only support h == 1
    // dim4 mv_out_shape = {n, c, 1, w + eu_num - align_w};
    // dim4 mv_in_shape = {n, c, 1, eu_num};
    // dim4 in_shape = {n, c, 1, w};
    // dim4 mv_offset = {0, 0, 0, align_w - eu_num};
    // auto tensor_mv_out = in_tensor.view(in_shape).sub_view(mv_out_shape,
    // mv_offset); auto tensor_mv_in =
    // in_tensor.view(in_shape).sub_view(mv_in_shape, mv_offset);
    // tiu::zero(tmp_tensor.view(mv_in_shape));
    // tiu::move(tmp_tensor.view(mv_out_shape), tensor_mv_out);
    // tiu::move(tensor_mv_in, tmp_tensor.view(mv_in_shape));
  }
  dim4 in_reduce_h = {n * h, c, slice, opti_w};
  dim4 out_reduce_h = {n * h, c, 1, opti_w};
  dim4 in_reduce_w = {n, c, h, opti_w};
  dim4 out_reduce_w = {n, c, h, 1};

  dim2 kernel = {slice, 1};
  padding_t pad = {0, 0, 0, 0};
  dim2 stride = {1, 1};
  dim2 dilation = {1, 1};
  if (mode == 0) {
    tiu::fpool_max(tmp_tensor.view(out_reduce_h), in_tensor.view(in_reduce_h),
                   &kernel, &pad, &stride, &dilation);
  } else {
    tiu::fpool_avg(tmp_tensor.view(out_reduce_h), in_tensor.view(in_reduce_h),
                   &kernel, &pad, &stride, &dilation, 1.0);
  }
  kernel.h = 1;
  kernel.w = opti_w;
  if (mode == 0) {
    tiu::fpool_max(out_tensor.view(out_reduce_w), tmp_tensor.view(in_reduce_w),
                   &kernel, &pad, &stride, &dilation);

  } else {
    tiu::fpool_avg(out_tensor.view(out_reduce_w), tmp_tensor.view(in_reduce_w),
                   &kernel, &pad, &stride, &dilation, 1.0);
  }
}

// only support fp16/bf16
template <typename T>
void mlir_attention(T *ptr_out, T *ptr_q, T *ptr_k, T *ptr_v, T *ptr_musk,
                    int b, int qm, int kvm, int d, int q_head, int kv_head,
                    float sqrt_d, int has_musk, const int dmax,
                    const int block_m, const int block_k, const int block_h) {
  // assert(d <= dmax);
  int head_rep = q_head / kv_head;
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

  int order[4] = {2, 1, 0, 3};
  dim4 qo_global_shape = {b, qm, q_head, cur_d};
  auto q_global_tensor =
      make_gtensor_permute<T>(qo_global_shape, GLOBAL, ptr_q, order);
  auto out_global_tensor =
      make_gtensor_permute<T>(qo_global_shape, GLOBAL, ptr_out, order);

  dim4 kv_global_shape = {b, kvm, kv_head, cur_d};
  auto k_global_tensor =
      make_gtensor_permute<T>(kv_global_shape, GLOBAL, ptr_k, order);
  auto v_global_tensor =
      make_gtensor_permute<T>(kv_global_shape, GLOBAL, ptr_v, order);

  dim4 musk_global_shape = {b, qm, 1, kvm};
  auto musk_global_tensor = gtensor<T>(musk_global_shape, GLOBAL, ptr_musk);
  for (int _b = 0; _b < b; _b += 1) {
    for (int _h = 0; _h < q_head; _h += block_h_iter) {
      int real_q_h = min(block_h_iter, q_head - _h);
      real_q_h = min(real_q_h, block_h);
      int real_kv_h = real_q_h / head_rep;
      for (int _m = 0; _m < qm; _m += block_m) {
        int real_m = min(block_m, qm - _m);
        dim4 qi_real_local_shape = {real_q_h, real_m, 1, cur_d};
        dim4 qi_offset = {_h, _m, _b, 0};
        auto qi_tensor = make_tensor<T>(q_shape, qi_real_local_shape);
        dma::load(qi_tensor,
                  q_global_tensor.sub_view(qi_real_local_shape, qi_offset));
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
          dim4 kvi_offset = {_h / head_rep, _k, _b, 0};
          dim4 qk_real_shape = {real_q_h, real_m, 1, real_k};
          dim4 musk_real_shape = {1, real_m, 1, real_k};
          dim4 musk_offset = {_b, _m, 0, _k};

          auto ki_tensor = make_tensor<T>(kv_shape, kvi_real_local_shape);
          auto vi_tensor = make_tensor<T>(kv_shape, kvi_real_local_shape);
          dma::load(ki_tensor,
                    k_global_tensor.sub_view(kvi_real_local_shape, kvi_offset));
          dma::load(vi_tensor,
                    v_global_tensor.sub_view(kvi_real_local_shape, kvi_offset));
          tensor<T> musk_tensor;
          if (has_musk) {
            dma::load(musk_tensor, musk_global_tensor.sub_view(musk_real_shape,
                                                               musk_offset));
          }
          dim4 qk_batch_shape = {1, real_m, 1, real_k};
          dim4 qi_batch_shape = {1, real_m, 1, cur_d};
          dim4 ki_batch_shape = {1, real_k, 1, cur_d};
          auto qk_sub_tensor = make_tensor<T>(qk_shape, qk_real_shape);
          auto ki_sub_tensor = ki_tensor.view(kvi_real_local_shape);
          for (int i = 0; i < real_q_h; i++) {
            dim4 batch_q_offset = {i, 0, 0, 0};
            dim4 batch_k_offset = {i / head_rep, 0, 0, 0};
            auto qk_tensor_batch =
                qk_sub_tensor.sub_view(qk_batch_shape, batch_q_offset);
            auto qi_tensor_batch =
                qi_tensor_scale.sub_view(qi_batch_shape, batch_q_offset);
            auto ki_tensor_batch =
                ki_sub_tensor.sub_view(ki_batch_shape, batch_k_offset);

            tiu::fmm2(qk_tensor_batch, qi_tensor_batch, ki_tensor_batch, false,
                      true, false, false, false);
          }
          if (has_musk) {
            // broadcast add (bc dim n)
            tiu::fadd(qk_sub_tensor, qk_sub_tensor, musk_tensor);
          }

          tensor<T> max_out, mi_new_tensor;
          int opti_w = get_eu_num<T>();
          dim4 tmp_shape = {block_h, block_m, 1, opti_w};
          dim4 tmp_real_shape = {real_q_h, block_m, 1, opti_w};
          auto tmp_tensor = make_tensor<T>(tmp_shape, tmp_real_shape);
          pooling2(max_out, qk_sub_tensor, tmp_tensor, real_q_h, real_m, real_k,
                   0);

          tiu::fmax(mi_new_tensor, mi_sub_tensor, max_out);

          tensor<T> alpha, sub_out, li_tmp_tensor;
          tiu::fsub(sub_out, mi_sub_tensor, mi_new_tensor);
          tiu::move(mi_sub_tensor, mi_new_tensor);
          exp_no_overflow(alpha, sub_out, &mi_shape, &mi_real_shape);
          // broadcast mul (w)
          tiu::fmul(acc_sub_tensor, acc_sub_tensor, alpha);

          tiu::fmul(li_tmp_tensor, li_sub_tensor, alpha);

          tensor<T> sub_out1;
          // broadcast sub (w)
          tiu::fsub(sub_out1, qk_sub_tensor, mi_new_tensor);

          tensor<T> p_T, sum;
          exp_no_overflow(p_T, sub_out1, &qk_shape, &qk_real_shape);

          auto tmp_tensor2 = make_tensor<T>(tmp_shape, tmp_real_shape);
          pooling2(sum, p_T, tmp_tensor2, real_q_h, real_m, real_k, 1);
          tiu::fadd(li_sub_tensor, li_tmp_tensor, sum);

          tensor<T> pv_tensor = make_tensor<T>(acc_shape, acc_real_shape);
          dim4 pv_batch_shape = {1, real_m, 1, cur_d};
          dim4 p_batch_shape = {1, real_m, 1, real_k};
          dim4 vi_batch_shape = {1, real_k, 1, cur_d};
          auto vi_sub_tensor = vi_tensor.view(kvi_real_local_shape);
          for (int i = 0; i < real_q_h; i++) {
            dim4 batch_p_offset = {i, 0, 0, 0};
            dim4 batch_v_offset = {i / head_rep, 0, 0, 0};
            auto pv_tensor_batch =
                pv_tensor.sub_view(pv_batch_shape, batch_p_offset);
            auto p_tensor_batch =
                p_T.view(qk_real_shape).sub_view(p_batch_shape, batch_p_offset);
            auto vi_tensor_batch =
                vi_sub_tensor.sub_view(vi_batch_shape, batch_v_offset);

            tiu::fmm2(pv_tensor_batch, p_tensor_batch, vi_tensor_batch, false,
                      false, false, false, false);
          }
          tiu::fadd(acc_sub_tensor, acc_sub_tensor, pv_tensor);
        }

        tensor<fp32> li_tensor_fp32, div_li_tensor;
        tensor<T> div_li_tensor_T, qkvo_tensor;
        tiu::cast(li_tensor_fp32, li_sub_tensor);
        tiu::fdiv(div_li_tensor, 1.0f, li_tensor_fp32, 3);
        tiu::cast(div_li_tensor_T, div_li_tensor);
        // broadcast mul (w)
        tiu::fmul(qkvo_tensor, acc_sub_tensor, div_li_tensor_T);

        dim4 qkv_offset = {_h, _m, _b, 0};
        dma::store(out_global_tensor.sub_view(acc_real_shape, qkv_offset),
                   qkvo_tensor);
      }
    }
  }
}

__KERNEL__ void mlir_attention_f16(fp16 *ptr_out, fp16 *ptr_q, fp16 *ptr_k,
                                   fp16 *ptr_v, fp16 *ptr_musk, int b, int qm,
                                   int kvm, int d, int q_head, int kv_head,
                                   float sqrt_d, int has_musk, const int dmax,
                                   const int block_m, const int block_k,
                                   const int block_h) {
  mlir_attention<fp16>(ptr_out, ptr_q, ptr_k, ptr_v, ptr_musk, b, qm, kvm, d,
                       q_head, kv_head, sqrt_d, has_musk, dmax, block_m,
                       block_k, block_h);
}

__KERNEL__ void mlir_attention_bf16(bf16 *ptr_out, bf16 *ptr_q, bf16 *ptr_k,
                                    bf16 *ptr_v, bf16 *ptr_musk, int b, int qm,
                                    int kvm, int d, int q_head, int kv_head,
                                    float sqrt_d, int has_musk, const int dmax,
                                    const int block_m, const int block_k,
                                    const int block_h) {
  mlir_attention<bf16>(ptr_out, ptr_q, ptr_k, ptr_v, ptr_musk, b, qm, kvm, d,
                       q_head, kv_head, sqrt_d, has_musk, dmax, block_m,
                       block_k, block_h);
}

using DTYPE = fp16;

__TEST__ void mlir_attention_main() {
// #define PPL_TEST
#ifdef PPL_TEST
  const int dmax = 128;
  const int block_m = 32;
  const int block_k = 32;
  const int block_h = 32;
#else
#ifdef KVCACHE
  const int dmax = 192;
  const int block_m = 64;
  const int block_k = 192;
  const int block_h = 32;
#else
  const int dmax = 128;
  const int block_m = 128;
  const int block_k = 128;
  const int block_h = 16;
  // int block_h = 32;
#endif
#endif
  int d = 100;
  float sqrt_d = 0.088388;
  int qm = 256;
  int kvm = 513;
  int b = 1;
  int q_head = 32;
  int kv_head = 8;

  dim4 q_shape = {b, qm, q_head, d};
  dim4 kv_shape = {b, kvm, kv_head, d};
  dim4 mask_shape = {b, 1, qm, kvm};
  dim4 tmp_shape = q_shape;
  // dim4 tmp_shape2 = {b, head, qm, d};
  // dim4 tmp_shape3 = {b, head, qm, 96};

  auto ptr_out = ppl::malloc<DTYPE>(&q_shape);
  auto ptr_q = ppl::malloc<DTYPE>(&q_shape);
  auto ptr_k = ppl::malloc<DTYPE>(&kv_shape);
  auto ptr_v = ppl::malloc<DTYPE>(&kv_shape);
  auto ptr_mask = ppl::malloc<DTYPE>(&mask_shape);
  auto ptr_tmp = ppl::malloc<DTYPE>(&tmp_shape);

  ppl::rand(ptr_q, &q_shape, -10.f, 10.f);
  ppl::rand(ptr_k, &kv_shape, -10.f, 10.f);
  ppl::rand(ptr_v, &kv_shape, -10.f, 10.f);

  dim4 mask_rand_shape = {b, 1, qm, kvm / 2};
  dim4 mask_stride = {qm * kvm, qm * kvm, kvm, 2};
  dim4 mask_offset = {0, 0, 0, 0};
  // ppl::rand(ptr_mask, &mask_rand_shape, &mask_stride, 0, -10000.f, -10000.f);
  ppl::rand(ptr_mask, &mask_rand_shape, &mask_stride, -10.f, -10000.f);

  int has_mask = 1;
  mlir_attention_f16(ptr_out, ptr_q, ptr_k, ptr_v, ptr_mask, b, qm, kvm, d,
                     q_head, kv_head, sqrt_d, has_mask, dmax, block_m, block_k,
                     block_h);
}
