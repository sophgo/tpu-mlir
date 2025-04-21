#include "ppl.h"  // PPL 代码依赖的头文件
#include "ppl_wrapper_func.h"
using namespace ppl;


template <typename T>
void correlation(T *ptr_dst, T *ptr_lsrc, T *ptr_rsrc,
                                const int N, const int C, const int H, const int W, const int cut_nums)
{

    dim4 in_gshape = {N, C, H, W};
    dim4 in_1b_shape = {1, C, H, W};
    dim4 in_view_gshape = {C, H, 1, W};
    auto in_lgtensor = gtensor<T>(in_gshape, GLOBAL, ptr_lsrc);
    auto in_rgtensor = gtensor<T>(in_gshape, GLOBAL, ptr_rsrc);

    dim4 out_gshape = {N, cut_nums, H, W};
    dim4 out_1b_shape = {1, cut_nums, H, W};
    dim4 out_view_gshape = {cut_nums, H, 1, W};
    auto res_gtensor = gtensor<T>(out_gshape, GLOBAL, ptr_dst);

    for (int n_idx = 0; n_idx < N; n_idx++) {
        // left_feature(1,24,184,320) ->left_feature(24,184,1,320)
        // right_feature(1,24,184,320)->right_feature(24,184,1,320)
        dim4 offset = {n_idx, 0, 0, 0};
        auto view_lgtensor = in_lgtensor.sub_view(in_1b_shape, offset);
        auto view_rgtensor = in_rgtensor.sub_view(in_1b_shape, offset);
        auto view_gtensor = res_gtensor.sub_view(out_1b_shape, offset);
        auto in_view_lgtensor = view_lgtensor.view(in_view_gshape);
        auto in_view_rgtensor = view_rgtensor.view(in_view_gshape);
        auto out_view_gtensor = view_gtensor.view(out_view_gshape);

        int block_h = LANE_NUM;
        dim4 block_shape = {C, block_h, 1, W};

        // left_feature(24,184,1,320) ->left_feature(24,32,1,320) cut
        // right_feature(24,184,1,320) ->right_feature(24,32,1,320) cut
        for (int h_idx = 0; h_idx < H; h_idx += block_h) {
            int tile_h = min(block_h, H - h_idx);
            dim4 cur_shape = {C, tile_h, 1, W};
            auto in_l_static = tensor<T>(block_shape, TPU_COMPACT);
            auto in_l = in_l_static.view(cur_shape);
            auto in_r_static = tensor<T>(block_shape, TPU_COMPACT);
            auto in_r = in_r_static.view(cur_shape);

            dim4 offset_gin = {0, h_idx, 0, 0};
            dma::load(in_l, in_view_lgtensor.sub_view(cur_shape, offset_gin));
            dma::load(in_r, in_view_rgtensor.sub_view(cur_shape, offset_gin));

            // left_feature(24,32,1,320) ->left_feature(1,32,24,320) reshape
            // right_feature(24,32,1,320) ->right_feature(1,32,24,320) reshpe
            dim4 view_shape = {1, tile_h, C, W};
            dim4 stride_in = {H * W, min(H * W, 32), W, 1};
            auto in_l_v =  in_l.view(view_shape, stride_in);
            auto in_r_v =  in_r.view(view_shape, stride_in);

            dim4 mul_shape = {1, block_h, C, W};
            auto mul_tensor = make_tensor<T>(mul_shape, view_shape);

            dim4 reduce_shape = {1, block_h, 1, W};
            dim4 reduce_real_shape = {1, tile_h, 1, W};
            auto reduce_tensor = make_tensor<T>(reduce_shape, reduce_real_shape);

            double scale = static_cast<double>(1.0 / C);
            dim2 kernel = {C, 1};
            padding_t padding = {0, 0, 0, 0};
            dim2 stride = {1, 1};
            dim2 dilation = {1, 1};

            for (int slice_idx = 0; slice_idx < cut_nums; slice_idx += 1) {
                ppl::enable_pipeline();
                // left_feature(1,32,24,320) -> left_feature(offset={0,0,0,slice_idx})(1,32,24,319) slice
                // right_feature(1,32,24,320) -> right_feature(offset={0,0,0,0})(1,32,24,319) slice
                dim4 slice_shape = {1, tile_h, C, W - slice_idx};
                dim4 offsetl = {0, 0, 0, slice_idx};
                dim4 offsetr = {0, 0, 0, 0};

                // left_feature(1,32,24,319) -> mul(1,32,24,319)
                // right_feature(1,32,24,319)->
                tiu::fmul(mul_tensor.sub_view(slice_shape, offsetl), in_l_v.sub_view(slice_shape, offsetl), in_r_v.sub_view(slice_shape, offsetr));
                // mul(1,32,24,319) -> reduce(dim=2)(1,32,1,319)
                tiu::fpool_avg(reduce_tensor, mul_tensor, &kernel, &padding, &stride, &dilation, scale);
                // reduce(1,32,1,319)                                      -> reduce(1,32,1,320)
                // fill_zero(offset={0,0,0,slice_idx})(1,32,1,slice_idx)   ->
                dim4 fill_shape = {1, tile_h, 1, slice_idx};
                if (slice_idx > 0) {
                    dma::fill(reduce_tensor.view(fill_shape), 0);
                }
                dim4 offset_gout = {slice_idx, h_idx, 0, 0};
                dma::store(out_view_gtensor.sub_view(reduce_real_shape, offset_gout), reduce_tensor);
            }
        }
    }
}

__KERNEL__ void correlation_f32(float *ptr_dst, float *ptr_lsrc, float *ptr_rsrc,
                                const int N, const int C, const int H, const int W, const int cut_nums) {
    correlation(ptr_dst, ptr_lsrc, ptr_rsrc, N, C, H, W, cut_nums);
}
__KERNEL__ void correlation_fp16(fp16 *ptr_dst, fp16 *ptr_lsrc, fp16 *ptr_rsrc,
                                const int N, const int C, const int H, const int W, const int cut_nums) {
    correlation(ptr_dst, ptr_lsrc, ptr_rsrc, N, C, H, W, cut_nums);
}
__KERNEL__ void correlation_bf16(bf16 *ptr_dst, bf16 *ptr_lsrc, bf16 *ptr_rsrc,
                                const int N, const int C, const int H, const int W, const int cut_nums) {
    correlation(ptr_dst, ptr_lsrc, ptr_rsrc, N, C, H, W, cut_nums);
}


__TEST__ void test()
{
    const int N = 2;
    const int C = 24;
    const int H = 184;
    const int W = 320;
    const int cut_nums = 48;
    dim4 shape = {N, C, H, W};
    dim4 out_shape = {N, cut_nums, H, W};
    auto dst = malloc<fp16>(&out_shape);
    auto src0 = malloc<fp16>(&shape);
    rand(src0, &shape, -1.f, 1.f);
    auto src1 = malloc<fp16>(&shape);
    rand(src1, &shape, -1.f, 1.f);
    correlation_fp16(dst, src0, src1, N, C, H, W, cut_nums);
}