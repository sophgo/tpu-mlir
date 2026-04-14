/**
 * ConcatVolume PPL Kernel for Stereo Matching
 * 
 * Build concatenated cost volume for stereo matching. Concatenate left and right
 * feature maps after shifting by disparity.
 * 
 * Input:
 *   - left_feature: Left image feature (N, C, H, W)
 *   - right_feature: Right image feature (N, C, H, W)
 *   - max_disp: Maximum disparity value
 * 
 * Output:
 *   - volume: Cost volume (N, 2*C, max_disp, H, W)
 *   - Memory layout: [n][c][d][h][w]
 * 
 * Mathematical formula:
 *   for d in range(max_disp):
 *     if d > 0:
 *       output[:, :C, d, :, :] = left[:, :, :, :]
 *       output[:, C:, d, :, d:] = right[:, :, :, :-d]
 *       output[:, C:, d, :, :d] = 0  # Zero padding
 *     else:
 *       output[:, :C, d, :, :] = left
 *       output[:, C:, d, :, :] = right
 */

#include "ppl.h"
#include "ppl_wrapper_func.h"
using namespace ppl;

template <typename T>
void concat_volume(T *ptr_dst, T *ptr_lsrc, T *ptr_rsrc,
                   const int N, const int C, const int H, const int W, const int max_disp) {
    // Input shape: (N, C, H, W)
    dim4 in_gshape = {N, C, H, W};
    dim4 in_1b_shape = {1, C, H, W};
    dim4 in_view_gshape = {C, H, 1, W};
    auto in_lgtensor = gtensor<T>(in_gshape, GLOBAL, ptr_lsrc);
    auto in_rgtensor = gtensor<T>(in_gshape, GLOBAL, ptr_rsrc);
    
    // Output shape: (N, 2*C, max_disp, H, W)
    // Memory layout: [n][c][d][h][w] -> linear index = n*(2C*D*H*W) + c*(D*H*W) + d*(H*W) + h*W + w
    // Equivalent to 4D [n][c*D+d][h][w] -> linear index = n*(2C*D*H*W) + (c*D+d)*(H*W) + h*W + w
    dim4 out_gshape = {N, 2 * C * max_disp, H, W};
    dim4 out_1b_shape = {1, 2 * C * max_disp, H, W};
    dim4 out_view_gshape = {2 * C * max_disp, H, 1, W};
    auto res_gtensor = gtensor<T>(out_gshape, GLOBAL, ptr_dst);
    
    for (int n_idx = 0; n_idx < N; n_idx++) {
        // Get view of current batch
        dim4 offset = {n_idx, 0, 0, 0};
        auto view_lgtensor = in_lgtensor.sub_view(in_1b_shape, offset);
        auto view_rgtensor = in_rgtensor.sub_view(in_1b_shape, offset);
        auto view_res_gtensor = res_gtensor.sub_view(out_1b_shape, offset);
        
        // Reinterpret shape for easier processing
        auto in_view_lgtensor = view_lgtensor.view(in_view_gshape);
        auto in_view_rgtensor = view_rgtensor.view(in_view_gshape);
        auto out_view_gtensor = view_res_gtensor.view(out_view_gshape);
        
        int block_h = LANE_NUM;
        dim4 block_shape = {C, block_h, 1, W};
        
        // Process by blocks along H dimension
        for (int h_idx = 0; h_idx < H; h_idx += block_h) {
            int tile_h = min(block_h, H - h_idx);
            dim4 cur_shape = {C, tile_h, 1, W};
            
            // Allocate local memory
            auto in_l_static = tensor<T>(block_shape, TPU_COMPACT);
            auto in_l = in_l_static.view(cur_shape);
            auto in_r_static = tensor<T>(block_shape, TPU_COMPACT);
            auto in_r = in_r_static.view(cur_shape);
            
            // Load left and right features
            dim4 offset_gin = {0, h_idx, 0, 0};
            dma::load(in_l, in_view_lgtensor.sub_view(cur_shape, offset_gin));
            dma::load(in_r, in_view_rgtensor.sub_view(cur_shape, offset_gin));
            
            // Allocate a local tensor for zero padding
            // Use maximum possible zero padding width (max_disp - 1)
            dim4 zero_block_shape = {1, block_h, 1, max_disp};
            auto zero_static = tensor<T>(zero_block_shape, TPU_COMPACT);
            dma::fill(zero_static, 0);
            
            // Process each disparity value
            for (int d_idx = 0; d_idx < max_disp; d_idx++) {
                ppl::enable_pipeline();
                
                int w_start = d_idx;
                int w_len = W - d_idx;
                
                // Process left feature: copy to first C channel groups
                // Output channel index: c * max_disp + d
                for (int c_idx = 0; c_idx < C; c_idx++) {
                    int out_c_left = c_idx * max_disp + d_idx;
                    dim4 out_shape_l = {1, tile_h, 1, W};
                    dim4 out_offset_l = {out_c_left, h_idx, 0, 0};
                    dim4 in_offset_c = {c_idx, 0, 0, 0};
                    
                    dma::store(out_view_gtensor.sub_view(out_shape_l, out_offset_l),
                               in_l.sub_view(out_shape_l, in_offset_c));
                }
                
                // Process right feature: copy to last C channel groups after shifting
                // Output channel index: (C + c) * max_disp + d
                for (int c_idx = 0; c_idx < C; c_idx++) {
                    int out_c_right = (C + c_idx) * max_disp + d_idx;
                    dim4 in_offset_c = {c_idx, 0, 0, 0};
                    
                    if (d_idx == 0) {
                        // When d=0, directly copy the entire right feature
                        dim4 out_shape_full = {1, tile_h, 1, W};
                        dim4 out_offset_r = {out_c_right, h_idx, 0, 0};
                        dma::store(out_view_gtensor.sub_view(out_shape_full, out_offset_r),
                                   in_r.sub_view(out_shape_full, in_offset_c));
                    } else {
                        // When d > 0, need two steps:
                        // 1. First write zero padding region [0, w_start)
                        dim4 zero_shape = {1, tile_h, 1, w_start};
                        dim4 out_offset_zero = {out_c_right, h_idx, 0, 0};
                        dma::store(out_view_gtensor.sub_view(zero_shape, out_offset_zero),
                                   zero_static.view(zero_shape));
                        
                        // 2. Then write data region [w_start, W)
                        if (w_len > 0) {
                            dim4 out_shape_r = {1, tile_h, 1, w_len};
                            dim4 out_offset_r = {out_c_right, h_idx, 0, w_start};
                            dma::store(out_view_gtensor.sub_view(out_shape_r, out_offset_r),
                                       in_r.sub_view(out_shape_r, in_offset_c));
                        }
                    }
                }
            }
        }
    }
}

__KERNEL__ void concat_volume_f32(float *ptr_dst, float *ptr_lsrc, float *ptr_rsrc,
                                  const int N, const int C, const int H, const int W, const int max_disp) {
    concat_volume(ptr_dst, ptr_lsrc, ptr_rsrc, N, C, H, W, max_disp);
}

__KERNEL__ void concat_volume_fp16(fp16 *ptr_dst, fp16 *ptr_lsrc, fp16 *ptr_rsrc,
                                   const int N, const int C, const int H, const int W, const int max_disp) {
    concat_volume(ptr_dst, ptr_lsrc, ptr_rsrc, N, C, H, W, max_disp);
}

__KERNEL__ void concat_volume_bf16(bf16 *ptr_dst, bf16 *ptr_lsrc, bf16 *ptr_rsrc,
                                   const int N, const int C, const int H, const int W, const int max_disp) {
    concat_volume(ptr_dst, ptr_lsrc, ptr_rsrc, N, C, H, W, max_disp);
}

__KERNEL__ void concat_volume_int8(int8 *ptr_dst, int8 *ptr_lsrc, int8 *ptr_rsrc,
                                   const int N, const int C, const int H, const int W, const int max_disp) {
    concat_volume(ptr_dst, ptr_lsrc, ptr_rsrc, N, C, H, W, max_disp);
}

__TEST__ void test() {
    const int N = 1;
    const int C = 12;
    const int H = 24;
    const int W = 30;
    const int max_disp = 8;
    
    dim4 in_shape = {N, C, H, W};
    dim4 out_shape = {N, 2 * C * max_disp, H, W};
    
    auto dst = malloc<fp16>(&out_shape);
    auto src0 = malloc<fp16>(&in_shape);
    rand(src0, &in_shape, -1.f, 1.f);
    auto src1 = malloc<fp16>(&in_shape);
    rand(src1, &in_shape, -1.f, 1.f);
    
    concat_volume_fp16(dst, src0, src1, N, C, H, W, max_disp);
}
