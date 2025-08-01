#pragma once

#include "tpuDNN.h"
#include "common_def.h"
#include "api/sg_api_struct.h"
#ifdef USING_PLD_TEST
#include "api/sg_api_pld.h"
#endif

extern "C"
{
    tpudnnStatus_t tpudnnActive(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        const int *shape,
        int shape_dim,
        sg_active_type_t active_type,
        const float *coeff,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnConcat(
        tpudnnHandle_t      handle,
        void ** inputs,
        void * output,
        const int        (*input_shapes)[FW_MAX_SHAPE_DIMS],
        const int*       st_by_concatway,
        int              input_num,
        int              concat_axis,
        int              input_dims,
        sg_data_type_t   sgdtype);

    tpudnnStatus_t tpudnnPooling(
        tpudnnHandle_t handle,
        // input
        void *input,
        int input_n,
        int input_c,
        int input_h,
        int input_w,
        int output_h,
        int output_w,
        int kh,
        int kw,
        int pad_h,
        int pad_w,
        int pad_h_after,
        int pad_w_after,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w,
        int is_avg_pooling,
        int avg_pooling_mode,
        int max_mask,
        int if_relu,
        float relu_upper_limit,
        sg_data_type_t dtype,
        int is_max,
        // output
        void *output,
        void *max_mask_addr);

    tpudnnStatus_t tpudnnArithmetic(
            tpudnnHandle_t        handle,
            void *    src0,
            void *    src1,
            void *    dst,
            int       op,
            int       n,
            int       c,
            int       h,
            int       w
#if defined(__mars3__) || defined(__sgtpuv8__)
            ,sg_data_type_t dType
#endif
        );

    tpudnnStatus_t tpudnnArithmeticShift(
      tpudnnHandle_t         handle,
      void *     input,
      void *     shift,
      void *     output,
      const int*          shape,
      int                 shape_dim,
      int                 is_per_channel,
      char                shift_value,
      sg_data_type_t      in_dtype,
      sg_data_type_t      shift_dtype,
      sg_data_type_t      out_dtype,
      sg_round_mode_t     sg_round_mode);

    tpudnnStatus_t tpudnnActiveMultiCore(
        tpudnnHandle_t      handle,
        void *              input,
        void *              output,
        const int*          shape,
        int                 dims,
        sg_active_type_t    active_type,
        const float*        coeff,
        sg_data_type_t      dtype);

    tpudnnStatus_t tpudnnLoraMatmulMultiCore(
        tpudnnHandle_t handle,
        void *X,
        void *loraA,
        void *loraB,
        void *W,
        void *output,
        const int *X_shape,
        const int *loraA_shape,
        const int *loraB_shape,
        const int X_dims,
        const int loraA_dims,
        const int loraB_dims,
        sg_data_type_t in_dtype,
        sg_data_type_t out_dtype,
        int do_scale,
        float scale_val);

    tpudnnStatus_t tpudnnPoolingFix8b(
        tpudnnHandle_t handle,
        // input
        void *input,
        int input_n,
        int input_c,
        int input_h,
        int input_w,
        int kh,
        int kw,
        int pad_h_top,
        int pad_h_bottom,
        int pad_w_left,
        int pad_w_right,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w,
        int is_avg_pooling,
        int avg_pooling_mode,
        int ceil_mode,
        sg_data_type_t output_dtype,
        sg_data_type_t input_dtype,
        // output
        void *output);

    tpudnnStatus_t tpudnnArg(
        tpudnnHandle_t handle,
        void * input,
        void * index,
        void * value,
        int *input_shape,
        int dims,
        int axis,
        int method, // 0: argmax, 1: argmin
        int is_index_int32,
        int select_last_index,
        int need_val,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnAdaptivePool(
        tpudnnHandle_t handle,
        void *input_mem,
        const int *input_shape,
        int input_dims,
        const int oh,
        const int ow,
        const int mode,
        void *output_mem,
        sg_data_type_t sgdtype);
    tpudnnStatus_t tpudnnBilinearInterpolate(
        tpudnnHandle_t     handle,
        void * input,
        void * map_y,
        void * map_x,
        void * output,
        int             input_c,
        int             input_h,
        int             input_w,
        int             map_yx_len,
        int             map_yx_c_need_bcast,
        int             mode,
        sg_data_type_t  dtype);

    tpudnnStatus_t tpudnnBinaryShift(
        tpudnnHandle_t             handle,
        void *         input_a,
        void *         input_b,
        void *         output,
        const int*              a_shape,
        const int*              b_shape,
        int                     shape_dim,
        sg_binary_type_t        binary_op,
        int                     rshift_num,
        int                     b_is_const,
        int                     b_const_val,
        int                     a_is_coeff,
        int                     b_is_coeff,
        int                     inversed,
        sg_data_type_t          a_dtype,
        sg_data_type_t          b_dtype,
        sg_data_type_t          out_dtype);

    tpudnnStatus_t tpudnnBatchNorm(
        tpudnnHandle_t      handle,
        // input
        void *  input,
        void *  mean_ma,
        void *  variance_ma,
        int              input_n,
        int              input_c,
        int              input_h,
        int              input_w,
        int              if_relu,
        float            relu_upper_limit,
        sg_data_type_t   dtype,
        // output
        void *  output);

    tpudnnStatus_t tpudnnBatchNormBw(
        tpudnnHandle_t      handle,
        // input
        void *  grad_output,
        void *  input,
        void *  weight,
        void *  save_mean,
        void *  save_invstd,
        int              N,
        int              C,
        int              H,
        int              W,
        sg_data_type_t    dtype,
        // output
        void *  grad_input,
        void *  grad_weight,
        void *  grad_bias);

    tpudnnStatus_t tpudnnBatchNormTrainMultiCore(
        tpudnnHandle_t handle,
        // input
        void *input,
        void *running_mean,
        void *running_variance,
        const int *in_shape,
        void *weight,
        void *bias,
        float momentum,
        float eps,
        sg_data_type_t dtype,
        // output
        void *output,
        void *saved_mean,
        void *saved_invstd,
        void *running_mean_update,
        void *running_variance_update);

    tpudnnStatus_t tpudnnBnscaleFix8b(
        tpudnnHandle_t       handle,
        // input
        void *   input,
        void *   scale,
        void *   bias,
        void *   shift,
        int               input_n,
        int               input_c,
        int               input_h,
        int               input_w,
        int               input_sign,
        int               scale_sign,
        int               bias_sign,
        int               if_relu,
        int               relu_upper_limit,
        sg_round_mode_t   round_mode,
        // output
        void *   output);

    tpudnnStatus_t tpudnnInterpForward(
        tpudnnHandle_t handle,
        // input
        void *input,
        int input_n,
        int input_c,
        int input_h,
        int input_w,
        int output_h,
        int output_w,
        int pad_bag,
        int pad_end,
        bool align_corners,
        bool half_pixel_centers,
        PLATFORM_SUPPORT platform_sp,
        sg_data_type_t dtype,
        // output
        void *output);

    tpudnnStatus_t tpudnnScaleForward(
        tpudnnHandle_t handle,
        // input
        void *input,
        void *scale,
        void *bias,
        int input_n,
        int input_c,
        int input_h,
        int input_w,
        int axis,     // scale begin axis
        int axis_num, // scale axis num
        int has_bias,
        int if_relu,
        float relu_upper_limit,
        sg_data_type_t dtype,
        // output
        void *output);

    tpudnnStatus_t tpudnnShift(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        const int *shape,
        int shape_dim,
        int shift_axis,
        sg_shift_type_t shift_dir,
        int shift_num,
        sg_data_type_t in_dtype);

    tpudnnStatus_t tpudnnShuffleChannel(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        const int *shape,
        int shape_dim,
        int group,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnSoftmax(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        int input_n,
        int input_c,
        int input_inner_dim,
        float scale_val,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnSoftmaxTfliteFix8b(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        void *table,
        int input_n,
        int input_c,
        int input_inner_dim,
        int zero_point,
        float scale_val,
        sg_data_type_t input_dtype,
        sg_data_type_t output_dtype);

    tpudnnStatus_t tpudnnSort(
        tpudnnHandle_t handle,
        void *src_data,
        void *src_idx,
        void *dst_data,
        void *dst_idx,
        int len,
        int topk,
        bool is_descend,
        bool idx_en,
        bool auto_idx,
        int dtype);

    tpudnnStatus_t tpudnnSparseConv3d(
        tpudnnHandle_t handle,
        void *input_features_global_addr,
        void *input_coor_global_addr,
        void *weight_global_addr,
        void *origin_input_shape_global_addr,
        void *intermedia_mem_pool_global_addr,
        void *intermedia_mem_pool_ex_global_addr,
        void *output_features_global_addr,
        void *output_coor_global_addr,
        void *origin_output_shape_global_addr,
        void *debug_pool_global_addr,
        int case_num,
        int batch_num,
        int limit_active_out_num,
        int ndim,
        int output_channel,
        int input_channel,
        int kz, // kernel
        int ky,
        int kx,
        int sz, // stride
        int sy,
        int sx,
        int pz, // padding
        int py,
        int px,
        int dz, // dilation
        int dy,
        int dx,
        unsigned long long input_feature_sz,
        unsigned long long input_coor_sz,
        unsigned long long weight_sz,
        unsigned long long origin_input_shape_sz,
        unsigned long long intermedia_mem_pool_sz,
        unsigned long long intermedia_mem_pool_ex_sz,
        unsigned long long output_feature_sz,
        unsigned long long output_coor_sz,
        unsigned long long origin_output_shape_sz,
        unsigned long long debug_sz,
        int has_bias,
        int subm,
        int opz, // output padding
        int opy,
        int opx,
        sg_data_type_t feature_dtype);

    tpudnnStatus_t tpudnnSplitForward(
        tpudnnHandle_t handle,
        // input
        void *input,
        const int *input_shape,
        int input_dim,
        int split_axis,
        const int *split_size,
        int split_num,
        sg_data_type_t dtype,
        // output
        void **output,
        int test_glb);

    tpudnnStatus_t tpudnnSortPerDim(
        tpudnnHandle_t handle,
        void *input_data,
        void *output_data,
        void *output_index,
        const int *input_shape,
        int input_dims,
        int sort_dim,
        bool is_argsort,
        bool stable,
        bool descending,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnSsdDetectOut(tpudnnHandle_t handle,
                                      void *location,
                                      void *confidence,
                                      void *prior,
                                      int batch_num,
                                      int num_prior,
                                      int num_classes,
                                      int num_loc_classes,
                                      int share_location,
                                      int background_label_id,
                                      int top_k,
                                      int code_type,
                                      int keep_top_k,
                                      int variance_encoded_in_target,
                                      float nms_threshold,
                                      float conf_threshold,
                                      float eta,
                                      int onnx_nms,
                                      void *output);

    tpudnnStatus_t tpudnnStrideSlice(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        const int *input_shape,
        int shape_dim,
        int begin_mask,
        int end_mask,
        const int *begin_index,
        const int *end_index,
        const int *strides,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnTileForward(
        tpudnnHandle_t handle,
        // input
        void *input,
        const int *input_shape,
        const int *tile_coeff,
        int input_dim,
        int type,
        sg_data_type_t dtype,
        // output
        void *output,
        int test_glb);

    tpudnnStatus_t tpudnnTiuLoopTest(
        tpudnnHandle_t handle,
        const void *input_data,
        void *output_data,
        sg_data_type_t dtype,
        int full_local_mem_size,
        int loop_num,
        int save_last,
        int test_power);

    tpudnnStatus_t tpudnnTopk(
        tpudnnHandle_t handle,
        void *input_data,
        void *input_index,
        void *output_data,
        void *output_index,
        bool input_index_valid,
        int k,
        int descending,
        int batchs,
        int batch_num,
        int batch_stride,
        sg_data_type_t dtype);

    #ifdef __sg2262__
    tpudnnStatus_t tpudnnTopkMultiCore(
        tpudnnHandle_t handle,
        void *input,
        int shape_dim,
        const int *shape,
        int k,
        int axis,
        bool largest,
        bool sorted,
        void *value,
        void *index,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnGroupTopk(
        tpudnnHandle_t handle,
        void *input,
        int shape_dim,
        const int *shape,
        int k,
        int group_k,
        int global_k,
        int axis,
        bool largest,
        bool sorted,
        void *value,
        void *index,
        sg_data_type_t dtype);
    #endif

    tpudnnStatus_t tpudnnTpuFullTest(
        tpudnnHandle_t handle,
        const void *input_data,
        void *output_data,
        int full_local_mem_size,
        int loop_num,
        int max_run_num,
        unsigned long long disable_mask);

    tpudnnStatus_t tpudnnTranspose(
        tpudnnHandle_t handle,
        void *input_mem,
        int *input_shape,
        int *order,
        int input_dims,
        void *output_mem,
        sg_data_type_t sgdtype);

    tpudnnStatus_t tpudnnTriangularize(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        int *shape,
        int dims,
        int is_upper,
        int diagonal,
        sg_data_type_t dtype,
        bool use_local_test = false);

    tpudnnStatus_t tpudnnUpsample(
        tpudnnHandle_t handle,
        void *input,
        int input_n,
        int input_c,
        int input_h,
        int input_w,
        int size,
        int if_relu,
        sg_data_type_t dtype,
        void *output);

    tpudnnStatus_t tpudnnUpsampleMaskForward(
        tpudnnHandle_t handle,
        void *input,
        void *input_mask,
        int input_n,
        int input_c,
        int input_h,
        int input_w,
        int size,
        void *output);

    tpudnnStatus_t sgdnn_yolo_forward(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        int input_n,
        int input_c,
        int input_h,
        int input_w,
        int num,
        int classes,
        int coords,
        int background,
        int softmax,
        sg_data_type_t dtype,
        int test_glb);

    tpudnnStatus_t tpudnnYolov3DetectOut(
        tpudnnHandle_t handle, int input_num, void *bottom[3],
        int batch_num, int hw_shape[3][2], int num_classes,
        int num_boxes, int mask_group_size, float nms_threshold, float confidence_threshold,
        int keep_top_k, float bias[18], float anchor_scale[3], float mask[9],
        void *output, int yolov5_flag, int len_per_batch,
        int scale, int *orig_image_shape, int model_h, int model_w);

#define MAX_YOLO_INPUT_NUM 8
    tpudnnStatus_t tpudnnYolov5PostProcessDecode(
        tpudnnHandle_t handle,
        void **input_data,
        void *output_data,
        void *detected_num,
        int input_num,
        int batch_num,
        int hw_shape[MAX_YOLO_INPUT_NUM][2],
        int num_classes,
        int num_boxes,
        int keep_top_k,
        float nms_threshold,
        float confidence_threshold,
        float *anchors,
        float *anchor_scale,
        int clip_box,
        int agnostic_nms,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnWhere(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        void *pos_num,
        const int *shape,
        int dims,
        int order,
        sg_data_type_t sgdtype);

    tpudnnStatus_t tpudnnEltwise(
        tpudnnHandle_t handle,
        int op_,
        int input_n,
        int input_c,
        int input_h,
        int input_w,
        int if_relu,
        int *coeffs_,
        int *index,
        void *input_A,
        void *input_B,
        void *mask_data,
        void *output,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnEltwiseFix8b(
        tpudnnHandle_t handle,
        // input
        void *input_A,
        void *input_B,
        int batch_size,
        int channels,
        int height,
        int width,
        int op_code, // 0: mul, 1: add, 2: max
        int scale_A,
        int scale_B,
        int sign_A,
        int sign_B,
        int rshift_A,
        int rshift_B,
        int if_relu,
        int round_mode,
        // output
        void *output);

    tpudnnStatus_t tpudnnEmbeddingBag(
        tpudnnHandle_t handle,
        void *input_data,
        void *medium_data,
        void *output_data,
        int num_embeddings,
        int embedding_dim,
        int mode,
        int indices_size,
        int offsets_size,
        void *indices,
        void *offsets,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnEmbeddingBagFix8b(
        tpudnnHandle_t handle,
        void *input_data,
        void *medium_data,
        void *output_data,
        int num_embeddings,
        int embedding_dim,
        int mode,
        int indices_size,
        int offsets_size,
        void *indices,
        void *offsets,
        void *scale,
        void *bias,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnFc(
        tpudnnHandle_t handle,
        void *L_addr,
        void *R_addr,
        void *bias_addr,
        void *Y_addr,
        int L_row_num,
        int L_col_num,
        int R_col_num,
        int transpose,
        int have_bias,
        sg_data_type_t L_dtype,
        sg_data_type_t R_dtype,
        sg_data_type_t Y_dtype,
        int if_relu,
        float relu_upper_limit);

    tpudnnStatus_t tpudnnFcQuantSym(
        tpudnnHandle_t handle,
        void *L_addr,
        void *R_addr,
        void *bias_addr,
        void *Y_addr,
        int L_row_num,
        int L_col_num,
        int R_col_num,
        int transpose,
        int have_bias,
        sg_data_type_t L_dtype,
        sg_data_type_t R_dtype,
        sg_data_type_t bias_dtype,
        sg_data_type_t Y_dtype,
        int if_relu,
        int rshift_bit);

    tpudnnStatus_t tpudnnGatherNdTf(
        tpudnnHandle_t handle,
        void *input,
        void *indices,
        int *input_shape,
        int shape_dims,
        int *indices_shape,
        int indices_dims,
        int const_val, // fill_value if index not found in input
        int batch_dims,
        sg_data_type_t dtype,
        void *output);

    tpudnnStatus_t tpudnnGdmaLoopTest(
        tpudnnHandle_t handle,
        void *input_mem,
        void *output_mem,
        const int *shape,
        int loop_num,
        int mode);

    tpudnnStatus_t tpudnnGemm(
        tpudnnHandle_t handle,
        void *L_addr,
        void *R_addr,
        void *Y_addr,
        int L_row_num,
        int L_col_num,
        int R_col_num,
        int transpose,
        sg_data_type_t L_dtype,
        sg_data_type_t R_dtype,
        sg_data_type_t Y_dtype);

    tpudnnStatus_t tpudnnGridSample(
        tpudnnHandle_t handle,
        void *input,
        void *grid,
        void *output,
        int input_n,
        int input_c,
        int input_h,
        int input_w,
        int output_h,
        int output_w,
        int align_corners,
        GridSampleInterpMode interp_mode,
        GridSamplePaddingMode padding_mode,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnGroupNorm(
        tpudnnHandle_t handle,
        void *input,
        void *weight,
        void *bias,
        void *output,
        const int *shape,
        int dims,
        int axis,
        int group_num,
        float eps,
        int affine,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnLayerNorm(
        tpudnnHandle_t handle,
        void *input,
        void *weight,
        void *bias,
        void *output,
        void *mean,
        void *rstd,
        const int *shape,
        int dims,
        int axis,
        float eps,
        int affine,
        int need_mean,
        int need_rstd,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnRmsNorm(
        tpudnnHandle_t handle,
        void *input,
        void *weight,
        void *output,
        const int *shape,
        int dims,
        float eps,
        int affine,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnClipFloat(
        tpudnnHandle_t handle,
        void* input,
        void* output,
        const int* shape,
        int dims,
        sg_data_type_t dtype,
        float min,
        float max,
        int if_relu,
        float relu_upper_limit
    );

    tpudnnStatus_t tpudnnColumnHash(
        tpudnnHandle_t handle,
        void *input[11],
        void *output,
        unsigned int len,
        int init,
        int mode);

    tpudnnStatus_t tpudnnConvFloat(
        tpudnnHandle_t handle,
        void *input,
        void *weight,
        void *bias,
        void *rescale,
        void *output,
        int n,
        int ic,
        int ih,
        int iw,
        int oc,
        int groups,
        int kh,
        int kw,
        int stride_h,
        int stride_w,
        int dh,
        int dw,
        int pht,
        int phb,
        int pwl,
        int pwr,
        bool has_bias,
        bool if_relu,
        float upper_limit,
        bool result_add,
        bool has_rescale,
        sg_data_type_t idtype,
        sg_data_type_t odtype,
        int weight_is_coeff,
        int weight_size,
        bool use_multicore);

    tpudnnStatus_t tpudnnConvLoopTest(
        tpudnnHandle_t handle,
        void *input_mem,
        void *output_mem,
        const int *shape,
        int loop_num);

    tpudnnStatus_t tpudnnConvQuantSym(
        tpudnnHandle_t handle,
        void *input,
        void *weight,
        void *bias,
        void *output,
        int n,
        int ic,
        int ih,
        int iw,
        int oc,
        int groups,
        int kh,
        int kw,
        int stride_h,
        int stride_w,
        int dh,
        int dw,
        int pht,
        int phb,
        int pwl,
        int pwr,
        bool has_bias,
        bool if_relu,
        int upper_limit,
        int rshift,
        bool isign,
        bool wsign,
        bool bsign,
        bool out_is_16bit,
        int wsize);

    tpudnnStatus_t tpudnnConvRequant(
        tpudnnHandle_t handle,
        void *input,
        void *weight,
        void *bias,
        void *requant,
        void *output,
        int n,
        int ic,
        int ih,
        int iw,
        int oc,
        int groups,
        int kh,
        int kw,
        int stride_h,
        int stride_w,
        int dh,
        int dw,
        int pht,
        int phb,
        int pwl,
        int pwr,
        bool has_bias,
        bool if_relu,
        bool if_requant,
        bool requant_is_const,
        int multiplier,
        int rshift,
        int yzp,
        int upper_limit,
        int nIC,
        sg_data_type_t idtype,
        sg_data_type_t wdtype,
        sg_data_type_t bdtype,
        sg_data_type_t odtype);

    tpudnnStatus_t tpudnnDeformGatherForward(
        tpudnnHandle_t handle,
        void *input,
        void *offset,
        void *mask,
        void *output,
        int input_n,
        int input_c,
        int input_h,
        int input_w,
        int modulated,
        int deform_groups,
        int kh,
        int kw,
        int pad_h,
        int pad_w,
        int pad_h_after,
        int pad_w_after,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w,
        int mode,
        int offset_interleave,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnDependIdWraparound(
        tpudnnHandle_t handle,
        void* input,
        void* output);

    tpudnnStatus_t tpudnnDepth2space(
        tpudnnHandle_t      handle,
        void *  input_mem,
        void *  output_mem,
        const int*       input_shape,
        const int*       block_sizes,
        int              in_is_nchw,
        int              out_is_nchw,
        int              is_inversed,
        int              is_crd_mode,
        int              swap_cr,
        sg_data_type_t   sgdtype);

    tpudnnStatus_t tpudnnDepthwise(
        tpudnnHandle_t         handle,
        // input
        void *     input,
        void *     weight,
        void *     bias,
        void *     rescale,
        int                 input_n,
        int                 input_c,
        int                 input_h,
        int                 input_w,
        int                 kh,
        int                 kw,
        int                 stride_h,
        int                 stride_w,
        int                 pad_h_t,
        int                 pad_h_b,
        int                 pad_w_l,
        int                 pad_w_r,
        int                 dh,
        int                 dw,
        int                 has_bias,
        int                 if_relu,
        float               relu_upper_limit,
        int                 if_rescale,
        sg_data_type_t      dtype,
        sg_data_type_t      out_dtype,
        // output
        void *     output);

    tpudnnStatus_t tpudnnDepthwiseFix8b(
        tpudnnHandle_t         handle,
        // input
        void *     input,
        void *     weight,
        void *     bias,
        int                 input_n,
        int                 input_c,
        int                 input_h,
        int                 input_w,
        int                 kh,
        int                 kw,
        int                 stride_h,
        int                 stride_w,
        int                 pad_h_t,
        int                 pad_h_b,
        int                 pad_w_l,
        int                 pad_w_r,
        int                 dh,
        int                 dw,
        int                 has_bias,
        int                 if_relu,
        int                 relu_upper_limit,
        int                 rshift_num,
        int                 input_sign,
        int                 weight_sign,
        int                 bias_sign,
        sg_data_type_t      output_dtype,
        sg_round_mode_t     round_mode,
        // output
        void *     output);

    tpudnnStatus_t tpudnnDequantFloat(
        tpudnnHandle_t      handle,
        void *  input,
        void *  output,
        void *  dequant,
        int              N,
        int              C,
        int              H,
        int              W,
        bool             is_perchannel,
        float            scale_val,
        int              offset_val,
        sg_data_type_t   bottom_dtype);

    tpudnnStatus_t tpudnnDequantInt(
        tpudnnHandle_t      handle,
        void *  input,
        void *  output,
        void *  dequant,
        int              N,
        int              C,
        int              H,
        int              W,
        bool             is_perchannel,
        int              scale_val,
        int              offset_val,
        int              shift_val,
        sg_data_type_t   bottom_dtype,
        sg_data_type_t   top_dtype,
        int              mode,
        int              lshift,
        sg_round_mode_t  round_mode);

    tpudnnStatus_t tpudnnBatch2space(
        tpudnnHandle_t      handle,
        void *  input_mem,
        const int*       input_shape,
        int              input_dims,
        const int*       block_sizes,
        const int*       crop_sizes,
        void *  output_mem,
        sg_data_type_t   sgdtype);

    tpudnnStatus_t tpudnnActiveMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        const int *shape,
        int dims,
        sg_active_type_t active_type,
        const float *coeff,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnMatmulMultiCore(
        tpudnnHandle_t handle,
        void *left,
        void *right,
        void *bias,
        void *output,
        const int *L_shape,
        const int *R_shape,
        const int L_dims,
        const int R_dims,
        const int L_trans,
        const int R_trans,
        sg_data_type_t in_dtype,
        sg_data_type_t out_dtype,
        int slice_m_core,
        int slice_n_core,
        int slice_m,
        int slice_n,
        int slice_k,
        const int left_slice_dim,
        const int right_slice_dim,
        const int result_slice_dim,
        const int left_8ch_buf_size,
        const int right_8ch_buf_size,
        const int result_8ch_buf_size,
        char *left_8ch_buf[8],
        char *right_8ch_buf[8],
        char *result_8ch_buf[8],
        int has_bias,
        sg_data_type_t bias_dtype);

    tpudnnStatus_t tpudnnRmsnormForwardMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *weight,
        void *bias,
        void *output,
        int *shape,
        int dims,
        int axis,
        float partial,
        float eps,
        int with_weight,
        int with_bias,
        sg_data_type_t dtype,
        const int enable_8ch,
        const int input_slice_dim,
        const int weight_slice_dim,
        const int bias_slice_dim,
        const int output_slice_dim,
        const int input_buffer_size,
        const int weight_buffer_size,
        const int bias_buffer_size,
        const int output_buffer_size,
        char **input_8ch_buffer,
        char **weight_8ch_buffer,
        char **bias_8ch_buffer,
        char **output_8ch_buffer);

    tpudnnStatus_t tpudnnRmsnormBackwardMultiCore(
        tpudnnHandle_t handle,
        void *grad_output_global_addr,
        void *input_global_addr,
        void *weight_global_addr,
        void *rms_global_addr,
        void *grad_input_global_addr,
        void *grad_weight_global_addr,
        void *grad_bias_global_addr,
        int with_weight,
        int with_bias,
        int *shape,
        int dims,
        int axis,
        int requires_grad_input,
        sg_data_type_t dtype,
        int input_slice_dim,
        int output_slice_dim,
        float eps,
        char *grad_output_8ch_global_addr[8],
        char *input_8ch_global_addr[8],
        char *grad_input_8ch_global_addr[8]);

    tpudnnStatus_t tpudnnGroupNormMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *weight,
        void *bias,
        void *output,
        const int *shape,
        int dims,
        int axis,
        int group_num,
        float eps,
        int affine,
        sg_data_type_t dtype);

    typedef struct
    {
        void *data;
        unsigned long long addr;
        unsigned int size;
    } io_mem_t;

    typedef struct
    {
        void *sys_mem_tpu_cmd;
        void *sys_mem_gdma_cmd;
        void *sys_mem_hau_cmd;
        void *sys_mem_sdma_cmd;
        void *sys_mem_imm_buf;
        void *sys_mem_pio_buffer;
        int tpu_cmd_num;
        int gdma_cmd_num;
        int hau_cmd_num;
        int sdma_cmd_num;
        unsigned int tpu_cmd_size;
        unsigned int gdma_cmd_size;
        unsigned int hau_cmd_size;
        unsigned int sdma_cmd_size;
        unsigned int imm_buf_size;
        unsigned int pio_buffer_size;
    } system_msg_sync_core_param_t;
    typedef struct
    {
        system_msg_sync_core_param_t core_param[8];
        int loop;
        int enable_pio_des_interleave;
        int core_num;
        io_mem_t input[20];  // Align with test_msg_sync_multi_core MAX_IO_NUM
        io_mem_t output[20]; // Align with test_msg_sync_multi_core MAX_IO_NUM
        int input_num;
        int output_num;
        unsigned long long total_io_size;
        void *placeholder;
    } system_msg_sync_multi_core_param_t;

    tpudnnStatus_t tpudnnCommDescMultiCore(
        tpudnnHandle_t handle,
        system_msg_sync_multi_core_param_t system_msg_sync_param);

    typedef struct
    {
        void *sys_mem_tpu_cmd;
        void *sys_mem_gdma_cmd;
        void *sys_mem_hau_cmd;
        void *sys_mem_sdma_cmd;
        void *sys_mem_cdma_cmd;
        void *sys_mem_imm_buf;
        void *sys_mem_pio_buffer;
        int tpu_cmd_num;
        int gdma_cmd_num;
        int hau_cmd_num;
        int sdma_cmd_num;
        int cdma_cmd_num;
        unsigned int tpu_cmd_size;
        unsigned int gdma_cmd_size;
        unsigned int hau_cmd_size;
        unsigned int sdma_cmd_size;
        unsigned int cdma_cmd_size;
        unsigned int imm_buf_size;
        unsigned int pio_buffer_size;
    } system_msg_sync_core_cdma_param_t;

    typedef struct
    {
        system_msg_sync_core_cdma_param_t core_param[8];
        int loop;
        int enable_pio_des_interleave;
        int core_num;
        io_mem_t input[20];  // Align with test_msg_sync_multi_core MAX_IO_NUM
        io_mem_t output[20]; // Align with test_msg_sync_multi_core MAX_IO_NUM
        int input_num;
        int output_num;
        unsigned long long total_io_size;
        void *placeholder;
        int nranks;
        int cur_rank;
        int *chip_map;
    } system_msg_sync_cdma_param_t;

    tpudnnStatus_t tpudnnCommDescCdma(
        tpudnnHandle_t handle,
        system_msg_sync_cdma_param_t system_param_cdma);

    tpudnnStatus_t tpudnnCommDesc(
        tpudnnHandle_t handle,
        void *tpu_cmd_addr,
        void *gdma_cmd_addr,
        void *hau_cmd_addr,
        void *sdma_cmd_addr,
        void *imm_buf_addr,
        void *pio_addr,
        void *pio_addr_o,
        void *input_data,
        void *output_data,
        void *io_data,
        unsigned long long input_origin_addr,
        unsigned long long output_origin_addr,
        unsigned long long io_origin_addr,
        int tpu_cmd_nums,
        int gdma_cmd_nums,
        int hau_cmd_nums,
        int sdma_cmd_nums,
        unsigned int tpu_cmd_byte_size,
        unsigned int gdma_cmd_byte_size,
        unsigned int hau_cmd_byte_size,
        unsigned int sdma_cmd_byte_size,
        unsigned int imm_buf_byte_size,
        unsigned int pio_byte_size,
        unsigned int input_byte_size,
        unsigned int output_byte_size,
        unsigned int io_byte_size,
        int output_offset,
        int loop,
        int enable_pio_des_interleave);

    tpudnnStatus_t tpudnnLlamaMultiCore(
        tpudnnHandle_t handle,
        int loop);
    tpudnnStatus_t tpudnnCdmaMsgCentralTestMultiCore(
        tpudnnHandle_t handle);
    tpudnnStatus_t tpudnnMsgCentralMultiCore(
        tpudnnHandle_t handle,
        void *blob_A_0,
        void *blob_B_0,
        void *blob_T_0,
        void *blob_A_1,
        void *blob_B_1,
        void *blob_T_1,
        int op,
        int n,
        int c,
        int h,
        int w,
        int test_core_idx0,
        int test_core_idx1);

    tpudnnStatus_t tpudnnMgmMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *weight0,
        void *weight1,
        void *bias0,
        void *bias1,
        void *output,
        const int *in_shape,
        const int *w0_shape,
        const int *w1_shape,
        int in_dims,
        int w0_dims,
        int w1_dims,
        sg_data_type_t in_dtype,
        sg_data_type_t out_dtype,
        int has_bias,
        bool use_fast);

    tpudnnStatus_t tpudnnMgmBackwardMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *weight0,
        void *weight1,
        void *mat0,
        void *gelu,
        void *grad_output,
        void *grad_weight0,
        void *grad_weight1,
        void *grad_bias0,
        void *grad_bias1,
        void *grad_mat0,
        void *grad_input,
        const int *in_shape,
        const int *w0_shape,
        const int *w1_shape,
        int in_dims,
        int w0_dims,
        int w1_dims,
        int has_bias,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnMlp0FuseMultiCore(
        tpudnnHandle_t handle,
        void *input0,
        void *input1,
        void *gamma,
        void *beta,
        void *weight,
        void *bias,
        void *output,
        void *gelu_output,
        void *norm_output,
        void *norm_mean,
        void *norm_rstd,
        const int *in_shape,
        const int *w_shape,
        int in_dims,
        int w_dims,
        sg_data_type_t dtype,
        float eps,
        int has_bias,
        bool use_fast);

    tpudnnStatus_t tpudnnLayernormMatmulFuseMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *gamma,
        void *beta,
        void *weight,
        void *bias,
        void *output,
        void *norm_mean,
        void *norm_rstd,
        const int *in_shape,
        const int *w_shape,
        int in_dims,
        int w_dims,
        sg_data_type_t dtype,
        float eps,
        int has_bias);

    tpudnnStatus_t tpudnnSoftmaxWhereBackwardFuseMultiCore(
        tpudnnHandle_t handle,
        void *grad_output,
        void *softmax_output,
        void *cond,
        void *grad_input,
        const int *in_shape,
        const int *cond_shape,
        int in_dims,
        sg_data_type_t dtype,
        float value);

    tpudnnStatus_t tpudnnWhereMultiCore(
        tpudnnHandle_t handle,
        void *output,
        void *cond,
        void *self,
        void *other,
        float self_val,
        float other_val,
        const int *out_shape,
        const int *cond_shape,
        const int *self_shape,
        const int *other_shape,
        int dim,
        bool self_is_scalar,
        bool other_is_scalar,
        sg_data_type_t cond_dtype,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnCrossEntropyMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *target,
        void *weight,
        void *output,
        void *sum,
        void *max,
        int ignore_index,
        int batch_num,
        int class_num,
        int reduction,
        float label_smoothing,
        sg_data_type_t dtype,
        bool target_is_int64);

    tpudnnStatus_t tpudnnCrossEntropyBackwardMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *target,
        void *weight,
        void *grad_output,
        void *sum,
        void *max,
        void *grad_input,
        int batch_num,
        int class_num,
        int reduction,
        float label_smoothing,
        sg_data_type_t dtype,
        bool target_is_int64);

    tpudnnStatus_t tpudnnMatmulBackwardMultiCore(
        tpudnnHandle_t handle,
        void *left,
        void *right,
        void *grad_out,
        void *grad_left,
        void *grad_right,
        const int *L_shape,        // left && grad_left shape
        const int *R_shape,        // right && grad_right shape
        const int *Y_shape,        // out && grad_out shape
        const int L_dims,          // left && grad_left dims
        const int R_dims,          // right && grad_right dims
        const int Y_dims,          // out && grad_out dims
        sg_data_type_t in_dtype,   // fwd l&r data type
        sg_data_type_t out_dtype); // fwd out data type

    tpudnnStatus_t tpudnnGptQkvMultiCore(
        tpudnnHandle_t handle,
        void *Q,
        void *K,
        void *V,
        void *Y,
        void *where_cond,
        float C,
        float where_other_val,
        const int batch,
        const int N,
        const int d,
        sg_data_type_t dtype,
        sg_data_type_t where_cond_dtype);

    tpudnnStatus_t tpudnnGeluForwardMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        const int *shape,
        int dims,
        const float *coeff,
        sg_data_type_t dtype,
        int input_slice_dim,
        int output_slice_dim,
        int input_8ch_buf_size,
        int output_8ch_buf_size,
        char *input_8ch_buf[8],
        char *output_8ch_buf[8]);

    tpudnnStatus_t tpudnnPerfMultiCore(
            tpudnnHandle_t handle,
            void *param,
            size_t size,
            uint32_t &time);

    tpudnnStatus_t tpudnnAttention(
        tpudnnHandle_t        handle,
        void *input,
        void *keys,
        void *values,
        void *weight,
        void *bias,
        void *weight_o,
        void *mask,
        void *Y,
        int batch_num,
        int M_queries_num,
        int M_keys_num,
        int N_num,
        int dim,
        int hasbias,
        float scale,
        int has_mask,
        bool has_keys,
        bool has_value,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnGeluBackwardMultiCore(
        tpudnnHandle_t handle,
        void *grad_input,
        void *grad_output,
        void *input,
        const int *shape,
        const int dims,
        sg_data_type_t dtype,
        int input_slice_dim,
        int output_slice_dim,
        int input_8ch_buf_size,
        int output_8ch_buf_size,
        char *grad_input_8ch_buf[8],
        char *grad_output_8ch_buf[8],
        char *input_output_8ch_buf[8]);

    tpudnnStatus_t tpudnnSoftmaxForwardMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        const int *shape,
        const int dims,
        int begin_dim,
        int end_dim,
        float scale_val,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnSoftmaxBackwardMultiCore(
        tpudnnHandle_t handle,
        void *grad_input,
        void *grad_output,
        void *output,
        int n,
        int c,
        int h,
        int w,
        int axis,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnLayernormForwardMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *weight,
        void *bias,
        void *output,
        void *mean,
        void *rstd,
        int *shape,
        int dims,
        int axis,
        float eps,
        int affine,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnLayernormBackwardMultiCore(
        tpudnnHandle_t handle,
        void *grad_input,
        void *grad_weight,
        void *grad_bias,
        void *grad_output,
        void *input,
        void *weight,
        void *mean,
        void *rstd,
        int *shape,
        int dims,
        int axis,
        int affine,
        int requires_grad_input,
        sg_data_type_t dtype,
        int input_slice_dim,
        int output_slice_dim,
        int input_8ch_buf_size,
        int output_8ch_buf_size,
        char *grad_input_8ch_buf[8],
        char *grad_output_8ch_buf[8],
        char *input_8ch_buf[8]);

    tpudnnStatus_t tpudnnAdamBackwardMultiCore(
        tpudnnHandle_t handle,
        void *weight_out,
        void *m_out,
        void *v_out,
        void *vmax_out,
        void *grad_weight,
        void *weight_in,
        void *m_in,
        void *v_in,
        void *vmax_in,
        void *t,
        float lr,
        float beta1,
        float beta2,
        float eps,
        float weight_decay,
        bool amsgrad,
        bool maximize,
        int *shape,
        int dims,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnLlamaMlpForwardMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *weight0,
        void *weight1,
        void *weight2,
        void *output,
        int batch,
        int input_w,
        int middle_w,
        sg_data_type_t dtype,
        const int enable_8ch,
        const int input_slice_dim,
        const int weight0_slice_dim,
        const int weight1_slice_dim,
        const int weight2_slice_dim,
        const int output_slice_dim,
        const int input_buffer_size,
        const int weight0_buffer_size,
        const int weight1_buffer_size,
        const int weight2_buffer_size,
        const int output_buffer_size,
        char **input_8ch_buffer,
        char **weight0_8ch_buffer,
        char **weight1_8ch_buffer,
        char **weight2_8ch_buffer,
        char **output_8ch_buffer);

    tpudnnStatus_t tpudnnLlamaMlpW8A8ForwardMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *weight0,
        void *weight1,
        void *weight2,
        void *rescale0,
        void *rescale1,
        void *rescale2,
        void *output,
        int batch,
        int input_w,
        int middle_w,
        float scale,
        bool do_rescale,
        bool rescale_is_const,
        sg_data_type_t dtype
    );

    tpudnnStatus_t tpudnnBinaryFloatMultiCore(
        tpudnnHandle_t handle,
        void *input_A,
        void *input_B,
        void *output,
        const int *A_shape,
        const int *B_shape,
        int A_dim,
        int B_dim,
        int binary_type,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnBinaryBackwardMultiCore(
        tpudnnHandle_t handle,
        void *input_A,
        void *input_B,
        void *grad_output,
        void *grad_input_A,
        void *grad_input_B,
        const int *A_shape,
        const int *B_shape,
        int A_dim,
        int B_dim,
        int binary_type,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnConstBinaryFloatMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        const int *shape,
        int dims,
        sg_data_type_t dtype,
        sg_binary_type_t binary_type,
        float const_value,
        int is_inversed);

    tpudnnStatus_t tpudnnConstBinaryFloatBackwardMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *grad_output,
        void *grad_input,
        const int *shape,
        int dims,
        sg_data_type_t dtype,
        sg_binary_type_t binary_type,
        float const_value,
        int is_inversed);

    tpudnnStatus_t tpudnnDropoutMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        void *mask,
        const int *shape,
        const int dims,
        const float drop_rate,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnMatmulAllReduceMultiCore(
        tpudnnHandle_t handle,
        void *left,
        void *right,
        void *output,
        int op_code,
        const int *L_shape,
        const int *R_shape,
        const int L_dims,
        const int R_dims,
        const int L_trans,
        const int R_trans,
        sg_data_type_t in_dtype,
        sg_data_type_t out_dtype);

    tpudnnStatus_t tpudnnLlama2QkvMultiCore(
        tpudnnHandle_t handle,
        void *Q,
        void *K,
        void *V,
        void *Q_buffer,
        void *K_buffer,
        void *V_buffer,
        void *Kcache,
        void *Vcache,
        void *RoPE_cos,
        void *RoPE_sin,
        void *Mask,
        void *Y,
        void *Input_length,
        void *Cache_length,
        void *Save_slots,
        void *block_tables,
        int slots_size,
        int fetch_slots_size,
        float C,
        const int batch,
        const int Mask_max,
        const int hidden_size,
        const int num_attention_heads,
        const int num_k_v_heads,
        const int embeddings,
        const int attention_mode,
        const int block_size,
        const int max_blocks,
        sg_data_type_t dtype,
        int qkv_packed,
        int page_kv_cache_layout);

    tpudnnStatus_t tpudnnLlamaAtteionForwardMultiCore(
        tpudnnHandle_t handle,
        void *Q,
        void *K,
        void *V,
        void *RoPE_cos,
        void *RoPE_sin,
        void *Mask,
        void *Y,
        void *Input_length,
        void *softmax_lse,
        float C,
        float dropout_rate,
        const int batch,
        const int Mask_max,
        const int hidden_size,
        const int num_attention_heads,
        const int num_k_v_heads,
        const int mask_batch,
        sg_data_type_t dtype,
        int qkv_packed);

    typedef struct
    {
        unsigned long long gdma_src_offset;
        unsigned long long gdma_dst_offset;
        unsigned long long gdma_reduce_src_offset[8];
        unsigned long long gdma_reduce_dst_offset[8];
        unsigned long long sdma_src_offset[8];
        unsigned long long sdma_dst_offset[8];
        unsigned long long cdma_src_offset[8];
        unsigned long long cdma_dst_offset[8];
        unsigned int gdma_shape[4];
        unsigned int gdma_reduce_shape[4];
        unsigned int sdma_shape[4];
        unsigned int sdma_reduce_shape[4];
        unsigned int cdma_shape[4];
        sg_data_type_t gdma_sg_dtype;
        sg_data_type_t gdma_reduce_sg_dtype;
        sg_data_type_t sdma_sg_dtype;
        sg_data_type_t sdma_reduce_sg_dtype;
        sg_data_type_t cdma_sg_dtype;
        sg_reduce_method_t gdma_sg_reduce_method;
        sg_reduce_method_t sdma_sg_reduce_method;
    } dma_k2k_stress_multi_core_param_t;

    tpudnnStatus_t tpudnnDmaK2kStressMultiCore(
        tpudnnHandle_t handle,
        dma_k2k_stress_multi_core_param_t dma_k2k_stress_multi_core_param);

    tpudnnStatus_t tpuDNNBatchMatmul(
        tpudnnHandle_t handle,
        void * L_addr,
        void * R_addr,
        void * B_addr,
        void * Y_addr,
        int batch_num,
        int L_row_num,
        int L_col_num,
        int R_col_num,
        sg_data_type_t L_dtype,
        sg_data_type_t R_dtype,
        sg_data_type_t Y_dtype,
        int if_relu,
        float relu_upper_limit,
        int use_bias,
        int do_rescale,
        int rescale_const_val);

    tpudnnStatus_t tpuDNNBatchMatmulFix8b(
        tpudnnHandle_t handle,
        void * L_addr,
        void * R_addr,
        void * zp_addr,
        void * Y_addr,
        int batch_num,
        int L_row_num,
        int L_col_num,
        int R_col_num,
        sg_data_type_t L_dtype,
        sg_data_type_t R_dtype,
        int zp_is_const,
        int zp_const_val);

    tpudnnStatus_t tpuDNNBatchMatmulFix8bExt(
        tpudnnHandle_t handle,
        void * L_mem,
        void * R_mem,
        void * rzp_mem,
        void * bias_mem,
        void * requant_mem,
        void * Y_mem,
        int batch_num,
        int hsize,
        int L_row_num,
        int L_col_num,
        sg_data_type_t L_dtype,
        int R_row_num,
        int R_col_num,
        sg_data_type_t R_dtype,
        int L_trans,
        int R_trans,
        int izp_const_val,
        sg_data_type_t rzp_dtype,
        int rzp_is_const,
        int rzp_const_val,
        sg_data_type_t bias_dtype,
        int bias_is_const,
        int bias_const_val,
        int do_relu,
        int requant_mode,
        int is_perchannel,
        int scale_val,
        int offset_val,
        int shift_val,
        sg_round_mode_t round_mode,
        int do_sym_saturate,
        sg_data_type_t Y_dtype);

    tpudnnStatus_t tpuDNNBcBinaryFloat(
        tpudnnHandle_t handle,
        void * input_A,
        void * input_B,
        void * output,
        const int* A_shape,
        const int* B_shape,
        int shape_dim,
        sg_data_type_t dtype,
        sg_binary_type_t binary_type);

    tpudnnStatus_t tpuDNNBcBinaryFix8b(
        tpudnnHandle_t handle,
        void * input_A,
        void * input_B,
        void * output,
        const int* A_shape,
        const int* B_shape,
        int shape_dim,
        int scale_A,
        int scale_B,
        int rshift_A,
        int rshift_B,
        sg_data_type_t A_dtype,
        sg_data_type_t B_dtype,
        sg_data_type_t res_dtype,
        sg_binary_type_t binary_type);

    tpudnnStatus_t tpudnnIndexPut(
        tpudnnHandle_t      handle,
        void *  input,
        void *  index,
        void *  value,
        void *  output,
        void *  buffer,
        const int        shape[FW_MAX_SHAPE_DIMS],
        int              dims,
        int              indice_len,
        int              mode,
        int              accumulate,
        sg_data_type_t   dtype);

    tpudnnStatus_t tpudnnIndexSelect(
        tpudnnHandle_t         handle,
        void *     input,
        void *     index,
        const int          *input_shape,
        int                 shape_dims,
        int                 index_num,
        int                 axis, // axis to do index_select
        int                 const_val, // fill_value if index not found in input
        sg_data_type_t      dtype,
        void *     output);

    tpudnnStatus_t tpudnnLLama2Attention(
        tpudnnHandle_t handle,
        void * Q,
        void * K,
        void * V,
        void * Kcache,
        void * Vcache,
        void * Mask,
        void * Y,
        void * Input_length,
        void * Save_slots,
        void * Block_tables,
        int slots_size,
        float C,
        const int batch,
        const int mask_max,
        const int head_size,
        const int num_attention_heads,
        const int num_k_v_heads,
        const int attention_mode,
        const int block_size,
        const int block_num,
        const int tokens_num,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnMaskedFill(
        tpudnnHandle_t      handle,
        void *  input,
        void *  mask,
        void *  output,
        const int*       input_shape,
        const int*       mask_shape,
        int              input_dims,
        int              mask_dims,
        float            value,
        sg_data_type_t   dtype);

    tpudnnStatus_t tpudnnMaskedSelect(
        tpudnnHandle_t      handle,
        void *  input,
        void *  mask,
        void *  output,
        void *  buffer,
        void *  mask_count,
        const int       *input_shape,
        const int       *mask_shape,
        int              input_dims,
        int              mask_dims,
        bool             bcast_from_begin,
        sg_data_type_t   input_dtype,
        sg_data_type_t   mask_dtype);

    tpudnnStatus_t tpudnnMemset(
        tpudnnHandle_t handle,
        void * output,
        unsigned int height,
        unsigned int width,
        int mode,
        int val);

    tpudnnStatus_t tpudnnMsgSyncDebug(
        tpudnnHandle_t        handle,
        void *    input,
        void *    weight,
        void *    output);

    tpudnnStatus_t tpudnnPixelNorm(
        tpudnnHandle_t handle,
        void * input,
        void * weight,
        void * bias,
        void * output,
        const int *shape,
        int dims,
        float eps,
        int affine,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnPixelNormFix8b(
        tpudnnHandle_t handle,
        void * input,
        void * weight,
        void * bias,
        void * output,
        const int *shape,
        int dims,
        float scale,
        float eps,
        int affine,
        sg_data_type_t idtype,
        sg_data_type_t odtype);

    tpudnnStatus_t tpudnnPrelu(
        tpudnnHandle_t         handle,
        // input
        void *     input,
        void *     slope,
        int                 input_n,
        int                 input_c,
        int                 input_h,
        int                 input_w,
        int                 channel_shared,
        float               slope_value,
        float               upper_limit,
        sg_data_type_t      dtype,
        // output
        void *     output);

    tpudnnStatus_t tpudnnPreluFix8b(
        tpudnnHandle_t         handle,
        // input
        void *     input,
        void *     slope,
        int                 input_n,
        int                 input_c,
        int                 input_h,
        int                 input_w,
        int                 channel_shared,
        int                 slope_value,
        int                 rshift_bit,
        int                 upper_limit,
        sg_data_type_t      dtype,
        // output
        void *     output);

    tpudnnStatus_t tpudnnPriorBox(
        tpudnnHandle_t      handle,
        void *  output,
        void *  buffer,
        const float*     mins,
        const float*     maxs,
        const float*     asps,
        const float*     vars,
        int              min_size,
        int              max_size,
        int              asp_size,
        int              var_size,
        float            step_h,
        float            step_w,
        int              img_h,
        int              img_w,
        int              fmp_h,
        int              fmp_w,
        int              num_priors,
        float            offset,
        bool             clip,
        float            thTop,
        sg_data_type_t   odtype);

    tpudnnStatus_t tpudnnProposalLayer(
        tpudnnHandle_t handle,
        void * score,
        void * box,
        void * src_info,
        int feat_stride,
        int min_size,
        int pre_nms_topN,
        int post_nms_topN,
        float nms_thresh,
        float score_thresh,
        int base_size,
        int scales_num,
        int ratios_num,
        int *anchor_scales,
        float *ratios,
        int batch_num,
        int map_height,
        int map_width,
        int distribute_fpn_proposals_flag,
        int score_out_flag,
        void * output,
        void * score_out);

    tpudnnStatus_t tpudnnQuantDiv(
        tpudnnHandle_t         handle,
        void *     input,
        void *     output,
        unsigned long long  length,
        int                 divisor,
        int                 quant_mode,
        sg_data_type_t      in_dtype,
        sg_data_type_t      out_dtype);

    tpudnnStatus_t tpudnnReduce(
        tpudnnHandle_t         handle,
        void *     input,
        void *     buffer,
        const int          *input_shape,
        const int          *axis_list,
        int                 shape_dims,
        int                 axis_num,
        int                 method,
        sg_data_type_t      dtype,
        void *     output);

    tpudnnStatus_t tpudnnReduceFix8b(
        tpudnnHandle_t         handle,
        void *     input,
        void *     buffer,
        const int          *input_shape,
        const int          *axis_list,
        int                 shape_dims,
        int                 axis_num,
        int                 method,
        float               input_scale,
        float               output_scale,
        sg_data_type_t      input_dtype,
        sg_data_type_t      output_dtype,
        void *     output);

    tpudnnStatus_t tpudnnRelativePositionEncoding(
        tpudnnHandle_t     handle,
        void * input_data,
        void * seq_len,
        void * output_data,
        const int*      shape,
        int             dims,
        sg_data_type_t  dtype);

    tpudnnStatus_t tpudnnRelu(
        tpudnnHandle_t         handle,
        // input
        void *     input,
        int                 input_n,
        int                 input_c,
        int                 input_h,
        int                 input_w,
        float               upper_limit,
        sg_data_type_t      dtype,
        // output
        void *     output,
        bool       use_local_test = false);

    tpudnnStatus_t tpudnnRequantFloat(
        tpudnnHandle_t      handle,
        void *  input,
        void *  output,
        void *  requant,
        int              N,
        int              C,
        int              H,
        int              W,
        bool             is_perchannel,
        float            scale_val,
        float            offset_val,
        sg_data_type_t   bottom_dtype,
        sg_data_type_t   top_dtype,
        int              mode);

    tpudnnStatus_t tpudnnRequantInt(
        tpudnnHandle_t      handle,
        void *  input,
        void *  output,
        void *  requant,
        int              N,
        int              C,
        int              H,
        int              W,
        bool             is_perchannel,
        int              scale_val,
        int              shift_val,
        int              offset_val,
        sg_data_type_t   bottom_dtype,
        sg_data_type_t   top_dtype,
        int              mode);

    tpudnnStatus_t tpudnnReverse(
        tpudnnHandle_t         handle,
        void *     input,
        void *     output,
        const int*          shape,
        int                 dims,
        sg_data_type_t      dtype,
        int                 axis);

    tpudnnStatus_t tpudnnRoiAlignForward(
        tpudnnHandle_t         handle,
        // input
        void *     input,
        void *     rois,
        int                 input_n,
        int                 input_c,
        int                 input_h,
        int                 input_w,
        int                 roi_num,
        int                 roi_len,
        int                 pooled_height,
        int                 pooled_width,
        float               spatial_scale,
        int                 sampling_ratio,
        int                 position_sensitive,
        int                 align_corners,
        int                 plat_sp,
        sg_data_type_t      dtype,
        // output
        void *     output);

    tpudnnStatus_t tpudnnRoiExtractorForward(
        tpudnnHandle_t         handle,
        // input
        void **    inputs,
        void *     rois,
        void *     target_lvls,
        int                (*input_shapes)[FW_MAX_SHAPE_DIMS],
        int                 num_levels,
        int                 input_dims,
        int                 roi_num,
        int                 roi_len,
        int                 pooled_height,
        int                 pooled_width,
        float*              spatial_scales,
        int                 sampling_ratio,
        int                 position_sensitive,
        int                 align_corners,
        int                 plat_sp,
        sg_data_type_t      dtype,
        // output
        void *     output);

    tpudnnStatus_t tpudnnRoiPooling(
        tpudnnHandle_t         handle,
        void *     input,
        void *     roi,
        int                 input_n,
        int                 input_c,
        int                 input_h,
        int                 input_w,
        int                 roi_num,
        int                 roi_len,
        int                 pooled_h,
        int                 pooled_w,
        float               spatial_scale,
        int                 position_sensitive,
        sg_data_type_t      dtype,
        void *     output);

    tpudnnStatus_t tpudnnRoundFp(
        tpudnnHandle_t handle,
        void * input,
        void * output,
        const int* shape,
        int shape_dim,
        sg_round_mode_t round_mode,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnConstantFill(
        tpudnnHandle_t         handle,
        void *     output,
        const int*          shape,
        int                 shape_dim,
        unsigned int        filled_value,
        sg_data_type_t      filled_sgdtype);

    tpudnnStatus_t tpudnnConstBinaryFix8b(
        tpudnnHandle_t handle,
        void * input,
        void * output,
        const int* shape,
        int shape_dim,
        int scale_A,
        int rshift_A,
        int B_const_val,
        int inversed,
        sg_data_type_t A_dtype,
        sg_data_type_t B_dtype,
        sg_data_type_t res_dtype,
        sg_binary_type_t binary_type);

    tpudnnStatus_t tpudnnConstBinaryFloat(
        tpudnnHandle_t handle,
        void * input,
        void * output,
        const int* shape,
        int dims,
        sg_data_type_t dtype,
        sg_binary_type_t binary_type,
        float const_value,
        int is_inversed
        );

    tpudnnStatus_t tpudnnConvBwFloat(
        tpudnnHandle_t      handle,
        void *  input,
        void *  grad_output,
        void *  padding_insert,
        void *  kernel_grad,
        int              n,
        int              ic,
        int              ih,
        int              iw,
        int              oc,
        int              oh,
        int              ow,
        int              kh,
        int              kw,
        int              ins_h,
        int              ins_w,
        int              dh,
        int              dw,
        int              stride_h,
        int              stride_w,
        int              pad_h_t,
        int              pad_h_b,
        int              pad_w_l,
        int              pad_w_r,
        unsigned int     ins_const_val,
        int              pad_ins_is_const,
        int              pad_mode,
        sg_data_type_t   input_dtype,
        sg_data_type_t   grad_dtype,
        sg_data_type_t   res_dtype,
        unsigned long long   osz_kgrad);

    tpudnnStatus_t tpudnnConvbwdMultiCore(
        tpudnnHandle_t      handle,
        void *  gradout,
        void *  input,
        void *  weight,
        void *  grad_input,
        void *  grad_weight,
        void *  grad_bias,
        void *  buffer,
        int              n,
        int              ic,
        int              ih,
        int              iw,
        int              oc,
        int              oh,
        int              ow,
        int              kh,
        int              kw,
        int              ins_h,
        int              ins_w,
        int              dh,
        int              dw,
        int              stride_h,
        int              stride_w,
        int              pad_h_t,
        int              pad_h_b,
        int              pad_w_l,
        int              pad_w_r,
        int              groups,
        unsigned int     ins_const_val,
        int              pad_ins_is_const,
        int              pad_mode,
        int              grad_input_enable,
        int              grad_weight_enable,
        int              grad_bias_enable,
        sg_data_type_t   input_dtype,
        sg_data_type_t   grad_dtype,
        sg_data_type_t   res_dtype);

    tpudnnStatus_t tpudnnSoftNms(tpudnnHandle_t     handle,
        void * input_proposal_addr,
        void * output_proposal_addr,
        void * all_mask_buf,
        int             nms_type,
        int             proposal_size,
        float           nms_threshold,
        float           score_threshold,
        float           sigma,
        int             weighting_method,
        float *         densities,
        float           eta);

    tpudnnStatus_t tpudnnNms(tpudnnHandle_t     handle,
        void * input_proposal_addr,
        void * score_addr,//just used at hard-nms V2
        void * output_proposal_addr,
        void * iou_buff,
        void * all_mask_buf,
        int             nms_type,
        int             proposal_size,
        float           nms_threshold,
        float           score_threshold,
        float           sigma,
        int             weighting_method,
        float *         densities,
        float           eta,
        int             hard_nms_version,
        int             keep_top_k);

    tpudnnStatus_t tpudnnOnnxNms(tpudnnHandle_t     handle,
        void * box_addr,
        void * score_addr,
        void * output_addr,
        void * buffer,
        int             batch_num,
        int             num_priors,
        int             num_classes,
        float           nms_threshold,
        float           score_threshold,
        int             top_k,
        int             center_point_box);

    tpudnnStatus_t tpudnnPowerStress(tpudnnHandle_t handle,
        void *input_ptrs[5],
        void *output_ptrs[5],
        void *buffer_ptr,
        void *weight_fp32_ptr,
        void *weight_fp16_ptr,
        void *weight_int8_ptr,
        int input_shapes[5][FW_MAX_SHAPE_DIMS],
        int output_shapes[5][FW_MAX_SHAPE_DIMS],
        sg_data_type_t dtypes[5],
        int buffer_size,
        int used_engine_num,
        int dims,
        int weight_element,
        int use_core_num,
        int device_loop_times,
        const char* api_name);

    tpudnnStatus_t tpudnnDDRStress(
            tpudnnHandle_t      handle,
            uint64_t            start_offset,
            uint64_t            stride,
            uint64_t            data_len,
            uint64_t            end_offset);

    tpudnnStatus_t tpudnnRotatedNms(
        tpudnnHandle_t handle,
        const float* input_dets_addr,
        const float* input_scores_addr,
        int* output_indx_addr,
        int dets_num,
        int dets_dim,
        float iou_threshold);

    tpudnnStatus_t tpudnnRope(
        tpudnnHandle_t handle,
        void * input_global_addr,
        void * output_global_addr,
        void * input_weight0_addr,
        void * input_weight1_addr,
        int shape_dim,
        int * shape,
        int * W_shape,
        sg_data_type_t      dtype
        );

    tpudnnStatus_t tpudnnQrHouseHolder(
        tpudnnHandle_t handle,
        void* spectral_embeddings_global_addr,
        void* num_spks_global_addr,
        void* eignvalue_global_addr,
        void* eignvector_global_addr,
        void* input_global_addr,
        void* buffer_global_addr,
        const int* shape,
        int   shape_dim,
        int   num_iter_QR,
        int   buffer_coeff,
        int   num_spks,
        int   max_num_spks,
        int   min_num_spks,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnKnnNaive(
        tpudnnHandle_t handle,
        void* centroids_global_addr,
        void* labels_global_addr,
        void* input_global_addr,
        void* weight_global_addr,
        void* buffer_global_addr,
        const int* Shape_Input,
        const int* Shape_Weight,
        int   dims_Input,
        int   dims_Weight,
        int   n_feat,
        int   k,
        int   num_iter,
        int   buffer_coeff,
        unsigned    buffer_max_cnt,
        sg_data_type_t dtype);

    tpudnnStatus_t tpudnnMaxpoolingWithMaskForwardMultiCore(
        tpudnnHandle_t      handle,
        void                *input,
        int                 input_n,
        int                 input_c,
        int                 input_h,
        int                 input_w,
        int                 output_h,
        int                 output_w,
        int                 kh,
        int                 kw,
        int                 pad_h,
        int                 pad_w,
        int                 pad_h_after,
        int                 pad_w_after,
        int                 stride_h,
        int                 stride_w,
        int                 dilation_h,
        int                 dilation_w,
        int                 is_avg_pooling,
        int                 avg_pooling_mode,
        int                 if_relu,
        float               relu_upper_limit,
        sg_data_type_t      dtype,
        void                *output,
        void                *max_mask_addr
    );

    tpudnnStatus_t tpudnnMaxpoolinIndicesBwd(
        tpudnnHandle_t handle,
        // input
        void           *grad,
        void           *indices,
        int            n,
        int            c,
        int            h,
        int            w,
        int            hout,
        int            wout,
        int            kernel,
        int            stride,
        int            padding,
        sg_data_type_t dtype,
        // output
        void           *output
    );

    tpudnnStatus_t tpudnnBatchNormTrain(
        tpudnnHandle_t handle,
        // input
        void           *input,
        void           *running_mean,
        void           *running_var,
        void           *weight,
        void           *bias,
        void           *output,
        void           *saved_mean,
        void           *saved_invstd,
        void           *running_mean_update,
        void           *running_var_update,
        const int *    *origin_shape,
        float          momentum,
        float          eps,
        int            if_relu,
        sg_data_type_t dtype
    );

    tpudnnStatus_t tpudnnTestStream(
        tpudnnHandle_t handle,
        void *input_A,
        void *input_B,
        void *output,
        const int *shape,
        int dims,
        int rounds);

    tpudnnStatus_t tpudnnTestLaunchLatency(
        tpudnnHandle_t handle,
        size_t launchNum);

    tpudnnStatus_t tpudnnReadTickTockLatency(tpudnnHandle_t handle);

    tpudnnStatus_t tpudnnTestRegLatency(tpudnnHandle_t handle);

    tpudnnStatus_t tpudnnInstructionPower(
        tpudnnHandle_t handle,
        int loop_times,
        InsType type,
        tpudnnDataType_t dtype,
        InsMode mode,
        int core_num,
        int use_multi_engine,
        int idle_max_interleave,
        void *input_data,
        void *output_data,
        size_t count);

    tpudnnStatus_t tpudnnSwapDim(
        tpudnnHandle_t      handle,
        // input
        void*               input,
        int                 input_n,
        int                 input_c,
        int                 input_h,
        int                 input_w,
        int                 axis_num,
        int                 axis_list[FW_MAX_SHAPE_DIMS],
        int                 offset_list[FW_MAX_SHAPE_DIMS],
        sg_data_type_t      dtype,
        // output
        void*               output);

    tpudnnStatus_t tpudnnTestEvent(
        tpudnnHandle_t handle,
        int rounds,
        size_t size);

    tpudnnStatus_t tpudnnPoolingFp8(
        tpudnnHandle_t handle,
        // input
        void *input,
        int input_n,
        int input_c,
        int input_h,
        int input_w,
        int output_h,
        int output_w,
        int kh,
        int kw,
        int pad_h,
        int pad_w,
        int pad_h_after,
        int pad_w_after,
        int stride_h,
        int stride_w,
        int dilation_h,
        int dilation_w,
        int is_avg_pooling,
        int avg_pooling_mode,
        float re_scale,
        sg_data_type_t output_dtype,
        sg_data_type_t input_dtype,
        // output
        void *output);

    tpudnnStatus_t tpudnnSdmaMultiThread(
        tpudnnHandle_t handle,
        void *input_data,
        void *output_data,
        void *buffer,
        unsigned int element_num,
        sg_data_type_t dtype,
        unsigned int case_);

    tpudnnStatus_t tpudnnLossyCompress(
        tpudnnHandle_t      handle,
        void * input0_addr,
        void * input1_addr,
        void * output_addr,
        void * buffer_addr,
        int n,
        int c,
        int h,
        int w,
        int reduce_opcode);
    tpudnnStatus_t tpudnnL2mTest(
        tpudnnHandle_t      handle,
        void *  input,
        void *  output);
    tpudnnStatus_t tpudnnMsgCentralStressTest(
        tpudnnHandle_t         handle,
        int                 loop
        );
    tpudnnStatus_t tpudnnPipelineSdmaCwtrans(
            tpudnnHandle_t     handle,
            void * input_data,
            void * output_data,
            const int*      shape,
            int             dims,
            sg_data_type_t  dtype);
    tpudnnStatus_t tpudnnMultiSdmaCwtrans(
            tpudnnHandle_t     handle,
            void * input_data,
            void * output_data,
            const int*      shape,
            int             dims,
            sg_data_type_t  dtype);
    tpudnnStatus_t tpudnnMultiSdmaSparseAdd(
            tpudnnHandle_t     handle,
            void * input_A_data,
            void * input_B_data,
            void * mask,
            void * output_data,
            const int*      shape,
            int             dims,
            sg_data_type_t  dtype);

    tpudnnStatus_t testTpudnnCrossEntropyLossBackwardAsync(
            tpudnnHandle_t handle,
            void* input,
            void* target,
            void* grad_output,
            void* grad_input,
            int batch,
            int class_num,
            int reduction,
            float label_smoothing,
            sg_data_type_t dtype,
            bool target_is_int64);

    tpudnnStatus_t tpudnnLlamaMatmulFp8(
        tpudnnHandle_t handle ,
        void *left,
        void *right,
        void *bias,
        void *output,
        void *scale,
        void *zp,
        const int *L_shape,
        const int *R_shape,
        const int L_dims,
        const int R_dims,
        const int L_trans,
        const int R_trans,
        sg_data_type_t in_dtype,
        sg_data_type_t bias_dtype,
        sg_data_type_t out_dtype,
        const int q_group_size,
        const int weight_bits,
        const bool has_bias
    );

    tpudnnStatus_t tpudnnDeepSeekMlpForwardMultiCore(
        tpudnnHandle_t handle,
        void *input,
        void *gate_weight,
        void *up_weight,
        void *down_weight,
        void *gate_scale,
        void *up_scale,
        void *down_scale,
        void *output,
        int batch,
        int input_w,
        int middle_w,
        int blocksize,
        int input_dtype,
        int weight_dtype,
        int weight_scale_dtype,
        int quant);

    tpudnnStatus_t tpudnnFastExpBF16(
        tpudnnHandle_t handle,
        void *input,
        void *output,
        const int* shape);

#ifdef USING_PLD_TEST
    tpudnnStatus_t tpudnnPldTest(
        tpudnnHandle_t        handle,
        void *    src,
        void *    dst,
        int       size_in,
        int       size_o,
        pld_test_id_t      id,
        unsigned long long *src_addr = NULL,
        unsigned long long *dst_addr = NULL);

    tpudnnStatus_t tpudnnPldTestGeneral(
        tpudnnHandle_t handle,
        const void *api,
        unsigned long api_size);

    tpudnnStatus_t tpudnnPldTestSysPower(
        tpudnnHandle_t        handle,
        void*    input,
        int      g_sh[4],
        int      t0_sh[4],
        int      t1_sh[4],
        int      res_sh[4],
        int      m_size,
        int       dtype,
        int        loops,
        int       cmd_id);
    tpudnnStatus_t tpudnnPldSendInstruction(
        tpudnnHandle_t        handle,
        void *             src,
        void *             dst,
        int                size_in,
        int                size_out,
        pld_test_id_t      id,
        int                loops,
        int                N,
        int                C,
        int                H,
        int                W,
        unsigned long long *input_global_addr = NULL,
        unsigned long long *output_global_addr = NULL,
        int core_num  = 1);

    tpudnnStatus_t tpudnnPldCdma(
        tpudnnHandle_t     handle,
        void *             src,
        void *             dst,
        pld_test_id_t      id,
        int                N,
        int                C,
        int                H,
        int                W,
        int                len,
        sg_data_type_t     data_type,
        int                reduce_op_index,
        int                chip_id,
        int                world_size,
        int*               chip_map,
        unsigned long long *input_global_addr,
        unsigned long long *output_global_addr);

    tpudnnStatus_t tpudnnPldCdmaMsg(
        tpudnnHandle_t     handle,
        void *             src,
        void *             index_list,
        pld_test_id_t      id,
        int                len,
        int                index_list_num,
        int                index_list_len,
        int                length_shift,
        sg_data_type_t     data_type,
        int                reduce_op_index,
        int                chip_id,
        int                world_size,
        int*               chip_map);

    tpudnnStatus_t tpudnnPldTestFusedCmp(
        tpudnnHandle_t        handle,
        void *    src,
        void *    dst0,
        void *    dst1,
        pld_test_id_t      id,
        int                src_size,
        int                dst0_size,
        int                dst1_size,
        int                N,
        int                C,
        int                H,
        int                W,
        int                a_is_sign,
        int                A_dtype,
        int                C_dtype);

#endif

}
