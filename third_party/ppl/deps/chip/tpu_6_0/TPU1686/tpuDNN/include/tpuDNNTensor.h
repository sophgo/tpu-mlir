#pragma once

#include <cstdint>
#include <cstddef>
#include<vector>
#include<string>
extern "C"
{

enum tpudnnDataType_t
{
    TPUDNN_DTYPE_FP32 = 0,
    TPUDNN_DTYPE_FP16 = 1,
    TPUDNN_DTYPE_INT8 = 2,
    TPUDNN_DTYPE_UINT8 = 3,
    TPUDNN_DTYPE_INT16 = 4,
    TPUDNN_DTYPE_UINT16 = 5,
    TPUDNN_DTYPE_INT32 = 6,
    TPUDNN_DTYPE_UINT32 = 7,
    TPUDNN_DTYPE_BF16 = 8,
    TPUDNN_DTYPE_INT4 = 9,
    TPUDNN_DTYPE_UINT4 = 10,
    TPUDNN_DTYPE_FP20 = 11,
    TPUDNN_DTYPE_FP8E5M2 = 12,
    TPUDNN_DTYPE_FP8E4M3 = 13,
    TPUDNN_DTYPE_INT64 = 14,
    TPUDNN_DTYPE_TF32 = 15,
    TPUDNN_DTYPE_BOOL = 16,

    TPUDNN_DTYPE_UNKNOWN = -1,
};

enum tpudnnReduceType_t {
    TPUDNN_REDUCE_MEAN = 0,
    TPUDNN_REDUCE_SUM  = 1,
    TPUDNN_REDUCE_MAX  = 2,
    TPUDNN_REDUCE_MIN  = 3,
    TPUDNN_REDUCE_PROD = 4,
    TPUDNN_REDUCE_L2   = 5,
    TPUDNN_REDUCE_L1   = 6,
};

typedef enum {
    TPUDNN_INTRA_CARD = 0,
    TPUDNN_INTER_CARD = 1,
    TPUDNN_INTER_CHIP = 2
} c2c_communication_type;

typedef enum {
    TPUDNN_DIRECT_LEFT2RIGHT = 0,
    TPUDNN_DIRECT_RIGHT2LEFT = 1,
    TPUDNN_DIRECT_BIDIR = 2,
} c2c_direct_type;

typedef enum {
    TPUDNN_COPY_S2S = 0,
    TPUDNN_COPY_S2L2 = 1,
    TPUDNN_COPY_L22S = 2,
    TPUDNN_COPY_L22L2 =3,
} c2c_copy_type;

typedef enum {
  TPUDNN_ACTIVE_TANH = 0,
  TPUDNN_ACTIVE_SIGMOID = 1,
  TPUDNN_ACTIVE_RELU = 2,
  TPUDNN_ACTIVE_EXP = 3,
  TPUDNN_ACTIVE_ELU = 4,
  TPUDNN_ACTIVE_SQRT = 5,
  TPUDNN_ACTIVE_SQUARE = 6,
  TPUDNN_ACTIVE_RSQRT = 7,
  TPUDNN_ACTIVE_ABSVAL = 8,
  TPUDNN_ACTIVE_LN = 9,
  TPUDNN_ACTIVE_ROUND = 10,
  TPUDNN_ACTIVE_CEIL = 11,
  TPUDNN_ACTIVE_FLOOR = 12,
  TPUDNN_ACTIVE_SIN = 13,
  TPUDNN_ACTIVE_COS = 14,
  TPUDNN_ACTIVE_IS_FINITE = 15,
  TPUDNN_ACTIVE_MISH = 16,
  TPUDNN_ACTIVE_SWISH = 17,
  TPUDNN_ACTIVE_HSWISH = 18,
  TPUDNN_ACTIVE_SILU = 19,
  TPUDNN_ACTIVE_ARCSIN = 20,
  TPUDNN_ACTIVE_ARCCOS = 21,
  TPUDNN_ACTIVE_ARCSINH = 22,
  TPUDNN_ACTIVE_ARCCOSH = 23,
  TPUDNN_ACTIVE_ARCTANH = 24,
  TPUDNN_ACTIVE_SINH = 25,
  TPUDNN_ACTIVE_COSH = 26,
  TPUDNN_ACTIVE_TAN = 27,
  TPUDNN_ACTIVE_SIGN = 28,
  TPUDNN_ACTIVE_GELU = 29,
  TPUDNN_ACTIVE_ERF = 30,
  TPUDNN_ACTIVE_HSIGMOID = 31,
  TPUDNN_ACTIVE_LOG_SIGMOID = 32,
  TPUDNN_ACTIVE_SOFT_PLUS = 33,
  TPUDNN_ACTIVE_SOFT_SIGN = 34,
  // only implemented in tpu-train
  TPUDNN_ACTIVE_ERFC = 35,
  TPUDNN_ACTIVE_ISINF = 36,
  TPUDNN_ACTIVE_ISNAN = 37,
  TPUDNN_ACTIVE_EXPM1 = 38,
  TPUDNN_ACTIVE_RECIPROCAL = 39,
  TPUDNN_ACTIVE_EXP2 = 40,
  TPUDNN_ACTIVE_TRUNC = 41,
} tensor_active_type_t;

typedef enum {
  TPUDNN_LOG_E = 0,
  TPUDNN_LOG_1P = 1,
  TPUDNN_LOG_2 = 2,
  TPUDNN_LOG_10 = 10,
} tensor_log_type_t;

typedef enum {
  TPUDNN_POOLING_MAX = 0,
  TPUDNN_POOLING_MIN = 1,
  TPUDNN_POOLING_AVG = 2,
} tensor_pooling_mode_t;

typedef enum {
  TPUDNN_UPSAMPLING_NEAREST = 0,
  TPUDNN_UPSAMPLING_BILINEAR = 1,
} tensor_resize_mode_t;
typedef struct
{
  int kernel_h;
  int kernel_w;
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int dilation_h;
  int dilation_w;
  int groups;
}tpudnnConv2dParam_t;

typedef enum
{
  TPUDNN_NO_FORMATED = 0,
  TPUDNN_CONV_W_INFER_FORMAT  = 1,
  TPUDNN_CONV_W_TRAIN_FORMAT  = 2,
  TPUDNN_CONV_DW_TRAIN_FORMAT = 3,
}
TpudnnFormatedType_t;

typedef struct
{
    void *addr;
    int dim;
    int shape[8];
    int stride[8];
    tpudnnDataType_t dtype;
    TpudnnFormatedType_t format_casted;
} tpudnnTensor_t;

typedef struct {
    void* cmd;
    unsigned cmd_num;
    unsigned cmd_size;
} tpudnnCmd_t;

typedef struct {
  int kh;
  int kw;
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int output_h;
  int output_w;
  tensor_pooling_mode_t mode;
} TPUDNN_PoolingDescriptor_t;

static inline size_t tpudnnTensorDataSize(tpudnnDataType_t dtype)
{
    if (dtype == TPUDNN_DTYPE_INT8 ||
        dtype == TPUDNN_DTYPE_UINT8)
    {
      return 1;
    }
    else if (dtype == TPUDNN_DTYPE_INT16 ||
             dtype == TPUDNN_DTYPE_UINT16 ||
             dtype == TPUDNN_DTYPE_FP16 ||
             dtype == TPUDNN_DTYPE_BF16)
    {
        return 2;
    }
    else if (dtype == TPUDNN_DTYPE_FP32 ||
             dtype == TPUDNN_DTYPE_INT32 ||
             dtype == TPUDNN_DTYPE_UINT32)
    {
        return 4;
    }
    else if ( dtype == TPUDNN_DTYPE_INT64 )
    {
        return 8;
    }
    return -1;
}

static inline size_t tpudnnTensorBytes(const tpudnnTensor_t *tensor)
{
    size_t bytes = tpudnnTensorDataSize(tensor->dtype);
    for ( int i = 0; i < tensor->dim; ++i)
    {
        bytes *= tensor->shape[i];
    }
    return bytes;
}

static inline bool tpudnnIsTensorContiguous(const tpudnnTensor_t *tensor)
{
    int stride = 1;
    for (int i = tensor->dim - 1; i >= 0; --i)
    {
        if (tensor->shape[i] > 1 && tensor->stride[i] != stride)
        {
            return false;
        }
        else
        {
            stride *= tensor->shape[i];
        }
    }
    return true;
}

static inline bool tpudnnIsTensorTransposed ( const tpudnnTensor_t * tensor )
{
    if ( tensor->dim < 2 || tpudnnIsTensorContiguous ( tensor ) )
    {
        return false;
    }
    else
    {
        int stride = 1;
        for ( int i = tensor->dim - 1; i >= 0; --i )
        {
            if ( ( i == tensor->dim - 1 && tensor->stride[i] != tensor->shape[tensor->dim - 2] ) ||
                 ( i == tensor->dim - 2 && tensor->stride[i] != 1 ) ||
                 ( i < tensor->dim - 2 && tensor->stride[i] != stride ) )
            {
                return false;
            }
            else
            {
                stride *= tensor->shape[i];
            }
        }
    }
    return true;
}

static inline bool tpudnnIsSameShape ( const tpudnnTensor_t * tensor1, const tpudnnTensor_t * tensor2 )
{
    if ( tensor1->dim == tensor2->dim )
    {
        for ( int i = 0; i < tensor1->dim; ++i )
        {
            if ( tensor1->shape[i] != tensor2->shape[i] )
            {
                return false;
            }
        }
    }
    else
    {
        return false;
    }
    return true;
}

static inline tpudnnTensor_t tpudnnUndefinedTensor()
{
    tpudnnTensor_t tensor = {.addr = 0};

    return tensor;
}

tpudnnStatus_t tpudnnBinaryAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t other,
    float scalar,
    tpudnnTensor_t output,
    int binary_type);

tpudnnStatus_t tpudnnBinaryBcastAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t other,
    float scalar,
    tpudnnTensor_t output,
    int binary_type);

tpudnnStatus_t tpudnnBinaryCAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    float scalar,
    tpudnnTensor_t output,
    int binary_type,
    bool inversed);

tpudnnStatus_t tpudnnMatmulAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t left,
    tpudnnTensor_t right,
    tpudnnTensor_t bias,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnSliceScatterAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t src,
    tpudnnTensor_t indices,
    int dim,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnLogSoftmaxAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    int dim,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnSoftmaxAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    int dim,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnSoftmaxBackwardAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t grad_output,
    tpudnnTensor_t output,
    int dim,
    tpudnnTensor_t grad_input);

tpudnnStatus_t tpudnnLogAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output,
    tensor_log_type_t log_type);

tpudnnStatus_t tpudnnSqueezeAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnWhereAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t cond,
    tpudnnTensor_t self,
    tpudnnTensor_t other,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnNorm2Async(
    tpudnnHandle_t handle,
    const tpudnnTensor_t input,
    int keepdim,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnNegAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnArangeAsync (
    tpudnnHandle_t handle,
    int start,
    int end,
    int step,
    tpudnnTensor_t out);

tpudnnStatus_t tpudnnRepeatAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    int* repeat_times,
    int repeat_dim,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnStridedCopyAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnConvertAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output,
    int is_bool);

tpudnnStatus_t tpudnnNonzeroAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t self,
    tpudnnTensor_t out,
    tpudnnTensor_t num);

tpudnnStatus_t tpudnnClampAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    float min,
    float max,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnNativeGroupNormAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t gamma,
    tpudnnTensor_t beta,
    int group,
    int affine,
    float eps,
    tpudnnTensor_t output,
    tpudnnTensor_t mean,
    tpudnnTensor_t rstd);

tpudnnStatus_t tpudnnLogicalAndAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t other,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnLogicalNotAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnLogicalOrAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t other,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnBitwiseNotAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnCbrtAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnAddCMulAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t tensor1,
    tpudnnTensor_t tensor2,
    float scalar,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnAddCMulBcastAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t tensor1,
    tpudnnTensor_t tensor2,
    float scalar,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnAddCDivAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t tensor1,
    tpudnnTensor_t tensor2,
    float scalar,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnCrossEntropyLossAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t target,
    int reduction,
    int ignore_index,
    float label_smoothing,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnCrossEntropyLossBackwardAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t target,
    tpudnnTensor_t grad_output,
    int ignore_index,
    int reduction,
    float label_smoothing,
    tpudnnTensor_t grad_input);

tpudnnStatus_t tpudnnMselossAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t self,
    tpudnnTensor_t target,
    tpudnnTensor_t out,
    int reduction);

tpudnnStatus_t tpudnnPoolingForwardAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output,
    TPUDNN_PoolingDescriptor_t pooling_desc);

tpudnnStatus_t tpudnnMaxPoolingWithMaskForwardAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output,
    tpudnnTensor_t max_mask_addr,
    TPUDNN_PoolingDescriptor_t pooling_desc);

tpudnnStatus_t tpudnnMaxpoolingIndicesBwdAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t grad,
    tpudnnTensor_t indices,
    tpudnnTensor_t output,
    int kernel,
    int stride,
    int padding);

tpudnnStatus_t tpudnnReduceMaxOrMinAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    int* reduction_dim,
    int reduction_dim_length,
    int keepdim,
    int mode,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnReduceVarAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    int* reduce_list,
    int reduce_dim,
    int correction,
    int keepdim,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnReduceVarAllAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    int correction,
    bool keepdim,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnReduceAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    int start_dim,
    int end_dim,
    int keepdim,
    int mode,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnReduceProdAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    int axis,
    int keepdim,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnIndexSelectAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t indices,
    int dim,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnEmbeddingBackwardAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t grad_output,
    tpudnnTensor_t indices,
    tpudnnTensor_t grad_input);

tpudnnStatus_t tpudnnConcatAsync (
    tpudnnHandle_t handle ,
    const tpudnnTensor_t * inputs,
    int input_num,
    int dim,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnGatherAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t index,
    tpudnnTensor_t output,
    int axis);

tpudnnStatus_t tpudnnArgAsync(
    tpudnnHandle_t resource,
    tpudnnTensor_t input,
    int axis,
    int mode,
    tpudnnTensor_t values,
    tpudnnTensor_t indices);

tpudnnStatus_t tpudnnTopkAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    int k,
    int axis,
    bool largest,
    bool sorted,
    tpudnnTensor_t value,
    tpudnnTensor_t index);

tpudnnStatus_t tpudnnConjAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnRealAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnBatchnorm2dAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t weight,
    tpudnnTensor_t bias,
    float eps,
    tpudnnTensor_t running_mean,
    tpudnnTensor_t running_var,
    float momentum,
    tpudnnTensor_t output,
    tpudnnTensor_t saved_mean,
    tpudnnTensor_t saved_invstd);

tpudnnStatus_t tpudnnBatchnorm2dBackwardAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t grad_output,
    tpudnnTensor_t input,
    tpudnnTensor_t weight,
    tpudnnTensor_t saved_mean,
    tpudnnTensor_t saved_invstd,
    tpudnnTensor_t grad_input,
    tpudnnTensor_t grad_weight,
    tpudnnTensor_t grad_bias);

tpudnnStatus_t tpudnnLayernormAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t weight,
    tpudnnTensor_t bias,
    int start_dim,
    float eps,
    tpudnnTensor_t output,
    tpudnnTensor_t mean,
    tpudnnTensor_t rstd);

tpudnnStatus_t tpudnnLayernormBackwardAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t grad_output,
    tpudnnTensor_t input,
    tpudnnTensor_t weight,
    tpudnnTensor_t mean,
    tpudnnTensor_t rstd,
    int start_dim,
    tpudnnTensor_t grad_input,
    tpudnnTensor_t grad_weight,
    tpudnnTensor_t grad_bias,
    int requires_grad_input);

tpudnnStatus_t tpudnnSignbitAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnReLUBackwardAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t grad_output,
    tpudnnTensor_t input,
    tpudnnTensor_t grad_input);

tpudnnStatus_t tpudnnGELUAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnGELUBackwardAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t grad_output,
    tpudnnTensor_t input,
    tpudnnTensor_t grad_input);

tpudnnStatus_t tpudnnLeakyReLUAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output,
    float negative_slope);

tpudnnStatus_t tpudnnHardtanhAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    float min_value,
    float max_value,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnActiveAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output,
    tensor_active_type_t active_type);

tpudnnStatus_t tpudnnReorderConv2dWeightAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    int mode,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnConv2dAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t weight,
    tpudnnTensor_t bias,
    tpudnnConv2dParam_t param,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnConv2dBackwardAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t grad_output,
    tpudnnTensor_t input,
    tpudnnTensor_t weight,
    tpudnnConv2dParam_t param,
    tpudnnTensor_t grad_input,
    tpudnnTensor_t grad_weight,
    tpudnnTensor_t grad_bias);

tpudnnStatus_t tpudnnUpsamplingAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output,
    bool align_corners,
    tensor_resize_mode_t upsampling_type);

tpudnnStatus_t tpudnnUpsampleNearest2dBackwardAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t grad_output,
    tpudnnTensor_t grad_input,
    int scale,
    TPUDNN_PoolingDescriptor_t pooling_desc);

tpudnnStatus_t tpudnnFlipAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    int axis,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnTriangularizeAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t self,
    int is_upper,
    int diagonal,
    tpudnnTensor_t out);

tpudnnStatus_t tpudnnInfCheckAndUnscaleAsync(
    tpudnnHandle_t handle,
    std::vector<tpudnnTensor_t>& inputs,
    tpudnnTensor_t found_inf,
    float inv_scale);

tpudnnStatus_t tpudnnRmsNormForwardAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t scale,
    tpudnnTensor_t bias,
    tpudnnTensor_t output,
    int axis,
    float eps
);

tpudnnStatus_t tpudnnAddRmsNormForwardAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t residule,
    tpudnnTensor_t scale,
    tpudnnTensor_t bias,
    tpudnnTensor_t output,
    int add_residule,
    int axis,
    float eps
);

tpudnnStatus_t tpudnnRmsNormBackwardAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t grad_output,
    tpudnnTensor_t input,
    tpudnnTensor_t scale,
    tpudnnTensor_t rms,
    tpudnnTensor_t grad_input,
    tpudnnTensor_t grad_scale,
    tpudnnTensor_t grad_bias,
    int axis,
    double eps);

tpudnnStatus_t tpudnnC2CPerf(
    tpudnnHandle_t handle,
    uint64_t count,
    c2c_communication_type comm_type,
    c2c_direct_type direct_type,
    c2c_copy_type copy_type,
    void * info_buffer,
    tpudnnDataType_t dtype,
    const char* uuid,
    int nranks,
    int cur_rank,
    const int *chip_map);

tpudnnStatus_t tpudnnC2CSend(
    tpudnnHandle_t handle,
    void *buff,
    uint64_t count,
    tpudnnDataType_t dtype,
    int dst_rank,
    const char* uuid,
    int nranks,
    int cur_rank,
    const int *chip_map);

tpudnnStatus_t tpudnnC2CRecv(
    tpudnnHandle_t handle,
    void *buff,
    uint64_t count,
    tpudnnDataType_t dtype,
    int src_rank,
    const char* uuid,
    int nranks,
    int cur_rank,
    const int *chip_map);

tpudnnStatus_t tpudnnC2CAllReduce(
    tpudnnHandle_t handle,
    void *send_buff,
    void *recv_buff,
    uint64_t count,
    tpudnnDataType_t dtype,
    tpudnnReduceType_t reduce_method,
    const char* uuid,
    int nranks,
    int cur_rank,
    const int *chip_map);

tpudnnStatus_t tpudnnC2CReduce(
    tpudnnHandle_t handle,
    void *send_buff,
    void *recv_buff,
    uint64_t count,
    tpudnnDataType_t dtype,
    tpudnnReduceType_t reduce_method,
    int root,
    const char* uuid,
    int nranks,
    int cur_rank,
    const int *chip_map);

tpudnnStatus_t tpudnnC2CGather(
    tpudnnHandle_t handle,
    void *send_buff,
    uint64_t send_count,
    void *recv_buff,
    uint64_t recv_count,
    tpudnnDataType_t dtype,
    int root,
    const char* uuid,
    int nranks,
    int cur_rank,
    const int *chip_map);

tpudnnStatus_t tpudnnC2CAllGather(
    tpudnnHandle_t handle,
    void *send_buff,
    uint64_t send_count,
    void *recv_buff,
    uint64_t recv_count,
    const char* uuid,
    tpudnnDataType_t dtype,
    int nranks,
    int cur_rank,
    const int *chip_map);

tpudnnStatus_t tpudnnC2CBroadcast(
    tpudnnHandle_t handle,
    void *buff,
    uint64_t count,
    tpudnnDataType_t dtype,
    int root,
    const char* uuid,
    int nranks,
    int cur_rank,
    const int *chip_map);

tpudnnStatus_t tpudnnC2CScatter(
    tpudnnHandle_t handle,
    void *send_mem,
    tpudnnDataType_t send_type,
    void *recv_mem,
    uint64_t recv_count,
    tpudnnDataType_t recv_type,
    int root,
    const char* uuid,
    int nranks,
    int cur_rank,
    const int *chip_map);

tpudnnStatus_t tpudnnC2CAllToAll(
    tpudnnHandle_t handle,
    void *send_mem,
    tpudnnDataType_t send_type,
    void *recv_mem,
    uint64_t recv_count,
    tpudnnDataType_t recv_type,
    const char* uuid,
    int nranks,
    int cur_rank,
    const int *chip_map);

tpudnnStatus_t tpudnnMaskedFillAsync(
    tpudnnHandle_t handle,
    const tpudnnTensor_t input,
    const tpudnnTensor_t mask,
    float value,
    tpudnnTensor_t out);

tpudnnStatus_t tpudnnFillAsync(
    tpudnnHandle_t handle,
    const void * scalar_ptr,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnAdamBackwardMultiCoreAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t weight_out,
    tpudnnTensor_t m_out,
    tpudnnTensor_t v_out,
    tpudnnTensor_t vmax_out,
    tpudnnTensor_t grad_weight,
    tpudnnTensor_t weight_in,
    tpudnnTensor_t m_in,
    tpudnnTensor_t v_in,
    tpudnnTensor_t vmax_in,
    tpudnnTensor_t t,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    bool amsgrad,
    bool maximize);

tpudnnStatus_t tpudnnDropoutMultiCoreAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t output,
    const float drop_rate);

tpudnnStatus_t tpudnnLoraMatmulForwardAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t loraA,
    tpudnnTensor_t loraB,
    tpudnnTensor_t weight,
    tpudnnTensor_t output,
    float scale);

tpudnnStatus_t tpudnnPagedAttentionAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t OUT,
    tpudnnTensor_t Q,
    tpudnnTensor_t K,
    tpudnnTensor_t V,
    tpudnnTensor_t Kcache,
    tpudnnTensor_t Vcache,
    tpudnnTensor_t cos,
    tpudnnTensor_t sin,
    tpudnnTensor_t save_slots,
    tpudnnTensor_t fetch_slots,
    tpudnnTensor_t mask,
    tpudnnTensor_t Qbuffer,
    tpudnnTensor_t Kbuffer,
    tpudnnTensor_t Vbuffer,
    tpudnnTensor_t input_lengths_tensor,
    int            rope_head_size,
    int*           input_lengths,
    int            num_input_lengths,
    int slots_size,
    int mask_size,
    int block_size,
    float C,
    int attention_mode // 2: prefile, 3: decode
    );

tpudnnStatus_t tpudnnLlamaAttentionForwardAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t O,
    tpudnnTensor_t Q,
    tpudnnTensor_t K,
    tpudnnTensor_t V,
    tpudnnTensor_t RoPE_cos,
    tpudnnTensor_t RoPE_sin,
    tpudnnTensor_t mask,
    tpudnnTensor_t softmax_lse,
    int* input_lengths,
    int num_input_lengths,
    int mask_max,
    float C,
    float dropout_rate,
    int batch);

tpudnnStatus_t tpudnnLlamaAttentionBackwardAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t Q,
    tpudnnTensor_t K,
    tpudnnTensor_t V,
    tpudnnTensor_t O,
    tpudnnTensor_t dO,
    tpudnnTensor_t L,
    tpudnnTensor_t dQ,
    tpudnnTensor_t dK,
    tpudnnTensor_t dV,
    tpudnnTensor_t RoPE_cos,
    tpudnnTensor_t RoPE_sin,
    tpudnnTensor_t mask,
    tpudnnTensor_t input_length,
    int mask_max,
    float C);

tpudnnStatus_t tpudnnLLamaMlpAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t weight0,
    tpudnnTensor_t weight1,
    tpudnnTensor_t weight2,
    tpudnnTensor_t silu,
    tpudnnTensor_t sigmoid,
    tpudnnTensor_t m0,
    tpudnnTensor_t output,
    bool save_mid_res);

tpudnnStatus_t tpudnnLLamaA16MlpAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t weight0,
    tpudnnTensor_t zp0,
    tpudnnTensor_t scale0,
    tpudnnTensor_t weight1,
    tpudnnTensor_t zp1,
    tpudnnTensor_t scale1,
    tpudnnTensor_t weight2,
    tpudnnTensor_t zp2,
    tpudnnTensor_t scale2,
    int group_size,
    int weight_bits,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnMlpW8A16DqAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t weight0,
    tpudnnTensor_t weight1,
    tpudnnTensor_t weight2,
    tpudnnTensor_t scale0,
    tpudnnTensor_t scale1,
    tpudnnTensor_t scale2,
    tpudnnTensor_t output,
    int blocksize);

tpudnnStatus_t tpudnnGDMAD2DAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t src,
    tpudnnTensor_t dst,
    size_t size);

tpudnnStatus_t tpudnnTanhBackwardAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t grad_output,
    tpudnnTensor_t output,
    tpudnnTensor_t grad_input);

tpudnnStatus_t tpudnnSigmoidBackwardAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t grad_output,
    tpudnnTensor_t output,
    tpudnnTensor_t grad_input);

tpudnnStatus_t tpudnnSiluBackwardAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t grad_output,
    tpudnnTensor_t input,
    tpudnnTensor_t grad_input);

tpudnnStatus_t tpudnnLLamaA16MatmulAsync(
    tpudnnHandle_t handle ,
    tpudnnTensor_t left,
    tpudnnTensor_t right,
    tpudnnTensor_t bias,
    tpudnnTensor_t scale,
    tpudnnTensor_t zp,
    int group_size,
    int weight_bits,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnTgiInputIdsUpdateAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t all_input_ids,
    tpudnnTensor_t next_ids,
    int* input_lengths,
    int  n_input_lengths,
    int  n_accept_ids);

tpudnnStatus_t tpudnnEnableProfile(
    tpudnnHandle_t handle,
    int max_record_num,
    int mode);

tpudnnStatus_t tpudnnDisableProfile(
    tpudnnHandle_t handle);

tpudnnStatus_t tpudnnMultiHeadAttentionAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t OUT, // {batch, seq, head, head_dim}
    tpudnnTensor_t Q, // {batch, seq, head, head_dim}
    tpudnnTensor_t K, // {batch, seq, head, head_dim}
    tpudnnTensor_t V, // {batch, seq, head, head_dim}
    tpudnnTensor_t mask, // {batch, 1, head, head_dim}
    float scale);

tpudnnStatus_t tpudnnLLaVaMlpAsync (
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t w1,
    tpudnnTensor_t w2,
    tpudnnTensor_t b1,
    tpudnnTensor_t b2,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnIndexSelectAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t indices,
    int dim,
    tpudnnTensor_t output);

tpudnnStatus_t tpudnnIndexAdd(
    tpudnnHandle_t handle,
    tpudnnTensor_t io,
    tpudnnTensor_t indices,
    tpudnnTensor_t add,
    int axis);

tpudnnStatus_t tpudnnC2CDescriptor(
    tpudnnHandle_t handle,
    tpudnnCmd_t* vsdma_cmd,
    int vsdma_engine_num,
    tpudnnCmd_t* cdma_cmd,
    int cdma_engine_num,
    tpudnnCmd_t gdma_cmd,
    void* reduce_data,
    int reduce_size,
    int nranks,
    int cur_rank,
    const int *chip_map);

tpudnnStatus_t tpudnnDynLibExecuteAsync(
                    tpudnnHandle_t handle,
                    const char *so_url,
                    const char *func_name,
                    std::vector<tpudnnTensor_t> tensors,
                    std::vector<int64_t> tensors_index,
                    std::vector<double> fp_scalars,
                    std::vector<int64_t> fp_scalars_index,
                    std::vector<int64_t> fixed_scalars,
                    std::vector<int64_t> fixed_scalars_index);

tpudnnStatus_t tpudnnScatterAddAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t input,
    tpudnnTensor_t src,
    tpudnnTensor_t indices,
    int dim);

tpudnnStatus_t tpudnnFusedMoEGroupedTopkMultiCoreAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t topk_experts_res,
    tpudnnTensor_t topk_weights_res_bf16,
    tpudnnTensor_t left_bf16,
    tpudnnTensor_t right_bf16,
    tpudnnTensor_t topk_weights_res,
    tpudnnTensor_t left,
    tpudnnTensor_t right,
    tpudnnTensor_t max,
    tpudnnTensor_t matmul_res,
    tpudnnTensor_t softmax_res);

tpudnnStatus_t tpudnnFusedMoEFusedExpertsMultiCoreAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t output,
    tpudnnTensor_t input,
    tpudnnTensor_t output_sample,
    tpudnnTensor_t input_sample,
    tpudnnTensor_t gate_weights, // [num_experts, middle_w, input_w]
    tpudnnTensor_t up_weights, // [num_experts, middle_w, input_w]
    tpudnnTensor_t down_weights, // [num_experts, middle_w, input_w]
    tpudnnTensor_t gate_scales, // [num_experts, block_m, block_n]
    tpudnnTensor_t up_scales, // [num_experts, block_m, block_n]
    tpudnnTensor_t down_scales, // [num_experts, block_m, block_n]
    tpudnnTensor_t select_experts, // [batch * seq_len, num_experts]
    tpudnnTensor_t routing_weights,
    tpudnnTensor_t num_select_experts,
    tpudnnTensor_t select_experts_middle,
    tpudnnTensor_t routing_weights_middle,
    int blocksize,
    int num_experts,
    int num_experts_per_tok,
    bool use_grouped_topk,
    int num_expert_group,
    int topk_group,
    tpudnnTensor_t silu,
    tpudnnTensor_t sigmoid,
    tpudnnTensor_t m0,
    bool save_mid_res);

} // extern "C"
