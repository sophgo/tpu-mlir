/*
* Copyright (C) Cvitek Co., Ltd. 2019-2020. All rights reserved.
*
* File Name: cviruntime.h
* Description:
*/

#ifndef _CVIRUNTIME_H_
#define _CVIRUNTIME_H_

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
#include "cvitpu_debug.h"

#ifdef __cplusplus
extern "C" {
#endif

// data type of tensor
typedef enum {
  CVI_FMT_FP32   = 0,
  CVI_FMT_INT32  = 1,
  CVI_FMT_UINT32 = 2,
  CVI_FMT_BF16   = 3,
  CVI_FMT_INT16  = 4,
  CVI_FMT_UINT16 = 5,
  CVI_FMT_INT8   = 6,
  CVI_FMT_UINT8  = 7
} CVI_FMT;

// memory source of Tensor buf.
typedef enum {
  CVI_MEM_SYSTEM  = 1,
  CVI_MEM_DEVICE  = 2
} CVI_MEM_TYPE_E;

// pixel format
typedef enum {
  CVI_NN_PIXEL_RGB_PACKED     = 0,
  CVI_NN_PIXEL_BGR_PACKED     = 1,
  CVI_NN_PIXEL_RGB_PLANAR     = 2,
  CVI_NN_PIXEL_BGR_PLANAR     = 3,
  CVI_NN_PIXEL_YUV_NV12       = 11,
  CVI_NN_PIXEL_YUV_NV21       = 12,
  CVI_NN_PIXEL_YUV_420_PLANAR = 13,
  CVI_NN_PIXEL_GRAYSCALE      = 15,
  CVI_NN_PIXEL_TENSOR         = 100,
  CVI_NN_PIXEL_RGBA_PLANAR = 1000,
  // please don't use below values,
  // only for backward compatibility
  CVI_NN_PIXEL_PLANAR         = 101,
  CVI_NN_PIXEL_PACKED         = 102
} CVI_NN_PIXEL_FORMAT_E;

typedef enum {
  /*
   * bool, default value is false,
   * if set to true, runtime will output all tensors as
   * output tensors for debugging.
   */
  OPTION_OUTPUT_ALL_TENSORS       = 4,
  /*
   * unsigned int, default value is 0,
   * set program id, for switch programs in cvimodel
   */
  OPTION_PROGRAM_INDEX            = 9,
  // DEPRECATED
  OPTION_BATCH_SIZE               = 1,
  // DEPRECATED
  OPTION_SKIP_POSTPROCESS         = 6,
  // DEPRECATED
  OPTION_PREPARE_BUF_FOR_INPUTS   = 2,
  // DEPRECATED
  OPTION_PREPARE_BUF_FOR_OUTPUTS  = 3,
  // DEPRECATED
  OPTION_SKIP_PREPROCESS          = 5,
  // DEPRECATED
  OPTION_INPUT_MEM_TYPE           = 7,
  // DEPRECATED
  OPTION_OUTPUT_MEM_TYPE          = 8
} CVI_CONFIG_OPTION;

#define CVI_DIM_MAX (6)
typedef struct {
  int32_t dim[CVI_DIM_MAX];
  size_t dim_size;
} CVI_SHAPE;

typedef struct {
  char                  *name;
  CVI_SHAPE             shape;
  CVI_FMT               fmt;
  size_t                count;
  size_t                mem_size;
  uint8_t               *sys_mem;
  uint64_t              paddr;
  CVI_MEM_TYPE_E        mem_type;
  float                 qscale;
  int                   zero_point;
  CVI_NN_PIXEL_FORMAT_E pixel_format;
  bool                  aligned;
  float                 mean[3];
  float                 scale[3];
  void                  *owner;
  char                  reserved[32];
} CVI_TENSOR;

typedef CVI_NN_PIXEL_FORMAT_E CVI_FRAME_TYPE;
#define CVI_FRAME_PLANAR  CVI_NN_PIXEL_PLANAR
#define CVI_FRAME_PACKAGE CVI_NN_PIXEL_PACKED

typedef struct {
  CVI_FRAME_TYPE type;
  CVI_SHAPE shape;
  CVI_FMT fmt;
  uint32_t stride[3];
  uint64_t pyaddr[3];
} CVI_VIDEO_FRAME_INFO;

typedef void *CVI_MODEL_HANDLE;

typedef int CVI_RC;
/*
 * Register a cvimodel file to runtime, and return a model handle.
 * @param [in] model_file,     file name of cvimodel.
 * @param [out] model,         handle to registered model.
 */
CVI_RC CVI_NN_RegisterModel(const char *model_file, CVI_MODEL_HANDLE *model);

/*
 * Register a cvimodel file from memory, and return a model handle.
 * @param [in] buf,            buffer to store cvimodel data.
 * @param [in] size,           bytes of cvimodel data.
 * @param [out] model,         handle to registered model.
 */
CVI_RC CVI_NN_RegisterModelFromBuffer(const int8_t *buf, uint32_t size, CVI_MODEL_HANDLE *model);

CVI_RC CVI_NN_RegisterModelFromFd(const int fd, const size_t ud_offset, CVI_MODEL_HANDLE *model);

/*
 * Clone model that pointed by previous model handle, it will increment
 * the refence count of model. The returned handle will share resources with
 * previous handle, and save considerable memory.
 * @param [in] model,  previous handle of model
 * @param [out] cloned, cloned handle of same model.
 */
CVI_RC CVI_NN_CloneModel(CVI_MODEL_HANDLE model, CVI_MODEL_HANDLE *cloned);

/*
 * Get version number of cvimodel.
 * @param [in] model,  previous handle of model
 * @param [out] major version number.
 * @param [out] minor version number.
 */
CVI_RC CVI_NN_GetModelVersion(CVI_MODEL_HANDLE model, int32_t *major, int32_t *minor);

/*
 * Get version number of cvimodel.
 * @param [in] model,  previous handle of model
 * @param [out] target name, cv182x,cv183x
 */
const char * CVI_NN_GetModelTarget(CVI_MODEL_HANDLE model);

/*
 * To set the configuration that specified by CVI_CONFIG_OPTION.
 * This API must to be called before GetInputOutputTensors if user
 * want to change default configuration.
 * It only needs to set all these configurations once.
 * @param [in] model,   handle of model
 * @param [in] option,  option defiend in enum CVI_CONFIG_OPTION
 * @param [in] variant value related to parameter option
 */
CVI_RC CVI_NN_SetConfig(CVI_MODEL_HANDLE model, CVI_CONFIG_OPTION option, ...);

/*
 * Get input and output tensors of model. It needs to be call before
 * Forward/ForwardAsync API.
 * @param [in] model,         handle of model.
 * @param [out] inputs,       array of input tensors.
 * @param [out] input_num,    number of input tensors.
 * @param [out] outputs,      array of output tensors.
 * @param [out] output_num,   number of output tensors.
 */
CVI_RC CVI_NN_GetInputOutputTensors(CVI_MODEL_HANDLE model, CVI_TENSOR **inputs,
    int32_t *input_num, CVI_TENSOR **outputs, int32_t *output_num);
/*
 * Inference forwarding in blocking mode.
 */
CVI_RC CVI_NN_Forward(CVI_MODEL_HANDLE model, CVI_TENSOR inputs[], int32_t input_num,
    CVI_TENSOR outputs[], int32_t output_num);
/*
 * Infernece forwarding in asynchronous mode and
 * waiting result by calling ForwardWait.
 */
CVI_RC CVI_NN_ForwardAsync(CVI_MODEL_HANDLE model, CVI_TENSOR inputs[], int32_t input_num,
    CVI_TENSOR outputs[], int32_t output_num, void **task_no);
/*
 * Waiting result after do inference forward in async mode.
 */
CVI_RC CVI_NN_ForwardWait(CVI_MODEL_HANDLE model, void *task_no);
/*
 * Decrement of the reference count of model.
 * It will cleanup all resources of model if reference
 * declined to zero.
 */
CVI_RC CVI_NN_CleanupModel(CVI_MODEL_HANDLE model);

///
/// Helper functions
///
CVI_RC CVI_NN_GetInputTensors(CVI_MODEL_HANDLE model, CVI_TENSOR **inputs, int32_t *input_num);
CVI_RC CVI_NN_GetOutputTensors(CVI_MODEL_HANDLE model, CVI_TENSOR **outputs, int32_t *output_num);

#define CVI_NN_DEFAULT_TENSOR (NULL)
/*
 * Get tensor from input or output tensors by name.
 * @param [in] name.     name of wanted tensor.
 *                       if value is CVI_NN_DEFAULT_TENSOR or NULL, return first tensor.
 *                       And it also support wild-card matching if name ended by '*' character.
 * @param [in] tensors,  array of input or output tensors.
 * @param [in] num,      number of input or output tensors.
 */
CVI_TENSOR *CVI_NN_GetTensorByName(const char *name, CVI_TENSOR *tensors, int32_t num);
/*
 * Get Name of tensor.
 */
char *CVI_NN_TensorName(CVI_TENSOR *tensor);
/*
 * Get Buffer pointer of tensor.
 */
void *CVI_NN_TensorPtr(CVI_TENSOR *tensor);
/*
 * Get Byte size of tensor's buffer.
 * tensor size = tensor count * sizeof(tensor data type)
 */
size_t CVI_NN_TensorSize(CVI_TENSOR *tensor);
/*
 * Get Count of elements stored in tensor.
 */
size_t CVI_NN_TensorCount(CVI_TENSOR *tensor);
/*
 * Get quant scale to do quantization(fp32 -> int8)
 */
float CVI_NN_TensorQuantScale(CVI_TENSOR *tensor);
/*
 * Get quant zero point to do asymmetric quantization(fp32 -> int8)
 */
int CVI_NN_TensorQuantZeroPoint(CVI_TENSOR *tensor);
/*
 * Get shape of a tensor.
 */
CVI_SHAPE CVI_NN_TensorShape(CVI_TENSOR *tensor);

/*
 * Set system memory for tensor.
 */
CVI_RC CVI_NN_SetTensorPtr(CVI_TENSOR *tensor, void *mem);

/*
 * Set physical Address for tensor.
 */
CVI_RC CVI_NN_SetTensorPhysicalAddr(CVI_TENSOR *tensor, uint64_t paddr);

/*
 * Do data copy from video frame to tensor
 * WARNNING, this API is DEPRECATED.
 */
CVI_RC CVI_NN_SetTensorWithVideoFrame(
    CVI_MODEL_HANDLE model, CVI_TENSOR* tensor,
    CVI_VIDEO_FRAME_INFO* video_frame_info);

/*
 * Do data copy from video frame to tensor
 * WARNNING, this API is DEPRECATED.
 */
CVI_RC CVI_NN_FeedTensorWithFrames(
    CVI_MODEL_HANDLE model, CVI_TENSOR *tensor,
    CVI_FRAME_TYPE type, CVI_FMT format,
    int32_t channel_num, uint64_t *channel_paddrs,
    int32_t height, int32_t width, uint32_t height_stride);

/*
 * Fill frames data from vpss to tensor.
 */
CVI_RC CVI_NN_SetTensorWithAlignedFrames(
    CVI_TENSOR *tensor, uint64_t frame_paddrs[],
    int32_t frame_num,  CVI_NN_PIXEL_FORMAT_E pixel_format);

/*
 * set shared memory size befor registering all cvimodels.
 */
void CVI_NN_Global_SetSharedMemorySize(size_t size);

#ifdef __cplusplus
}
#endif

#endif // _CVIRUNTIME_H_
