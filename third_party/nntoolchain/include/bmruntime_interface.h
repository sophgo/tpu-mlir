/*****************************************************************************
 *
 *    Copyright (c) 2016-2026 by Sophgo Technologies Inc. All rights reserved.
 *
 *    The material in this file is confidential and contains trade secrets
 *    of Sophgo Technologies Inc. This is proprietary information owned by
 *    Sophgo Technologies Inc. No part of this work may be disclosed,
 *    reproduced, copied, transmitted, or used in any way for any purpose,
 *    without the express written permission of Sophgo Technologies Inc.
 *
 *****************************************************************************/

/*****************************************************************************
 * BMRuntime Interface is mainly for inference.
 * Also we can use it for device computation from BMLang programming.
 * Note: please use interface from bmlib_runtime.h for device memory operation.
 ****************************************************************************/

#ifndef BMRUNTIME_INTERFACE_H_
#define BMRUNTIME_INTERFACE_H_

#include "bmdef.h"

#ifdef _WIN32
#define DECL_EXPORT _declspec(dllexport)
#define DECL_IMPORT _declspec(dllimport)
#else
#define DECL_EXPORT
#define DECL_IMPORT
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/* --------------------------------------------------------------------------*/
/* interface for basic data type */

/* get data type byte size */
DECL_EXPORT size_t bmrt_data_type_size(bm_data_type_t dtype);

/*
dims array to bm_shape_t,
shape and dims should not be NULL, num_dims should not be larger than BM_MAX_DIMS_NUM */
DECL_EXPORT void bmrt_shape(bm_shape_t* shape, const int* dims, int num_dims);

/*
number of shape elements, shape should not be NULL and num_dims should not large than
BM_MAX_DIMS_NUM */
DECL_EXPORT uint64_t bmrt_shape_count(const bm_shape_t* shape);

/* compare whether two shape is same */
DECL_EXPORT bool bmrt_shape_is_same(const bm_shape_t* left, const bm_shape_t* right);

/*
fill a tensor with data type and shape, and st_mode = 0 as default.
tensor and p_bmrt should not be NULL, shape count should not be 0.
it will alloc device mem to tensor->device_mem, so user should bmrt_free_device(p_bmrt,
tensor->device_mem) to free it.*/
DECL_EXPORT bool bmrt_tensor(bm_tensor_t* tensor, void* p_bmrt, bm_data_type_t dtype, bm_shape_t shape);

/*
fill a tensor with data type and shape, and st_mode = 0 as default.
tensor and p_bmrt should not be NULL, shape count should not be 0.
it will alloc device mem to tensor->device_mem on devid-th device.*/
DECL_EXPORT bool bmrt_tensor_ex(bm_tensor_t* tensor, void* p_bmrt, int devid, bm_data_type_t dtype, bm_shape_t shape);

/* fill a tensor with device mem existed, tensor byte size should not large than device mem size */
DECL_EXPORT void bmrt_tensor_with_device(bm_tensor_t* tensor, bm_device_mem_t device_mem,
                             bm_data_type_t dtype, bm_shape_t shape);

/* get tensor bytes size, tensor should not be NULL */
DECL_EXPORT size_t bmrt_tensor_bytesize(const bm_tensor_t* tensor);

/* get tensor mem size allocated in device mem, tensor should not be NULL */
DECL_EXPORT size_t bmrt_tensor_device_size(const bm_tensor_t* tensor);

/* print net info for debug */
DECL_EXPORT void bmrt_print_network_info(const bm_net_info_t* net_info);

/* --------------------------------------------------------------------------*/
/**
 * @name    bmrt_create
 * @brief   To create the bmruntime with bm_handle.
 * @ingroup bmruntime
 *
 * This API creates the bmruntime. It returns a void* pointer which is the pointer
 * of bmruntime. Device id is set when get bm_handle;
 *
 * @param [in] bm_handle     bm handle. It must be initialized by using bmlib.
 *
 * @retval void* the pointer of bmruntime
 */
DECL_EXPORT void* bmrt_create(bm_handle_t bm_handle);

/* --------------------------------------------------------------------------*/
/**
 * @name    bmrt_create_ex
 * @brief   To create the bmruntime with one or more bm_handle.
 * @ingroup bmruntime
 *
 * This API creates the bmruntime. It returns a void* pointer which is the pointer
 * of bmruntime.
 *
 * @param [in] bm_handles   bm handles. They must be initialized by using bmlib.
 * @param [in] num_handles  number of bm_handles.
 *
 * @retval void* the pointer of bmruntime
 */
DECL_EXPORT void *bmrt_create_ex(bm_handle_t *bm_handles, int num_handles);

/**
 * @name    bmrt_destroy
 * @brief   To destroy the bmruntime pointer
 * @ingroup bmruntime
 *
 * This API destroy the bmruntime.
 *
 * @param [in]     p_bmrt        Bmruntime that had been created
 */
DECL_EXPORT void bmrt_destroy(void* p_bmrt);

/**
 * @name    bmrt_get_bm_handle
 * @brief   To get the BM runtime context.
 * @ingroup bmruntime
 *
 * This API get the BM runtime context for using BMDNN, BMCV or BMLIB
 *
 * @param [in]     p_bmrt        Bmruntime that had been created
 */
DECL_EXPORT void * bmrt_get_bm_handle(void* p_bmrt);

/* --------------------------------------------------------------------------*/
/**
 * @name    bmrt_set_flags
 * @brief   set runtime flags for different situations
 * @ingroup bmruntime
 *
 * This API set runtime flags, for various situations. flag defined by bm_runtime_flag_t
 *
 * @param [in]     p_bmrt        Bmruntime that had been created
 *
 */
DECL_EXPORT void bmrt_set_flags(void* p_bmrt, uint32_t flags);

/* --------------------------------------------------------------------------*/
/**
 * @name    bmrt_get_flags
 * @brief   get runtime flags for different situations
 * @ingroup bmruntime
 *
 * This API get runtime flags, for various situations. flag defined by bm_runtime_flag_t
 *
 * @param [in]     p_bmrt        Bmruntime that had been created
 *
 */
DECL_EXPORT uint32_t bmrt_get_flags(void* p_bmrt);

/**
 * @name    bmrt_load_bmodel
 * @brief   To load the bmodel which is created by BM compiler
 * @ingroup bmruntime
 *
 * This API is to load bmodel created by BM compiler.
 * After loading bmodel, we can run the inference of neuron network.
 *
 * @param   [in]   p_bmrt        Bmruntime that had been created
 * @param   [in]   bmodel_path   Bmodel file directory.
 *
 * @retval true    Load context sucess.
 * @retval false   Load context failed.
 */
DECL_EXPORT bool bmrt_load_bmodel(void* p_bmrt, const char *bmodel_path);

/**
 * @name    bmrt_load_bmodel_data
 * @brief   To load the bmodel which is created by BM compiler from buffer
 * @ingroup bmruntime
 *
 * This API is to load bmodel created by BM compiler.
 * After loading bmodel, we can run the inference of neuron network.
 * Different with bmrt_load_bmodel, bmodel is the data in host memory.
 *
 * @param   [in]   p_bmrt        Bmruntime that had been created
 * @param   [in]   bmodel_data   Bmodel data pointer to buffer
 * @param   [in]   size          Bmodel data size
 *
 * @retval true    Load context sucess.
 * @retval false   Load context failed.
 */
DECL_EXPORT bool bmrt_load_bmodel_data(void* p_bmrt, const void * bmodel_data, size_t size);

/**
 * @name    bmrt_show_neuron_network
 * @brief   To print the name of all neuron network
 * @ingroup bmruntime
 *
 * @param [in]     p_bmrt         Bmruntime that had been created
 */
DECL_EXPORT void bmrt_show_neuron_network(void* p_bmrt);

/**
 * @name    bmrt_get_network_number
 * @brief   To get the number of neuron network in the bmruntime
 * @ingroup bmruntime
 *
 * @param [in]     p_bmrt         Bmruntime that had been created
 *
 * @retval  int value     The number of neuron networks.
 */
DECL_EXPORT int bmrt_get_network_number(void* p_bmrt);

/**
 * @name    bmrt_get_network_names
 * @brief   To get the names of all neuron network in the bmruntime
 * @ingroup bmruntime
 *
 * @param [in]     p_bmrt         Bmruntime that had been created
 * @param [out]    network_names  The names of all neuron networks. It should be declare as (const char** networks_ = NULL),
 *                                and use as the param &networks_. After this API, user need to free(networks_) if user
 *                                do not need it.
 */
DECL_EXPORT void bmrt_get_network_names(void* p_bmrt, const char*** network_names);

/**
 * @name    bmrt_get_network_info
 * @brief   To get network info by net name
 * @ingroup bmruntime
 *
 * @param [in]     p_bmrt         Bmruntime that had been created
 * @param [in]     net_name       Network name
 *
 * @retval  bm_net_info_t*        Pointer to net info, needn't free by user; if net name not found, will return NULL.
 */
DECL_EXPORT const bm_net_info_t* bmrt_get_network_info(void* p_bmrt, const char* net_name);

/**
 * @name    bmrt_launch_tensor
 * @brief   To launch the inference of the neuron network with setting input tensors
 * @ingroup bmruntime
 *
 * This API supports the neuron nework that is static-compiled or dynamic-compiled
 * After calling this API, inference on TPU is launched. And the CPU program will not
 * be blocked. bm_thread_sync should be called to make sure inference finished.
 * This API support multiple inputs, and multi thread safety
 *
 * @param [in]    p_bmrt         Bmruntime that had been created
 * @param [in]    net_name       The name of the neuron network
 * @param [in]    input_tensors  Array of input tensor, defined like bm_tensor_t input_tensors[input_num].
 *                               User should initialize each input tensor.
 * @param [in]    input_num      Input number
 * @param [out]   output_tensors Array of output tensor, defined like bm_tensor_t output_tensors[output_num].
 *                               This interface will alloc devcie mem to store output data. User should free each
 *                               device mem by bm_free_device after the result data not used.
 * @param [in]    output_num     Output number
 *
 * @retval true    Launch success.
 * @retval false   Launch failed.
 */
DECL_EXPORT bool bmrt_launch_tensor(void* p_bmrt, const char * net_name, const bm_tensor_t input_tensors[], int input_num,
                        bm_tensor_t output_tensors[], int output_num);

/**
 * @name    bmrt_launch_tensor_ex
 * @brief   To launch the inference of the neuron network with setting input tensors
 * @ingroup bmruntime
 *
 * This API supports the neuron nework that is static-compiled or dynamic-compiled
 * After calling this API, inference on TPU is launched. And the CPU program will not
 * be blocked. bm_thread_sync should be called to make sure inference finished.
 * This API support multiple inputs, and multi thread safety
 *
 * @param [in]    p_bmrt            Bmruntime that had been created
 * @param [in]    net_name          The name of the neuron network
 * @param [in]    input_tensors     Array of input tensor, defined like bm_tensor_t input_tensors[input_num],
 *                                  User should initialize each input tensor.
 * @param [in]    input_num         Input number
 * @param [out]   output_tensors    Array of output tensor, defined like bm_tensor_t output_tensors[output_num].
 *                                  User can set device_mem or stmode of output tensors. If user_mem is true, this interface
 *                                  will use device mem of output_tensors to store output data, and not alloc device mem;
 *                                  Or it will alloc device mem to store output. If user_stmode is true, it will use stmode in
 *                                  each output tensor; Or stmode will be BM_STORE_1N as default.
 * @param [in]    output_num        Output number
 * @param [in]    user_mem          whether device_mem of output tensors are set
 * @param [in]    user_stmode       whether stmode of output tensors are set
 *
 * @retval true    Launch success.
 * @retval false   Launch failed.
 */
DECL_EXPORT bool bmrt_launch_tensor_ex(void* p_bmrt, const char * net_name, const bm_tensor_t input_tensors[], int input_num,
                           bm_tensor_t output_tensors[], int output_num, bool user_mem, bool user_stmode);

/**
 * @name    bmrt_launch_data
 * @brief   To launch the inference of the neuron network with setting input datas in system memory
 * @ingroup bmruntime
 *
 * This API supports the neuron nework that is static-compiled or dynamic-compiled
 * After calling this API, inference on TPU is launched. And the CPU
 * program will be blocked.
 * This API support multiple inputs, and multi thread safety
 *
 * @param [in]    p_bmrt         Bmruntime that had been created
 * @param [in]    net_name       The name of the neuron network
 * @param [in]    input_datas    Array of input data, defined like void * input_datas[input_num]. User should
 *                               initialize each data pointer as input.
 * @param [in]    input_shapes   Array of input shape, defined like bm_shape_t input_shapes[input_num].
 *                               User should set each input shape
 * @param [in]    input_num      Input number
 * @param [out]   output_datas   Array of output data, defined like void * output_datas[output_num].
 *                               If user don't alloc each output data, set user_mem to false, and this api will alloc
 *                               output mem, user should free each output mem when output data not used. Also
 *                               user can alloc system memory for each output data by self and set user_mem = true.
 * @param [out]   output_shapes  Array of output shape, defined like bm_shape_t output_shapes[output_num].
 *                               It will store each output shape.
 * @param [in]    output_num     Output number
 * @param [in]    user_mem       whether output_datas[i] have allocated memory
 *
 * @retval true    Launch success.
 * @retval false   Launch failed.
 */
DECL_EXPORT bool bmrt_launch_data(void* p_bmrt, const char* net_name, void* const input_datas[],
                      const bm_shape_t input_shapes[], int input_num, void * output_datas[],
                      bm_shape_t output_shapes[], int output_num, bool user_mem);

/**
 * @name    bmrt_launch_data_multi_core
 * @brief   To launch the inference of the neuron network with setting input datas in system memory on the assigned cores
 * @ingroup bmruntime
 *
 * This API supports the neuron nework that is static-compiled or dynamic-compiled
 * After calling this API, inference on TPU is launched. And the CPU
 * program will be blocked.
 * This API support multiple inputs, and multi thread safety
 *
 * @param [in]    p_bmrt         Bmruntime that had been created
 * @param [in]    net_name       The name of the neuron network
 * @param [in]    input_datas    Array of input data, defined like void * input_datas[input_num]. User should
 *                               initialize each data pointer as input.
 * @param [in]    input_shapes   Array of input shape, defined like bm_shape_t input_shapes[input_num].
 *                               User should set each input shape
 * @param [in]    input_num      Input number
 * @param [out]   output_datas   Array of output data, defined like void * output_datas[output_num].
 *                               If user don't alloc each output data, set user_mem to false, and this api will alloc
 *                               output mem, user should free each output mem when output data not used. Also
 *                               user can alloc system memory for each output data by self and set user_mem = true.
 * @param [out]   output_shapes  Array of output shape, defined like bm_shape_t output_shapes[output_num].
 *                               It will store each output shape.
 * @param [in]    output_num     Output number
 * @param [in]    user_mem       whether output_datas[i] have allocated memory
 * @param [in]    core_list      the cores to launch on. If core_list = NULL, core_num must be 0
 * @param [in]    core_num       number of cores to use. If core_num=0, bmruntime will alloc the proper cores automatically to launch
 *
 * @retval true    Launch success.
 * @retval false   Launch failed.
 */
DECL_EXPORT bool bmrt_launch_data_multi_cores(void* p_bmrt, const char* net_name, void* const input_datas[],
                      const bm_shape_t input_shapes[], int input_num, void * output_datas[],
                      bm_shape_t output_shapes[], int output_num, bool user_mem, const int* core_list, int core_num);


/**
 * @name    bmrt_trace
 * @brief   To check runtime environment, and collect info for DEBUG
 * @ingroup bmruntime
 *
 * This API is to collect runtime info for DEBUG. Expecially when launch result sudden mistake, call bmrt_trace
 * will show whether device mems are broken, and other check info.
 *
 * @param [in]    p_bmrt         Bmruntime that had been created
 */
DECL_EXPORT void bmrt_trace(void* p_bmrt);

/**
 * @name    bmrt_launch_tensor_multi_cores
 * @brief   To launch the inference of the neuron network with setting input tensors, and support multi core inference.
 * @ingroup bmruntime
 *
 * This API supports the neuron nework that is static-compiled or dynamic-compiled
 * After calling this API, inference on TPU is launched. And the CPU program will not
 * be blocked. bm_thread_sync_from_core should be called to make sure inference is finished.
 * This API support multiple inputs, and multi thread safety
 *
 * @param [in]    p_bmrt            Bmruntime that had been created
 * @param [in]    net_name          The name of the neuron network
 * @param [in]    input_tensors     Array of input tensor, defined like bm_tensor_t input_tensors[input_num],
 *                                  User should initialize each input tensor.
 * @param [in]    input_num         Input number
 * @param [out]   output_tensors    Array of output tensor, defined like bm_tensor_t output_tensors[output_num].
 *                                  User can set device_mem or stmode of output tensors. If user_mem is true, this interface
 *                                  will use device mem of output_tensors to store output data, and not alloc device mem;
 *                                  Or it will alloc device mem to store output. If user_stmode is true, it will use stmode in
 *                                  each output tensor; Or stmode will be BM_STORE_1N as default.
 * @param [in]    output_num        Output number
 * @param [in]    user_mem          whether device_mem of output tensors are set
 * @param [in]    user_stmode       whether stmode of output tensors are set
 * @param [in]    core_list         core id list those will be used to inference
 * @param [in]    core_num          number of the core list
 *
 * @retval true    Launch success.
 * @retval false   Launch failed.
 */
DECL_EXPORT bool bmrt_launch_tensor_multi_cores(
    void *p_bmrt,
    const char *net_name,
    const bm_tensor_t input_tensors[],
    int input_num,
    bm_tensor_t output_tensors[],
    int output_num,
    bool user_mem,
    bool user_stmode,
    const int *core_list,
    int core_num);

/**
 * @name    bmrt_pre_alloc_neuron_multi_cores
 * @brief   To pre-allocate the neuron network compute memory during multi-cores arch inference.
 * @ingroup bmruntime
 *
 * This API only used for multi-cores arch runtime, need call before bmrt_launch_tensor_multi_cores API.
 * After calling this API, the memory during neuron network inference is pre-allocated, can reduce first bmrt_launch_tensor_multi_cores API time cost.
 * If no use this API, is also OK, bmrt will auto alloc compute memory during first launch tensor.
 *
 * @param [in]    p_bmrt            Bmruntime that had been created
 * @param [in]    net_name          The name of the neuron network
 * @param [in]    stage_idx         Witch network stage need to be pre-allocate
 * @param [in]    core_list         core id list those will be used to inference
 * @param [in]    core_num          number of the core list
 *
 * @retval true    Pre-allocate success.
 * @retval false   Pre-allocate failed.
 */
DECL_EXPORT bool bmrt_pre_alloc_neuron_multi_cores(
    void *p_bmrt,
    const char *net_name,
    int stage_idx,
    const int *core_list,
    int core_num);

/**
 *  @name    bmrt_memcpy_s2d_parallel
 *  @brief   To copy data from system memory to muti-devices memory in parallel
 *  @ingroup bmruntime
 *
 *  This API only could be used when the p_bmrt is created with bmrt_create_ex on multi devices.
 *  After calling this API, datas[:tensor_num[0]] will be copied to the first device, and
 *  datas[tensor_num[0]:tensor_num[0]+tensor_num[1]] will be copied to the second device and so on.
 *  The process of copying data to different devices is done in parallel and to the same device is in sequence.
 *
 *  @param [in]     p_bmrt      Bmruntime that had been created with multi bm_handles
 *  @param [in]     tensors     Array of tensors that will be copied to devices
 *  @param [in]     datas       Array of datas allocated in system memory
 *  @param [in]     tensor_num  Array of tensor_num that will be copied to each device
 *  @param [in]     device_num  Device number
*/
DECL_EXPORT bool bmrt_memcpy_s2d_parallel(
    void *p_bmrt,
    bm_tensor_t tensors[],
    void *datas[],
    int tensor_num[],
    int device_num);

/**
 *  @name    bmrt_memcpy_d2s_parallel
 *  @brief   To copy data from muti-devices memory to system memory in parallel
 *  @ingroup bmruntime
 *
 *  This API only could be used when the p_bmrt is created with bmrt_create_ex on multi devices.
 *  After calling this API, tensors on the first device will be copied to datas[:tensor_num[0]] , and
 *  tensors on the second device will be copied to datas[tensor_num[0]:tensor_num[0]+tensor_num[1]] and so on.
 *  The process of copying data from different devices is done in parallel and from the same device is in sequence.
 *
 *  @param [in]     p_bmrt      Bmruntime that had been created with multi bm_handles
 *  @param [in]     datas       Array of datas allocated in system memory
 *  @param [in]     tensors     Array of tensors that will be copied from devices
 *  @param [in]     tensor_num  Array of tensor_num that will be copied from each device
 *  @param [in]     device_num  Device number
*/
DECL_EXPORT bool bmrt_memcpy_d2s_parallel(
    void *p_bmrt,
    void *datas[],
    bm_tensor_t tensors[],
    int tensor_num[],
    int device_num);

/**
 *  @name    bmrt_memcpy_d2d_byte_parallel
 *  @brief   To copy specified bytes of data from one piece of device memory to
 *           another piece of device memory within one device and this will be
 *           done in parallel across multi-devices. Both source and destination
 *           offsets can be specified.
 *  @ingroup bmruntime
 *
 *  This API only could be used when the p_bmrt is created with bmrt_create_ex on multi devices.
 *  After calling this API, data in src_tensors[:tensor_num[0]] on the first device will be copied
 *  to dst_tensors[:tensor_num[0]] , and src_tensors[tensor_num[0]:tensor_num[0]+tensor_num[1]] on the
 *  second device will be copied to dst_tensors[tensor_num[0]:tensor_num[0]+tensor_num[1]] and so on.
 *  The process is in parallel across different devices and is in sequence within the same device.
 *
 *  @param [in]     p_bmrt      Bmruntime that had been created with multi bm_handles
 *  @param [in]     dst_tensors Array of tensors that will be copied to devices
 *  @param [in]     dst_offsets Array of offsets for each dst_tensor (in bytes)
 *  @param [in]     src_tensors Array of tensors that will be copied from devices
 *  @param [in]     src_offsets Array of offsets for each src_tensor (in bytes)
 *  @param [in]     sizes       Array of sizes that will be copyied for each tensor (in bytes)
 *  @param [in]     tensor_num  Array of tensor_num that will be copied for each device
 *  @param [in]     device_num  Device number
*/
DECL_EXPORT bool bmrt_memcpy_d2d_byte_parallel(
    void *p_bmrt,
    bm_tensor_t dst_tensors[],
    size_t dst_offsets[],
    bm_tensor_t src_tensors[],
    size_t src_offsets[],
    size_t sizes[],
    int tensor_num[],
    int device_num);

#if defined (__cplusplus)
}
#endif

#endif
