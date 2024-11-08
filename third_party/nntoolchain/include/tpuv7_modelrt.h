#ifndef __TPU_MODEL_RT__
#define __TPU_MODEL_RT__
#include "tpuv7_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

/* tpuRtDataType_t holds the type for a scalar value */
typedef enum {
  TPU_FLOAT32 = 0,
  TPU_FLOAT16 = 1,
  TPU_INT8 = 2,
  TPU_UINT8 = 3,
  TPU_INT16 = 4,
  TPU_UINT16 = 5,
  TPU_INT32 = 6,
  TPU_UINT32 = 7,
  TPU_BFLOAT16 = 8,
  TPU_INT4 = 9,
  TPU_UINT4 = 10,
} tpuRtDataType_t;

/* tpuRtShape_t holds the shape info */
#define TPU_MAX_DIMS_NUM 8
typedef struct {
  int num_dims;
  int dims[TPU_MAX_DIMS_NUM];
} tpuRtShape_t;

/*
tpu_tensor_t holds a multi-dimensional array of elements of a single data type
and tensor are in device memory */
typedef struct {
  tpuRtDataType_t dtype;
  tpuRtShape_t shape;
  void *data; // data in device mem
  unsigned char reserved[64];
} tpuRtTensor_t;

/* --------------------------------------------------------------------------*/
/* network information structure */

/* tpu_stage_info_t holds input/output shapes and device mems; every network can
 * contain one or more stages */
typedef struct tpu_stage_info_s {
  // input_shapes[0] / [1] / ... / [input_num-1]
  tpuRtShape_t *input_shapes;
  // output_shapes[0] / [1] / ... / [output_num-1]
  tpuRtShape_t *output_shapes;
  // inputs device memory which has been malloc after the net is loaded, and
  // user can reuse these device memory to store inputs' data which can decrease
  // latency.
  void **input_mems;
  // outputs device memory which has been malloc after the net is loaded, and
  // user can reuse these device memory to store outputs' data which can
  // decrease latency.
  void **output_mems;
} tpuRtStageInfo_t;

typedef struct {
  int num;
  const char **names;
  float *scales;
  int *zero_points;
  tpuRtDataType_t *dtypes;
} tpuRtIOInfo_t;

typedef struct tpu_net_info_s {
  char const *name; /* net name */
  bool is_dynamic;  /* dynamic or static */
  tpuRtIOInfo_t input;
  tpuRtIOInfo_t output;
  int stage_num;            /* number of stages */
  tpuRtStageInfo_t *stages; /* stages[0] / [1] / ... / [stage_num-1] */
  unsigned char reserved[64];
} tpuRtNetInfo_t;

typedef void *tpuRtNet_t;
typedef void *tpuRtNetContext_t;

/**
 * @brief Create Context for a Net.
 *
 * This function create context for a net and save the result in the
 * provided pointer.
 *
 * @param context The pointer of tpuRtNetContext_t.
 *
 * @return tpuRtStatus_t Returns tpuRtSuccess on success or other value on fail.
 */
tpuRtStatus_t tpuRtCreateNetContext(tpuRtNetContext_t *context);

/**
 * @brief Destroy Context
 *
 * This function destroy the net context created by tpuRtCreateNetContext.
 *
 * @param context The value of tpuRtNetContext_t.
 *
 * @return tpuRtStatus_t Returns tpuRtSuccess on success or other value on fail.
 */
tpuRtStatus_t tpuRtDestroyNetContext(tpuRtNetContext_t context);

/**
 * @brief Load a net file to memory on device.
 *
 * This function load a net from file, save weight and commands in memory.
 * And analyse net file to get related net infomation.
 *
 * @param net_path The full path to a net.
 * @param context The context created by tpuRtCreateNetContext.
 * @param net The pointer to tpuRtNet_t.
 *
 * @return tpuRtStatus_t Returns tpuRtSuccess on success or other value on fail.
 *
 * @note The net_path can not be NULL and Context must be created first.
 */
tpuRtStatus_t tpuRtLoadNet(const char *net_path, tpuRtNetContext_t context,
                           tpuRtNet_t *net);

/**
 * @brief Load a net from buffer to device.
 *
 * This function load a net from buffer, save weight and commands in memory.
 * Analyse net file to get related net infomation.
 *
 * @param net_data The buffer pointer that hold the net.
 * @param size The data length of net_data.
 * @param context The context created by tpuRtCreateNetContext.
 * @param net The pointer to tpuRtNet_t.
 *
 * @return tpuRtStatus_t Returns tpuRtSuccess on success or other value on fail.
 *
 * @note The net_data can not be NULL and Context must be created first.
 */
tpuRtStatus_t tpuRtLoadNetFromMem(const void *net_data, size_t size,
                                  tpuRtNetContext_t context, tpuRtNet_t *net);

/**
 * @brief Unload net.
 *
 * This function unload the net loaded by tpuRtLoadNet*** and release resources
 * created.
 *
 * @param net The net to unload.
 *
 * @return tpuRtStatus_t Returns tpuRtSuccess on success or other value on fail.
 */
tpuRtStatus_t tpuRtUnloadNet(tpuRtNet_t net);

/**
 * @brief Get information of a net.
 *
 * This function is used to get the information of a net loaded by
 * tpuRtLoadNet***. Since a net may be combined by one or more subnet, the name
 * of the subnet is need.
 *
 * @param net The net loaded by tpuRtLoadNet***.
 * @param name The name of subnet you need to get the net information.
 *
 * @return tpuRtNetInfo_t The information structure get by the function.
 *
 * @note Net must be loaded first and name can not be NULL.
 */
tpuRtNetInfo_t tpuRtGetNetInfo(const tpuRtNet_t net, const char *name);

/**
 * @brief Launch net in async mode.
 *
 * This function is used to launch a net with related input and name to get
 * output in asynchronous mode by a specified stream.
 *
 * @param net The net loaded by tpuRtLoadNet***.
 * @param input The array of input tensors.
 * @param output The array to hold the output tensors.
 * @param net_name The net name need to launch.
 * @param stream The stream to do the launch operation.
 *
 * @return tpuRtStatus_t Returns tpuRtSuccess on success or other value on fail.
 *
 * @note Net must be loaded first, net_name is get from tpuRtGetNetNames.
 * Stream must be created before launch.
 */
tpuRtStatus_t tpuRtLaunchNetAsync(tpuRtNet_t net, const tpuRtTensor_t input[],
                                  tpuRtTensor_t output[], const char *net_name,
                                  tpuRtStream_t stream);

/**
 * @brief Launch net in sync mode.
 *
 * This function is used to launch a net with related input and name to get
 * output in synchronous mode by a specified stream.
 *
 * @param net The net loaded by tpuRtLoadNet***.
 * @param input The array of input tensors.
 * @param output The array to hold the output tensors.
 * @param net_name The net name need to launch.
 * @param stream The stream to do the launch operation.
 *
 * @return tpuRtStatus_t Returns tpuRtSuccess on success or other value on fail.
 *
 * @note Net must be loaded first, net_name is get from tpuRtGetNetNames.
 * Stream must be created before launch.
 */
tpuRtStatus_t tpuRtLaunchNet(tpuRtNet_t net, const tpuRtTensor_t input[],
                             tpuRtTensor_t output[], const char *net_name,
                             tpuRtStream_t stream);

/**
 * @brief Get net names from loaded net.
 *
 * Since a net maybe combined with one or more subnet, this function is used to
 * get all the subnet names.
 *
 * @param net The net loaded by tpuRtLoadNet***.
 * @param names The pointer to pointer of names.
 *
 * @return int Return the subnet numbers.
 */
int tpuRtGetNetNames(const tpuRtNet_t net, char ***names);

/**
 * @brief Free net names.
 *
 * This function is used to free the names get form tpuRtGetNetNames.
 *
 * @param names The names array need to free.
 *
 * @return No return value.
 */
void tpuRtFreeNetNames(char **names);

#ifdef __cplusplus
}
#endif

#endif // end of __TPU_MODEL_RT__
