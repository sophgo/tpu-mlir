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
  void *input_mems;
  void *output_mems;
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

tpuRtStatus_t tpuRtCreateNetContext(tpuRtNetContext_t *context);
tpuRtStatus_t tpuRtDestroyNetContext(tpuRtNetContext_t context);
tpuRtStatus_t tpuRtLoadNet(const char *net_path, tpuRtNetContext_t context,
                           tpuRtNet_t *net);
tpuRtStatus_t tpuRtLoadNetFromMem(const void *net_data, size_t size,
                                  tpuRtNetContext_t context, tpuRtNet_t *net);
tpuRtStatus_t tpuRtUnloadNet(tpuRtNet_t net);
tpuRtNetInfo_t tpuRtGetNetInfo(const tpuRtNet_t net, const char *name);
tpuRtStatus_t tpuRtLaunchNetAsync(tpuRtNet_t net, const tpuRtTensor_t input[],
                                  tpuRtTensor_t output[], const char *net_name,
                                  tpuRtStream_t stream);
tpuRtStatus_t tpuRtLaunchNet(tpuRtNet_t net, const tpuRtTensor_t input[],
                             tpuRtTensor_t output[], const char *net_name,
                             tpuRtStream_t stream);
int tpuRtGetNetNames(const tpuRtNet_t net, char ***names);
void tpuRtFreeNetNames(char **names);

#ifdef __cplusplus
}
#endif

#endif // end of __TPU_MODEL_RT__
