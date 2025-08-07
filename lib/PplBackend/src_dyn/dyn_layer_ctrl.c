#include "ppl_dyn_fw.h"

#ifdef USING_CMODEL
#define fw_log printf
#else
extern void fw_log(char *, ...);
#endif

void *ppl_global_func_map[PPL_FW_LAYER_TYPE_UNKNOWN - PPL_FW_LAYER_START] = {0};
void *ppl_local_func_map[PPL_FW_LAYER_TYPE_UNKNOWN - PPL_FW_LAYER_START] = {0};

void set_out_info(local_output_info_t *top, local_tensor_ref_t *in_ref,
                  local_tensor_ref_t *out_ref, local_sec_info_t *sec_info) {
  int dims = in_ref->spec->dims;
  memset(top, 0, sizeof(local_output_info_t));
  top->reference_id = -1;
  top->local_offset = out_ref->spec->addr;
  top->real_h_slice = sec_info->h_slice;
  top->real_c_slice = in_ref->spec->shape[1];
  top->info.dtype = out_ref->spec->dtype;
  top->info.id = out_ref->id;
  top->info.h_idx = in_ref->h_idx;
  top->info.h_slice = in_ref->h_slice;
  top->info.pad_h_top = in_ref->pad_h_top;
  top->info.dims = in_ref->spec->dims;
  memcpy(top->info.shape, in_ref->spec->shape, sizeof(int) * dims);
  top->info.consume_num = out_ref->spec->consume_num;
}

// TPU1686 interface
typedef void (*ppl_global_layer_func_t)(void *ctx, void *param,
                                        global_tensor_spec_t *input_spec,
                                        global_tensor_spec_t *output_spec);
void *call_ppl_global_func(void *ctx, int type_id, void *param,
                           global_tensor_spec_t *input_spec,
                           global_tensor_spec_t *output_spec) {
  if (type_id >= PPL_FW_LAYER_TYPE_UNKNOWN || type_id < PPL_FW_LAYER_START) {
    TPUKERNEL_ERR("%s, invalid layer_type=%d\n", __func__, type_id);
  }
  ppl_global_layer_func_t layer_func =
      (ppl_global_layer_func_t)ppl_global_func_map[type_id - PPL_FW_LAYER_START];
  if (!layer_func)
    TPUKERNEL_ERR("%s, invalid layer_type=%d, init ctrl func first\n", __func__, type_id);
  layer_func(ctx, param, input_spec, output_spec);
}

typedef void (*ppl_local_layer_func_t)(void *ctx, void *param, 
                                       local_sec_info_t *sec_info,
                                       local_output_info_t *top,
                                       dynamic_local_tensor_info_t *tensor_info);
void *call_ppl_local_func(void* ctx, int type_id, void *param, 
                          local_sec_info_t *sec_info,
                          local_output_info_t *top,
                          dynamic_local_tensor_info_t *tensor_info) {
  if (type_id >= PPL_FW_LAYER_TYPE_UNKNOWN || type_id < PPL_FW_LAYER_START) {
    TPUKERNEL_ERR("%s, invalid layer_type=%d\n", __func__, type_id);
  }
  ppl_local_layer_func_t layer_func =
      (ppl_local_layer_func_t)ppl_local_func_map[type_id - PPL_FW_LAYER_START];
  if (!layer_func)
    TPUKERNEL_ERR("%s, invalid layer_type=%d, init ctrl func first\n", __func__, type_id);
  layer_func(ctx, param, sec_info, top, tensor_info);
}
