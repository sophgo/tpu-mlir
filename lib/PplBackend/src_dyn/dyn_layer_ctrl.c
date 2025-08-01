#include "add_const_fp.c"
#include "fattention_bf16.c"
#include "fattention_fp16.c"
#include "fattention_v2.c"

#include "ppl_dyn_fw.h"

/*============================================================================*/
/*==================== protected code borrowed from TPU1686 ==================*/
/*============================================================================*/
#ifdef USING_CMODEL
#define fw_log printf
#else
extern void fw_log(char *, ...);
#endif

#define MAX_SHAPE_DIMS 8
typedef struct local_tensor_spec {
  uint64_t addr;
  int32_t dtype;
  int32_t dims;
  int32_t shape[MAX_SHAPE_DIMS];
  uint8_t consume_num;
  int *host_data;
  int elem_num;
} tensor_spec_t;

typedef tensor_spec_t local_tensor_spec_t;
typedef tensor_spec_t global_tensor_spec_t;

// dynamic local ref relate
typedef enum fw_data_type {
  FW_DTYPE_FP32 = 0,
  FW_DTYPE_FP16 = 1,
  FW_DTYPE_INT8 = 2,
  FW_DTYPE_UINT8 = 3,
  FW_DTYPE_INT16 = 4,
  FW_DTYPE_UINT16 = 5,
  FW_DTYPE_INT32 = 6,
  FW_DTYPE_UINT32 = 7,
  FW_DTYPE_BFP16 = 8,
} FW_DATA_TYPE_T;

typedef struct local_tensor_ref {
  int type;
  int32_t id;
  int pad_h_top;
  int h_idx;
  int h_slice;
  int pad_w_left;
  int w_idx;
  int w_slice;
  int consume_num;
  int sign;
  local_tensor_spec_t *spec;
} local_tensor_ref_t;

typedef struct {
  int id;
  int pad_h_top;
  int h_idx;
  int h_slice;
  int pad_w_left;
  int w_idx;
  int w_slice;
  int dims;
  int shape[MAX_SHAPE_DIMS];
  u8 consume_num;
  u8 sign;
  int elem_num; // record the real element number. If 0, do not need to count
                // elem num
  FW_DATA_TYPE_T dtype;
} fw_local_tensor_info_t;

typedef struct {
  u32 reference_id;
  u32 local_offset;
  u32 real_h_slice;
  u32 real_c_slice;
  fw_local_tensor_info_t info;
} local_output_info_t;

typedef struct local_sec_info {
  int32_t group_type;

  int32_t n_slice;
  int32_t out_n_slice;

  int32_t d_slice;

  int32_t is_h_split;
  int32_t h_idx;
  int32_t h_slice;
  int32_t out_h_idx;
  int32_t out_h_slice;

  int32_t is_w_split;
  int32_t w_idx;
  int32_t w_slice;
  int32_t out_w_idx;
  int32_t out_w_slice;

  int32_t is_c_split;
  int32_t c_idx;
  int32_t c_slice;

  int32_t hw_margins_opA;
  int32_t hw_margins_opB;
} local_sec_info_t;

static void set_out_info(local_output_info_t *top, local_tensor_ref_t *in_ref,
                         local_tensor_ref_t *out_ref,
                         local_sec_info_t *sec_info) {
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
typedef void (*ppl_global_layer_func_t)(void *param,
                                        global_tensor_spec_t *input_spec,
                                        global_tensor_spec_t *output_spec);
void *call_ppl_global_func(int type_id, void *param,
                           global_tensor_spec_t *input_spec,
                           global_tensor_spec_t *output_spec) {
  if (type_id >= PPL_FW_LAYER_TYPE_UNKNOWN || type_id < PPL_FW_LAYER_START) {
    TPUKERNEL_ERR("%s, invalid layer_type=%d\n", __func__, type_id);
  }
  fw_init_ppl_func_map();
  ppl_global_layer_func_t layer_func =
      (ppl_global_layer_func_t)ppl_global_func_map[type_id];
  layer_func(param, input_spec, output_spec);
}

typedef void (*ppl_local_layer_func_t)(void *param, local_sec_info_t *sec_info,
                                       local_output_info_t *top,
                                       local_tensor_ref_t *input_ref,
                                       local_tensor_ref_t *output_ref);
void *call_ppl_local_func(int type_id, void *param, local_sec_info_t *sec_info,
                          local_output_info_t *top,
                          local_tensor_ref_t *input_ref,
                          local_tensor_ref_t *output_ref) {
  if (type_id >= PPL_FW_LAYER_TYPE_UNKNOWN || type_id < PPL_FW_LAYER_START) {
    TPUKERNEL_ERR("%s, invalid layer_type=%d\n", __func__, type_id);
  }
  fw_init_ppl_func_map();
  ppl_local_layer_func_t layer_func =
      (ppl_local_layer_func_t)ppl_local_func_map[type_id];
  layer_func(param, sec_info, top, input_ref, output_ref);
}
/* protected code end*/

/*============================================================================*/
/*========================= edit custom code form here =======================*/
/*============================================================================*/
void dynamic_glb_add_const_fp_layer_ctrl(void *param,
                                         global_tensor_spec_t *input_spec,
                                         global_tensor_spec_t *output_spec) {
  output_spec->dtype = input_spec->dtype;
  memcpy(output_spec->shape, input_spec->shape, input_spec->dims * sizeof(int));
  output_spec->dims = input_spec->dims;
  output_spec->elem_num = input_spec->elem_num;

  if (input_spec->dtype == FW_DTYPE_FP16) {
    tpu_kernel_api_add_const_f16_t *_param =
        (tpu_kernel_api_add_const_f16_t *)param;
    _param->ptr_dst = output_spec->addr;
    _param->ptr_src = input_spec->addr;
    _param->N = input_spec->shape[0];
    _param->C = input_spec->shape[1];
    _param->H = input_spec->shape[2];
    _param->W = input_spec->shape[3];
    add_const_f16(_param);
  } else if (input_spec->dtype == FW_DTYPE_BFP16) {
    tpu_kernel_api_add_const_bf16_t *_param =
        (tpu_kernel_api_add_const_bf16_t *)param;
    _param->ptr_dst = output_spec->addr;
    _param->ptr_src = input_spec->addr;
    _param->N = input_spec->shape[0];
    _param->C = input_spec->shape[1];
    _param->H = input_spec->shape[2];
    _param->W = input_spec->shape[3];
    add_const_bf16(_param);
  } else if (input_spec->dtype == FW_DTYPE_FP32) {
    tpu_kernel_api_add_const_f32_t *_param =
        (tpu_kernel_api_add_const_f32_t *)param;
    _param->ptr_dst = output_spec->addr;
    _param->ptr_src = input_spec->addr;
    _param->N = input_spec->shape[0];
    _param->C = input_spec->shape[1];
    _param->H = input_spec->shape[2];
    _param->W = input_spec->shape[3];
    add_const_f32(_param);
  }
}

void dynamic_glb_flash_attention_layer_ctrl(void *param,
                                            global_tensor_spec_t *input_spec,
                                            global_tensor_spec_t *output_spec) {
  output_spec->dtype = input_spec->dtype;
  output_spec->dims = 3;
  output_spec->shape[0] = input_spec->shape[0];
  output_spec->shape[1] = input_spec->shape[1];
  output_spec->shape[2] = input_spec->shape[2] * input_spec->shape[3];
  bool is_mha = input_spec->shape[2] == (input_spec + 1)->shape[2];
  if (input_spec->dtype == FW_DTYPE_FP16) {
    if (is_mha) {
      tpu_kernel_api_flash_attention_mha_f16_t *_param =
          (tpu_kernel_api_flash_attention_mha_f16_t *)param;
      _param->ptr_out = output_spec->addr;
      _param->ptr_q = input_spec[0].addr;
      _param->ptr_k = input_spec[1].addr;
      _param->ptr_v = input_spec[2].addr;
      _param->ptr_mask = input_spec[3].addr;
      _param->b = input_spec->shape[0];
      _param->qm = input_spec->shape[1];
      _param->kvm = (input_spec + 1)->shape[1];
      _param->d = input_spec->shape[3];
      _param->q_head = input_spec->shape[2];
      _param->kv_head = (input_spec + 1)->shape[2];
      flash_attention_mha_f16(_param);
    } else {
      tpu_kernel_api_flash_attention_gqa_f16_t *_param =
          (tpu_kernel_api_flash_attention_gqa_f16_t *)param;
      _param->ptr_out = output_spec->addr;
      _param->ptr_q = input_spec[0].addr;
      _param->ptr_k = input_spec[1].addr;
      _param->ptr_v = input_spec[2].addr;
      _param->ptr_mask = input_spec[3].addr;
      _param->b = input_spec->shape[0];
      _param->qm = input_spec->shape[1];
      _param->kvm = (input_spec + 1)->shape[1];
      _param->d = input_spec->shape[3];
      _param->q_head = input_spec->shape[2];
      _param->kv_head = (input_spec + 1)->shape[2];
      flash_attention_gqa_f16(_param);
    }
  } else if (input_spec->dtype == FW_DTYPE_BFP16) {
    if (is_mha) {
      tpu_kernel_api_flash_attention_mha_bf16_t *_param =
          (tpu_kernel_api_flash_attention_mha_bf16_t *)param;
      _param->ptr_out = output_spec->addr;
      _param->ptr_q = input_spec[0].addr;
      _param->ptr_k = input_spec[1].addr;
      _param->ptr_v = input_spec[2].addr;
      _param->ptr_mask = input_spec[3].addr;
      _param->b = input_spec->shape[0];
      _param->qm = input_spec->shape[1];
      _param->kvm = (input_spec + 1)->shape[1];
      _param->d = input_spec->shape[3];
      _param->q_head = input_spec->shape[2];
      _param->kv_head = (input_spec + 1)->shape[2];
      flash_attention_mha_bf16(_param);
    } else {
      tpu_kernel_api_flash_attention_gqa_bf16_t *_param =
          (tpu_kernel_api_flash_attention_gqa_bf16_t *)param;
      _param->ptr_out = output_spec->addr;
      _param->ptr_q = input_spec[0].addr;
      _param->ptr_k = input_spec[1].addr;
      _param->ptr_v = input_spec[2].addr;
      _param->ptr_mask = input_spec[3].addr;
      _param->b = input_spec->shape[0];
      _param->qm = input_spec->shape[1];
      _param->kvm = (input_spec + 1)->shape[1];
      _param->d = input_spec->shape[3];
      _param->q_head = input_spec->shape[2];
      _param->kv_head = (input_spec + 1)->shape[2];
      flash_attention_gqa_bf16(_param);
    }
  }
}

void dynamic_glb_flash_attention_heigh_prec_layer_ctrl(
    void *param, global_tensor_spec_t *input_spec,
    global_tensor_spec_t *output_spec) {
  output_spec->dtype = input_spec->dtype;
  output_spec->dims = 3;
  output_spec->shape[0] = input_spec->shape[0];
  output_spec->shape[1] = input_spec->shape[1];
  output_spec->shape[2] = input_spec->shape[2] * input_spec->shape[3];
  bool is_mha = input_spec->shape[2] == (input_spec + 1)->shape[2];
  if (input_spec->dtype == FW_DTYPE_FP16) {
    if (is_mha) {
      tpu_kernel_api_flash_attention_mha_f16_high_precision_t *_param =
          (tpu_kernel_api_flash_attention_mha_f16_high_precision_t *)param;
      _param->ptr_out = output_spec->addr;
      _param->ptr_q = input_spec[0].addr;
      _param->ptr_k = input_spec[1].addr;
      _param->ptr_v = input_spec[2].addr;
      _param->ptr_mask = input_spec[3].addr;
      _param->b = input_spec->shape[0];
      _param->qm = input_spec->shape[1];
      _param->kvm = (input_spec + 1)->shape[1];
      _param->d = input_spec->shape[3];
      _param->q_head = input_spec->shape[2];
      _param->kv_head = (input_spec + 1)->shape[2];
      flash_attention_mha_f16_high_precision(_param);
    } else {
      tpu_kernel_api_flash_attention_gqa_f16_high_precision_t *_param =
          (tpu_kernel_api_flash_attention_gqa_f16_high_precision_t *)param;
      _param->ptr_out = output_spec->addr;
      _param->ptr_q = input_spec[0].addr;
      _param->ptr_k = input_spec[1].addr;
      _param->ptr_v = input_spec[2].addr;
      _param->ptr_mask = input_spec[3].addr;
      _param->b = input_spec->shape[0];
      _param->qm = input_spec->shape[1];
      _param->kvm = (input_spec + 1)->shape[1];
      _param->d = input_spec->shape[3];
      _param->q_head = input_spec->shape[2];
      _param->kv_head = (input_spec + 1)->shape[2];
      flash_attention_gqa_f16_high_precision(_param);
    }
  } else if (input_spec->dtype == FW_DTYPE_BFP16) {
    if (is_mha) {
      tpu_kernel_api_flash_attention_mha_bf16_high_precision_t *_param =
          (tpu_kernel_api_flash_attention_mha_bf16_high_precision_t *)param;
      _param->ptr_out = output_spec->addr;
      _param->ptr_q = input_spec[0].addr;
      _param->ptr_k = input_spec[1].addr;
      _param->ptr_v = input_spec[2].addr;
      _param->ptr_mask = input_spec[3].addr;
      _param->b = input_spec->shape[0];
      _param->qm = input_spec->shape[1];
      _param->kvm = (input_spec + 1)->shape[1];
      _param->d = input_spec->shape[3];
      _param->q_head = input_spec->shape[2];
      _param->kv_head = (input_spec + 1)->shape[2];
      flash_attention_mha_bf16_high_precision(_param);
    } else {
      tpu_kernel_api_flash_attention_gqa_bf16_high_precision_t *_param =
          (tpu_kernel_api_flash_attention_gqa_bf16_high_precision_t *)param;
      _param->ptr_out = output_spec->addr;
      _param->ptr_q = input_spec[0].addr;
      _param->ptr_k = input_spec[1].addr;
      _param->ptr_v = input_spec[2].addr;
      _param->ptr_mask = input_spec[3].addr;
      _param->b = input_spec->shape[0];
      _param->qm = input_spec->shape[1];
      _param->kvm = (input_spec + 1)->shape[1];
      _param->d = input_spec->shape[3];
      _param->q_head = input_spec->shape[2];
      _param->kv_head = (input_spec + 1)->shape[2];
      flash_attention_gqa_bf16_high_precision(_param);
    }
  }
}

void fw_init_ppl_func_map() {
  if (ppl_func_map_inited)
    return;
  GLOBAL_FUNC_BIND(PPL_FW_FLASH_ATTENTION,
                   dynamic_glb_flash_attention_layer_ctrl);
  GLOBAL_FUNC_BIND(PPL_FW_FLASH_ATTENTION_HEIGH_PRECISION,
                   dynamic_glb_flash_attention_heigh_prec_layer_ctrl);
  GLOBAL_FUNC_BIND(PPL_FW_ADD_CONST, dynamic_glb_add_const_fp_layer_ctrl);
  ppl_func_map_inited = 1;
}
// edit custom code to this end
