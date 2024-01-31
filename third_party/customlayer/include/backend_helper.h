#ifndef BACKEND_HELPER_H
#define BACKEND_HELPER_H

#include "common.h"
#include "tpu_kernel.h"

static void create_id_node_guard(CMD_ID_NODE* pid_node) {
  if(!pid_node) return;
  if(pid_node->in_parallel_state){
    tpu_set_parallel_id_node(pid_node, NULL);
  } else {
    tpu_set_id_node(pid_node);
  }
  tpu_disable_check_id_node();
}
static void free_id_node_guard(CMD_ID_NODE* pid_node){
  if(!pid_node) return;
  if(pid_node->in_parallel_state){
    tpu_get_parallel_id_node(pid_node, NULL);
  } else {
    tpu_get_id_node(pid_node);
  }
  tpu_enable_check_id_node();
}

#define IMPL_CUSTOM_API_GLB(name)                                         \
    int backend_api_##name##_global(                                      \
        const void *param,                                                \
        int param_size,                                                   \
        const global_tensor_spec_t *inputs,                               \
        global_tensor_spec_t *outputs,                                    \
        void *pid_node)                                                   \
    {                                                                     \
      create_id_node_guard((CMD_ID_NODE*)pid_node);                       \
      api_##name##_global(inputs, outputs, param);                        \
      free_id_node_guard((CMD_ID_NODE*)pid_node);                         \
      return 0;                                                           \
    }                                                                     \

#define IMPL_CUSTOM_API_LOC(name)                                         \
    int backend_api_##name##_local(                                       \
        const void *param,                                                \
        int param_size,                                                   \
        local_sec_info_t *sec_info,                                       \
        const local_tensor_spec_t *inputs,                                \
        local_tensor_spec_t *outputs,                                       \
        void *pid_node)                                                   \
    {                                                                     \
      create_id_node_guard((CMD_ID_NODE*)pid_node);                       \
      api_##name##_local(sec_info, inputs, outputs, param);               \
      free_id_node_guard((CMD_ID_NODE*)pid_node);                         \
      return 0;                                                           \
    }                                                                     \

#define IMPL_CUSTOM_API_LOC_BFSZ(name)                                    \
    int backend_api_##name##_local_bfsz(                                  \
        const void *param,                                                \
        int param_size,                                                   \
        local_sec_info_t *sec_info,                                       \
        local_tensor_spec_t *inputs,                                      \
        local_tensor_spec_t *outputs)                                     \
    {                                                                     \
      return api_##name##_local_bfsz(sec_info, inputs, outputs, param);   \
    }                                                                     \

#endif
