#ifndef BACKEND_HELPER_H
#define BACKEND_HELPER_H

#include "common.h"
#include "tpu_kernel.h"
#include "backend_custom_param.h"

class id_node_guard {
public:
  id_node_guard(void* pid_node_):pid_node((CMD_ID_NODE*)pid_node_) {
    if(!pid_node) return;
    if(pid_node->in_parallel_state){
      tpu_set_parallel_id_node(pid_node, NULL);
    } else {
      tpu_set_id_node(pid_node);
    }
    tpu_disable_check_id_node();
  }
  ~id_node_guard(){
    if(!pid_node) return;
    if(pid_node->in_parallel_state){
      tpu_get_parallel_id_node(pid_node, NULL);
    } else {
      tpu_get_id_node(pid_node);
    }
    tpu_enable_check_id_node();
  }
private:
  CMD_ID_NODE* pid_node;
};

#define USE_NODE(node) id_node_guard __guard(node)

#define IMPL_CUSTOM_API_GLB(name, type)                                   \
  extern "C"{                                                             \
    int backend_api_##name##_global(                                      \
        custom_param_t *param,                                            \
        int param_size,                                                   \
        global_tensor_spec_t *inputs,                                     \
        global_tensor_spec_t *outputs,                                    \
        void *pid_node)                                                   \
    {                                                                     \
      USE_NODE(pid_node);                                                 \
      api_##name##_global(inputs, outputs, param);                        \
      return 0;                                                           \
    }                                                                     \
  }                                                                       \

#define IMPL_CUSTOM_API_LOC(name, type)                                   \
  extern "C"{                                                             \
    int backend_api_##name##_local(                                       \
        custom_param_t *param,                                            \
        int param_size,                                                   \
        local_sec_info_t *sec_info,                                       \
        local_tensor_spec *inputs,                                        \
        local_tensor_spec *outputs,                                       \
        void *pid_node)                                                   \
    {                                                                     \
      USE_NODE(pid_node);                                                 \
      api_##name##_local(sec_info, inputs, outputs, param);               \
      return 0;                                                           \
    }                                                                     \
  }

#endif
