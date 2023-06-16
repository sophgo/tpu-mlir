#ifndef BACKEND_HELPER_H
#define BACKEND_HELPER_H

#include "common.h"
#include "tpu_kernel.h"
#include "backend_api_param.h"

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

#define _BACKEND_PARSE_API(type, name, param, param_size)          \
  TPUKERNEL_ASSERT((size_t)param_size <= sizeof(type));            \
  type name = {0};                                                 \
  memcpy(&name, param, param_size);

#define BACKEND_PARSE_API(type, name, param, param_size, pid_node) \
  _BACKEND_PARSE_API(type, name, param, param_size)                \
  USE_NODE(pid_node)

#define IMPL_BACKEND_API_GLB(name, type)                          \
  extern "C"{                                                             \
    int backend_api_##name##_global(                                      \
        void *param,                                                      \
        int param_size,                                                   \
        global_tensor_spec_t *inputs,                                     \
        global_tensor_spec_t *outputs,                                    \
        void *pid_node)                                                   \
    {                                                                     \
      BACKEND_PARSE_API(type, api, param, param_size, pid_node);          \
      api_##name##_global(inputs, outputs, &api);                         \
      return 0;                                                           \
    }                                                                     \
  }                                                                       \

#endif

