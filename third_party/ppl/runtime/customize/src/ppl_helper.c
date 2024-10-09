#include "ppl_helper.h"

void set_id_node(CMD_ID_NODE *pid_node) {
  if (pid_node == NULL) {
    return;
  }
  if (pid_node->in_parallel_state) {
    tpu_set_parallel_id_node(pid_node, NULL);
  } else {
    tpu_set_id_node(pid_node);
  }
  tpu_disable_check_id_node();
}

void get_id_node(CMD_ID_NODE *pid_node) {
  if (!pid_node)
    return;
  if (pid_node->in_parallel_state) {
    tpu_get_parallel_id_node(pid_node, NULL);
  } else {
    tpu_get_id_node(pid_node);
  }
  tpu_enable_check_id_node();
}

char *__ppl_to_string(const __ppl_tensor_info *tensor) {
  static char buffer[512];
  snprintf(buffer, sizeof(buffer),
           "\n\tshape: {%d, %d, %d, %d}\n"
           "\tstride: {%d, %d, %d, %d}\n"
           "\taddr: %llu\n"
           "\tmode: %d\n"
           "\tunsigned_flag: %d\n",
           tensor->shape.n, tensor->shape.c, tensor->shape.h, tensor->shape.w,
           tensor->stride.n, tensor->stride.c, tensor->stride.h,
           tensor->stride.w, tensor->addr, tensor->mode, tensor->unsigned_flag);
  return buffer;
}
