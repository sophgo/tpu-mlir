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

char *__ppl_to_string(const void *tensor) {
  const char *base = (const char *)tensor;
  // shape
  const dim4 *shape = (const dim4 *)(base + 0);
  // stride
  const dim4 *stride =
      (const dim4 *)(base + sizeof(dim4));
  // addr
  const global_addr_t *addr =
      (const global_addr_t *)(base +
                              2 * sizeof(dim4));
  // mode
  const int *mode = (const int *)(base + 2 * sizeof(dim4) +
                                  sizeof(global_addr_t) + sizeof(data_type_t));
  // unsigned_flag
  const bool *unsigned_flag =
      (const bool *)(base + 2 * sizeof(dim4) + sizeof(global_addr_t) +
                     sizeof(data_type_t) + 4 * sizeof(int));

  static char buffer[512];
  snprintf(buffer, sizeof(buffer),
           "\n\tshape: {%d, %d, %d, %d}\n"
           "\tstride: {%d, %d, %d, %d}\n"
           "\taddr: %llu\n"
           "\tmode: %d\n"
           "\tunsigned_flag: %d\n",
           shape->n, shape->c, shape->h, shape->w,
           stride->n, stride->c, stride->h, stride->w,
           *addr, *mode, *unsigned_flag);
  return buffer;
}
