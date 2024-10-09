#ifndef __PPL_HELPER__
#define __PPL_HELPER__

#include "common.h"

#ifndef __sg2380__
#include "tpu_kernel.h"
#endif

typedef struct {
  dim4 shape;
  dim4 stride;
  global_addr_t addr;
  data_type_t dtype;
  int mode;       // 0-LOCAL 1-L2 2-GLOBAL
  int align_mode; // 0-continuous 1-eu_align 2-compact 3-row_align
  int size;
  int offset;
  bool unsigned_flag;
  bool default_stride;
} __ppl_tensor_info;

void set_id_node(CMD_ID_NODE *pid_node);

void get_id_node(CMD_ID_NODE *pid_node);

char *__ppl_to_string(const __ppl_tensor_info *tensor);

#endif
