#ifndef __PPL_HELPER__
#define __PPL_HELPER__

#include "common.h"

#ifndef __sg2380__
#include "tpu_kernel.h"
#endif

void set_id_node(CMD_ID_NODE *pid_node);

void get_id_node(CMD_ID_NODE *pid_node);

char *__ppl_to_string(const void *tensor);

#endif
