#ifndef __PPL_HELPER__
#define __PPL_HELPER__

#include "common.h"
#include "tpu_kernel.h"

void set_id_node(CMD_ID_NODE *pid_node);

void get_id_node(CMD_ID_NODE *pid_node);

char *__ppl_to_string(const void *tensor);

void print_local_mem_data(local_addr_t local_offset, int start_idx,
                          const dim4 *shape, const dim4 *stride,
                          data_type_t dtype, int lane_num);

void print_global_mem_data(global_addr_t addr, const dim4 *shape,
                           const dim4 *stride, data_type_t dtype,
                           bool is_global);
void __attribute__((weak)) fw_log(char *fmt, ...) {}
#endif
