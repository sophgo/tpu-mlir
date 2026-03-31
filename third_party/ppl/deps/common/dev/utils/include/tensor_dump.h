#include "common.h"
#include "tpu_kernel.h"
#include "dev_cnpy.h"


void dump_local_mem_data(local_addr_t local_offset, int start_idx,
                         const dim4 *shape, const dim4 *stride,
                         data_type_t dtype, const char *file_path,
                         const char *tensor_name, int lane_num, bool is_rv);

void dump_global_mem_data(global_addr_t addr, const dim4 *shape,
                          const dim4 *stride, data_type_t dtype,
                          const char *file_path, const char *tensor_name,
                          int is_l2, bool is_rv);

