#include "api_common.h"
#include "backend_helper.h"
#include "common_def.h"
// 1. include head file of api function
#include "api_swapchannel.h"
#include "api_absadd.h"

// 2. global backend api functions
// e.g., IMPL_CUSTOM_API_GLB(op_name, param_t)
IMPL_BACKEND_API_GLB(swapchannel, swapchannel_param_t)
