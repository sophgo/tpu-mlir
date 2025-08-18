#include "selective_scan.h"
#include "helper.h"
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <ppl_mem.h>
#include <sstream>
#include <unistd.h>
#include <vector>
// #include "tpu_defs.h"
#include "tpu_mlir/Backend/BM168x/Param.h"
#define MIN(x, y) (((x)) < ((y)) ? (x) : (y))
#define MAX(x, y) (((x)) > ((y)) ? (x) : (y))

#ifdef __cplusplus
extern "C" {
#endif

int type_call(int dtype, uint64_t output_addr, uint64_t Cs_addr,
              uint64_t deltaA_addr, uint64_t deltaB_u_addr, uint64_t us_addr,
              uint64_t Ds_addr, int Batch, int KC_dim, int L) {
  int ret = -1;
  switch (dtype) {
  case DTYPE_FP32:
    ret = selective_scan_f32(output_addr, Cs_addr, deltaA_addr, deltaB_u_addr,
                             us_addr, Ds_addr, Batch, KC_dim, L);
    CHECK_PPL_RET(ret);
    break;
  case DTYPE_FP16:
    ret = selective_scan_fp16(output_addr, Cs_addr, deltaA_addr, deltaB_u_addr,
                              us_addr, Ds_addr, Batch, KC_dim, L);
    CHECK_PPL_RET(ret);
    break;
  case DTYPE_BFP16:
    ret = selective_scan_bf16(output_addr, Cs_addr, deltaA_addr, deltaB_u_addr,
                              us_addr, Ds_addr, Batch, KC_dim, L);
    CHECK_PPL_RET(ret);
    break;
  default:
    assert(0 && "not supported, need fix selective_scan_fp.pl\n");
  }
  if (ret == 0) {
    return 0;
  } else {
    assert(0 && "not supported\n");
    return ret;
  }
  return 1;
}

void api_selective_scan_global(void *param, size_t param_size) {

  selective_scan_common_spec_t *_param = (selective_scan_common_spec_t *)param;

  type_call(_param->dtype, _param->output_addr, _param->Cs_addr,
            _param->deltaA_addr, _param->deltaB_u_addr, _param->us_addr,
            _param->Ds_addr, _param->Batch, _param->KC_dim, _param->L);
}

#ifdef __cplusplus
}
#endif
