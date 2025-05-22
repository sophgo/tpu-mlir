//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "dlfcn.h"
#include "tpu_mlir/Backend/BM168x/BM1688.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <fcntl.h>
#include <sys/file.h>

using namespace tpu_mlir::backend;

uint64_t BM168x::get_cmodel_gmem_start_addr() {
  return dl_tpu_global_mem_get_start_addr();
}

uint64_t BM168x::get_cmodel_l2mem_start_addr() {
  return dl_tpu_l2_sram_get_start_addr();
}

void *BM168x::get_gmem_addr(uint64_t addr) {
  auto start = static_cast<char *>(this->dl_get_global_memaddr(0));
  return start + addr - GMEM_START_ADDR;
}

void *BM168x::get_l2mem_addr(uint64_t addr) {
  auto start = static_cast<char *>(this->dl_get_l2_sram(0));
  return start + addr - L2_SRAM_START_ADDR;
}

void *BM168x::get_system_mem_ptr(uint64_t addr) {
  const uint64_t cmodel_gmem_start_addr = get_cmodel_gmem_start_addr();
  const uint64_t cmodel_l2mem_start_addr = get_cmodel_l2mem_start_addr();
  if (addr >= (GMEM_START_ADDR | cmodel_gmem_start_addr)) {
    if (GMEM_START_ADDR != cmodel_gmem_start_addr) {
      return get_gmem_addr(addr & (~cmodel_gmem_start_addr));
    } else {
      return get_gmem_addr(addr);
    }
  } else if (L2_SRAM_SIZE > 0 &&
             (L2_SRAM_START_ADDR | cmodel_l2mem_start_addr) <= addr &&
             addr < (L2_SRAM_START_ADDR | cmodel_l2mem_start_addr) +
                        L2_SRAM_SIZE) {
    if (L2_SRAM_START_ADDR != cmodel_l2mem_start_addr) {
      return get_l2mem_addr(addr & (~cmodel_l2mem_start_addr));
    } else {
      return get_l2mem_addr(addr);
    }
  }
  return NULL;
}

void *BM168x::get_local_mem_ptr(int npu_idx, uint64_t addr) {
  auto start =
      static_cast<char *>(this->dl_get_local_memaddr_by_node(0, npu_idx));
  return start + addr;
}

void *BM168x::get_gmem_addr(const bm_device_mem_t &mem) {
  auto start = static_cast<char *>(this->dl_get_global_memaddr(0));
  return start + mem.offset;
}

void BM168x::bm_memcpy_s2d(const bm_device_mem_t &dst, void *src) {
  memcpy(get_gmem_addr(dst), src, dst.size);
}

void BM168x::bm_memcpy_d2s(void *dst, const bm_device_mem_t &src) {
  memcpy(dst, get_gmem_addr(src), src.size);
}

void BM168x::value_s2d(Value v, void *src) {
  auto addr = module::getAddress(v);
  auto bytes = module::getBytes(v);
  memcpy(get_gmem_addr(addr), src, bytes);
}

void BM168x::value_d2s(Value v, void *dst) {
  auto addr = module::getAddress(v);
  auto bytes = module::getBytes(v);
  memcpy(dst, get_gmem_addr(addr), bytes);
}

void BM168x::divide_sync_id() {
  dl_cmd_id_divide(code->cmdid_node, code->bdc_node, code->gdma_node);
}

void BM168x::merge_sync_id() {
  dl_cmd_id_merge(code->cmdid_node, code->bdc_node, code->gdma_node);
}

DATA_TYPE_T BM168x::getDataType(Value v) {
  auto type = module::getStorageType(v);
  return getDataType(type);
}

DATA_TYPE_T BM168x::getDataType(mlir::Type type) {
  auto bits = type.getIntOrFloatBitWidth();
  if (type.isUnsignedInteger()) {
    switch (bits) {
    case 4:
      return DTYPE_UINT4;
    case 8:
      return DTYPE_UINT8;
    case 16:
      return DTYPE_UINT16;
    case 32:
      return DTYPE_UINT32;
    default:
      break;
    }
  } else if (type.isSignedInteger() || type.isSignlessInteger()) {
    switch (bits) {
    case 4:
      return DTYPE_INT4;
    case 8:
      return DTYPE_INT8;
    case 16:
      return DTYPE_INT16;
    case 32:
      return DTYPE_INT32;
    default:
      break;
    }
  } else if (type.isF32()) {
    return DTYPE_FP32;
  } else if (type.isBF16()) {
    return DTYPE_BFP16;
  } else if (type.isF16()) {
    return DTYPE_FP16;
  } else if (type.isFloat8E4M3FN()) {
    return DTYPE_F8E4M3;
  } else if (type.isFloat8E5M2()) {
    return DTYPE_F8E5M2;
  }
  type.dump();
  llvm_unreachable("Unsupport type \n");
  return DTYPE_FP32;
}

int BM168x::getGdmaFormat(DATA_TYPE_T data_type) {
  switch (data_type) {
  case DTYPE_INT8:
  case DTYPE_UINT8:
  case DTYPE_F8E4M3:
  case DTYPE_F8E5M2:
    return GDMA_VALUE_FORMAT_INT8;
  /*case DTYPE_INT4:   // for BM1690
  case DTYPE_UINT4:
    return GDMA_VALUE_FORMAT_INT4;*/
  case DTYPE_FP16:
    return GDMA_VALUE_FORMAT_FLOAT16;
  case DTYPE_BFP16:
    return GDMA_VALUE_FORMAT_BFLOAT16;
  case DTYPE_INT16:
  case DTYPE_UINT16:
    return GDMA_VALUE_FORMAT_INT16;
  default:
    return GDMA_VALUE_FORMAT_FLOAT32;
  }
  return 0;
}

double BM168x::getFmtBytes(DATA_TYPE_T data_type) {
  double data_byte_size = 0;
  switch (data_type) {
  case DTYPE_FP32:
    data_byte_size = 4;
    break;
  case DTYPE_FP16:
  case DTYPE_BFP16:
  case DTYPE_INT16:
  case DTYPE_UINT16:
    data_byte_size = 2;
    break;
  case DTYPE_INT8:
  case DTYPE_UINT8:
  case DTYPE_F8E4M3:
  case DTYPE_F8E5M2:
    data_byte_size = 1;
    break;
  case DTYPE_INT4:
  case DTYPE_UINT4:
    data_byte_size = 0.5;
    break;
  default:
    data_byte_size = 4;
    break;
  }
  return (double)data_byte_size;
}

STORE_MODE_T BM168x::getStoreMode(Value v) {
  STORE_MODE_T stmode = STORE_MODE_1N;
  if (module::isBM1684Family()) {
    auto type = module::getStorageType(v);
    auto typeBytes = type.getIntOrFloatBitWidth() / 8;
    if (typeBytes == 1) {
      stmode = STORE_MODE_4N;
    } else if (typeBytes == 2) {
      stmode = STORE_MODE_2N;
    } else if (typeBytes != 4) {
      llvm_unreachable("stmode type error");
    }
  }
  return stmode;
}

tensor_spec_t BM168x::value_to_spec(mlir::Value v, group_type_t group_type,
                                    int64_t n_step, int64_t c_step,
                                    int64_t h_step, int64_t d_step,
                                    int64_t w_step) {
  tensor_spec_t spec;
  group_info_t ginfo = {0};
  memset(&spec, 0, sizeof(spec));
  auto shape = module::getShape(v);
  auto pre_op = v.getDefiningOp();
  if (module::isOpInGroup(pre_op)) {
    if (isa<tpu::MoveOp>(pre_op)) {
      auto moveOp = dyn_cast<tpu::MoveOp>(pre_op);
      auto vec_input_move_addr = *module::getI64Array(moveOp.getMoveDestAdd());
      int idx = v.cast<OpResult>().getResultNumber();
      spec.addr = vec_input_move_addr[idx];
      llvm::errs() << "value_to_spec, v:" << module::getName(v).str()
                   << ", idx:" << idx
                   << ", vec_input_move_addr[idx]:" << spec.addr << "\n";
      if (group_type == GROUP_MM_OPT3 &&
          !isa<tpu::LoadToL2MOp>(pre_op->getOperand(idx).getDefiningOp())) {
        ginfo = LocalGenInterface::getGroupInfo(pre_op->getOperand(idx), n_step,
                                                h_step, d_step, w_step, c_step);
      }
    } else if (isa<tpu::LoadToL2MOp>(pre_op)) {
      spec.addr = module::getAddress(pre_op->getOperand(1));
    } else {
      ginfo = LocalGenInterface::getGroupInfo(v, n_step, h_step, d_step, w_step,
                                              c_step);
      spec.addr = ginfo.out_addr;
    }
  } else {
    spec.addr = module::getAddress(v);
  }
  spec.dtype = getDataType(v);
  if (group_type == GROUP_NORMAL || group_type == GROUP_3D ||
      group_type == GROUP_MM) {
    spec.dims = shape.size();
    for (int i = 0; i < spec.dims; i++) {
      spec.shape[i] = shape[i];
    }
  } else if (group_type == GROUP_MM_OPT3) {
    if (module::IsSliceOpInOrOut(v)) {
      spec.dims = shape.size();
      for (int i = 0; i < spec.dims; i++) {
        spec.shape[i] = 1;
      }
      spec.shape[spec.dims - 2] = ginfo.c_slice;
      spec.shape[spec.dims - 1] = ginfo.h_slice;
      if (spec.dims >= 3) {
        spec.shape[0] = ginfo.n_slice;
      }
    } else {
      spec.dims = 4;
      if (module::IsHdimIsBatch(v)) {
        spec.shape[0] = ginfo.n_slice;
        spec.shape[1] = ginfo.c_slice;
        spec.shape[2] = ginfo.h_slice;
        spec.shape[3] = ginfo.w_slice;
      } else {
        spec.shape[0] = ginfo.n_slice;
        spec.shape[1] = ginfo.c_slice;
        spec.shape[2] = ginfo.h_slice;
        spec.shape[3] = 1;
      }
    }
  } else if (group_type == GROUP_SMALL_C) {
    int64_t n, c, h, w;
    module::getNCHW(v, n, c, h, w, group_type);
    spec.dims = 3;
    spec.shape[0] = n;
    spec.shape[1] = c;
    spec.shape[2] = h;
    spec.shape[3] = w;
  }
  spec.elem_num = 0;
  return spec;
}
std::shared_ptr<std::vector<tensor_spec_t>>
BM168x::get_input_spec(Operation *op, group_type_t group_type, int64_t n_step,
                       int64_t c_step, int64_t h_step, int64_t d_step,
                       int64_t w_step) {
  return get_spec(op->getOperands(), group_type, n_step, h_step, d_step, w_step,
                  c_step);
}

std::shared_ptr<std::vector<tensor_spec_t>>
BM168x::get_output_spec(Operation *op, group_type_t group_type, int64_t n_step,
                        int64_t c_step, int64_t h_step, int64_t d_step,
                        int64_t w_step) {
  return get_spec(op->getResults(), group_type, n_step, h_step, d_step, w_step,
                  c_step);
}

std::shared_ptr<std::vector<tensor_spec_t>>
BM168x::get_spec(ValueRange values, group_type_t group_type, int64_t n_step,
                 int64_t c_step, int64_t h_step, int64_t d_step,
                 int64_t w_step) {
  auto specs = std::make_shared<std::vector<tensor_spec_t>>();
  for (auto v : values) {
    if (module::isNone(v)) {
      continue;
    }
    specs->push_back(
        value_to_spec(v, group_type, n_step, h_step, d_step, w_step, c_step));
  }
  return std::move(specs);
}

void BM168x::fix_shape(tensor_spec_t &spec,
                       const std::vector<int64_t> &new_shape) {
  assert(new_shape.size() <= MAX_SHAPE_DIMS);
  auto &old_shape = spec.shape;
  int64_t new_num = std::accumulate(new_shape.begin(), new_shape.end(), 1,
                                    std::multiplies<int32_t>());
  int64_t old_num = std::accumulate(old_shape, old_shape + spec.dims, 1,
                                    std::multiplies<int32_t>());
  assert(new_num == old_num);
  memset(old_shape, 0, sizeof(old_shape));
  std::copy(new_shape.begin(), new_shape.end(), old_shape);
  spec.dims = new_shape.size();
}

stride_4D_t BM168x::getGlobalStride(int64_t N, int64_t C, int64_t H,
                                    int64_t W) {
  stride_4D_t s;
  s.N = C * H * W;
  s.C = H * W;
  s.H = W;
  s.W = 1;
  return s;
}

stride_4D_t BM168x::getLocalStride(int64_t N, int64_t C, int64_t H, int64_t W,
                                   double fmtBytes, bool eu_align) {
  stride_4D_t s;
  s.W = 1;
  s.H = W;
  if (eu_align) {
    s.C = align_up(H * W, eu_num(fmtBytes));
  } else {
    s.C = H * W;
  }
  s.N = ceiling_func(C, BM168x::NPU_NUM) * s.C;
  return s;
}

int BM168x::get_reduce_type(llvm::StringRef mode) {
  if (mode == "ReduceMean") {
    return SG_REDUCE_MEAN;
  } else if (mode == "ReduceSum") {
    return SG_REDUCE_SUM;
  } else if (mode == "ReduceMax") {
    return SG_REDUCE_MAX;
  } else if (mode == "ReduceMin") {
    return SG_REDUCE_MIN;
  } else if (mode == "ReduceProd") {
    return SG_REDUCE_PROD;
  } else if (mode == "ReduceL2") {
    return SG_REDUCE_L2;
  } else if (mode == "ReduceL1") {
    return SG_REDUCE_L1;
  } else {
    llvm_unreachable("unsupport reduce mode.");
  }
}

typedef int (*backend_api_t)(void *params, int param_size, void *pid_node);
void BM168x::call_global_func(const char *symbolName, void *params,
                              int param_size) {
  auto func = instance()->CastToFPtr<backend_api_t>(symbolName);
  func(params, param_size, (*instance())->cmdid_node);
}

void BM168x::call_local_func(const char *symbolName, void *params,
                             int param_size) {
  auto func = instance()->CastToFPtr<backend_api_t>(symbolName);
  func(params, param_size, (*instance())->bdc_node);
}

typedef int (*global_backend_api_t)(void *params, int param_size, void *input,
                                    void *output, void *pid_node);

void BM168x::call_global_func(const char *symbolName, void *params,
                              int param_size, void *input, void *output) {
  auto func = instance()->CastToFPtr<global_backend_api_t>(symbolName);
  func(params, param_size, input, output, (*instance())->cmdid_node);
}

typedef int (*local_backend_api_t)(void *params, int param_size, void *input,
                                   void *info, void *output, void *pid_node);
void BM168x::call_local_func(const char *symbolName, void *params,
                             int param_size, void *info, void *input,
                             void *output) {
  auto func = instance()->CastToFPtr<local_backend_api_t>(symbolName);
  func(params, param_size, info, input, output, (*instance())->bdc_node);
}

typedef int (*ppl_set_node)(void *cmdid_node);
typedef int (*ppl_global_backend_api_t)(void *params, int param_size,
                                        void *input, void *output);
void BM168x::call_ppl_global_func(const char *symbolName, void *params,
                                  int param_size, void *input, void *output) {
  auto set_node_chip = instance()->PplCastToFPtr<ppl_set_node>("ppl_set_node");
  set_node_chip((*instance())->cmdid_node);
  auto kernel_func =
      instance()->PplCastToFPtr<ppl_global_backend_api_t>(symbolName);
  kernel_func(params, param_size, input, output);
}

typedef int (*ppl_local_backend_api_t)(void *params, int param_size,
                                       void *input, void *info, void *output);
void BM168x::call_ppl_local_func(const char *symbolName, void *params,
                                 int param_size, void *info, void *input,
                                 void *output) {
  auto set_node_chip = instance()->PplCastToFPtr<ppl_set_node>("ppl_set_node");
  auto func = instance()->PplCastToFPtr<ppl_local_backend_api_t>(symbolName);
  set_node_chip((*instance())->cmdid_node);
  func(params, param_size, info, input, output);
}

typedef bool (*force_dynamic_run_func_t)(void *params, int param_size);
typedef bool (*local_gen_support_func_t)(void *params, int param_size);
typedef bool (*allow_data_split_func_t)(void *params, int param_size, int axis,
                                        group_type_t group_type);
typedef bool (*backward_slice_func_t)(void *params, int param_size, int *in_idx,
                                      int *in_slice, int out_idx,
                                      int out_slice);
typedef bool (*inference_func_t)(void *params, int param_size,
                                 const int **input_shapes,
                                 const int *input_dims, const float **inputs,
                                 float **outputs);
void BM168x::call_custom_plugin_func(kCustomPluginTypes plugin_type, void *ret,
                                     const char *symbolName, void *params,
                                     int param_size, void *args) {
  switch (plugin_type) {
  case kCustomPluginTypes::PLUGIN_FORCEDYNAMICRUN:
  case kCustomPluginTypes::PLUGIN_LOCALGENSUPPORT: {
    auto func =
        instance()->CastToCustomPluginPtr<local_gen_support_func_t>(symbolName);
    *(bool *)ret = func ? func(params, param_size) : false;
  } break;
  case kCustomPluginTypes::PLUGIN_ALLOWDATASPLIT: {
    int *_args = (int *)args; // {axis, group_type}
    auto func =
        instance()->CastToCustomPluginPtr<allow_data_split_func_t>(symbolName);
    if (func) {
      *(bool *)ret = func(params, param_size, _args[0], (group_type_t)_args[1]);
    } else {
      *(bool *)ret = true;
    }
  } break;
  case kCustomPluginTypes::PLUGIN_BACKWARDH:
  case kCustomPluginTypes::PLUGIN_BACKWARDW: {
    int *_args = (int *)args; // {in_idx, in_slice, out_idx, out_slice}
    auto func =
        instance()->CastToCustomPluginPtr<backward_slice_func_t>(symbolName);
    if (func) {
      *(bool *)ret =
          func(params, param_size, &_args[0], &_args[1], _args[2], _args[3]);
    } else {
      _args[0] = _args[2], _args[1] = _args[3];
      *(bool *)ret = true;
    }
  } break;
  case kCustomPluginTypes::PLUGIN_INFERENCE: {
    void *_args[4] = {
        ((void **)args)[0], ((void **)args)[1], ((void **)args)[2],
        ((void **)args)[3]}; // {input_shapes, input_dims, inputs, outputs}
    auto func = instance()->CastToCustomPluginPtr<inference_func_t>(symbolName);
    if (func) {
      func(params, param_size, (const int **)(_args[0]),
           (const int *)(_args[1]), (const float **)(_args[2]),
           (float **)(_args[3]));
      *(bool *)ret = true;
    } else
      *(bool *)ret = false;
  } break;
  default:
    break;
  }
}

typedef int (*global_dq_custom_api_t)(void *params, int param_size, void *input,
                                      void *output, void *pid_node);
void BM168x::call_dq_custom_global_func(const char *libName,
                                        const char *symbolName, void *params,
                                        int param_size, void *input,
                                        void *output) {
  auto func =
      instance()->CastToDQFPtr<global_dq_custom_api_t>(libName, symbolName);
  func(params, param_size, input, output, (*instance())->cmdid_node);
}

typedef int (*global_custom_api_t)(void *params, int param_size, void *input,
                                   void *output, void *pid_node);
void BM168x::call_global_custom_func(const char *symbolName, void *params,
                                     int param_size, void *input,
                                     void *output) {
  auto func = instance()->CastToCustomFPtr<global_custom_api_t>(symbolName);
  func(params, param_size, input, output, (*instance())->cmdid_node);
}

typedef int (*local_custom_api_t)(void *params, int param_size, void *info,
                                  void *input, void *output, void *pid_node);
void BM168x::call_local_custom_func(const char *symbolName, void *params,
                                    int param_size, void *info, void *input,
                                    void *output) {
  auto func = instance()->CastToCustomFPtr<local_custom_api_t>(symbolName);
  func(params, param_size, info, input, output, (*instance())->bdc_node);
}

typedef int64_t (*global_bfsz_custom_api_t)(void *params, int param_size,
                                            void *input, void *output);
int64_t BM168x::call_global_bfsz_custom_func(const char *symbolName,
                                             void *params, int param_size,
                                             void *input, void *output) {
  auto func =
      instance()->CastToCustomFPtr<global_bfsz_custom_api_t>(symbolName, false);
  return func ? func(params, param_size, input, output) : 0;
}

typedef int (*local_bfsz_custom_api_t)(void *params, int param_size, void *info,
                                       void *input, void *output);
int BM168x::call_local_bfsz_custom_func(const char *symbolName, void *params,
                                        int param_size, void *info, void *input,
                                        void *output) {
  auto func =
      instance()->CastToCustomFPtr<local_bfsz_custom_api_t>(symbolName, false);
  return func ? func(params, param_size, info, input, output) : 0;
}

typedef int64_t (*global_bfsz_backend_api_t)(void *params, int param_size,
                                             void *input, void *output);
int64_t BM168x::call_global_bfsz_func(const char *symbolName, void *params,
                                      int param_size, void *input,
                                      void *output) {
  auto func = instance()->CastToFPtr<global_bfsz_backend_api_t>(symbolName);
  return func(params, param_size, input, output);
}

typedef int (*local_bfsz_backend_api_t)(void *params, int param_size,
                                        void *info, void *input, void *output);
int BM168x::call_local_bfsz_func(const char *symbolName, void *params,
                                 int param_size, void *info, void *input,
                                 void *output) {
  auto func = instance()->CastToFPtr<local_bfsz_backend_api_t>(symbolName);
  return func(params, param_size, info, input, output);
}

uint64_t BM168x::COEFF_START_ADDR = 0;
uint64_t BM168x::CTX_START_ADDR = 0;
uint64_t BM168x::IO_ADDR[5] = {0};
int64_t BM168x::IC_PARALLEL = 0;
uint64_t BM168x::GMEM_START_ADDR = 0;
int64_t BM168x::ALIGNMENT = 0;
uint64_t BM168x::L2_SRAM_START_ADDR = 0;
uint64_t BM168x::L2_SRAM_SIZE = 0;
bool BM168x::SUPPORT_MEM_TAG = false;

int BM168x::GDMA_VALUE_FORMAT_UINT8 = 0;
int BM168x::GDMA_VALUE_FORMAT_INT8 = 0;
int BM168x::GDMA_VALUE_FORMAT_FLOAT16 = 0;
int BM168x::GDMA_VALUE_FORMAT_FLOAT32 = 0;
int BM168x::GDMA_VALUE_FORMAT_INT16 = 0;
int BM168x::GDMA_VALUE_FORMAT_INT32 = 0;
int BM168x::GDMA_VALUE_FORMAT_BFLOAT16 = 0;
int BM168x::GDMA_VALUE_FORMAT_INT4 = 0;
int BM168x::GDMA_VALUE_FORMAT_FLOAT20 = 0;
int BM168x::GDMA_VALUE_FORMAT_NUM = 0;

void BM168x::load_functions() {
  CAST_FUNCTION(cmodel_init);
  CAST_FUNCTION(cmodel_deinit);
  CAST_FUNCTION(cmodel_nodechip_runtime_init);
  CAST_FUNCTION(cmodel_nodechip_runtime_exit);
  CAST_FUNCTION(create_cmd_id_node);
  CAST_FUNCTION(destroy_cmd_id_node);
  CAST_FUNCTION(set_cmd_id_cycle);
  CAST_FUNCTION(get_cmd_id_cycle);
  CAST_FUNCTION(reset_cmd_id);
  CAST_FUNCTION(allow_store_cmd);
  CAST_FUNCTION(forbid_store_cmd);
  CAST_FUNCTION(use_atomic_cmodel);
  CAST_FUNCTION(forbid_atomic_cmodel);
  CAST_FUNCTION(get_global_memaddr);
  CAST_FUNCTION(get_l2_sram);
  CAST_FUNCTION(get_local_memaddr_by_node);
  CAST_FUNCTION(set_cmd_buffer_ptr);
  CAST_FUNCTION(set_cmd_id_prefix);
  CAST_FUNCTION(set_cmd_check_param);
  CAST_FUNCTION(enable_profile);
  CAST_FUNCTION(allow_atomic_cmodel_assert);
  CAST_FUNCTION(forbid_atomic_cmodel_assert);
  CAST_FUNCTION(tensor_stride_move_gen_cmd);
  CAST_FUNCTION(tensor_compact_move_gen_cmd);
  CAST_FUNCTION(tensor_align_move_gen_cmd);
  CAST_FUNCTION(set_total_id_ptr);
  CAST_FUNCTION(tpu_global_mem_get_start_addr);
  CAST_FUNCTION(tpu_l2_sram_get_start_addr);
  CAST_CPU_FUNCTION(bmcpu_init);
  CAST_CPU_FUNCTION(bmcpu_uninit);
  CAST_CPU_FUNCTION(bmcpu_process);
  CAST_CPU_FUNCTION(bmcpu_reshape);
  CAST_CPU_FUNCTION(bmcpu_dtype);
}

BM168x::~BM168x() {
  if (code->bmcpu_handle != nullptr) {
    dl_bmcpu_uninit(code->bmcpu_handle);
  }
}

void BM168x::start_env() {
  load_library();
  bmcpu_setup();
  load_functions();
  if (0 != dl_cmodel_init(0, CMODEL_GMEM_SIZE)) {
    llvm_unreachable("cmodel init failed");
  }
  code->bmcpu_handle = dl_bmcpu_init();
  code->cmdid_node = dl_create_cmd_id_node();
  code->bdc_node = dl_create_cmd_id_node();
  code->gdma_node = dl_create_cmd_id_node();
}

void BM168x::end_env() {
  BM168x::instance()->reset_command_flag();
  if (DL.isValid()) {
    if (code->cmdid_node != nullptr) {
      dl_destroy_cmd_id_node(code->gdma_node);
      dl_destroy_cmd_id_node(code->bdc_node);
      dl_destroy_cmd_id_node(code->cmdid_node);
    }
    dl_cmodel_deinit(0);
  }
}

void BM168x::bmcpu_setup() {
  std::string Err;
  cpuopDL =
      llvm::sys::DynamicLibrary::getPermanentLibrary(libcpuop.data(), &Err);
  if (cpuopDL.isValid() == false) {
    llvm_unreachable(Err.c_str());
  }
}

int BM168x::compare_mode(StringRef mode) {
  if (mode == "Equal") {
    return BINARY_EQ;
  }
  if (mode == "Greater") {
    return BINARY_GT;
  }
  if (mode == "GreaterOrEqual") {
    return BINARY_GE;
  }
  if (mode == "Less") {
    return BINARY_LT;
  }
  if (mode == "LessOrEqual") {
    return BINARY_LE;
  }
  if (mode == "NotEqual" || mode == "Xor") {
    return BINARY_NE;
  }
  if (mode == "And") {
    return BINARY_MUL;
  }
  if (mode == "Not") {
    return BINARY_EQ;
  }
  llvm_unreachable("Not Implemented");
}

int BM168x::binary_mode(StringRef mode) {
  if (mode == "Add") {
    return BINARY_ADD;
  }
  if (mode == "Sub") {
    return BINARY_SUB;
  }
  if (mode == "Mul") {
    return BINARY_MUL;
  }
  llvm_unreachable("Not Implemented");
}

static void size_to_2dim(int64_t size, int64_t &small, int64_t &big) {
  int64_t div = std::sqrt(size);
  for (small = div; small >= 1; small--) {
    if (size % small == 0) {
      big = size / small;
      break;
    }
  }
}

// TODO: nodechip should be optimized, not here.
// because 1684x nodechip should slice by num element, but by n,c,h,w;
// it would be not efficient
void BM168x::getBetterNCHW(Value v, int64_t &n, int64_t &c, int64_t &h,
                           int64_t &w) {
  auto num = module::getNumElements(v);
  auto bytes = module::getDtypeSize(v);
  auto EU = eu_num(bytes);
  n = 1;
  c = 1;
  h = 1;
  w = 1;
  int64_t left;
  if (num % NPU_NUM == 0) {
    c = NPU_NUM;
    left = num / c;
    if (left % EU == 0) {
      w = EU;
      h = left / w;
    } else {
      size_to_2dim(num / c, w, h);
    }
    return;
  }
  if (num % EU == 0) {
    w = EU;
    size_to_2dim(num / EU, h, c);
    return;
  }
  int64_t a[3];
  size_to_2dim(num, a[0], a[1]);
  size_to_2dim(a[1], a[2], a[1]);
  // most similar to NPU_NUM
  int64_t b = a[0] % NPU_NUM;
  int max_idx = 0;
  for (int idx = 1; idx < 3; idx++) {
    if (a[idx] > b) {
      max_idx = idx;
    }
  }
  c = a[max_idx];
  size_to_2dim(num / c, w, h);
}

void BM168x::before_codegen() {
  // set_command_issue_flag(true);
  dl_sg_set_profile_path("./");
  reset_cmd_id_node();
}

void BM168x::after_codegen(int64_t flops) {
  dl_sg_stas_dump(code->cmdid_node);
  if (flops) {
    dl_sg_flops_dump(flops, code->cmdid_node);
  }
}

void BM168x::set_command_issue_flag(bool value) {
  really_issue_command = value;
  if (really_issue_command) {
    dl_allow_store_cmd();
    dl_use_atomic_cmodel();
    dl_allow_atomic_cmodel_assert();
  } else {
    dl_forbid_store_cmd();
    dl_forbid_atomic_cmodel();
    dl_forbid_atomic_cmodel_assert();
    dl_set_cmd_check_param(nullptr, false);
  }
}

void BM168x::reset_cmd_id_node() {
  dl_reset_cmd_id(code->cmdid_node);
  dl_reset_cmd_id(code->bdc_node);
  dl_reset_cmd_id(code->gdma_node);
}

int64_t BM168x::get_gdma_cycle() {
  return dl_get_cmd_id_cycle(code->gdma_node);
}

int64_t BM168x::get_bdc_cycle() { return dl_get_cmd_id_cycle(code->bdc_node); }

int64_t BM168x::get_cmd_cycle() {
  return dl_get_cmd_id_cycle(code->cmdid_node);
}

void BM168x::enter_runtime() {
  dl_use_atomic_cmodel();
  // for (int core_idx = 0; core_idx < module::getCoreNum(); ++core_idx) {
  //   dl_cmodel_nodechip_runtime_init(core_idx);
  // }
}

void BM168x::exit_runtime() {
  // for (int core_idx = 0; core_idx < module::getCoreNum(); ++core_idx) {
  //   dl_cmodel_nodechip_runtime_exit(core_idx);
  // }
  dl_forbid_atomic_cmodel();
}
