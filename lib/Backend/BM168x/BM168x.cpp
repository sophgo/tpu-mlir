//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Backend/BM168x/BM1686.h"
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Interfaces/LocalGenInterface.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

using namespace tpu_mlir;
using namespace tpu_mlir::backend;
using namespace tpu_mlir::helper;

void *BM168x::get_gmem_addr(uint64_t addr) {
  auto start = static_cast<char *>(this->dl_get_global_memaddr(0));
  return start + addr - GMEM_START_ADDR;
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
  auto addr = Module::getAddress(v);
  auto bytes = Module::getBytes(v);
  memcpy(get_gmem_addr(addr), src, bytes);
}

void BM168x::value_d2s(Value v, void *dst) {
  auto addr = Module::getAddress(v);
  auto bytes = Module::getBytes(v);
  memcpy(dst, get_gmem_addr(addr), bytes);
}

void BM168x::divide_sync_id() {
  dl_cmd_id_divide(cmdid_node, bdc_node, gdma_node);
}

void BM168x::merge_sync_id() {
  dl_cmd_id_merge(cmdid_node, bdc_node, gdma_node);
}

DATA_TYPE_T BM168x::getDataType(Value v) {
  auto type = Module::getStorageType(v);
  return getDataType(type);
}

DATA_TYPE_T BM168x::getDataType(mlir::Type type) {
  auto bits = type.getIntOrFloatBitWidth();
  if (type.isUnsignedInteger()) {
    switch (bits) {
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
  }
  type.dump();
  llvm_unreachable("Unsupport type \n");
  return DTYPE_FP32;
}

int BM168x::getGdmaFormat(DATA_TYPE_T data_type) {
  int gdma_format = GDMA_VALUE_FORMAT_FLOAT32;
  switch (data_type) {
  case DTYPE_INT8:
  case DTYPE_UINT8:
    gdma_format = GDMA_VALUE_FORMAT_INT8;
    break;
  case DTYPE_INT16:
  case DTYPE_UINT16:
  case DTYPE_FP16:
  case DTYPE_BFP16:
    gdma_format = GDMA_VALUE_FORMAT_INT16;
    break;
  default:
    gdma_format = GDMA_VALUE_FORMAT_FLOAT32;
    break;
  }
  return gdma_format;
}

int BM168x::getFmtBytes(DATA_TYPE_T data_type) {
  int data_byte_size = 0;
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
    data_byte_size = 1;
    break;
  default:
    data_byte_size = 4;
    break;
  }
  return data_byte_size;
}

tensor_spec_t BM168x::value_to_spec(mlir::Value v) {
  tensor_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  if (Module::isOpInGroup(v.getDefiningOp())) {
    auto gi = LocalGenInterface::getGroupInfo(v);
    spec.addr = gi.out_addr;
  } else {
    spec.addr = Module::getAddress(v);
  }
  spec.dtype = getDataType(v);
  auto shape = Module::getShape(v);
  spec.dims = shape.size();
  for (int i = 0; i < spec.dims; i++) {
    spec.shape[i] = shape[i];
  }
  spec.elem_num = 0;
  return spec;
}
std::shared_ptr<std::vector<tensor_spec_t>>
BM168x::get_input_spec(Operation *op) {
  return get_spec(op->getOperands());
}

std::shared_ptr<std::vector<tensor_spec_t>>
BM168x::get_output_spec(Operation *op) {
  return get_spec(op->getResults());
}

std::shared_ptr<std::vector<tensor_spec_t>>
BM168x::get_spec(ValueRange values) {
  auto specs = std::make_shared<std::vector<tensor_spec_t>>();
  for (auto v : values) {
    if (v.getType().isa<mlir::NoneType>()) {
      continue;
    }
    specs->push_back(value_to_spec(v));
  }
  return std::move(specs);
}

void BM168x::fix_shape(tensor_spec_t &spec,
                       const std::vector<int32_t> &new_shape) {
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
                                   int fmtBytes, bool eu_align) {
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

typedef int (*backend_api_t)(void *params, int param_size, void *pid_node);
void BM168x::call_global_func(const char *symbolName, void *params,
                              int param_size) {
  auto func = instance()->CastToFPtr<backend_api_t>(symbolName);
  func(params, param_size, instance()->cmdid_node);
}

void BM168x::call_local_func(const char *symbolName, void *params,
                             int param_size) {
  auto func = instance()->CastToFPtr<backend_api_t>(symbolName);
  func(params, param_size, instance()->bdc_node);
}

typedef int (*global_backend_api_t)(void *params, int param_size, void *input,
                                    void *output, void *pid_node);
void BM168x::call_global_func(const char *symbolName, void *params,
                              int param_size, void *input, void *output) {
  auto func = instance()->CastToFPtr<global_backend_api_t>(symbolName);
  func(params, param_size, input, output, instance()->cmdid_node);
}

typedef int (*local_backend_api_t)(void *params, int param_size, void *input,
                                   void *info, void *output, void *pid_node);
void BM168x::call_local_func(const char *symbolName, void *params,
                             int param_size, void *info, void *input,
                             void *output) {
  auto func = instance()->CastToFPtr<local_backend_api_t>(symbolName);
  func(params, param_size, info, input, output, instance()->bdc_node);
}

uint64_t BM168x::CTX_START_ADDR = 0;
int64_t BM168x::IC_PARALLEL = 0;
uint64_t BM168x::GMEM_START_ADDR = 0;
int64_t BM168x::ALIGNMENT = 0;

void BM168x::load_functions() {
  CAST_FUNCTION(cmodel_init);
  CAST_FUNCTION(cmodel_deinit);
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
  CAST_FUNCTION(set_cmd_buffer_ptr);
  CAST_FUNCTION(set_cmd_id_prefix);
  CAST_FUNCTION(allow_atomic_cmodel_assert);
  CAST_FUNCTION(forbid_atomic_cmodel_assert);
  CAST_FUNCTION(tensor_stride_move_gen_cmd);
  CAST_FUNCTION(tensor_compact_move_gen_cmd);
  CAST_FUNCTION(tensor_broadcast_move_gen_cmd);
  CAST_FUNCTION(set_total_id_ptr);
  CAST_FUNCTION(sg_set_profile_dump);
  CAST_FUNCTION(sg_stas_dump);
  CAST_FUNCTION(sg_flops_dump);
}

BM168x::~BM168x() {}

void BM168x::start_env() {
  load_library();
  load_functions();
  dl_cmodel_init(0, CMODEL_GMEM_SIZE);
  cmdid_node = dl_create_cmd_id_node();
  bdc_node = dl_create_cmd_id_node();
  gdma_node = dl_create_cmd_id_node();
  gdma_buffer.reserve(0x1000000);
  bdc_buffer.reserve(0x1000000);
  dl_set_cmd_buffer_ptr((void *)&gdma_buffer, (void *)&bdc_buffer);
  dl_set_total_id_ptr(&gdma_total_id, &bdc_total_id, cmdid_node,
                      (void *)&gdma_group_id, (void *)&bdc_group_id,
                      &cmdid_groupnum);
  dl_allow_store_cmd();
  dl_forbid_atomic_cmodel(); // TODO:(no compare)
  dl_sg_set_profile_dump(true);
}

void BM168x::end_env() {
  if (DL.isValid()) {
    if (cmdid_node != nullptr) {
      dl_destroy_cmd_id_node(gdma_node);
      dl_destroy_cmd_id_node(bdc_node);
      dl_destroy_cmd_id_node(cmdid_node);
    }
    dl_cmodel_deinit(0);
  }
}

void BM168x::before_codegen() {
  dl_reset_cmd_id(cmdid_node);
  dl_reset_cmd_id(bdc_node);
  dl_reset_cmd_id(gdma_node);
  // set_command_issue_flag(true);
  gdma_group_id.clear();
  gdma_group_id.push_back(0);
  bdc_group_id.clear();
  bdc_group_id.push_back(0);
  gdma_bytes.clear();
  bdc_bytes.clear();
  gdma_buffer.clear();
  bdc_buffer.clear();
  cmdid_groupnum = 1;
}

void BM168x::after_codegen(int64_t flops) {
  dl_sg_stas_dump(cmdid_node);
  if (flops)
    dl_sg_flops_dump(flops, cmdid_node);
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
  }
}
