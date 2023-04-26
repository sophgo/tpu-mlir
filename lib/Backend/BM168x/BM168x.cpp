//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Backend/BM168x/BM1686.h"
#include "tpu_mlir/Interfaces/LocalGenInterface.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace tpu_mlir::backend;

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
  dl_cmd_id_divide(cmdid_node, bdc_node, gdma_node);
}

void BM168x::merge_sync_id() {
  dl_cmd_id_merge(cmdid_node, bdc_node, gdma_node);
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
  }
  type.dump();
  llvm_unreachable("Unsupport type \n");
  return DTYPE_FP32;
}

int BM168x::getGdmaFormat(DATA_TYPE_T data_type) {
  switch (data_type) {
  case DTYPE_INT8:
  case DTYPE_UINT8:
    return GDMA_VALUE_FORMAT_INT8;
  /*case DTYPE_INT4:   // for SG2260
  case DTYPE_UINT4:
    return GDMA_VALUE_FORMAT_INT4;*/
  case DTYPE_INT16:
  case DTYPE_UINT16:
  case DTYPE_FP16:
  case DTYPE_BFP16:
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

tensor_spec_t BM168x::value_to_spec(mlir::Value v, group_type_t group_type) {
  tensor_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  if (module::isOpInGroup(v.getDefiningOp())) {
    auto gi = LocalGenInterface::getGroupInfo(v);
    spec.addr = gi.out_addr;
  } else {
    spec.addr = module::getAddress(v);
  }
  spec.dtype = getDataType(v);
  auto shape = module::getShape(v);
  if (group_type == GROUP_NORMAL || group_type == GROUP_3D) {
    spec.dims = shape.size();
    for (int i = 0; i < spec.dims; i++) {
      spec.shape[i] = shape[i];
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
BM168x::get_input_spec(Operation *op, group_type_t group_type) {
  return get_spec(op->getOperands(), group_type);
}

std::shared_ptr<std::vector<tensor_spec_t>>
BM168x::get_output_spec(Operation *op, group_type_t group_type) {
  return get_spec(op->getResults(), group_type);
}

std::shared_ptr<std::vector<tensor_spec_t>>
BM168x::get_spec(ValueRange values, group_type_t group_type) {
  auto specs = std::make_shared<std::vector<tensor_spec_t>>();
  for (auto v : values) {
    if (module::isNone(v)) {
      continue;
    }
    specs->push_back(value_to_spec(v, group_type));
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

typedef int64_t (*global_bfsz_backend_api_t)(void *params, int param_size, void *input,
                                             void *output);
int64_t BM168x::call_global_bfsz_func(const char *symbolName, void *params,
                                      int param_size, void *input, void *output) {
  auto func = instance()->CastToFPtr<global_bfsz_backend_api_t>(symbolName);
  return func(params, param_size, input, output);
}

typedef int (*local_bfsz_backend_api_t)(void *params, int param_size, void *input,
                                        void *info, void *output);
int BM168x::call_local_bfsz_func(const char *symbolName, void *params,
                                 int param_size, void *info, void *input,
                                 void *output) {
  auto func = instance()->CastToFPtr<local_bfsz_backend_api_t>(symbolName);
  return func(params, param_size, info, input, output);
}

uint64_t BM168x::CTX_START_ADDR = 0;
int64_t BM168x::IC_PARALLEL = 0;
uint64_t BM168x::GMEM_START_ADDR = 0;
int64_t BM168x::ALIGNMENT = 0;
uint64_t BM168x::L2_SRAM_START_ADDR = 0;

int BM168x::GDMA_VALUE_FORMAT_UINT8 = 0;
int BM168x::GDMA_VALUE_FORMAT_INT8 = 0;
int BM168x::GDMA_VALUE_FORMAT_FLOAT16 = 0;
int BM168x::GDMA_VALUE_FORMAT_FLOAT32 = 0;
int BM168x::GDMA_VALUE_FORMAT_INT16 = 0;
int BM168x::GDMA_VALUE_FORMAT_INT32 = 0;
int BM168x::GDMA_VALUE_FORMAT_BFLOAT16 = 0;
int BM168x::GDMA_VALUE_FORMAT_INT4 = 0;
int BM168x::GDMA_VALUE_FORMAT_NUM = 0;

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
  CAST_FUNCTION(enable_profile);
  CAST_FUNCTION(allow_atomic_cmodel_assert);
  CAST_FUNCTION(forbid_atomic_cmodel_assert);
  CAST_FUNCTION(tensor_stride_move_gen_cmd);
  CAST_FUNCTION(tensor_compact_move_gen_cmd);
  CAST_FUNCTION(tensor_align_move_gen_cmd);
  CAST_FUNCTION(set_total_id_ptr);
  CAST_CPU_FUNCTION(bmcpu_init);
  CAST_CPU_FUNCTION(bmcpu_uninit);
  CAST_CPU_FUNCTION(bmcpu_process);
  CAST_CPU_FUNCTION(bmcpu_reshape);
  CAST_CPU_FUNCTION(bmcpu_dtype);
}

BM168x::~BM168x() {
  if(bmcpu_handle != nullptr) {
    dl_bmcpu_uninit(bmcpu_handle);
  }
}

void BM168x::start_env() {
  load_library();
  bmcpu_setup();
  load_functions();
  if (0 != dl_cmodel_init(0, CMODEL_GMEM_SIZE)) {
    llvm_unreachable("cmodel init failed");
  }
  bmcpu_handle  = dl_bmcpu_init();
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
}

void BM168x::end_env() {
  BM168x::instance()->reset_command_flag();
  if (DL.isValid()) {
    if (cmdid_node != nullptr) {
      dl_destroy_cmd_id_node(gdma_node);
      dl_destroy_cmd_id_node(bdc_node);
      dl_destroy_cmd_id_node(cmdid_node);
    }
    dl_cmodel_deinit(0);
  }
}

void BM168x::bmcpu_setup() {
  std::string Err;
  cpuopDL = llvm::sys::DynamicLibrary::getPermanentLibrary(libcpuop.data(), &Err);
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
  if (mode == "NotEqual") {
    return BINARY_NE;
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
  dl_sg_set_profile_dump(true);
  reset_cmd_id_node();
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
  if (flops) {
    dl_sg_flops_dump(flops, cmdid_node);
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
  }
}

void BM168x::reset_cmd_id_node() {
  dl_reset_cmd_id(cmdid_node);
  dl_reset_cmd_id(bdc_node);
  dl_reset_cmd_id(gdma_node);
}

int64_t BM168x::get_gdma_cycle() { return dl_get_cmd_id_cycle(gdma_node); }

int64_t BM168x::get_bdc_cycle() { return dl_get_cmd_id_cycle(bdc_node); }

int64_t BM168x::get_cmd_cycle() { return dl_get_cmd_id_cycle(cmdid_node); }
