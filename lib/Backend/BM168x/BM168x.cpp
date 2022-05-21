#include "sophgo/Backend/BM168x/BM168x.h"
#include "sophgo/Backend/BM168x/BM1684.h"
#include "sophgo/Backend/BM168x/BM1686.h"
#include "sophgo/Interfaces/LocalGenInterface.h"
#include "sophgo/Support/Helper/Module.h"
#include "sophgo/Support/MathUtils.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

using namespace sophgo;
using namespace sophgo::backend;
using namespace sophgo::helper;

void *BM168x::get_gmem_addr(uint64_t addr) {
  auto start = static_cast<char *>(this->dl_get_global_memaddr(0));
  return start + addr - get_gmem_start();
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

bm_data_type_t BM168x::getDataType(Value v) {
  auto type = Module::getStorageType(v);
  return getDataType(type);
}

bm_data_type_t BM168x::getDataType(mlir::Type type) {
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

int BM168x::getFmtBytes(bm_data_type_t data_type) {
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

global_tensor_spec_t BM168x::value_to_global(mlir::Value v) {
  global_tensor_spec_t spec;
  memset(&spec, sizeof(spec), 0);
  spec.addr = Module::getAddress(v);
  spec.dtype = getDataType(v);
  auto shape = Module::getShape(v);
  spec.dims = shape.size();
  for (int i = 0; i < spec.dims; i++) {
    spec.shape[i] = shape[i];
  }
  return spec;
}
std::shared_ptr<std::vector<global_tensor_spec_t>>
BM168x::get_input_global_spec(Operation *op) {
  std::vector<Value> inputs;
  for (auto in : op->getOperands()) {
    if (in.getType().isa<NoneType>()) {
      continue;
    }
    inputs.push_back(in);
  }
  auto global_specs =
      std::make_shared<std::vector<global_tensor_spec_t>>(inputs.size());
  std::transform(inputs.begin(), inputs.end(), global_specs->begin(),
                 value_to_global);
  return std::move(global_specs);
}

std::shared_ptr<std::vector<global_tensor_spec_t>>
BM168x::get_output_global_spec(Operation *op) {
  auto outputs = op->getResults();
  auto global_specs =
      std::make_shared<std::vector<global_tensor_spec_t>>(outputs.size());
  std::transform(outputs.begin(), outputs.end(), global_specs->begin(),
                 value_to_global);
  return std::move(global_specs);
}

local_tensor_spec_t BM168x::value_to_local(mlir::Value v) {
  local_tensor_spec_t spec;
  memset(&spec, sizeof(spec), 0);
  auto gi = LocalGenInterface::getGroupInfo(v);
  spec.addr = gi.out_addr;
  spec.dtype = getDataType(v);
  auto shape = Module::getShape(v);
  spec.dims = shape.size();
  for (int i = 0; i < spec.dims; i++) {
    spec.shape[i] = shape[i];
  }
  return spec;
}

std::shared_ptr<std::vector<local_tensor_spec_t>>
BM168x::get_input_local_spec(mlir::Operation *op) {
  std::vector<Value> inputs;
  for (auto in : op->getOperands()) {
    if (in.getType().isa<NoneType>()) {
      continue;
    }
    inputs.push_back(in);
  }
  auto local_specs =
      std::make_shared<std::vector<local_tensor_spec_t>>(inputs.size());
  std::transform(inputs.begin(), inputs.end(), local_specs->begin(),
                 value_to_local);
  return std::move(local_specs);
}

std::shared_ptr<std::vector<local_tensor_spec_t>>
BM168x::get_output_local_spec(mlir::Operation *op) {
  auto outputs = op->getResults();
  auto local_specs =
      std::make_shared<std::vector<local_tensor_spec_t>>(outputs.size());
  std::transform(outputs.begin(), outputs.end(), local_specs->begin(),
                 value_to_local);
  return std::move(local_specs);
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
                                   int fmtBytes) {
  stride_4D_t s;
  s.W = 1;
  s.H = W;
  s.C = align_up(H * W, get_eu_num(fmtBytes));
  s.N = ceiling_func(C, get_npu_num()) * s.C;
  return s;
}

int64_t BM168x::get_lmem_bytes(int64_t n, int64_t c, int64_t h, int64_t w,
                               mlir::Type type, bool eu_align, bool is_4N) {
  int64_t npu_num = get_npu_num();
  int64_t dbytes = type.getIntOrFloatBitWidth() / 8;
  int64_t eu_num = get_eu_num(dbytes);
  int64_t c_per_npu = ceiling_func(c, npu_num);
  int64_t n_align = is_4N ? 1 : get_n_align(dbytes);
  int64_t n_aligned = align_up(n, n_align);
  int64_t eu_aligned =
      eu_align ? align_up(h * w, eu_num) * dbytes : (h * w * dbytes);
  return n_aligned * c_per_npu * eu_aligned;
}

int64_t BM168x::get_tensor_lmem_bytes(mlir::Value v, int64_t slice_n,
                                      int64_t slice_h, bool eu_align) {
  int64_t n, c, h, w;
  Module::getNCHW(v, n, c, h, w);
  auto type = Module::getStorageType(v);
  bool is_4N = false;
  if (chip == Module::Chip::BM1684) {
    is_4N = true;
  }
  return get_lmem_bytes(slice_n, c, slice_h, w, type, eu_align, is_4N);
}

int64_t BM168x::get_weight_lmem_bytes(mlir::Value v, bool eu_align) {
  int64_t n, c, h, w;
  Module::getNCHW(v, n, c, h, w);
  auto type = Module::getStorageType(v);
  return get_lmem_bytes(n, c, h, w, type, eu_align);
}

template <typename FPtrTy> FPtrTy BM168x::CastToFPtr(const char *symbolName) {
  assert(DL.isValid());
  auto fPtr = DL.getAddressOfSymbol(symbolName);
  if (fPtr == nullptr) {
    llvm::errs() << "can't find symbol: " << symbolName << "\n";
    llvm_unreachable(symbolName);
  }
  return reinterpret_cast<FPtrTy>(fPtr);
}

#define CAST_FUNCTION(name) dl_##name = CastToFPtr<name>(#name)

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
  CAST_FUNCTION(tensor_stride_move_gen_cmd);
  CAST_FUNCTION(set_total_id_ptr);
}

void BM168x::init() {
  if (!DL.isValid()) {
    std::string Err;
    DL = llvm::sys::DynamicLibrary::getPermanentLibrary(get_lib_name(), &Err);
    if (DL.isValid() == false) {
      llvm_unreachable(Err.c_str());
    }
    load_functions();
  }
  dl_cmodel_init(0, get_cmodel_gmem_size());
  cmdid_node = dl_create_cmd_id_node();
  bdc_node = dl_create_cmd_id_node();
  gdma_node = dl_create_cmd_id_node();
  bdc_buffer = std::make_shared<std::vector<uint32_t>>(0x1000000);
  gdma_buffer = std::make_shared<std::vector<uint32_t>>(0x1000000);
  dl_set_cmd_buffer_ptr((void *)gdma_buffer->data(),
                        (void *)bdc_buffer->data());
  dl_set_total_id_ptr(&gdma_total_id, &bdc_total_id, cmdid_node,
                      (void *)&gdma_group_id, (void *)&bdc_group_id,
                      &cmdid_groupnum);
  dl_forbid_atomic_cmodel(); // TODO:(no compare)
}

void BM168x::deinit() {
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
  set_command_issue_flag(true);
  gdma_group_id.clear();
  gdma_group_id.push_back(0);
  bdc_group_id.clear();
  bdc_group_id.push_back(0);
  gdma_bytes.clear();
  bdc_bytes.clear();
  cmdid_groupnum = 1;
}

void BM168x::after_codegen() {}

BM168x *BM168x::instance(const StringRef chip) {
  BM168x *p_backend;
  if (chip == Module::Chip::BM1684) {
    return &BM1684::instance();
  } else if (chip == Module::Chip::BM1686) {
    return &BM1686::instance();
  } else {
    llvm_unreachable("unsupport chip");
  }
}

void BM168x::set_command_issue_flag(bool value) {
  really_issue_command = value;
  if (really_issue_command) {
    dl_allow_store_cmd();
    dl_use_atomic_cmodel();
  } else {
    dl_forbid_store_cmd();
    dl_forbid_atomic_cmodel();
  }
}
