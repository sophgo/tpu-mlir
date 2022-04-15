#include "sophgo/Backend/BM168x/BM168x.h"
#include "sophgo/Backend/BM168x/BM1684.h"
#include "sophgo/Backend/BM168x/BM1686.h"
#include "sophgo/Support/Helper/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

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

uint64_t BM168x::get_cmodel_gmem_size() { return 0x100000000ull; }

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
  CAST_FUNCTION(set_total_id_ptr);
}

void BM168x::init() {
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
