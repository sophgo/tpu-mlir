#include "sophgo/Backend/BM168x/BM168x.h"
#include "sophgo/Support/Helper/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

using namespace sophgo::backend;
using namespace sophgo::helper;

BM168x::~BM168x() {}

bm_data_type_t BM168x::getType(mlir::Type type) {
  if (type.isF32()) {
    return DTYPE_FP32;
  }
  if (type.isBF16()) {
    return DTYPE_BFP16;
  }
  if (type.isF16()) {
    return DTYPE_FP16;
  }
  if (type.isSignedInteger(8) || type.isSignlessInteger(8)) {
    return DTYPE_INT8;
  }
  if (type.isSignedInteger(16) || type.isSignlessInteger(16)) {
    return DTYPE_INT16;
  }
  if (type.isSignedInteger(32) || type.isSignlessInteger(32)) {
    return DTYPE_INT32;
  }
  if (type.isUnsignedInteger(8)) {
    return DTYPE_UINT8;
  }
  if (type.isUnsignedInteger(16)) {
    return DTYPE_UINT16;
  }
  if (type.isUnsignedInteger(32)) {
    return DTYPE_UINT32;
  }
  type.dump();
  llvm_unreachable("unknow type");
  return DTYPE_FP32;
}

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

uint64_t BM168x::get_cmodel_gmem_size() {
  return 0x100000000ull;
}

bm_data_type_t BM168x::getDataType(Value v) {
  auto type = Module::getStorageType(v);
  auto bits = type.getIntOrFloatBitWidth();
  if (type.isUnsignedInteger()) {
    switch(bits) {
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
    switch(bits) {
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
  v.dump();
  llvm_unreachable("Unsupport type \n");
  return DTYPE_FP32;
}
