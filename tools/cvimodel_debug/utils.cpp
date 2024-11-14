#include "utils.hpp"
namespace cvi_debug {

std::string dtypeToStr(cvi::model::DType type) {
  switch (type) {
  case cvi::model::DType_FP32:
    return "fp32";
  case cvi::model::DType_INT32:
    return "int32";
  case cvi::model::DType_UINT32:
    return "uint32";
  case cvi::model::DType_BF16:
    return "bf16";
  case cvi::model::DType_INT16:
    return "int16";
  case cvi::model::DType_UINT16:
    return "uint16";
  case cvi::model::DType_INT8:
    return "int8";
  case cvi::model::DType_UINT8:
    return "uint8";
  default:
    printf("unknown dtype\n");
  }
  return "";
}

uint32_t dtypeSize(cvi::model::DType type) {
  switch (type) {
  case cvi::model::DType_FP32:
    return 4;
  case cvi::model::DType_INT32:
    return 4;
  case cvi::model::DType_UINT32:
    return 4;
  case cvi::model::DType_BF16:
    return 2;
  case cvi::model::DType_INT16:
    return 2;
  case cvi::model::DType_UINT16:
    return 2;
  case cvi::model::DType_INT8:
    return 1;
  case cvi::model::DType_UINT8:
    return 1;
  default:
    printf("unknown dtype\n");
  }
  return 0;
}

uint32_t getNpuSize(std::string chip) {
  uint32_t size = 0;
  if (chip == "cv180x") {
    size = 2 * 32 * 1024;
  } else if (chip == "cv181x" || chip == "cv182x") {
    size = 8 * 32 * 1024;
  } else if (chip == "cv183x") {
    size = 32 * 32 * 1024;
  } else {
    assert(0 && "unsupport chip");
  }
  return size;
}

uint32_t align_up(int x, int n) {
  if (n == 0) {
    return x;
  }
  return ((x + n - 1) / n) * n;
}

tpu_mlir::module::Chip getChip(std::string chip) {
  if (chip == "cv180x") {
    return tpu_mlir::module::Chip::CV180x;
  } else if (chip == "cv181x") {
    return tpu_mlir::module::Chip::CV181x;
  } else if (chip == "cv182x") {
    return tpu_mlir::module::Chip::CV182x;
  } else if (chip == "cv183x") {
    return tpu_mlir::module::Chip::CV183x;
  } else {
    assert(0 && "unsupport chip type");
  }
  return tpu_mlir::module::Chip::CV183x;
}

CVIKERNEL_FMT_E getCviKernelFmt(std::string type) {
  if (type == "bf16") {
    return CVIKERNEL_FMT_E::CVK_FMT_BF16;
  } else if (type == "int8") {
    return CVIKERNEL_FMT_E::CVK_FMT_I8;
  } else {
    assert(0 && "cv18xx local op only support bf16 and int8 type");
  }
  return CVIKERNEL_FMT_E::CVK_FMT_BF16;
}

void reset_tiu_info(uint32_t *p, uint32_t tiuCnt, uint32_t tdmaCnt,
                    uint8_t magicNum) {
  if (tiuCnt == 0 && tdmaCnt == 0) {
    return;
  }
  if (magicNum == 0xA5) {
    // reset cv183x tiu_info
    // cal des_cmd_id_tpu
    uint32_t ori_cmd_id_tpu =
        (p[0] >> 3) & ((1u << 16) - 1); // get p[0][4:19] bit
    uint32_t des_cmd_id_tpu = ori_cmd_id_tpu - tiuCnt;
    if (des_cmd_id_tpu > 65535) {
      llvm::errs() << "ori_cmd_id_tpu:" << ori_cmd_id_tpu
                   << ",tiuCnt:" << tiuCnt
                   << ",des_cmd_id_tpu:" << des_cmd_id_tpu << "\n";
    }
    assert(des_cmd_id_tpu <= 65535);
    // reflect to p
    p[0] = p[0] & 0xFFF80007;            // set p[0][4:19] = 0
    p[0] = p[0] | (des_cmd_id_tpu << 3); // set p[0][4:19] = des_cmd_id_tpu
    // cal des_cmd_id_gdma
    uint32_t ori_cmd_id_gdma = (p[0] >> 19) & ((1u << 13) - 1);
    ori_cmd_id_gdma |= (uint64_t)(p[1] & ((1u << 3) - 1))
                       << 13; // get p[0][20:32] and p[1][1:3](left shift is to
                              // put it to the high bit part)
    uint32_t des_cmd_id_gdma = 0;
    if (ori_cmd_id_gdma <= tdmaCnt) {
      des_cmd_id_gdma = 0;
    } else {
      des_cmd_id_gdma = ori_cmd_id_gdma - tdmaCnt;
    }
    // reflect to p
    uint32_t low13 = des_cmd_id_gdma & 0x1FFF;
    uint32_t high3 = des_cmd_id_gdma & 0xEFFF;
    p[0] = p[0] & 0x7FFFF;
    p[0] = p[0] | (low13 << 19);
    p[1] = p[1] & 0xFFFFFFF8;
    p[1] = p[1] | (high3 >> 29);
  } else if (magicNum == 0xA6 || magicNum == 0xA7 || magicNum == 0xA8) {
    // reset cv182x/cv181x/cv180x
    // cal des_cmd_id_tpu
    uint32_t ori_cmd_id_tpu = p[1] & ((1u << 16) - 1);
    uint32_t des_cmd_id_tpu = ori_cmd_id_tpu - tiuCnt;
    assert(des_cmd_id_tpu <= 65535);
    // reflect to p
    p[1] = p[1] & 0xFFFF0000;
    p[1] = p[1] | des_cmd_id_tpu;
    // cal des_cmd_id_gdma
    uint32_t ori_cmd_id_gdma = (p[1] >> 16) & ((1u << 16) - 1);
    uint32_t des_cmd_id_gdma = 0;
    if (ori_cmd_id_gdma <= tdmaCnt) {
      des_cmd_id_gdma = 0;
    } else {
      des_cmd_id_gdma = ori_cmd_id_gdma - tdmaCnt;
    }
    // reflect to p
    p[1] = p[1] & 0xFFFF;
    p[1] = p[1] | (des_cmd_id_gdma << 16);
  } else {
    printf("invalid magic number %x\n", magicNum);
    assert(0);
  }
}

void reset_tdma_info(uint32_t *p, uint32_t tiuCnt, uint32_t tdmaCnt,
                     uint8_t magicNum) {
  if (tiuCnt == 0 && tdmaCnt == 0) {
    return;
  }
  if (magicNum == 0xA5 || magicNum == 0xA6 || magicNum == 0xA7 ||
      magicNum == 0xA8) {
    // reset cv183x/cv182x/cv181x/cv180x
    // cal des_cmd_id_tdma
    uint32_t ori_cmd_id = (p[0] >> 16) & ((1u << 16) - 1);
    uint32_t des_cmd_id = ori_cmd_id - tdmaCnt;
    assert(des_cmd_id <= 65535);
    // reflect
    p[0] = p[0] & 0xFFFF;
    p[0] = p[0] | (des_cmd_id << 16);
    // cal des_wait_id_tpu
    uint32_t ori_wait_id_tpu = (p[1] >> 16) & ((1u << 16) - 1);
    uint32_t des_wait_id_tpu = 0;
    if (ori_wait_id_tpu <= tiuCnt) {
      des_wait_id_tpu = 0;
    } else {
      des_wait_id_tpu = ori_wait_id_tpu - tiuCnt;
    }
    // reflect
    p[1] = p[1] & 0xFFFF;
    p[1] = p[1] | (des_wait_id_tpu << 16);
  } else {
    printf("invalid magic number %x\n", magicNum);
    assert(0);
  }
}

void strSplit(const std::string &str, const std::string &splits,
              std::vector<std::string> &res) {
  if (str == "") {
    return;
  }
  std::string strs = str + splits;
  size_t pos = strs.find(splits);
  int step = splits.size();
  while (pos != strs.npos) {
    std::string temp = strs.substr(0, pos);
    res.emplace_back(temp);
    strs = strs.substr(pos + step, strs.size());
    pos = strs.find(splits);
  }
}

void ConvertFp32ToInt8(float *src, int8_t *dst, int count, float qscale) {
  for (int i = 0; i < count; i++) {
    int val = std::round((*src++) / qscale);
    if (val > 127) {
      val = 127;
    } else if (val < -128) {
      val = -128;
    }
    *dst++ = (int8_t)val;
  }
}

void ConvertFp32ToBF16(float *src, uint16_t *dst, int count) {
  for (int i = 0; i < count; i++) {
    float src_val = src[i];
    uint16_t dst_val = ((uint16_t *)(&src_val))[1];
    dst[i] = dst_val;
  }
}

void ConvertFp32ToUint16(float *src, uint16_t *dst, int count) {
  for (int i = 0; i < count; i++) {
    uint16_t val = (uint16_t)(src[i]);
    dst[i] = val;
  }
}

void ConvertFp32ToInt32(float *src, int32_t *dst, int count) {
  for (int i = 0; i < count; i++) {
    int32_t val = (int32_t)(src[i]);
    dst[i] = val;
  }
}

void ConvertFp32ToInt8NoScale(float *src, int8_t *dst, int count) {
  for (int i = 0; i < count; i++) {
    int8_t val = (int8_t)(src[i]);
    dst[i] = val;
  }
}

void ConvertInt8ToFp32(int8_t *src, float *dst, int count, float qscale) {
  for (int i = 0; i < count; i++) {
    float val = (*src++) * qscale;
    *dst++ = val;
  }
}

void ConvertInt8ToFp32NoScale(int8_t *src, float *dst, int count) {
  for (int i = 0; i < count; i++) {
    float val = (float)(src[i]);
    dst[i] = val;
  }
}

void ConvertUint8ToFp32(uint8_t *src, float *dst, int count, float qscale) {
  for (int i = 0; i < count; i++) {
    float val = (*src++) * qscale;
    *dst++ = val;
  }
}

void ConvertUint16ToFp32(uint16_t *src, float *dst, int count) {
  for (int i = 0; i < count; i++) {
    float val = (float)(src[i]);
    dst[i] = val;
  }
}

void ConvertInt32ToFp32(int32_t *src, float *dst, int count) {
  for (int i = 0; i < count; i++) {
    float val = (float)(src[i]);
    *dst++ = val;
  }
}

void ConvertBF16ToFp32(uint16_t *src, float *dst, int count) {
  for (int i = 0; i < count; i++) {
    unsigned int tmp = *src;
    tmp = tmp << 16;
    float val = *((float *)&tmp);
    *dst = val;
    src++;
    dst++;
  }
}
} // namespace cvi_debug
