#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Builder/CV18xx/cvimodel_generated.h"
#include "tpu_mlir/Builder/CV18xx/parameter_generated.h"
using namespace tpu_mlir::backend;
namespace cvi_debug {

std::string dtypeToStr(cvi::model::DType type);

uint32_t dtypeSize(cvi::model::DType type);

uint32_t getNpuSize(std::string chip);

tpu_mlir::module::Chip getChip(std::string chip);

CVIKERNEL_FMT_E getCviKernelFmt(std::string type);

uint32_t align_up(int x, int n);

void strSplit(const std::string &str, const std::string &splits,
              std::vector<std::string> &res);

void reset_tiu_info(uint32_t *p, uint32_t tiuCnt, uint32_t tdmaCnt,
                    uint8_t magicNum);

void reset_tdma_info(uint32_t *p, uint32_t tiuCnt, uint32_t tdmaCnt,
                     uint8_t magicNum);

void ConvertFp32ToInt8(float *src, int8_t *dst, int count, float qscale);

void ConvertFp32ToBF16(float *src, uint16_t *dst, int count);

void ConvertFp32ToUint16(float *src, uint16_t *dst, int count);

void ConvertFp32ToInt32(float *src, int32_t *dst, int count);

// just convert int8 data with float storage to int8 storage
void ConvertFp32ToInt8NoScale(float *src, int8_t *dst, int count);

void ConvertInt8ToFp32(int8_t *src, float *dst, int count, float qscale);

// just convert int8 data to float storage
void ConvertInt8ToFp32NoScale(int8_t *src, float *dst, int count);

void ConvertUint8ToFp32(uint8_t *src, float *dst, int count, float qscale);

void ConvertUint16ToFp32(uint16_t *src, float *dst, int count);

void ConvertInt32ToFp32(int32_t *src, float *dst, int count);

void ConvertBF16ToFp32(uint16_t *src, float *dst, int count);

// void ConvertFp32ToUint8(float *src, uint8_t dst, int count, float qscale, int
// zero_point = 0); void ConvertFp32ToInt16(float *src, int16_t *dst, int
// count); void ConvertFp32ToUInt16(float *src, uint16_t *dst, int count); void
// ConvertFp32ToBf16(float *src, uint16_t *dst, int size, bool rounding);
} // namespace cvi_debug
