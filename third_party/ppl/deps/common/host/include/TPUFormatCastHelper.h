#ifndef __TPU_FORMATCAST
#define __TPU_FORMATCAST

#include <ATen/ATen.h>
#include <unordered_map>
#include "TPUStorageImpl.h"

using namespace torch_tpu;

namespace at_tpu
{
constexpr int MAX_FORMAT_SHAPE_SIZE = 8;
using FormatShape = c10::SmallVector<int64_t, MAX_FORMAT_SHAPE_SIZE>;
using baseFormatConverter = std::function<FormatShape(c10::IntArrayRef storage_dims, c10::IntArrayRef base_dims)>;

/**************************************************************
*****************   TPU Storage Desc Helper   *****************
***************************************************************/
class StorageDescHelper {
public:
  // Get Attribute
  static tpuFormat GetBaseFormat(tpuFormat format) {
    const auto& itr = info_.find(format);
    if (itr == info_.end())
    {
    AT_ERROR("unknown format type:", format);
    return TPU_DFORMAT_ND;
    }
    return itr->second.baseFormat;
  }

  // Check Part
  static bool IsBaseFormatType(tpuFormat format) {
    return GetBaseFormat(format) == format;
  }
  static bool IsBaseFormatType(const at::Tensor &tensor) {
    if (tensor.device().type() == c10::DeviceType::CPU) return true;
    auto format = torch_tpu::GetTpuStorageImplDesc(tensor).tpu_format_;
    return IsBaseFormatType(format);
  }

private:
  using shapeInfer = std::function<FormatShape(c10::IntArrayRef dims)>;
  typedef struct FormatInfo_
  {
    tpuFormat format = TPU_DFORMAT_ND;
    tpuFormat baseFormat = TPU_DFORMAT_ND;
    shapeInfer func = nullptr;
    char formatName[30] = {0};
  } FormatInfo;
  inline static std::unordered_map<tpuFormat, FormatInfo> info_ = {
    {TPU_DFORMAT_ND, (FormatInfo){TPU_DFORMAT_ND, TPU_DFORMAT_ND, nullptr, "ND"}},
    {TPU_DFORMAT_CONV_W_Infer, (FormatInfo){TPU_DFORMAT_CONV_W_Infer, TPU_DFORMAT_ND, nullptr, "CONV_W_Infer"}},
    {TPU_DFORMAT_CONV_W_Train, (FormatInfo){TPU_DFORMAT_CONV_W_Train, TPU_DFORMAT_ND, nullptr, "CONV_W_Train"}},
    {TPU_DFORMAT_CONV_DW, (FormatInfo){TPU_DFORMAT_CONV_DW, TPU_DFORMAT_ND, nullptr, "CONV_DW"}},
  };
}; // struct StorageDescHelper
}  // namespace:at_tpu
#endif
