//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022  Technologies Inc.  All rights reserved.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "ppl_defs.h"
#include "ppl_types.h"
#include <inttypes.h>

namespace ppl {
namespace conv::param {
template <typename DataType>
conv_param padding(DataType padv, int up_pad, int down_pad, int left_pad, int right_pad, int mode=0);
template <typename DataType>
conv_param padding(tensor<DataType> &padv, int up_pad, int down_pad, int left_pad, int right_pad, int mode=0);
template <typename DataType>
conv_param insert(DataType insert_val=0, int dilation_h=1, int dilation_w=1, int ins_h=0, int ins_w=0);
conv_param kernel(int kh, int kw, int stride_h, int stride_w, bool do_relu=false, bool kernel_rotate=false);
} // namespace conv::param
} // namespace ppl
