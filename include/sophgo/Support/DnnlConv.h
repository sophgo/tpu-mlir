#pragma once
#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {
class Conv {
public:
  Conv();

  void setup(float *input, float *weight, float *bias, float *output, int n,
             int ic, int ih, int iw, int oc, int oh, int ow, int kh, int kw,
             int sh, int sw, int dh, int dw, int pt, int pb, int pl, int pr,
             int g);

  void run();

private:
  engine eng;
  stream eng_stream;
  std::vector<primitive> net;
  std::vector<std::unordered_map<int, memory>> net_args;
  convolution_forward::primitive_desc conv_prim_desc;
  memory prim_filter_memory;
  memory prim_bias_memory;
  memory::dims src_shape;
  memory::dims dst_shape;
};
} // namespace dnnl
