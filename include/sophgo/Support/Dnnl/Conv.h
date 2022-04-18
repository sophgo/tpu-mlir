#pragma once
#include "oneapi/dnnl/dnnl.hpp"
using namespace dnnl;
namespace sophgo {
class Conv {
public:
  Conv();

  void setup(float *input, float *weight, float *bias, float *output, int n,
             int ic, int ih, int iw, int oc, int oh, int ow, int kh, int kw,
             int sh, int sw, int dh, int dw, int pt, int pb, int pl, int pr,
             int g, bool do_relu, int izp = 0, int ozp = 0, int* rshift = nullptr, int* multiplier = nullptr,
             memory::data_type idt = memory::data_type::f32,
             memory::data_type wdt = memory::data_type::f32,
             memory::data_type bdt = memory::data_type::f32,
             memory::data_type odt = memory::data_type::f32, bool per_channel = false, int chip = 0);

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
