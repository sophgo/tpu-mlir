
#pragma once

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {
class Pooling {
public:
  Pooling();

  void setup(float *input, float*output, int n, int c, int ih, int iw, int oh, int ow, int kh, int kw,
             int sh, int sw, int pt, int pb, int pl, int pr, bool is_avg,
             bool count_include_pad, int pad_value = 0);

  void run();

private:
  engine eng;
  stream eng_stream;
  std::vector<primitive> net;
  std::vector<std::unordered_map<int, memory>> net_args;
  pooling_forward::primitive_desc prim_desc;
};
} // namespace dnnl
