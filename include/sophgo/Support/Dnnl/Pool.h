
#pragma once

#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;
namespace sophgo {
class Pooling {
public:
  Pooling();
  ~Pooling();

  void pad_init(float *input, int n, int ic, int ih, int iw, int& pt, int& pb, int& pl, int& pr, int izp);
  void setup(float *input, float*output, int n, int c, int ih, int iw, int oh, int ow, int kh, int kw,
             int sh, int sw, int pt, int pb, int pl, int pr, bool is_avg,
             bool count_include_pad, int izp = 0, int pad_value = 0, memory::data_type dt = memory::data_type::f32);

  void run();

private:
  engine eng;
  stream eng_stream;
  std::vector<primitive> net;
  std::vector<std::unordered_map<int, memory>> net_args;
  pooling_forward::primitive_desc prim_desc;
  memory::dims src_shape;
  memory::dims dst_shape;
  float* _input;
  float* _input_paded1;
  float* _input_paded2;
  int _pt;
  int _pb;
  int _pl;
  int _pr;
  int _izp;
};
} // namespace dnnl
