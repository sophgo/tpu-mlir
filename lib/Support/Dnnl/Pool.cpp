
#include "sophgo/Support/Dnnl/Pool.h"
#include "sophgo/Support/MathUtils.h"

using namespace dnnl;
using namespace sophgo;

Pooling::Pooling() {
  eng = dnnl::engine(engine::kind::cpu, 0);
  eng_stream = dnnl::stream(eng);
  _input_paded1 = nullptr;
  _input_paded2 = nullptr;
  _pt = _pb = _pr = _pl = 0;
  _izp = 0;
}

Pooling::~Pooling() {
  if (_izp && (_pt > 0 || _pb > 0 || _pr > 0 || _pl > 0)) {
    if (_input_paded1) {
      delete []_input_paded1;
      _input_paded1 = nullptr;
    }

    if (_input_paded2) {
      delete []_input_paded2;
      _input_paded2 = nullptr;
    }
  }
}

void Pooling::pad_init(float *input, int n, int ic, int ih, int iw, int& pt, int& pb, int& pl, int& pr, int izp) {
  _input = input;
  _pt = pt;
  _pb = pb;
  _pr = pr;
  _pl = pl;
  _izp = izp;
  if (_izp && (pt > 0 || pb > 0 || pr > 0 || pl > 0)) {
    int input_paded_size = n*ic*(ih+pt+pb)*(iw+pr+pl);
    _input_paded1 = new float[input_paded_size];
    _input_paded2 = new float[input_paded_size];
    for (int i = 0; i < input_paded_size; i++) {
      _input_paded1[i] = izp;
      _input_paded2[i] = izp;
    }
    src_shape = {n, ic, ih+pt+pb, iw+pr+pl};
    pt = pb = pr = pl = 0;
  } else {
    src_shape = {n, ic, ih, iw};
    _input_paded2 = input;
  }
}

void Pooling::setup(float *input, float *output, int n, int c, int ih, int iw,
                    int oh, int ow, int kh, int kw, int sh, int sw, int pt,
                    int pb, int pl, int pr, bool is_avg, bool count_include_pad,
                    int izp, int pad_value, memory::data_type dt) {
  pad_init(input, n, c, ih, iw, pt, pb, pl, pr, izp);
  memory::dims dst_shape = {n, c, oh, ow};
  memory::dims strides = {sh, sw};
  memory::dims kernel = {kh, kw};
  memory::dims padding_tl = {pt, pl};
  memory::dims padding_br = {pb, pr};
  auto src_md = memory::desc({src_shape}, dt,
                             memory::format_tag::nchw);
  auto dst_md = memory::desc({dst_shape}, dt,
                             memory::format_tag::nchw);
  auto pool_avg_algo = count_include_pad
                           ? algorithm::pooling_avg_include_padding
                           : algorithm::pooling_avg_exclude_padding;
  // pool desc
  auto pool_desc = pooling_forward::desc(
      prop_kind::forward_inference,
      is_avg ? pool_avg_algo : algorithm::pooling_max, src_md, dst_md, strides,
      kernel, padding_tl, padding_br);

  prim_desc = pooling_forward::primitive_desc(pool_desc, eng);
  memory src_memory =
      memory({{src_shape}, memory::data_type::f32, memory::format_tag::nchw},
             eng, _input_paded2);
  memory dst_memory =
      memory({{dst_shape}, memory::data_type::f32, memory::format_tag::nchw},
             eng, output);
  net.clear();
  net_args.clear();
  auto prim_src_memory = src_memory;
  if (prim_desc.src_desc() != src_memory.get_desc()) {
    prim_src_memory = memory(prim_desc.src_desc(), eng);
    net.push_back(reorder(src_memory, prim_src_memory));
    net_args.push_back(
        {{DNNL_ARG_FROM, src_memory}, {DNNL_ARG_TO, prim_src_memory}});
  }
  auto prim_dst_memory = memory(prim_desc.dst_desc(), eng);
  net.push_back(pooling_forward(prim_desc));
  net_args.push_back(
      {{DNNL_ARG_SRC, prim_src_memory}, {DNNL_ARG_DST, prim_dst_memory}});
  if (prim_dst_memory != dst_memory) {
    net.push_back(reorder(prim_dst_memory, dst_memory));
    net_args.push_back(
        {{DNNL_ARG_FROM, prim_dst_memory}, {DNNL_ARG_TO, dst_memory}});
  }
}

void Pooling::run() {
  if (_izp)
    pad_tensor(_input, _input_paded1, _input_paded2, src_shape[0], src_shape[1], src_shape[2], src_shape[3], _pt, _pb, _pl, _pr);
  for (size_t i = 0; i < net.size(); ++i)
    net.at(i).execute(eng_stream, net_args.at(i));
  eng_stream.wait();
}
