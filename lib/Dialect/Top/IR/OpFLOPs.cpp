#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

int64_t top::AddOp::getFLOPs() {
  return Module::getNumElements(output()) *
         (inputs().size() - 1 + do_relu() ? 1 : 0);
}

int64_t top::AvgPoolOp::getFLOPs() {
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool has_relu, is_global, count_include_pad;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             has_relu, is_global, count_include_pad);
  return Module::getNumElements(output()) * (kh * kw + has_relu ? 1 : 0);
}

int64_t top::BatchNormOp::getFLOPs() {
  return Module::getNumElements(output()) * 2;
}

int64_t top::ConvOp::getFLOPs() {
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, has_relu;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, has_relu);
  auto extra = with_bias ? 1 : 0 + has_relu ? 1 : 0;
  return Module::getNumElements(output()) * (kw * kw * ic / g * 2 + extra);
}

int64_t top::MatMulOp::getFLOPs() {
  int64_t batch, M, K, N;
  bool has_relu, with_bias;
  parseParam(batch, M, K, N, with_bias, has_relu);
  auto extra = with_bias ? 1 : 0 + has_relu ? 1 : 0;
  return batch * (2 * K + extra) * N * M;
}

int64_t top::MaxPoolOp::getFLOPs() {
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool has_relu, is_global, count_include_pad;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             has_relu, is_global, count_include_pad);
  auto extra = has_relu ? 1 : 0;
  return Module::getNumElements(output()) * (kh * kw + extra);
}

int64_t top::ReluOp::getFLOPs() { return Module::getNumElements(output()); }

int64_t top::SoftmaxOp::getFLOPs() {
  //   2*n          -- compute shifted logits
  //   n            -- exp of shifted logits
  //   2*n          -- compute softmax from exp of shifted logits
  return Module::getNumElements(input()) * 5;
}

// ###########################
// Ops needn't care FLOPs
// ##########################
int64_t top::PadOp::getFLOPs() { return 0; }

int64_t top::ReshapeOp::getFLOPs() { return 0; }
