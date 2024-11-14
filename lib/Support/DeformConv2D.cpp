#include "tpu_mlir/Support/Dnnl/Dnnl.h"

namespace tpu_mlir {

template <typename T>
static inline T index_2d(const T *arr, int height, int width, int y, int x) {
  if (y <= -1 || height <= y || x <= -1 || width <= x)
    return 0;
  return arr[y * width + x];
}

template <typename T>
T bilinear_interpolate(const T *in, int height, int width, T h, T w) {
  if (h <= -1 || height <= h || w <= -1 || width <= w) {
    return 0;
  }

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  T lh = h - h_low;
  T lw = w - w_low;
  T hh = 1 - lh, hw = 1 - lw;

  T v1 = index_2d(in, height, width, h_low, w_low);
  T v2 = index_2d(in, height, width, h_low, w_high);
  T v3 = index_2d(in, height, width, h_high, w_low);
  T v4 = index_2d(in, height, width, h_high, w_high);

  T w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename T>
static inline void deform_process_per_kernel(
    const T *data_img, const T *data_offset, const T *data_mask, T *data_out,
    const int H, const int W, const int kh, const int kw, const int dilation_h,
    const int dilation_w, const int num_kernels_per_channel, const int in_h,
    const int in_w) {
  for (int h = 0; h < kh; ++h) {
    for (int w = 0; w < kw; ++w) {
      const int out_idx = (h * kw + w) * num_kernels_per_channel;
      const int mask_idx = (h * kw + w) * num_kernels_per_channel;
      const int offset_h_idx = mask_idx * 2;
      const int offset_w_idx = offset_h_idx + num_kernels_per_channel;
      const T offset_h = data_offset[offset_h_idx];
      const T offset_w = data_offset[offset_w_idx];
      const T h_im = in_h + h * dilation_h + offset_h;
      const T w_im = in_w + w * dilation_w + offset_w;

      T val = bilinear_interpolate(data_img, H, W, h_im, w_im);
      if (data_mask) {
        val *= data_mask[mask_idx];
      }
      data_out[out_idx] = val;
    }
  }
}

template <typename T>
static inline void deform_process_v1(
    const T *data_img, const T *data_offset, const T *data_mask, T *data_out,
    const int H, const int W, const int conved_H, const int conved_W,
    const int kh, const int kw, const int dilation_h, const int dilation_w,
    const int stride_h, const int stride_w, const int pad_t, const int pad_l) {
  // process input features as the next layer is still conv layer
  // with new params, such as stride, dilation and pad.
  const int num_kernels_per_channel = conved_H * conved_W;

  for (int h = 0; h < conved_H; ++h) {
    for (int w = 0; w < conved_W; ++w) {
      const int in_h = stride_h * h - pad_t;
      const int in_w = stride_w * w - pad_l;
      const T *data_offset_ptr = data_offset + h * conved_W + w;
      const T *data_mask_ptr =
          (data_mask ? data_mask + h * conved_W + w : nullptr);
      T *data_out_ptr = data_out + h * conved_W + w;
      deform_process_per_kernel(
          data_img, data_offset_ptr, data_mask_ptr, data_out_ptr, H, W, kh, kw,
          dilation_h, dilation_w, num_kernels_per_channel, in_h, in_w);
    }
  }
}

void processDeformGather(InferenceParameter &p,
                         const deform_gather_attr_t &attr, float *data_out,
                         bool top_flag) {
  // data: [N, C, H, W]
  // offset: [N, num_deform_group*kh*kw*2, conved_H, conved_W]
  // mask: [N, num_deform_group*kh*kw, conved_H, conved_W]
  // out: [N, C*kh*kw, conved_H, conved_W]

  const int N = attr.n;
  const int C = attr.ic;
  const int H = attr.ih;
  const int W = attr.iw;
  const int conved_H =
      ((H - (attr.dh * (attr.kh - 1) + 1) + attr.pht + attr.phb) / attr.sh + 1);
  const int conved_W =
      ((W - (attr.dw * (attr.kw - 1) + 1) + attr.pwl + attr.pwr) / attr.sw + 1);
  const int channel_per_deform_group = C / attr.deform_groups;
  int offset_offset = attr.ofc * attr.ofh * attr.ofw / attr.deform_groups;

  const float *data_img = (float *)p.inputs[0];
  const float *data_offset = nullptr;
  if (top_flag)
    data_offset = (float *)p.inputs[2];
  else
    data_offset = (float *)p.inputs[1];

  float *data_mask = nullptr;
  int mask_offset = 0;
  if (attr.use_mask) {
    mask_offset = attr.mkc * attr.mkh * attr.mkw / attr.deform_groups;
    if (top_flag)
      data_mask = p.inputs[3];
    else
      data_mask = p.inputs[2];
  }

  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      const int g = c / channel_per_deform_group;
      const int in_idx = (n * C + c) * H * W;
      const int out_idx = (n * C + c) * attr.kh * conved_H * attr.kw * conved_W;
      const int offset_idx = (n * attr.deform_groups + g) * offset_offset;
      const int mask_idx = (n * attr.deform_groups + g) * mask_offset;
      deform_process_v1<float>(data_img + in_idx, data_offset + offset_idx,
                               (attr.use_mask ? data_mask + mask_idx : nullptr),
                               data_out + out_idx, H, W, conved_H, conved_W,
                               attr.kh, attr.kw, attr.dh, attr.dw, attr.sh,
                               attr.sw, attr.pht, attr.pwl);
    }
  }
}

void parseGatherParam(const deform_conv2d_attr_t &attr,
                      deform_gather_attr_t &gattr) {
  gattr.n = attr.n;
  gattr.ic = attr.ic;
  gattr.ih = attr.ih;
  gattr.iw = attr.iw;
  gattr.oc = attr.ic * attr.kh * attr.kw;
  gattr.kh = attr.kh;
  gattr.kw = attr.kw;
  gattr.sh = attr.sh;
  gattr.sw = attr.sw;
  gattr.dh = attr.dh;
  gattr.dw = attr.dw;
  gattr.phb = attr.phb;
  gattr.pht = attr.pht;
  gattr.pwl = attr.pwl;
  gattr.pwr = attr.pwr;
  gattr.ofc = attr.ofc;
  gattr.ofh = attr.ofh;
  gattr.ofw = attr.ofw;
  gattr.mkc = attr.mkc;
  gattr.mkh = attr.mkh;
  gattr.mkw = attr.mkw;
  gattr.use_mask = attr.use_mask;
  gattr.deform_groups = attr.deform_groups;
}

void parseConvParam(const deform_conv2d_attr_t &attr, conv_attr_t &cattr) {
  cattr.n = attr.n;
  cattr.ic = attr.ic * attr.kh * attr.kw;
  cattr.oc = attr.oc;
  cattr.oh = attr.oh;
  cattr.ow = attr.ow;
  cattr.kh = cattr.kw = 1;
  cattr.dh = cattr.dw = 1;
  cattr.sh = cattr.sw = 1;
  cattr.id = cattr.od = cattr.kd = cattr.dd = cattr.sd = 1;
  cattr.groups = attr.groups;
  cattr.has_bias = attr.has_bias;
  cattr.do_relu = attr.do_relu;
  cattr.relu_limit = attr.relu_limit;
}

void processDeformConv2D(InferenceParameter &p,
                         const deform_conv2d_attr_t &attr) {
  // p.inputs: input weight offset mask bias
  // p.outputs: output

  const int conved_H =
      ((attr.ih - (attr.dh * (attr.kh - 1) + 1) + attr.pht + attr.phb) /
           attr.sh +
       1);
  const int conved_W =
      ((attr.iw - (attr.dw * (attr.kw - 1) + 1) + attr.pwl + attr.pwr) /
           attr.sw +
       1);

  int buffer_size = attr.n * attr.ic * attr.kh * attr.kw * conved_H * conved_W;
  float *buffer = new float[buffer_size];

  deform_gather_attr_t gattr = {0};
  gattr.oh = conved_H;
  gattr.ow = conved_W;
  parseGatherParam(attr, gattr);
  processDeformGather(p, gattr, buffer, true);

  float *output = p.outputs[0];
  float *weight = p.inputs[1];
  float *bias = p.inputs[4];

  conv_attr_t cattr = {0};
  cattr.ih = conved_H;
  cattr.iw = conved_W;
  parseConvParam(attr, cattr);

  auto conv = new Conv();
  conv->setup(buffer, weight, bias, output, cattr);
  conv->run();
}

} // namespace tpu_mlir
