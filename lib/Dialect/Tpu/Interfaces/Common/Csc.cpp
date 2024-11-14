//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"
static inline int align_up(int x, int n) {
  if (n == 0 || n == 1) {
    return x;
  }
  return ((x + n - 1) / n) * n;
}

static inline float UINT8(float data) {
  return static_cast<float>(to_uint8(data, ROUNDING_HALF_TO_EVEN));
}

enum YuvType { YUV_UNKNOWN = 0, YUV420_PLANAR = 1, YUV_NV12 = 2, YUV_NV21 = 3 };

struct AlignParam {
  YuvType pixel_type;
  int y_align;
  int w_align;
  int channel_align;
};

void yuv_csc(float *input, float *output, int n, int c, int h, int w,
             std::vector<int> &order, int quant_type, AlignParam align_param) {
  YuvType pixel_type = align_param.pixel_type;
  int y_align = align_param.y_align;
  int w_align = align_param.w_align;
  int channel_align = align_param.channel_align;
  int y_w_aligned = align_up(w, y_align);
  int uv_w_aligned = 0;
  int y_offset = 0;
  int u_offset = 0;
  int v_offset = 0;
  if (pixel_type == YUV420_PLANAR) {
    uv_w_aligned = align_up(w / 2, w_align);
    u_offset = align_up(h * y_w_aligned, channel_align);
    v_offset = align_up(u_offset + h / 2 * uv_w_aligned, channel_align);
  } else {
    uv_w_aligned = align_up(w, w_align);
    u_offset = align_up(h * y_w_aligned, channel_align);
    v_offset = u_offset;
  }
  int n_stride = align_up(v_offset + h / 2 * uv_w_aligned, channel_align);
  for (int idx_n = 0; idx_n < n; idx_n++) {
    for (int idx_h = 0; idx_h < h; idx_h++) {
      for (int idx_w = 0; idx_w < w; idx_w++) {
        int y_idx = y_offset + idx_n * n_stride + idx_h * y_w_aligned + idx_w;
        int u_idx = 0;
        int v_idx = 0;
        if (pixel_type == YUV420_PLANAR) { // i420
          u_idx = u_offset + idx_n * n_stride + idx_h / 2 * uv_w_aligned +
                  idx_w / 2;
          v_idx = v_offset + idx_n * n_stride + idx_h / 2 * uv_w_aligned +
                  idx_w / 2;
        } else if (pixel_type == YUV_NV12) { // nv12
          u_idx = u_offset + idx_n * n_stride + idx_h / 2 * uv_w_aligned +
                  idx_w / 2 * 2;
          v_idx = v_offset + idx_n * n_stride + idx_h / 2 * uv_w_aligned +
                  idx_w / 2 * 2 + 1;
        } else { // nv21
          u_idx = u_offset + idx_n * n_stride + idx_h / 2 * uv_w_aligned +
                  idx_w / 2 * 2 + 1;
          v_idx = v_offset + idx_n * n_stride + idx_h / 2 * uv_w_aligned +
                  idx_w / 2 * 2;
        }
        float y = input[y_idx];
        float u = input[u_idx];
        float v = input[v_idx];
        float r, g, b;
        if (quant_type == 0) {
          y = 1.164 * (y - 16.0f);
          u -= 128;
          v -= 128;
          // float:
          r = y + 1.596 * v;
          g = y - 0.813 * v - 0.391 * u;
          b = y + 2.018 * u;
        } else {
          // u8 or bf16

          y = (float)(uint8_t)y;
          u = (float)(uint8_t)u;
          v = (float)(uint8_t)v;

          y = BF16(BF16(1.164) * (y - 16.0f));
          u -= 128.0f;
          v -= 128.0f;
          r = BF16(y + BF16(1.596f) * v);
          g = BF16(BF16(y + BF16(-0.813f) * v) + BF16(-0.391f) * u);
          b = BF16(y + BF16(2.018f) * u);
        }
        r = UINT8(r);
        g = UINT8(g);
        b = UINT8(b);

        float color[3] = {b, g, r};
        int c0_idx = idx_n * 3 * h * w + idx_h * w + idx_w;
        int c1_idx = c0_idx + h * w;
        int c2_idx = c1_idx + h * w;
        output[c0_idx] = color[order[0]];
        output[c1_idx] = color[order[1]];
        output[c2_idx] = color[order[2]];
      }
    }
  }
}

inline int crop_offset(const std::vector<int> &indices, const int64_t *shape) {
  int offset = 0;
  for (int i = 0; i < 4; ++i) {
    offset *= shape[i];
    if ((int)indices.size() > i) {
      offset += indices[i];
    }
  }
  return offset;
}

inline int copy_offset(const int *indices, int32_t *stride) {
  int offset = 0;
  for (int i = 0; i < 4; ++i) {
    offset += indices[i] * stride[i];
  }
  return offset;
}

void crop(float *input, float *output, const int64_t *input_shape,
          long int *output_shape, int cur_dim, int *offsets, int *indices) {
  // for loop if dim is not last
  if (cur_dim + 1 < 4) {
    for (int i = 0; i < output_shape[cur_dim]; ++i) {
      indices[cur_dim] = i;
      crop(input, output, input_shape, output_shape, cur_dim + 1, offsets,
           indices);
    }
  } else {
    std::vector<int> ind_red(cur_dim, 0);
    std::vector<int> ind_off(cur_dim + 1, 0);

    for (int j = 0; j < cur_dim; ++j) {
      ind_red[j] = indices[j];

      ind_off[j] = indices[j] + offsets[j];
    }
    ind_off[cur_dim] = offsets[cur_dim];

    std::memcpy(output + crop_offset(ind_red, output_shape),
                input + crop_offset(ind_off, input_shape),
                sizeof(float) * output_shape[cur_dim]);
  }
}

void stride_copy(float *input, float *output, const int64_t *shape,
                 int32_t *input_stride, int32_t *output_stride, int cur_dim,
                 int *indices) {
  if (cur_dim + 1 < 4) {
    for (int i = 0; i < shape[cur_dim]; ++i) {
      indices[cur_dim] = i;
      stride_copy(input, output, shape, input_stride, output_stride,
                  cur_dim + 1, indices);
    }
  } else {
    indices[cur_dim] = 0;
    std::memcpy(output + copy_offset(indices, output_stride),
                input + copy_offset(indices, input_stride),
                sizeof(float) * shape[cur_dim]);
  }
}

LogicalResult tpu::CscOp::init(InferenceParameter &p) { return success(); }
void tpu::CscOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::CscOp::inference(InferenceParameter &p) {
  float *input_data = p.inputs[0];
  float *output_data = p.outputs[0];
  auto input_shape = module::getShape(this->getInput());
  auto output_shape = module::getShape(this->getOutput());
  if (output_shape.size() < 4 || input_shape.size() < 4) {
    dump();
    llvm_unreachable("wrong shape size");
  }
  int y_align = this->getYAlign();
  int w_align = this->getWAlign();
  int channel_align = this->getChannelAlign();
  bool aligned = this->getAligned();
  std::string pixel_format = this->getPixelFormat().str();

  int on = output_shape[0];
  int oc = output_shape[1];
  int oh = output_shape[2];
  int ow = output_shape[3];
  int ic = input_shape[1];
  int ih = input_shape[2];

  AlignParam param;
  if (pixel_format == "YUV420_PLANAR") {
    std::vector<int> orders{0, 1, 2};
    // here DataType is UI8/BF16
    param = {YUV420_PLANAR, y_align, w_align, channel_align};
    yuv_csc(input_data, output_data, on, oc, oh, ow, orders, 1, param);
  } else if (pixel_format == "YUV_NV12") {
    std::vector<int> orders{0, 1, 2};
    param = {YUV_NV12, y_align, w_align, channel_align};
    yuv_csc(input_data, output_data, on, oc, oh, ow, orders, 1, param);
  } else if (pixel_format == "YUV_NV21") {
    std::vector<int> orders{0, 1, 2};
    param = {YUV_NV21, y_align, w_align, channel_align};
    yuv_csc(input_data, output_data, on, oc, oh, ow, orders, 1, param);
  } else if (aligned) {
    if (pixel_format == "RGB_PLANAR" || pixel_format == "BGR_PLANAR" ||
        pixel_format == "RGBA_PLANAR") {
      // do stride copy to make data unaligned
      std::vector<int> indices(4, 0);
      int iw = ::align_up(output_shape[3], w_align);
      int ic_stride = ::align_up(iw * output_shape[2], channel_align);
      int in_stride = ic_stride * ic;
      std::vector<int32_t> input_stride = {in_stride, ic_stride, iw, 1};
      std::vector<int32_t> output_stride = {ow * oh * oc, ow * oh, ow, 1};
      stride_copy(input_data, output_data, output_shape.data(),
                  input_stride.data(), output_stride.data(), 0, indices.data());
    } else {
      // do crop to make data unaligned
      std::vector<int64_t> crop_shape(input_shape.begin(), input_shape.end());
      crop_shape[3] = (int)(oc * oh * ow / (ic * ih));
      std::vector<int> crop_offset{0, 0, 0, 0};
      std::vector<int> indices(4, 0);
      crop(input_data, output_data, input_shape.data(), crop_shape.data(), 0,
           crop_offset.data(), indices.data());
    }
  } else {
    memcpy(output_data, input_data, on * oc * oh * ow);
  }
  return success();
}

bool tpu::CscOp::support_multi_core() { return false; }
