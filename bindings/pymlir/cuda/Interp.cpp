#include "../pycuda.h"
#include "cuda_helper.h"

static float calc_resize_scale(int in_size, int out_size, bool align_corners,
                               cuda::interp_platform_t platform_sp) {
  int _in_size = in_size, _out_size = out_size;
  switch (platform_sp) {
  case cuda::TENSORFLOW_NEAREST:
  case cuda::TENSORFLOW_SUPPORT:
  case cuda::PYTORCH_SUPPORT:
  case cuda::ONNX_NEAREST: {
    if (!align_corners)
      break;
  }
  case cuda::CAFFE_NEAREST:
  case cuda::CAFFE_SUPPORT: {
    if (out_size <= 1)
      return 0.0f;
    --_in_size;
    --_out_size;
  } break;
  default:;
  }

  return _in_size / (float)_out_size;
}

void py_cuda::cudaInterpOp(tpu::InterpOp op) {
    auto mode = op.getMode();
    int coord = 0;
    auto coord_mode = op.getCoordMode();
    bool align_corners = (coord_mode == tpu::ResizeCoordMode::align_corners);
    bool half_pixel = (coord_mode == tpu::ResizeCoordMode::half_pixel);
    auto platform = module::getPlatform();
    cuda::interp_platform_t platform_sp;
    if (coord_mode == tpu::ResizeCoordMode::half_pixel)
      coord = 0;
    else if (coord_mode == tpu::ResizeCoordMode::pytorch_half_pixel)
      coord = 1;
    else if (coord_mode == tpu::ResizeCoordMode::align_corners)
      coord = 2;
    else if (coord_mode == tpu::ResizeCoordMode::asymmetric) {
      coord = 3;
    } else {
      llvm_unreachable("Unsupport coord mode.");
    }
    if (mode == tpu::ResizeMode::nearest) {
        switch (platform) {
        case module::Platform::ONNX:
            platform_sp = cuda::ONNX_NEAREST;
            break;
        case module::Platform::CAFFE:
            platform_sp = cuda::CAFFE_NEAREST;
            break;
        case module::Platform::TORCH:
            platform_sp = cuda::PYTORCH_NEAREST;
            break;
        case module::Platform::TFLITE:
            platform_sp = cuda::TENSORFLOW_NEAREST;
            break;
        default:
            platform_sp = cuda::ONNX_NEAREST;
            break;
        }
        align_corners = true;
        half_pixel = false;
        if (coord == 3) align_corners = false;
    } else if (mode == tpu::ResizeMode::linear) {
        switch (platform) {
        case module::Platform::TORCH:
            platform_sp = cuda::PYTORCH_SUPPORT;
            break;
        case module::Platform::CAFFE:
            platform_sp = cuda::CAFFE_SUPPORT;
            break;
        default:
            platform_sp = cuda::PYTORCH_SUPPORT;
            break;
        }
        align_corners = (coord == 2) ? 1 : 0;
        half_pixel = (coord == 0 || coord == 1) ? 1 : 0;
    }
    void *input = getCudaData(op.getInput());
    void *output = getCudaData(op.getOutput());
    auto input_shape = module::getShape(op.getInput());
    auto output_shape = module::getShape(op.getOutput());
    if (input_shape.size() != 4 || output_shape.size() != 4) {
      llvm_unreachable("only support 4d input and output for InterpOp");
    }
    float scale_h = calc_resize_scale(input_shape[2], output_shape[2], align_corners, platform_sp);
    float scale_w = calc_resize_scale(input_shape[3], output_shape[3], align_corners, platform_sp);
    if (module::isUniformQuantized(op.getOutput())) {
      llvm_unreachable("not support quantized InterpOp");
    } else if (module::getStorageType(op.getOutput()).isF32()) {
      cuda::interp(input, output, input_shape[0], input_shape[1], input_shape[2], input_shape[3],
                  output_shape[2], output_shape[3], scale_h, scale_w, align_corners, half_pixel,
                  platform_sp);
    } else {
      if (module::getStorageType(op.getOutput()).isFloat8E4M3FN()) {
        llvm_unreachable("Not Implemented");
      }
      auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      auto output_f32 = newCudaData(op.getOutput(), cuda::DT_F32);
      cuda::interp(input_f32.get(), output_f32.get(), input_shape[0], input_shape[1], input_shape[2], input_shape[3],
                  output_shape[2], output_shape[3], scale_h, scale_w, align_corners, half_pixel,
                  platform_sp);
      cuda::convertType(output_f32.get(), output, module::getNumElements(op.getOutput()), cuda::DT_F32, getCudaType(op.getOutput()));
      input_f32.reset();
      output_f32.reset();
    }
}

void py_cuda::cudaInterpOp(top::InterpOp op) {
    auto mode = op.getMode();
    int coord = 0;
    auto coord_mode = op.getCoordMode();
    bool align_corners = (coord_mode == "align_corners");
    bool half_pixel = (coord_mode == "half_pixel");
    auto platform = module::getPlatform();
    cuda::interp_platform_t platform_sp;
    if (coord_mode == "half_pixel")
      coord = 0;
    else if (coord_mode == "pytorch_half_pixel")
      coord = 1;
    else if (coord_mode == "align_corners")
      coord = 2;
    else if (coord_mode == "asymmetric") {
      coord = 3;
    } else {
      llvm_unreachable("Unsupport coord mode.");
    }
    if (mode == "nearest") {
        switch (platform) {
        case module::Platform::ONNX:
            platform_sp = cuda::ONNX_NEAREST;
            break;
        case module::Platform::CAFFE:
            platform_sp = cuda::CAFFE_NEAREST;
            break;
        case module::Platform::TORCH:
            platform_sp = cuda::PYTORCH_NEAREST;
            break;
        case module::Platform::TFLITE:
            platform_sp = cuda::TENSORFLOW_NEAREST;
            break;
        default:
            platform_sp = cuda::ONNX_NEAREST;
            break;
        }
        align_corners = true;
        half_pixel = false;
        if (coord == 3) align_corners = false;
    } else if (mode == "linear") {
        switch (platform) {
        case module::Platform::TORCH:
            platform_sp = cuda::PYTORCH_SUPPORT;
            break;
        case module::Platform::CAFFE:
            platform_sp = cuda::CAFFE_SUPPORT;
            break;
        default:
            platform_sp = cuda::PYTORCH_SUPPORT;
            break;
        }
        align_corners = (coord == 2) ? 1 : 0;
        half_pixel = (coord == 0 || coord == 1) ? 1 : 0;
    }
    void *input = getCudaData(op.getInput());
    void *output = getCudaData(op.getOutput());
    auto input_shape = module::getShape(op.getInput());
    auto output_shape = module::getShape(op.getOutput());
    if (input_shape.size() != 4 || output_shape.size() != 4) {
      llvm_unreachable("only support 4d input and output for InterpOp");
    }
    float scale_h = calc_resize_scale(input_shape[2], output_shape[2], align_corners, platform_sp);
    float scale_w = calc_resize_scale(input_shape[3], output_shape[3], align_corners, platform_sp);
    cuda::interp(input, output, input_shape[0], input_shape[1], input_shape[2], input_shape[3],
                output_shape[2], output_shape[3], scale_h, scale_w, align_corners, half_pixel,
                platform_sp);
}