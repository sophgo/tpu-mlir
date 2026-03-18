//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "cuda/cuda_helper.h"
#include "pymlir.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Float8.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <cuda_runtime.h>
#include <cudnn.h>

// Error checking macros
#define CHECK_CUDNN(status)                                                    \
  if (status != CUDNN_STATUS_SUCCESS) {                                        \
    std::cerr << "[" << __FILE__ << ":" << __LINE__                            \
              << "] CUDNN failure: " << cudnnGetErrorString(status)            \
              << std::endl;                                                    \
    exit(EXIT_FAILURE);                                                        \
  }

#define CHECK_CUDA(status)                                                     \
  if (status != cudaSuccess) {                                                 \
    std::cerr << "[" << __FILE__ << ":" << __LINE__                            \
              << "] CUDA failure: " << cudaGetErrorString(status)              \
              << std::endl;                                                    \
    exit(EXIT_FAILURE);                                                        \
  }

struct cudaDeleter {
  void operator()(void *ptr) {
    if (ptr != nullptr) {
      CHECK_CUDA(cudaFree(ptr));
    }
  }
};

typedef std::unique_ptr<void, cudaDeleter> cuda_ptr;

class py_cuda {
public:
  py_cuda();
  ~py_cuda();
  void load(std::string filename);

  // only can set input data
  void set_tensor(
      std::string name,
      py::array_t<float, py::array::c_style | py::array::forcecast> data);
  void invoke(bool dump_all, const std::vector<std::string>& extra_outputs);
  py::array get_tensor(std::string name);
  py::dict get_all_tensor();

  bool is_cuda_support_op(Operation *op);
  std::vector<Operation *> get_weight_target_op(Operation *weight_op);

private:
  // -------------------------------------------------------------------
  // -------------- helper functions -----------------------------------
  // get data in cuda by activation_map_ and weight_map_; if not find, will
  // assert
  void *getCudaData(mlir::Value v);
  // get cudnn type from mlir type
  cuda::data_type_t getCudaType(mlir::Value v);
  // convert cuda data from one type to another type
  cuda_ptr newCudaData(void *data, size_t num, cuda::data_type_t src_type,
                       cuda::data_type_t dst_type);
  // alloc new buffer to store new type
  cuda_ptr newCudaData(mlir::Value v, cuda::data_type_t dst_type);

  // -------------------------------------------------------------------
  // -------------- op inference by cuda -------------------------------
  void cudaActiveOp(tpu::ActiveOp op);
  void cudaAddConstOp(tpu::AddConstOp op);
  void cudaAddOp(tpu::AddOp op);
  void cudaArgOp(tpu::ArgOp op);
  void cudaCastOp(tpu::CastOp op);
  void cudaConcatOp(tpu::ConcatOp op);
  void cudaConv2DOp(tpu::Conv2DOp op);
  void cudaDeconvOp(tpu::DeconvOp op);
  void cudaDepth2SpaceOp(tpu::Depth2SpaceOp op);
  void cudaDivOp(tpu::DivOp op);
  void cudaFAttentionOp(tpu::FAttentionOp op);
  void cudaGatherElementsOp(tpu::GatherElementsOp op);
  void cudaGatherOp(tpu::GatherOp op);
  void cudaGenericCpuOp(tpu::GenericCpuOp op);
  void cudaGridSamplerOp(tpu::GridSamplerOp op);
  void cudaInterpOp(tpu::InterpOp op);
  void cudaLayerNormOp(tpu::LayerNormOp op);
  void cudaLutOp(tpu::LutOp op);
  void cudaMatMulOp(tpu::MatMulOp op);
  void cudaMaxConstOp(tpu::MaxConstOp op);
  void cudaMinConstOp(tpu::MinConstOp op);
  void cudaMulConstOp(tpu::MulConstOp op);
  void cudaMulOp(tpu::MulOp op);
  void cudaMulShiftOp(tpu::MulShiftOp op);
  void cudaPadOp(tpu::PadOp op);
  void cudaPermuteOp(tpu::PermuteOp op);
  void cudaPool2DOp(tpu::Pool2DOp op);
  void cudaPReluOp(tpu::PReluOp op);
  void cudaReduceOp(tpu::ReduceOp op);
  void cudaReluOp(tpu::ReluOp op);
  void cudaRequantFpOp(tpu::RequantFpOp op);
  void cudaRequantIntAxisOp(tpu::RequantIntAxisOp op);
  void cudaReshapeOp(tpu::ReshapeOp op);
  void cudaSliceOp(tpu::SliceOp op);
  void cudaSoftmaxOp(tpu::SoftmaxOp op);
  void cudaSqueezeOp(tpu::SqueezeOp op);
  void cudaSubConstOp(tpu::SubConstOp op);
  void cudaSubOp(tpu::SubOp op);
  void cudaSwapDimInnerOp(tpu::SwapDimInnerOp op);
  void cudaTileOp(tpu::TileOp op);
  void cudaUpsampleOp(tpu::UpsampleOp op);
  void cudaUnsqueezeOp(tpu::UnsqueezeOp op);

  void cudaAddConstOp(top::AddConstOp op);
  void cudaAddOp(top::AddOp op);
  void cudaArgOp(top::ArgOp op);
  void cudaAvgPoolOp(top::AvgPoolOp op);
  void cudaCastOp(top::CastOp op);
  void cudaConcatOp(top::ConcatOp op);
  void cudaConvOp(top::ConvOp op);
  void cudaDepth2SpaceOp(top::Depth2SpaceOp op);
  void cudaDivOp(top::DivOp op);
  void cudaFloorOp(top::FloorOp op);
  void cudaGatherElementsOp(top::GatherElementsOp op);
  void cudaGatherOp(top::GatherOp op);
  void cudaGridSamplerOp(top::GridSamplerOp op);
  void cudaGELUOp(top::GELUOp op);
  void cudaInterpOp(top::InterpOp op);
  void cudaLayerNormOp(top::LayerNormOp op);
  void cudaMatMulOp(top::MatMulOp op);
  void cudaMaxConstOp(top::MaxConstOp op);
  void cudaMaxPoolOp(top::MaxPoolOp op);
  void cudaMinConstOp(top::MinConstOp op);
  void cudaMulConstOp(top::MulConstOp op);
  void cudaMulOp(top::MulOp op);
  void cudaPadOp(top::PadOp op);
  void cudaPermuteOp(top::PermuteOp op);
  void cudaReduceOp(top::ReduceOp op);
  void cudaReluOp(top::ReluOp op);
  void cudaRequantFpOp(top::RequantFpOp op);
  void cudaReshapeOp(top::ReshapeOp op);
  void cudaScaleOp(top::ScaleOp op);
  void cudaSigmoidOp(top::SigmoidOp op);
  void cudaSiLUOp(top::SiLUOp op);
  void cudaSliceOp(top::SliceOp op);
  void cudaSoftmaxOp(top::SoftmaxOp op);
  void cudaSqueezeOp(top::SqueezeOp op);
  void cudaSubConstOp(top::SubConstOp op);
  void cudaSubOp(top::SubOp op);
  void cudaSwapDimInnerOp(top::SwapDimInnerOp op);
  void cudaTileOp(top::TileOp op);
  void cudaUnsqueezeOp(top::UnsqueezeOp op);
  void cudaUpsampleOp(top::UpsampleOp op);

private:
  cuda_ptr cuda_malloc(size_t bytes);
  void cuda_malloc(std::map<std::string, cuda_ptr> &map, mlir::Value v);
  void cuda_to_host(const std::string &name, bool for_infer);
  bool is_no_mem_op(Operation *op);

  void mix_load(ModuleOp m);
  void gpu_load(ModuleOp m);
  void gpu_invoke(bool dump_all, const std::vector<std::string>& extra_outputs);
  void mix_invoke(bool dump_all, const std::vector<std::string>& extra_outputs);

public:
  py::list input_names;
  py::list output_names;

private:
  std::unique_ptr<mlir::MLIRContext> context_;
  OwningOpRef<ModuleOp> module_;
  cudnnHandle_t cudnn_;
  bool dump_all_;
  bool mix_mode_ = false;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::map<std::string, mlir::Value> value_map_;
  std::map<std::string, cuda_ptr> input_map_;
  std::map<std::string, cuda_ptr> weight_map_;
  std::map<std::string, cuda_ptr> activation_map_;
  //should remove buffer map, convert type only wen get tensor... FIXME
  std::map<std::string, std::shared_ptr<std::vector<float>>> buffer_map_; // cpu mems, only active for dump
  std::map<std::string, std::shared_ptr<std::vector<float>>> infer_map_; // cpu mems, for cpu inference, including weights and active
  std::map<std::string, std::shared_ptr<InferenceParameter>> inference_map; // cpu infer ops
};
