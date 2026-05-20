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
  void cudaAddOp(tpu::AddOp op);
  void cudaConv2DOp(tpu::Conv2DOp op);
  void cudaConvBwdWeightOp(tpu::ConvBwdWeightOp op);
  void cudaCastOp(tpu::CastOp op);
  void cudaConcatOp(tpu::ConcatOp op);
  void cudaDeconvOp(tpu::DeconvOp op);
  void cudaDequantIntOp(tpu::DequantIntOp op);
  void cudaDevice2HostOp(tpu::Device2HostOp op);
  void cudaDepackRawOp(tpu::DepackRawOp op);
  void cudaDivOp(tpu::DivOp op);
  void cudaDtypeCastOp(tpu::DtypeCastOp op);
  void cudaGatherOp(tpu::GatherOp op);
  void cudaGenericCpuOp(tpu::GenericCpuOp op);
  void cudaHost2DeviceOp(tpu::Host2DeviceOp op);
  void cudaLutOp(tpu::LutOp op);
  void cudaMatMulOp(tpu::MatMulOp op);
  void cudaMulOp(tpu::MulOp op);
  void cudaMulShiftOp(tpu::MulShiftOp op);
  void cudaNmsOp(tpu::NmsOp op);
  void cudaNonZeroOp(tpu::NonZeroOp op);
  void cudaReshapeOp(tpu::ReshapeOp op);
  void cudaRequantIntAxisOp(tpu::RequantIntAxisOp op);
  void cudaPool2DOp(tpu::Pool2DOp op);
  void cudaPReluOp(tpu::PReluOp op);
  void cudaPermuteOp(tpu::PermuteOp op);
  void cudaSliceOp(tpu::SliceOp op);
  void cudaSoftmaxOp(tpu::SoftmaxOp op);
  void cudaShapeOp(tpu::ShapeOp op);
  void cudaShapeSliceOp(tpu::ShapeSliceOp op);
  void cudaShapeCastOp(tpu::ShapeCastOp op);
  void cudaSqueezeOp(tpu::SqueezeOp op);
  void cudaTileOp(tpu::TileOp op);
  void cudaUpsampleOp(tpu::UpsampleOp op);
  void cudaUnsqueezeOp(tpu::UnsqueezeOp op);
  void cudaArgOp(tpu::ArgOp op);
  void cudaCopyOp(tpu::CopyOp op);
  void cudaCorrelationOp(tpu::CorrelationOp op);
  void cudaActiveOp(tpu::ActiveOp op);
  void cudaSubOp(tpu::SubOp op);
  void cudaCompareOp(tpu::CompareOp op);
  void cudaCompareConstOp(tpu::CompareConstOp op);
  void cudaAddConstOp(tpu::AddConstOp op);
  void cudaMulConstOp(tpu::MulConstOp op);
  void cudaLayerNormOp(tpu::LayerNormOp op);
  void cudaDepth2SpaceOp(tpu::Depth2SpaceOp op);
  void cudaRangeOp(tpu::RangeOp op);
  void cudaReciprocalOp(tpu::ReciprocalOp op);
  void cudaReluOp(tpu::ReluOp op);
  void cudaRMSNormOp(tpu::RMSNormOp op);
  void cudaReduceOp(tpu::ReduceOp op);
  void cudaRoiAlignOp(tpu::RoiAlignOp op);
  void cudaRoiExtractorOp(tpu::RoiExtractorOp op);
  void cudaSwapDimInnerOp(tpu::SwapDimInnerOp op);
  void cudaSubConstOp(tpu::SubConstOp op);
  void cudaRequantFpOp(tpu::RequantFpOp op);
  void cudaReverseOp(tpu::ReverseOp op);
  void cudaClipOp(tpu::ClipOp op);
  void cudaConstantFillOp(tpu::ConstantFillOp op);
  void cudaCumSumOp(tpu::CumSumOp op);
  void cudaBatchNormOp(tpu::BatchNormTrainOp op);
  void cudaBatchNormBwdOp(tpu::BatchNormBwdOp op);


  void cudaA16MatMulOp(top::A16MatMulOp op);
  void cudaAttentionOp(top::AttentionOp op);
  void cudaFAttentionOp(top::FAttentionOp op);
  void cudaBinaryShiftOp(top::BinaryShiftOp op);
  void cudaBinaryConstShiftOp(top::BinaryConstShiftOp op);
  void cudaCeilOp(top::CeilOp op);
  void cudaAddOp(top::AddOp op);
  void cudaConvOp(top::ConvOp op);
  void cudaConvBwdWeightOp(top::ConvBwdWeightOp op);
  void cudaScaleOp(top::ScaleOp op);
  void cudaMaxPoolOp(top::MaxPoolOp op);
  void cudaAvgPoolOp(top::AvgPoolOp op);
  void cudaAdaptiveAvgPoolOp(top::AdaptiveAvgPoolOp op);
  void cudaMatMulOp(top::MatMulOp op);
  void cudaReshapeOp(top::ReshapeOp op);
  void cudaSiLUOp(top::SiLUOp op);
  void cudaCosOp(top::CosOp op);
  void cudaCoshOp(top::CoshOp op);
  void cudaCopyOp(top::CopyOp op);
  void cudaCorrelationOp(top::CorrelationOp op);
  void cudaConcatOp(top::ConcatOp op);
  void cudaUpsampleOp(top::UpsampleOp op);
  void cudaPermuteOp(top::PermuteOp op);
  void cudaSliceOp(top::SliceOp op);
  void cudaSoftmaxOp(top::SoftmaxOp op);
  void cudaSubOp(top::SubOp op);
  void cudaCompareOp(top::CompareOp op);
  void cudaCompareConstOp(top::CompareConstOp op);
  void cudaAddConstOp(top::AddConstOp op);
  void cudaMulConstOp(top::MulConstOp op);
  void cudaMulOp(top::MulOp op);
  void cudaNmsOp(top::NmsOp op);
  void cudaSigmoidOp(top::SigmoidOp op);
  void cudaLayerNormOp(top::LayerNormOp op);
  void cudaSqueezeOp(top::SqueezeOp op);
  void cudaAbsOp(top::AbsOp op);
  void cudaArgOp(top::ArgOp op);
  void cudaArccosOp(top::ArccosOp op);
  void cudaArctanhOp(top::ArctanhOp op);
  void cudaGELUOp(top::GELUOp op);
  void cudaHardSigmoidOp(top::HardSigmoidOp op);
  void cudaHardSwishOp(top::HardSwishOp op);
  void cudaEluOp(top::EluOp op);
  void cudaErfOp(top::ErfOp op);
  void cudaExpOp(top::ExpOp op);
  void cudaFloorOp(top::FloorOp op);
  void cudaGatherElementsOp(top::GatherElementsOp op);
  void cudaGatherElementsOp(tpu::GatherElementsOp op);
  void cudaGatherNDOp(top::GatherNDOp op);
  void cudaGatherNDOp(tpu::GatherNDOp op);
  void cudaGridSamplerOp(top::GridSamplerOp op);
  void cudaGridSamplerOp(tpu::GridSamplerOp op);
  void cudaGroupNormOp(top::GroupNormOp op);
  void cudaGroupNormOp(tpu::GroupNormOp op);
  void cudaGroupNormTrainOp(top::GroupNormTrainOp op);
  void cudaGroupNormTrainOp(tpu::GroupNormTrainOp op);
  void cudaInstanceNormOp(top::InstanceNormOp op);
  void cudaInstanceNormOp(tpu::InstanceNormOp op);
  void cudaIndexPutOp(top::IndexPutOp op);
  void cudaIndexPutOp(tpu::IndexPutOp op);
  void cudaInterpOp(top::InterpOp op);
  void cudaInterpOp(tpu::InterpOp op);
  void cudaLRNOp(top::LRNOp op);
  void cudaLRNOp(tpu::LRNOp op);
  void cudaLSTMOp(top::LSTMOp op);
  void cudaLSTMOp(tpu::LSTMOp op);
  void cudaLeakyReluOp(top::LeakyReluOp op);
  void cudaLayerNormTrainOp(top::LayerNormTrainOp op);
  void cudaLayerNormTrainOp(tpu::LayerNormTrainOp op);
  void cudaLeakyReluOp(tpu::LeakyReluOp op);
  void cudaLogOp(top::LogOp op);
  void cudaLogBOp(top::LogBOp op);
  void cudaLogicalAndOp(top::LogicalAndOp op);
  void cudaLogicalAndOp(tpu::LogicalAndOp op);
  void cudaGRUOp(top::GRUOp op);
  void cudaGRUOp(tpu::GRUOp op);
  void cudaExpandOp(top::ExpandOp op);
  void cudaEmbDenseBwdOp(top::EmbDenseBwdOp op);
  void cudaEmbDenseBwdOp(tpu::EmbDenseBwdOp op);
  void cudaDequantIntOp(top::DequantIntOp op);
  void cudaDequantizeLinearOp(top::DequantizeLinearOp op);
  void cudaDepackRawOp(top::DepackRawOp op);
  void cudaDepth2SpaceOp(top::Depth2SpaceOp op);
  void cudaDivOp(top::DivOp op);
  void cudaDivConstOp(top::DivConstOp op);
  void cudaDtypeCastOp(top::DtypeCastOp op);
  void cudaRangeOp(top::RangeOp op);
  void cudaReciprocalOp(top::ReciprocalOp op);
  void cudaReluOp(top::ReluOp op);
  void cudaRMSNormOp(top::RMSNormOp op);
  void cudaRoundOp(top::RoundOp op);
  void cudaRsqrtOp(top::RsqrtOp op);
  void cudaReduceOp(top::ReduceOp op);
  void cudaRoiAlignOp(top::RoiAlignOp op);
  void cudaRoiExtractorOp(top::RoiExtractorOp op);
  void cudaSwapDimInnerOp(top::SwapDimInnerOp op);
  void cudaUnsqueezeOp(top::UnsqueezeOp op);
  void cudaSubConstOp(top::SubConstOp op);
  void cudaGatherOp(top::GatherOp op);
  void cudaRequantFpOp(top::RequantFpOp op);
  void cudaReverseOp(top::ReverseOp op);
  void cudaBatchNormBwdOp(top::BatchNormBwdOp op);
  void cudaBatchNormOp(top::BatchNormOp op);
  void cudaClipOp(top::ClipOp op);
  void cudaConstantFillOp(top::ConstantFillOp op);
  void cudaCumSumOp(top::CumSumOp op);

  void cudaConv3DOp(tpu::Conv3DOp op);
  void cudaConv3DOp(top::ConvOp op);

private:
  cuda_ptr cuda_malloc(size_t bytes);
  void cuda_malloc(std::map<std::string, cuda_ptr> &map, mlir::Value v);
  void cuda_to_host(const std::string &name);

public:
  py::list input_names;
  py::list output_names;

private:
  std::unique_ptr<mlir::MLIRContext> context_;
  OwningOpRef<ModuleOp> module_;
  cudnnHandle_t cudnn_;
  bool dump_all_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::map<std::string, mlir::Value> value_map_;
  std::map<std::string, cuda_ptr> input_map_;
  std::map<std::string, cuda_ptr> weight_map_;
  std::map<std::string, cuda_ptr> activation_map_;
  std::map<std::string, std::shared_ptr<std::vector<float>>> buffer_map_;
};
