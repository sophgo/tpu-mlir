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
#ifndef CHECK_CUDNN
#define CHECK_CUDNN(status)                                                    \
  if (status != CUDNN_STATUS_SUCCESS) {                                        \
    std::cerr << "[" << __FILE__ << ":" << __LINE__                            \
              << "] CUDNN failure: " << cudnnGetErrorString(status)            \
              << std::endl;                                                    \
    exit(EXIT_FAILURE);                                                        \
  }
#endif

#ifndef CHECK_CUDA
#define CHECK_CUDA(status)                                                     \
  if (status != cudaSuccess) {                                                 \
    std::cerr << "[" << __FILE__ << ":" << __LINE__                            \
              << "] CUDA failure: " << cudaGetErrorString(status)              \
              << std::endl;                                                    \
    exit(EXIT_FAILURE);                                                        \
  }
#endif

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
  void cudaCastOp(tpu::CastOp op);
  void cudaConcatOp(tpu::ConcatOp op);
  void cudaDeconvOp(tpu::DeconvOp op);
  void cudaGatherOp(tpu::GatherOp op);
  void cudaGenericCpuOp(tpu::GenericCpuOp op);
  void cudaLutOp(tpu::LutOp op);
  void cudaMatMulOp(tpu::MatMulOp op);
  void cudaMulOp(tpu::MulOp op);
  void cudaMulShiftOp(tpu::MulShiftOp op);
  void cudaReshapeOp(tpu::ReshapeOp op);
  void cudaRequantIntAxisOp(tpu::RequantIntAxisOp op);
  void cudaPool2DOp(tpu::Pool2DOp op);
  void cudaPReluOp(top::PReluOp op);
  void cudaPackOp(top::PackOp op);
  void cudaPadOp(top::PadOp op);
  void cudaPadOp(tpu::PadOp op);
  void cudaPowOp(top::PowOp op);
  void cudaPow2Op(top::Pow2Op op);
  void cudaPow3Op(top::Pow3Op op);
  void cudaPReluOp(tpu::PReluOp op);
  void cudaPermuteOp(tpu::PermuteOp op);
  void cudaQuantizeLinearOp(top::QuantizeLinearOp op);
  void cudaSliceOp(tpu::SliceOp op);
  void cudaSoftmaxOp(tpu::SoftmaxOp op);
  void cudaSqueezeOp(tpu::SqueezeOp op);
  void cudaTileOp(top::TileOp op);
  void cudaTileOp(tpu::TileOp op);
  void cudaUnpackOp(top::UnpackOp op);
  void cudaUpsampleOp(tpu::UpsampleOp op);
  void cudaWhereOp(top::WhereOp op);
  void cudaWhereOp(tpu::WhereOp op);
  void cudaUnsqueezeOp(tpu::UnsqueezeOp op);
  void cudaActiveOp(tpu::ActiveOp op);
  void cudaSubOp(tpu::SubOp op);
  void cudaMulConstOp(tpu::MulConstOp op);
  void cudaLayerNormOp(tpu::LayerNormOp op);
  void cudaDepth2SpaceOp(tpu::Depth2SpaceOp op);
  void cudaReduceOp(tpu::ReduceOp op);
  void cudaSwapDimInnerOp(tpu::SwapDimInnerOp op);
  void cudaClipOp(tpu::ClipOp op);
  void cudaAddConstOp(tpu::AddConstOp op);
  void cudaDivOp(tpu::DivOp op);
  void cudaMaskedFillOp(tpu::MaskedFillOp op);
  void cudaMaskRCNNBboxPoolerOp(tpu::MaskRCNNBboxPoolerOp op);
  void cudaMaskRCNNGetBboxBOp(tpu::MaskRCNNGetBboxBOp op);
  void cudaMaskRCNNMaskPoolerOp(tpu::MaskRCNNMaskPoolerOp op);
  void cudaMatchTemplateOp(tpu::MatchTemplateOp op);
  void cudaMaxOp(tpu::MaxOp op);
  void cudaMaxConstOp(tpu::MaxConstOp op);
  void cudaMaxPoolWithMaskOp(tpu::MaxPoolWithMaskOp op);
  void cudaMaxPoolingIndicesBwdOp(tpu::MaxPoolingIndicesBwdOp op);
  void cudaMaxUnpoolOp(tpu::MaxUnpoolOp op);
  void cudaMeanRstdOp(tpu::MeanRstdOp op);
  void cudaMeanStdScaleOp(tpu::MeanStdScaleOp op);
  void cudaMinOp(tpu::MinOp op);
  void cudaMinConstOp(tpu::MinConstOp op);
  void cudaScatterElementsOp(tpu::ScatterElementsOp op);
  void cudaScatterNDOp(tpu::ScatterNDOp op);
  void cudaSelectiveScanOp(tpu::SelectiveScanOp op);
  void cudaShapeOp(tpu::ShapeOp op);
  void cudaShapeSliceOp(tpu::ShapeSliceOp op);
  void cudaShuffleChannelOp(tpu::ShuffleChannelOp op);
  void cudaSortOp(tpu::SortOp op);
  void cudaStridedSliceOp(tpu::StridedSliceOp op);
  void cudaSwapChannelOp(tpu::SwapChannelOp op);
  void cudaSubConstOp(tpu::SubConstOp op);
  void cudaRequantFpOp(tpu::RequantFpOp op);

  void cudaAddOp(top::AddOp op);
  void cudaConvOp(top::ConvOp op);
  void cudaScaleOp(top::ScaleOp op);
  void cudaMaxPoolOp(top::MaxPoolOp op);
  void cudaAvgPoolOp(top::AvgPoolOp op);
  void cudaMatMulOp(top::MatMulOp op);
  void cudaReshapeOp(top::ReshapeOp op);
  void cudaSiLUOp(top::SiLUOp op);
  void cudaConcatOp(top::ConcatOp op);
  void cudaUpsampleOp(top::UpsampleOp op);
  void cudaPermuteOp(top::PermuteOp op);
  void cudaSliceOp(top::SliceOp op);
  void cudaSoftmaxOp(top::SoftmaxOp op);
  void cudaSubOp(top::SubOp op);
  void cudaMulConstOp(top::MulConstOp op);
  void cudaMulOp(top::MulOp op);
  void cudaSigmoidOp(top::SigmoidOp op);
  void cudaTanhOp(top::TanhOp op);
  void cudaTopKOp(top::TopKOp op);
  void cudaTopKOp(tpu::TopKOp op);
  void cudaTriluOp(top::TriluOp op);
  void cudaTriluOp(tpu::TriluOp op);
  void cudaLayerNormOp(top::LayerNormOp op);
  void cudaSqueezeOp(top::SqueezeOp op);
  void cudaGELUOp(top::GELUOp op);
  void cudaDepth2SpaceOp(top::Depth2SpaceOp op);
  void cudaReduceOp(top::ReduceOp op);
  void cudaSwapDimInnerOp(top::SwapDimInnerOp op);
  void cudaUnsqueezeOp(top::UnsqueezeOp op);
  void cudaClipOp(top::ClipOp op);
  void cudaAddConstOp(top::AddConstOp op);
  void cudaDivOp(top::DivOp op);
  void cudaDivConstOp(top::DivConstOp op);
  void cudaEinsumOp(top::EinsumOp op);
  void cudaEluOp(top::EluOp op);
  void cudaErfOp(top::ErfOp op);
  void cudaExpOp(top::ExpOp op);
  void cudaMaskedFillOp(top::MaskedFillOp op);
  void cudaMaskRCNNBboxPoolerOp(top::MaskRCNNBboxPoolerOp op);
  void cudaMaskRCNNGetBboxBOp(top::MaskRCNNGetBboxBOp op);
  void cudaMaskRCNNMaskPoolerOp(top::MaskRCNNMaskPoolerOp op);
  void cudaMatchTemplateOp(top::MatchTemplateOp op);
  void cudaMaxOp(top::MaxOp op);
  void cudaMaxConstOp(top::MaxConstOp op);
  void cudaMaxPoolWithMaskOp(top::MaxPoolWithMaskOp op);
  void cudaMaxPoolingIndicesBwdOp(top::MaxPoolingIndicesBwdOp op);
  void cudaMaxUnpoolOp(top::MaxUnpoolOp op);
  void cudaMeanRstdOp(top::MeanRstdOp op);
  void cudaMeanStdScaleOp(top::MeanStdScaleOp op);
  void cudaMeshGridOp(top::MeshGridOp op);
  void cudaMinOp(top::MinOp op);
  void cudaMinConstOp(top::MinConstOp op);
  void cudaMishOp(top::MishOp op);
  void cudaModOp(top::ModOp op);
  void cudaScaleLutOp(top::ScaleLutOp op);
  void cudaScaleLutOp(tpu::ScaleLutOp op);
  void cudaScatterElementsOp(top::ScatterElementsOp op);
  void cudaScatterNDOp(top::ScatterNDOp op);
  void cudaSelectiveScanOp(top::SelectiveScanOp op);
  void cudaShapeOp(top::ShapeOp op);
  void cudaShuffleChannelOp(top::ShuffleChannelOp op);
  void cudaSignOp(top::SignOp op);
  void cudaSinOp(top::SinOp op);
  void cudaSinhOp(top::SinhOp op);
  void cudaTanOp(top::TanOp op);
  void cudaSliceAxisOp(top::SliceAxisOp op);
  void cudaSoftplusOp(top::SoftplusOp op);
  void cudaSoftsignOp(top::SoftsignOp op);
  void cudaSortOp(top::SortOp op);
  void cudaSplitOp(top::SplitOp op);
  void cudaSqrtOp(top::SqrtOp op);
  void cudaStridedSliceOp(top::StridedSliceOp op);
  void cudaSwapChannelOp(top::SwapChannelOp op);
  void cudaSwishOp(top::SwishOp op);
  void cudaSubConstOp(top::SubConstOp op);
  void cudaGatherOp(top::GatherOp op);
  void cudaRequantFpOp(top::RequantFpOp op);

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
