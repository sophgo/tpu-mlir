//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaMaskRCNNMaskPoolerOp(top::MaskRCNNMaskPoolerOp op) {
  auto feat0 = getCudaData(op.getX_0());
  auto feat1 = getCudaData(op.getX_1());
  auto feat2 = getCudaData(op.getX_2());
  auto feat3 = getCudaData(op.getX_3());
  auto bboxes = getCudaData(op.getDetBboxesMultiBatch());
  auto scale_factor_data = getCudaData(op.getScaleFactor());
  auto output = getCudaData(op.getResultRes());

  auto shape0 = module::getShape(op.getX_0());
  auto shape1 = module::getShape(op.getX_1());
  auto shape2 = module::getShape(op.getX_2());
  auto shape3 = module::getShape(op.getX_3());

  int batch_size = shape0[0];
  int C = op.getCHANNEL_ROI();
  int total_dets = op.getROI_SLICE() * batch_size;
  int roi_len = op.getROI_LEN();
  int PH = op.getROI_PH();
  int PW = op.getROI_PW();
  int num_levels = op.getROI_NUM_LEVELS();

  // Read scale_factor from GPU
  float scale_factor = 1.0f;
  CHECK_CUDA(cudaMemcpy(&scale_factor, scale_factor_data, sizeof(float),
                        cudaMemcpyDeviceToHost));

  cuda::maskRCNNMaskPoolerF32(feat0, feat1, feat2, feat3, bboxes, output,
                              shape0[2], shape0[3], shape1[2], shape1[3],
                              shape2[2], shape2[3], shape3[2], shape3[3],
                              batch_size, C, total_dets, roi_len, PH, PW,
                              num_levels, scale_factor);
}

void py_cuda::cudaMaskRCNNMaskPoolerOp(tpu::MaskRCNNMaskPoolerOp op) {
  auto feat0 = getCudaData(op.getX_0());
  auto feat1 = getCudaData(op.getX_1());
  auto feat2 = getCudaData(op.getX_2());
  auto feat3 = getCudaData(op.getX_3());
  auto bboxes = getCudaData(op.getDetBboxesMultiBatch());
  auto scale_factor_data = getCudaData(op.getScaleFactor());
  auto output = getCudaData(op.getResultRes());

  auto shape0 = module::getShape(op.getX_0());
  auto shape1 = module::getShape(op.getX_1());
  auto shape2 = module::getShape(op.getX_2());
  auto shape3 = module::getShape(op.getX_3());

  int batch_size = shape0[0];
  int C = op.getCHANNEL_ROI();
  int total_dets = op.getROI_SLICE() * batch_size;
  int roi_len = op.getROI_LEN();
  int PH = op.getROI_PH();
  int PW = op.getROI_PW();
  int num_levels = op.getROI_NUM_LEVELS();

  float scale_factor = 1.0f;
  CHECK_CUDA(cudaMemcpy(&scale_factor, scale_factor_data, sizeof(float),
                        cudaMemcpyDeviceToHost));

  auto stype = module::getStorageType(op.getResultRes());
  if (stype.isF32()) {
    cuda::maskRCNNMaskPoolerF32(feat0, feat1, feat2, feat3, bboxes, output,
                                shape0[2], shape0[3], shape1[2], shape1[3],
                                shape2[2], shape2[3], shape3[2], shape3[3],
                                batch_size, C, total_dets, roi_len, PH, PW,
                                num_levels, scale_factor);
  } else {
    auto num = module::getNumElements(op.getResultRes());
    auto output_f32 = cuda_malloc(num * sizeof(float));

    cuda::maskRCNNMaskPoolerF32(feat0, feat1, feat2, feat3, bboxes,
                                output_f32.get(), shape0[2], shape0[3],
                                shape1[2], shape1[3], shape2[2], shape2[3],
                                shape3[2], shape3[3], batch_size, C, total_dets,
                                roi_len, PH, PW, num_levels, scale_factor);

    cuda::convertType(output_f32.get(), output, num, cuda::DT_F32,
                      getCudaType(op.getResultRes()));
  }
}
