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

void py_cuda::cudaMaskRCNNBboxPoolerOp(top::MaskRCNNBboxPoolerOp op) {
  auto feat0 = getCudaData(op.getPtrFeat0());
  auto feat1 = getCudaData(op.getPtrFeat1());
  auto feat2 = getCudaData(op.getPtrFeat2());
  auto feat3 = getCudaData(op.getPtrFeat3());
  auto rois = getCudaData(op.getRoisMultiBatch());
  auto output = getCudaData(op.getResultRes());
  auto output_rois = getCudaData(op.getResultRois());

  auto shape0 = module::getShape(op.getPtrFeat0());
  auto shape1 = module::getShape(op.getPtrFeat1());
  auto shape2 = module::getShape(op.getPtrFeat2());
  auto shape3 = module::getShape(op.getPtrFeat3());

  int batch_size = shape0[0];
  int C = op.getCHANNEL_ROI();
  int roi_slice = op.getROI_SLICE();
  int roi_len = op.getROI_LEN();
  int PH = op.getROI_PH();
  int PW = op.getROI_PW();
  int num_levels = op.getROI_NUM_LEVELS();

  cuda::maskRCNNBboxPoolerF32(feat0, feat1, feat2, feat3, rois, output,
                              output_rois, shape0[2], shape0[3], shape1[2],
                              shape1[3], shape2[2], shape2[3], shape3[2],
                              shape3[3], batch_size, C, roi_slice, roi_len, PH,
                              PW, num_levels);
}

void py_cuda::cudaMaskRCNNBboxPoolerOp(tpu::MaskRCNNBboxPoolerOp op) {
  auto feat0 = getCudaData(op.getPtrFeat0());
  auto feat1 = getCudaData(op.getPtrFeat1());
  auto feat2 = getCudaData(op.getPtrFeat2());
  auto feat3 = getCudaData(op.getPtrFeat3());
  auto rois = getCudaData(op.getRoisMultiBatch());
  auto output = getCudaData(op.getResultRes());
  auto output_rois = getCudaData(op.getResultRois());

  auto shape0 = module::getShape(op.getPtrFeat0());
  auto shape1 = module::getShape(op.getPtrFeat1());
  auto shape2 = module::getShape(op.getPtrFeat2());
  auto shape3 = module::getShape(op.getPtrFeat3());

  int batch_size = shape0[0];
  int C = op.getCHANNEL_ROI();
  int roi_slice = op.getROI_SLICE();
  int roi_len = op.getROI_LEN();
  int PH = op.getROI_PH();
  int PW = op.getROI_PW();
  int num_levels = op.getROI_NUM_LEVELS();

  auto stype = module::getStorageType(op.getResultRes());
  if (stype.isF32()) {
    cuda::maskRCNNBboxPoolerF32(feat0, feat1, feat2, feat3, rois, output,
                                output_rois, shape0[2], shape0[3], shape1[2],
                                shape1[3], shape2[2], shape2[3], shape3[2],
                                shape3[3], batch_size, C, roi_slice, roi_len, PH,
                                PW, num_levels);
  } else {
    auto num = module::getNumElements(op.getResultRes());
    auto num_rois_out = module::getNumElements(op.getResultRois());
    auto output_f32 = cuda_malloc(num * sizeof(float));
    auto output_rois_f32 = cuda_malloc(num_rois_out * sizeof(float));

    cuda::maskRCNNBboxPoolerF32(feat0, feat1, feat2, feat3, rois,
                                output_f32.get(), output_rois_f32.get(),
                                shape0[2], shape0[3], shape1[2], shape1[3],
                                shape2[2], shape2[3], shape3[2], shape3[3],
                                batch_size, C, roi_slice, roi_len, PH, PW,
                                num_levels);

    cuda::convertType(output_f32.get(), output, num, cuda::DT_F32,
                      getCudaType(op.getResultRes()));
    cuda::convertType(output_rois_f32.get(), output_rois, num_rois_out,
                      cuda::DT_F32, getCudaType(op.getResultRois()));
  }
}
