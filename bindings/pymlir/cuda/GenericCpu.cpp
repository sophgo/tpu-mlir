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

void py_cuda::cudaGenericCpuOp(tpu::GenericCpuOp op) {
  if (op.getCpuOpName() == "quant") {
    if (!module::isUniformQuantized(op.getOutputs()[0])) {
      UNREACHABLE_OP("Not Implemented", op);
    }
    auto param = op.getParam().value();
    float scale = param.get("scale").cast<FloatAttr>().getValueAsDouble();
    void *input = getCudaData(op.getInputs()[0]);
    void *output = getCudaData(op.getOutputs()[0]);
    int num_elems = module::getNumElements(op.getInputs()[0]);
    cuda::f32ScaleToInt8(input, output, scale, num_elems, true,
                         cuda::RD_HALF_AWAY_FROM_ZERO);
    return;
  } else if (op.getCpuOpName() == "embedding") {
    auto in = op.getInputs()[0];
    auto embed = op.getInputs()[1];
    auto out = op.getOutputs()[0];
    void *in_ptr = getCudaData(in);
    void *embed_ptr = getCudaData(embed);
    void *out_ptr = getCudaData(out);
    auto in_type = getCudaType(in);
    auto out_type = getCudaType(out);
    int num_in = module::getNumElements(in);
    int num_embed = module::getNumElements(embed);
    auto embed_shape = module::getShape(embed);
    int embed_dim = embed_shape[0];
    int inner_dim = num_embed / embed_dim;
    cuda::gather(in_ptr, embed_ptr, out_ptr, num_in, embed_dim, inner_dim,
                 in_type, out_type);
  } else {
  }
}
