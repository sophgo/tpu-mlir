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

#include <cstring>
#include <vector>

#define EINSUM_MAX_DIMS 6

void py_cuda::cudaEinsumOp(top::EinsumOp op) {
  auto inputs = op.getInputs();
  auto out = op.getOutput();
  auto mode = op.getMode().str();
  auto num_inputs = inputs.size();

  // unsupported: 3-input modes (should be canonicalised away)
  if (num_inputs != 2) {
    UNREACHABLE_OP("Einsum with != 2 inputs not supported in CUDA runtime", op);
  }

  auto lhs = inputs[0];
  auto rhs = inputs[1];

  auto lhs_shape64 = module::getShape(lhs);
  auto rhs_shape64 = module::getShape(rhs);

  // ---- parse mode string -----------------------------------------------
  auto arrow = mode.find("->");
  auto comma = mode.find(',');
  auto lhs_spec = mode.substr(0, comma);
  auto rhs_spec = mode.substr(comma + 1, arrow - comma - 1);
  auto out_spec = mode.substr(arrow + 2);

  // letter -> {dim_in_lhs, dim_in_rhs, dim_in_out}, -1 = absent
  int letter_dim[26][3];
  memset(letter_dim, -1, sizeof(letter_dim));
  for (size_t i = 0; i < lhs_spec.size(); i++)
    letter_dim[lhs_spec[i] - 'a'][0] = (int)i;
  for (size_t i = 0; i < rhs_spec.size(); i++)
    letter_dim[rhs_spec[i] - 'a'][1] = (int)i;
  for (size_t i = 0; i < out_spec.size(); i++)
    letter_dim[out_spec[i] - 'a'][2] = (int)i;

  int lhs_rank = (int)lhs_shape64.size();
  int rhs_rank = (int)rhs_shape64.size();
  int out_rank = (int)out_spec.size();

  int lhs_shape[EINSUM_MAX_DIMS] = {0};
  int rhs_shape[EINSUM_MAX_DIMS] = {0};
  int out_shape[EINSUM_MAX_DIMS] = {0};
  int lhs_out_dim[EINSUM_MAX_DIMS];
  int rhs_out_dim[EINSUM_MAX_DIMS];
  int lhs_contract_dim[EINSUM_MAX_DIMS];
  int rhs_contract_dim[EINSUM_MAX_DIMS];
  int contract_shapes[EINSUM_MAX_DIMS];

  // zero-init all arrays
  memset(lhs_shape, 0, sizeof(lhs_shape));
  memset(rhs_shape, 0, sizeof(rhs_shape));
  memset(out_shape, 0, sizeof(out_shape));
  memset(lhs_out_dim, -1, sizeof(lhs_out_dim));
  memset(rhs_out_dim, -1, sizeof(rhs_out_dim));
  memset(lhs_contract_dim, -1, sizeof(lhs_contract_dim));
  memset(rhs_contract_dim, -1, sizeof(rhs_contract_dim));
  memset(contract_shapes, 0, sizeof(contract_shapes));

  for (int i = 0; i < lhs_rank; i++)
    lhs_shape[i] = (int)lhs_shape64[i];
  for (int i = 0; i < rhs_rank; i++)
    rhs_shape[i] = (int)rhs_shape64[i];

  // output shapes + output->input dimension mapping
  int total_out_elems = 1;
  for (int i = 0; i < out_rank; i++) {
    int letter = out_spec[i] - 'a';
    int ld = letter_dim[letter][0];
    int rd = letter_dim[letter][1];
    int sz = (ld >= 0) ? (int)lhs_shape64[ld] : (int)rhs_shape64[rd];
    out_shape[i] = sz;
    total_out_elems *= sz;
    lhs_out_dim[i] = ld;
    rhs_out_dim[i] = rd;
  }

  // contracted dimensions (in both inputs, not in output)
  int num_contract = 0;
  int total_contract_elems = 1;
  for (int letter = 0; letter < 26; letter++) {
    if (letter_dim[letter][0] >= 0 && letter_dim[letter][1] >= 0 &&
        letter_dim[letter][2] < 0) {
      int ld = letter_dim[letter][0];
      int sz = (int)lhs_shape64[ld];
      lhs_contract_dim[num_contract] = ld;
      rhs_contract_dim[num_contract] = letter_dim[letter][1];
      contract_shapes[num_contract] = sz;
      total_contract_elems *= sz;
      num_contract++;
    }
  }

  // ---- dispatch to CUDA kernel -----------------------------------------
  auto in0 = getCudaData(lhs);
  auto in1 = getCudaData(rhs);
  auto output = getCudaData(out);

  if (module::getStorageType(lhs).isF32()) {
    cuda::einsumF32(in0, in1, output,
                    lhs_shape, rhs_shape, out_shape,
                    lhs_rank, rhs_rank, out_rank, num_contract,
                    lhs_out_dim, rhs_out_dim,
                    lhs_contract_dim, rhs_contract_dim,
                    contract_shapes, total_out_elems, total_contract_elems);
  } else {
    auto total_elems = module::getNumElements(out);
    auto in0_f32 = newCudaData(lhs, cuda::DT_F32);
    auto in1_f32 = newCudaData(rhs, cuda::DT_F32);
    auto out_f32 = cuda_malloc(total_elems * sizeof(float));
    cuda::einsumF32(in0_f32.get(), in1_f32.get(), out_f32.get(),
                    lhs_shape, rhs_shape, out_shape,
                    lhs_rank, rhs_rank, out_rank, num_contract,
                    lhs_out_dim, rhs_out_dim,
                    lhs_contract_dim, rhs_contract_dim,
                    contract_shapes, total_out_elems, total_contract_elems);
    cuda::convertType(out_f32.get(), output, total_elems,
                      cuda::DT_F32, getCudaType(out));
    in0_f32.reset();
    in1_f32.reset();
    out_f32.reset();
  }
}
