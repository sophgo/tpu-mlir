//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaSelectiveScanOp(top::SelectiveScanOp op) {
  auto c_ptr = getCudaData(op.getCs());
  auto deltaA = getCudaData(op.getDeltaA());
  auto deltaB_u = getCudaData(op.getDeltaBU());
  auto output = getCudaData(op.getOutput());

  auto dA_shape = module::getShape(op.getDeltaA());
  int Kcdim = dA_shape[1], L = dA_shape[2], Batch = dA_shape[3];

  bool has_u = !op.getUs().getType().isa<mlir::NoneType>();
  bool has_D = !op.getDs().getType().isa<mlir::NoneType>();
  void *u_ptr = has_u ? getCudaData(op.getUs()) : nullptr;
  void *D_ptr = has_D ? getCudaData(op.getDs()) : nullptr;

  cuda::selectiveScan(c_ptr, deltaA, deltaB_u, u_ptr, D_ptr, output,
                      Kcdim, L, Batch, has_u && has_D ? 1 : 0);
}

void py_cuda::cudaSelectiveScanOp(tpu::SelectiveScanOp op) {
  auto dA_shape = module::getShape(op.getDeltaA());
  int Kcdim = dA_shape[1], L = dA_shape[2], Batch = dA_shape[3];
  bool has_u = !op.getUs().getType().isa<mlir::NoneType>();
  bool has_D = !op.getDs().getType().isa<mlir::NoneType>();

  auto stype = module::getStorageType(op.getOutput());
  if (stype.isF32()) {
    void *u_ptr = has_u ? getCudaData(op.getUs()) : nullptr;
    void *D_ptr = has_D ? getCudaData(op.getDs()) : nullptr;
    cuda::selectiveScan(getCudaData(op.getCs()), getCudaData(op.getDeltaA()),
                        getCudaData(op.getDeltaBU()), u_ptr, D_ptr,
                        getCudaData(op.getOutput()),
                        Kcdim, L, Batch, has_u && has_D ? 1 : 0);
    return;
  }

  auto c_shape = module::getShape(op.getCs());
  auto out_shape = module::getShape(op.getOutput());
  auto dBu_shape = module::getShape(op.getDeltaBU());

  int c_elems = 1, out_elems = 1, dA_elems = 1, dBu_elems = 1;
  for (auto d : c_shape) c_elems *= d;
  for (auto d : out_shape) out_elems *= d;
  for (auto d : dA_shape) dA_elems *= d;
  for (auto d : dBu_shape) dBu_elems *= d;

  auto c_f32 = cuda_malloc(c_elems * sizeof(float));
  auto dA_f32 = cuda_malloc(dA_elems * sizeof(float));
  auto dBu_f32 = cuda_malloc(dBu_elems * sizeof(float));
  auto out_f32 = cuda_malloc(out_elems * sizeof(float));

  auto c_dtype = getCudaType(op.getCs());
  auto dA_dtype = getCudaType(op.getDeltaA());
  auto dBu_dtype = getCudaType(op.getDeltaBU());
  cuda::convertType(getCudaData(op.getCs()), c_f32.get(), c_elems, c_dtype, cuda::DT_F32);
  cuda::convertType(getCudaData(op.getDeltaA()), dA_f32.get(), dA_elems, dA_dtype, cuda::DT_F32);
  cuda::convertType(getCudaData(op.getDeltaBU()), dBu_f32.get(), dBu_elems, dBu_dtype, cuda::DT_F32);

  void *u_f32 = nullptr, *D_f32 = nullptr;
  std::shared_ptr<void> u_guard, D_guard;
  if (has_u) {
    auto u_shape = module::getShape(op.getUs());
    int u_elems = 1;
    for (auto d : u_shape) u_elems *= d;
    u_guard = cuda_malloc(u_elems * sizeof(float));
    cuda::convertType(getCudaData(op.getUs()), u_guard.get(), u_elems,
                      getCudaType(op.getUs()), cuda::DT_F32);
    u_f32 = u_guard.get();
  }
  if (has_D) {
    auto D_shape = module::getShape(op.getDs());
    int D_elems = 1;
    for (auto d : D_shape) D_elems *= d;
    D_guard = cuda_malloc(D_elems * sizeof(float));
    cuda::convertType(getCudaData(op.getDs()), D_guard.get(), D_elems,
                      getCudaType(op.getDs()), cuda::DT_F32);
    D_f32 = D_guard.get();
  }

  cuda::selectiveScan(c_f32.get(), dA_f32.get(), dBu_f32.get(),
                      u_f32, D_f32, out_f32.get(),
                      Kcdim, L, Batch, has_u && has_D ? 1 : 0);

  cuda::convertType(out_f32.get(), getCudaData(op.getOutput()), out_elems,
                    cuda::DT_F32, getCudaType(op.getOutput()));
}
