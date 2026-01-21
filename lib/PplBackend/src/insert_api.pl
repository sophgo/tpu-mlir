#include "ppl.h"
#include "ppl_wrapper_func.h"
using namespace ppl;

template <typename T>
void insert_tensor_data(T *ptr_dst, T *ptr_src, int axis, int offset, int N,
                        int C, int H, int W, int D) {
  if (axis < 0 || axis > 3)
    return;
  if (offset < 0)
    return;
  int shape0 = N, shape1 = C, shape2 = H, shape3 = W;
  int offset0 = 0, offset1 = 0, offset2 = 0, offset3 = 0;
  if (axis == 0) {
    shape0 = D;
    offset0 = offset;
  } else if (axis == 1) {
    shape1 = D;
    offset1 = offset;
  } else if (axis == 2) {
    shape2 = D;
    offset2 = offset;
  } else if (axis == 3) {
    shape3 = D;
    offset3 = offset;
  }
  dim4 dst_shape = {N, C, H, W};
  dim4 dst_offset = {offset0, offset1, offset2, offset3};
  dim4 src_shape = {shape0, shape1, shape2, shape3};
  auto dst_gtensor = gtensor<T>(dst_shape, GLOBAL, ptr_dst);
  auto src_gtensor = gtensor<T>(src_shape, GLOBAL, ptr_src);
  auto dst_view = dst_gtensor.sub_view(src_shape, dst_offset);
  dma::move(dst_view, src_gtensor);
}

__KERNEL__ void insert_tensor(void *ptr_dst, void *ptr_src, int axis,
                              int offset, int N, int C, int H, int W, int D,
                              int dbytes) {
  if (dbytes == 1) {
    insert_tensor_data((int8 *)ptr_dst, (int8 *)ptr_src, axis, offset, N, C, H,
                       W, D);
  } else if (dbytes == 2) {
    insert_tensor_data((fp16 *)ptr_dst, (fp16 *)ptr_src, axis, offset, N, C, H,
                       W, D);
  } else if (dbytes == 4) {
    insert_tensor_data((fp32 *)ptr_dst, (fp32 *)ptr_src, axis, offset, N, C, H,
                       W, D);
  }
}

__TEST__ void topk_test_main() {

  const int N = 4;
  const int C = 1024;
  const int H = 8;
  const int W = 128;
  const int D = 1;

  dim4 dst_shape = {N, C, H, W};
  dim4 src_shape = {N, D, H, W};

  auto dst = rand<bf16>(&dst_shape, 0.0, 1000.0);
  auto src = rand<bf16>(&src_shape, 0.0, 1000.0);
  int offset = 1023;

  insert_tensor(dst, src, 1, offset, N, C, H, W, D, 2);
}
