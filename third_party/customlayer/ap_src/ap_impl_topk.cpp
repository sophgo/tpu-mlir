#include "ap_impl_topk.h"
#include "bmcpu.h"
#include "cpu_layer.h"
#include "cpu_layer_factory.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include <stdio.h>
#include <utility>
#include <vector>

using namespace std;
namespace bmcpu {

template <typename T>
struct Descending {
  bool operator()(const std::pair<T, int> &lhs,
                  const std::pair<T, int> &rhs) const {
    return lhs.first > rhs.first ||
           (lhs.first == rhs.first && lhs.second < rhs.second);
  }
};

template <typename T, typename I>
struct IndexComp {
  bool operator()(const std::pair<T, I> &lhs,
                  const std::pair<T, I> &rhs) const {
    return lhs.second < rhs.second;
  }
};

template <typename T, typename I, class ValueComp>
void GetTopK(const T *input, const int n, const int k, const int src_offset,
             const int dst_offset, const int stride, T *values, I *indices) {
  const T *src_ptr = input + src_offset;
  std::vector<std::pair<T, I>> heap_data;
  heap_data.reserve(k);
  for (int i = 0; i < k && i < n; ++i) {
    heap_data.emplace_back(*src_ptr, i);
    src_ptr += stride;
  }
  std::priority_queue<std::pair<T, I>, std::vector<std::pair<T, I>>, ValueComp>
      pq(ValueComp(), std::move(heap_data));
  for (int i = k; i < n; ++i) {
    if (pq.top().first < *src_ptr) {
      pq.pop();
      pq.emplace(*src_ptr, i);
    }
    src_ptr += stride;
  }
  int dst_pos = dst_offset + (std::min(k, n) - 1) * stride;
  while (!pq.empty()) {
    const auto &item = pq.top();
    values[dst_pos] = item.first;
    indices[dst_pos] = item.second;
    pq.pop();
    dst_pos -= stride;
  }
}

template <typename T, typename I, class ValueComp>
bool TopK(const T *input_data, const vector<int> &input_dims, const int k,
          const int axis, T *values_data, I *indices_data) {
  if (k == 0)
    return true;
  CPU_ASSERT(axis >= 0);
  CPU_ASSERT(axis < (int)input_dims.size());
  const int prev_size =
      std::accumulate(input_dims.cbegin(), input_dims.cbegin() + axis, int(1),
                      std::multiplies<int>());
  const int next_size =
      std::accumulate(input_dims.cbegin() + axis + 1, input_dims.cend(), int(1),
                      std::multiplies<int>());
  const int src_offset_stride = input_dims[axis] * next_size;
  const int dst_offset_stride = k * next_size;
  int src_offset = 0;
  int dst_offset = 0;
  for (int i = 0; i < prev_size; ++i) {
    for (int j = 0; j < next_size; ++j) {
      GetTopK<T, I, ValueComp>(input_data, input_dims[axis], k, src_offset + j,
                               dst_offset + j, next_size, values_data,
                               indices_data);
    }
    src_offset += src_offset_stride;
    dst_offset += dst_offset_stride;
  }
  return true;
}

struct ParallelExecutor {
  template <typename Func, typename... ArgTypes>
  static void run(Func functor, const int N, ArgTypes... args) {
    // TODO: use std::thread or openmp
    for (int i = 0; i < N; i++) {
      functor(i, args...);
    }
  }
};

int cpu_topklayer::get_param(void *param, int param_size) {
  axis_ = ((custom_param_t *)param)[1].int_t;
  K_ = ((custom_param_t *)param)[2].int_t;
  printf("topk get_param !\n");
  return 0;
}

int cpu_topklayer::forward(void *raw_param, int param_size) {
  CPU_ASSERT(output_tensors_.size() >= 1);
  const float *bottom_data = input_tensors_[0];
  int k = K_;
  if (input_tensors_.size() == 2)
    k = *(reinterpret_cast<int *>(input_tensors_[1]));
  int axis = axis_ == -1 ? input_shapes_[0].size() - 1 : axis_;
  if (k > 0) {
    float *values;
    int *indices;
    float *unused_data;
    if (output_tensors_.size() == 1) {
      unsigned long long output_len = 1;
      for (int i = 0; i < (int)output_shapes_[0][0].size(); i++)
        output_len *= output_shapes_[0][0][i];
      unused_data = new float[output_len];
      values = output_tensors_[0];
      indices = reinterpret_cast<int *>(unused_data);
    } else {
      values = output_tensors_[0];
      indices = reinterpret_cast<int *>(output_tensors_[1]);
    }
    TopK<float, int, Descending<float>>(bottom_data, input_shapes_[0], k, axis,
                                        values, indices);
    if (output_tensors_.size() == 1)
      delete[] unused_data;
  }

  for (int i = 0; i < (int)output_shapes_->size(); i++)
    (*output_shapes_)[i][axis] = k;
  printf("topk forward !\n");
  return 0;
}

int cpu_topklayer::shepe_infer(void *param, int param_size,
                               const vector<vector<int>> &input_shapes,
                               vector<vector<int>> &output_shapes) {
  get_param(param, param_size);
  for (const auto& array : input_shapes) {
    output_shapes.emplace_back(array);
  }
  output_shapes[0][axis_] = std::min(K_, input_shapes[0][axis_]);
  printf("topk_layer shape inference !\n");
  return 0;
}


REGISTER_APLAYER_CLASS(AP_CUSTOM_TOPK, ap_topk);
} /* namespace bmcpu */
