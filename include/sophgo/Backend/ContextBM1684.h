#pragma once

#include "sophgo/Backend/Context.h"

typedef int (*CModelInitFn)(int node_idx, unsigned long long global_mem_size);
typedef void (*CModelDeinitFn)(int node_idx);

namespace sophgo {
namespace backend {
class BM1684Context : public Context {
protected:
  BM1684Context();
  CModelInitFn cmodel_init;
  CModelDeinitFn cmodel_deinit;
};
} // namespace backend
} // namespace sophgo
