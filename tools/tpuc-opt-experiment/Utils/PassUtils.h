#pragma once
#include "include/Utils.h"
namespace mlir {
template <typename... OpTys>
struct MultiOpNest {
public:
  MultiOpNest(OpPassManager &parentPm) : parentPm(parentPm) {
    addNest<0, OpTys...>();
  }

  template <typename F>
  MultiOpNest &addPass(F constructor) {
    addPassInternal(constructor);
    return *this;
  }

  MultiOpNest &addPass(std::unique_ptr<Pass> (*constructor)()) {
    addPassInternal(constructor);
    return *this;
  }

  template <typename F>
  MultiOpNest &addPredicatedPass(bool enable, F constructor) {
    if (enable) {
      addPassInternal(constructor);
    }
    return *this;
  }

private:
  // Initialize a nest.
  template <int index, typename T, typename... Rest>
  void addNest() {
    std::get<index>(nestedPassManagers) = &parentPm.nest<T>();
    addNest<index + 1, Rest...>();
  }
  template <int index>
  void addNest() {}

  // Add a pass to all nests by constructor.
  template <typename F>
  void addPassInternal(F constructor) {
    addPassRecurse<F, 0, OpTys...>(constructor);
  }
  template <typename F, int index, typename T, typename... Rest>
  void addPassRecurse(F constructor) {
    std::get<index>(nestedPassManagers)->addPass(constructor());
    addPassRecurse<F, index + 1, Rest...>(constructor);
  }
  template <typename F, int index>
  void addPassRecurse(F constructor) {}

  OpPassManager &parentPm;
  std::array<OpPassManager *, sizeof...(OpTys)> nestedPassManagers;
};
} // namespace mlir
