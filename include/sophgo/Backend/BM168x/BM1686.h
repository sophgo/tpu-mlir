#pragma once

#include "sophgo/Backend/BM168x/BM168x.h"

typedef void (*load_lookup_tables)();
typedef void (*store_cmd_end)();
typedef void (*set_cmd_len_ptr)(void *gdma_cmd_len_ptr, void *bdc_cmd_len_ptr);

namespace sophgo {
namespace backend {
class BM1686 : public BM168x {
public:
  static BM1686 &instance() {
    static BM1686 inst;
    return inst;
  }
public:
  // -------------------------------------------------------------------
  // functions from nodechip
  // -------------------------------------------------------------------
  load_lookup_tables dl_load_lookup_tables;
  store_cmd_end dl_store_cmd_end;
  set_cmd_len_ptr dl_set_cmd_len_ptr;

public:
  virtual void init() override;
  virtual void after_codegen() override;
  virtual uint64_t get_gmem_start() override;
  virtual uint64_t get_ctx_start_addr() override;
  virtual uint32_t get_bdc_len(int bdc_num, int group_id) override;
  virtual uint32_t get_gdma_len(int gdma_num, int group_id) override;
  void call_global_func(const char * symbolName, void *params, int param_size);
  static const int64_t NPU_NUM = 64;
  static const int64_t EU_BYTES = 64;
  static int64_t get_eu_num(int64_t dtype_bytes);

protected:
  BM1686();
  ~BM1686();

  template <typename FPtrTy> FPtrTy CastToFPtr(const char *symbolName);
  virtual void load_functions() override;
};
}
}
