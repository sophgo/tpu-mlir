#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/Helper/Module.h"
#include "sophgo/Backend/BM168x/BM168x.h"
#include "mlir/Support/LLVM.h"
#include <map>
#include <set>
namespace sophgo {
namespace tpu {

typedef enum {
  LMEM_WEIGHT,
  LMEM_TENSOR,
  LMEM_OPERATION,
  LMEM_UNKNOWN,
} lmem_type_t;

typedef std::pair<int64_t, int64_t> slice_pair_t; // idx and slice
typedef std::pair<int64_t, int64_t> group_pair_t; // start_idx, and end_idx
struct slice_info_t {
  std::vector<slice_pair_t> h; // h_idx and h_slice
  std::vector<slice_pair_t> n; // h_idx and n_slice
};

struct lmem_info_t {
  int64_t addr;
  int64_t size;
  int64_t id;
  int64_t start_id;
  int64_t end_id;
  // memory for value or operation
  lmem_type_t type;
  Value value;
  Operation *op;
  // slice info
  slice_info_t slice_info;
  bool is_input;  // input tensor
  bool is_output; // output tensor
  // init
  explicit lmem_info_t(lmem_type_t type, int64_t id, int64_t start_id,
                       int64_t end_id, Value v = nullptr,
                       Operation *op = nullptr)
      : id(id), start_id(start_id), end_id(end_id), value(v), op(op),
        is_input(false), is_output(false) {}
};

typedef std::shared_ptr<std::vector<lmem_info_t>> group_lmem_t;

class GroupOps {
public:
  GroupOps(::mlir::func::FuncOp func);
  void process();
  ::mlir::func::FuncOp func;
  backend::BM168x *bm168x;

protected:
  group_lmem_t list_lmems(int64_t start_idx, int64_t end_idx);
  group_lmem_t CreateGroup(int64_t start_idx, int64_t end_idx,
                           int64_t &new_start_idx);
  group_lmem_t CreateGroupBySecs(int64_t start_idx, int64_t end_idx,
                                 int64_t nsecs, int64_t hsecs);
  bool isWeightValue(mlir::Value v);
  void group_search();
  bool isLgSupport(int64_t op_idx);
  bool check_group(int64_t start_idx, int64_t end_idx);
  bool check_hsecs(lmem_info_t &lmem_info);
  void slice_all_outputs(group_lmem_t group_lmem, int64_t nsecs, int64_t hsecs);
  bool backward_entry(group_lmem_t group_lmem);
  bool backward_from_tensor(group_lmem_t group_lmem, Value v);
  void get_max_slice_nh(const lmem_info_t &lmem_info, int64_t &max_n,
                        int64_t &max_h);
  void get_op_buffer_size(group_lmem_t group_lmem);
  void union_slice(slice_pair_t &target, slice_pair_t &from);
  void union_slice(std::vector<slice_pair_t> &targets,
                   std::vector<slice_pair_t> &froms);
  lmem_info_t *find_lmem_info(group_lmem_t group_lmem, mlir::Value v);
  lmem_info_t *find_lmem_info(group_lmem_t group_lmem, mlir::Operation *op);
  std::vector<group_lmem_t> all_lmems;
  std::vector<mlir::Operation *> all_ops;
  std::vector<group_pair_t> groups;
  int64_t n_align;
  bool no_more_try_hsecs;
};

} // namespace tpu
} // namespace sophgo
