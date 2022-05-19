#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/Helper/Module.h"
#include "sophgo/Backend/BM168x/BM168x.h"
#include "mlir/Support/LLVM.h"
#include <map>
#include <set>
#include <list>
namespace sophgo {
namespace tpu {

typedef enum {
  LMEM_WEIGHT,
  LMEM_TENSOR,
  LMEM_OPERATION,
  LMEM_ANY,
} lmem_type_t;

typedef std::pair<int64_t, int64_t> slice_pair_t; // idx and slice
typedef std::pair<int64_t, int64_t> group_pair_t; // start_idx, and end_idx
typedef std::pair<int64_t, int64_t> addr_pair_t;  // lmem addr, and size
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
  int64_t timestep;
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
      : addr(-1), size(0), id(id), start_id(start_id), end_id(end_id),
        type(type), value(v), op(op), is_input(false), is_output(false) {}
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
  void buildGroups();
  void buildMlir();
  bool isLgSupport(int64_t op_idx);
  bool check_group(int64_t start_idx, int64_t end_idx);
  bool check_hsecs(lmem_info_t &lmem_info);
  void slice_all_outputs(group_lmem_t group_lmem, int64_t nsecs, int64_t hsecs);
  bool backward_entry(group_lmem_t group_lmem);
  bool backward_from_tensor(group_lmem_t group_lmem, lmem_info_t *linfo);
  void get_max_slice_nh(const lmem_info_t &linfo, int64_t &max_n,
                        int64_t &max_h);
  lmem_info_t *find_max_unalloc_lmem(group_lmem_t group_lmem,
                                     int64_t op_id = -1,
                                     lmem_type_t type = LMEM_ANY);
  void free_unuse_lmem(group_lmem_t group_lmem, int64_t op_id);
  void set_lmem_size(group_lmem_t group_lmem);
  void assign_timestep(group_lmem_t group_lmem);
  void adjust_lmem_id(group_lmem_t group_lmem, int64_t nsecs, int64_t hsecs);
  bool assign_lmem_addr(group_lmem_t group_lmem, int64_t nsecs, int64_t hsecs);
  int64_t alloc_lmem(int64_t size);
  void free_lmem(int64_t addr);
  inline bool is_same_slice(const slice_pair_t &a, const slice_pair_t &b) {
    return a.first == b.first && a.second == b.second;
  }
  bool is_same_slice(const std::vector<slice_pair_t> &a,
                     const std::vector<slice_pair_t> &b);
  lmem_info_t *find_lmem_info(group_lmem_t group_lmem, mlir::Value v);
  lmem_info_t *find_lmem_info(group_lmem_t group_lmem, mlir::Operation *op);
  void CreateLoadOp(lmem_info_t &linfo,
                    const std::vector<mlir::Operation *> &ops);
  tpu::StoreOp CreateStoreOp(lmem_info_t &linfo);
  void UpdateOpLgParam(group_lmem_t group_lmem, lmem_info_t &linfo);
  tpu::LayerGroup getLgParam(lmem_info_t &linfo, int64_t buffer_addr = 0,
                             int64_t buffer_size = 0);
  bool need_none(group_lmem_t group_lmem);
  void buildGroupOp(group_lmem_t group_lmem);

protected:
  std::vector<group_lmem_t> all_lmems;
  std::vector<mlir::Operation *> all_ops;
  std::vector<group_pair_t> groups;
  std::list<addr_pair_t> allocated_lmems;
  int64_t n_align;
  bool no_more_try_secs;
  mlir::MLIRContext *ctx;
  Operation *current_op;
  Block *body;
};

} // namespace tpu
} // namespace sophgo
