//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/TimeStepMethod.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/IlpTimeStep.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>

namespace tpu_mlir {
namespace tpu {

class AutoIndent {
public:
    static int indent;
    AutoIndent() {
      indent++;
    }

    ~AutoIndent() {
      indent--;
    }
};
int AutoIndent::indent = 0;

bool SortByMemStruct(const std::pair<std::string, mem_struct> &v1, const std::pair<std::string, mem_struct> &v2)
{
    return v1.second.addr < v2.second.addr;//升序排列
}

bool SortByMemSize(const mem_alloc_req_info &v1, const mem_alloc_req_info &v2)
{
    return v1.size > v2.size;//升序排列
}

inline int64_t align(int64_t input, int64_t align_size)
{
    if (input % align_size != 0) {
      printf("warning, input:%ld is not align %ld\n", input, align_size);
    }
    return (int64_t)((input + align_size - 1)/align_size*align_size);
}

inline std::string convert_name_to_key(const std::string& name, int slice_idx) {
  return llvm::formatv("{0}_slice{1}", name, slice_idx).str();
}

bool is_range_overlap(int start1, int end1, int start2, int end2) {
  if (std::max(start1,start2) < std::min(end1, end2)) {
    return true;
  }
  return false;
}

std::string vector_to_string(std::vector<int> vec_data) {
  std::string tmp = "[";
  for (auto it: vec_data) {
    tmp = tmp + std::to_string(it) + ",";
  }
  if (vec_data.size() > 0)
    tmp.pop_back();
  tmp = tmp + "]";
  return tmp;
}

lmem_alloc::lmem_alloc(std::map<std::string, std::vector<std::string>>& banked_tensors, ILPTimeStep* pILPTimeStep, int ts_count)
:banked_tensors_(banked_tensors), m_pILPTimeStep(pILPTimeStep) {
  total_size = 256*1024;
  lmem_buf = new bool[total_size];
  for (int i = 0; i < total_size; i++) {
    lmem_buf[i] = false;
  }

  for (int i = 0; i < 16; i++) {
    bank_num[i] = i;
    bank_area_start_addr[i] = i*total_size/16; //16*1024
  }
  bank_area_start_addr[16] = total_size;
  m_ts_count = ts_count;
}

lmem_alloc::~lmem_alloc() {
  delete []lmem_buf;
}

std::shared_ptr<std::vector<std::pair<std::string, mem_struct>>> lmem_alloc::show_mem(int& total_free_size, int& max_free_mem_idx, int& max_free_mem_size) {
  int free_start_addr = -1, free_count = 0, total_free_count = 0, start_bank = -1, end_bank = -1;
  for (int i = 0; i < total_size; i++) {
    if (!lmem_buf[i]) {
      if (free_start_addr == -1) {
        free_start_addr = i;
        start_bank = i / (total_size/16);
      }
      free_count++;
      total_free_count++;
    } else {
      if (free_start_addr >= 0) {
        end_bank = i / (total_size/16);
        printf("        >>>free_start_addr:%d, end_addr:%d, size:%d, start_bank:%d, end_bank:%d\n",
              free_start_addr, free_start_addr + free_count - 1, free_count, start_bank, end_bank);
        free_start_addr = -1;
        free_count = 0;
      }
    }
  }
  if (free_start_addr >= 0) {
    printf("        >>>free_start_addr:%d, end_addr:%d, size:%d, start_bank:%d, end_bank:15\n",
          free_start_addr, free_start_addr + free_count - 1, free_count, start_bank);
  }
  printf("        >>>total_free_count:%d\n", total_free_count);

  std::vector<std::pair<std::string, mem_struct>> vec_mem_struct;
  printf("        >>>mem_dict:\n");
  for (auto itr = mem_dict.begin(); itr != mem_dict.end(); ++itr) {
    vec_mem_struct.push_back(std::make_pair(itr->first.c_str(), itr->second));
    printf("        name:%s, addr:%d, size:%d\n", itr->first.c_str(), itr->second.addr, itr->second.size);
  }
  std::sort(vec_mem_struct.begin(), vec_mem_struct.end(), SortByMemStruct);
  int pre_s_addr = 0, free_mem_idx = 0;
  auto vec_mem_struct2 = std::make_shared<std::vector<std::pair<std::string, mem_struct>>>();
  int idx = 0;
  max_free_mem_idx = 0;
  max_free_mem_size = 0;
  total_free_size = 0;
  for (auto itr:vec_mem_struct) {
    if (itr.second.addr - pre_s_addr >= 64) {
      mem_struct mem_s;
      memset(&mem_s, 0, sizeof(mem_struct));
      mem_s.addr = pre_s_addr;
      mem_s.size = itr.second.addr - pre_s_addr;
      if (mem_s.size > max_free_mem_size) {
        max_free_mem_size = mem_s.size;
        max_free_mem_idx = idx;
      }
      total_free_size += mem_s.size;
      mem_s.type = 1;
      vec_mem_struct2->push_back(std::make_pair("free_mem "+std::to_string(free_mem_idx++), mem_s));
      idx++;
    }
    // itr.second.type = 0;
    vec_mem_struct2->push_back(itr);
    idx++;
    pre_s_addr = itr.second.addr + itr.second.size;
  }
  if (pre_s_addr < total_size) {
    mem_struct mem_s;
    memset(&mem_s, 0, sizeof(mem_struct));
    mem_s.addr = pre_s_addr;
    mem_s.size = total_size - pre_s_addr;
    if (mem_s.size > max_free_mem_size) {
      max_free_mem_size = mem_s.size;
      max_free_mem_idx = idx;
    }
    total_free_size += mem_s.size;
    mem_s.type = 1;
    vec_mem_struct2->push_back(std::make_pair("free_mem "+std::to_string(free_mem_idx++), mem_s));
  }
  printf("        >>>mem_dict:\n");
  int idx2 = 0;
  for (auto itr: *vec_mem_struct2) {
    start_bank = itr.second.addr / (total_size/16);
    int e_addr = itr.second.addr + itr.second.size - 1;
    end_bank = e_addr / (total_size/16);
    printf("           idx:%3d, key:%40s, s_addr:%8d, e_addr:%8d, size:%8d, bank_id:%d, type:%d, start_bank:%d, end_bank:%d\n",
      idx2++, itr.first.c_str(), itr.second.addr, e_addr, itr.second.size,
      itr.second.bank_id.size() > 0?itr.second.bank_id[0]:-1, itr.second.type, start_bank, end_bank);
  }
  printf("max_free_mem_idx:%d, max_free_mem_size:%d\n", max_free_mem_idx, max_free_mem_size);
  return std::move(vec_mem_struct2);
}

bool lmem_alloc::_alloc(int slice_idx, const std::string& name, Value value, int size, std::vector<int>& ret_bank_id,
                        int& free_addr, int& confict_size, bool force_not_care_bank) {
  std::vector<int> vec_bank_id, all_conf_bank_id;
  bool care_bank = false;
  if (!force_not_care_bank) {
    for (auto it: banked_tensors_[name]) {
      std::string key = convert_name_to_key(it, slice_idx);
      std::vector<int> bank_id = get_bank(key);
      all_conf_bank_id.insert(all_conf_bank_id.end(), bank_id.begin(), bank_id.end());
      if (bank_id.size() > 0) {
        printf("        confict to %s, bank_id:%s\n", key.c_str(), vector_to_string(bank_id).c_str());
        vec_bank_id.insert(vec_bank_id.end(), bank_id.begin(), bank_id.end());
        care_bank = true;
      }
    }
  }

  std::vector<int> searched_bank;
  int bidx = 0;
  while(true) {
    if (care_bank && !force_not_care_bank) {
      bidx = -1;
      for (int i = 0; i < 16; i++) {
        if ((std::find(vec_bank_id.begin(), vec_bank_id.end(), i) == vec_bank_id.end())
          && (std::find(searched_bank.begin(), searched_bank.end(), i) == searched_bank.end())) {
            searched_bank.push_back(i);
            bidx = i;
            break;
          }
      }
      if (bidx == -1) {
        printf("warning: not find valid bank, force no bank\n");
        care_bank = false;
      }
    }
    free_addr = -1;
    int saddr = care_bank? bank_area_start_addr[bidx]:0;
    // printf("saddr:%d\n", saddr);
    int count = 0;
    for (int i = saddr; i < total_size; i++) {
      if (lmem_buf[i]) {
        free_addr = -1;
        count = 0;
      } else {
        if (free_addr == -1) {
          if (i%64 == 0) {
            free_addr = i;
            // printf("find free_addr:%d\n", free_addr); //todo, have a bug
          }
        }
        if (free_addr == -1)
          continue;
        count++;
        if (count == size){
          break;
        }
      }
    }
    if (count == size) {
      int s_bidx = free_addr / (total_size/16);
      int e_bidx = (free_addr + size - 1) / (total_size/16);
      ret_bank_id.clear();
      std::map<int,int> bank_size;
      for (int i = s_bidx; i <= e_bidx; i++) {
        ret_bank_id.push_back(i);
        if (i == s_bidx) {
          if (free_addr + size > bank_area_start_addr[i + 1]) //分配区域横跨2个bank
            bank_size[i] = bank_area_start_addr[i + 1] - free_addr;
          else
            bank_size[i] = size; //分配区域在s_bidx这个bank内
        } else if (i == e_bidx) {
          bank_size[i] = free_addr + size - bank_area_start_addr[e_bidx];
        } else {
          //中间的bank不可能冲突
        }
      }
      printf("        ret_bank_id:%s\n", vector_to_string(ret_bank_id).c_str());

      confict_size = 0;
      for (auto itr: bank_size) {
        if ((std::find(all_conf_bank_id.begin(), all_conf_bank_id.end(), itr.first) != all_conf_bank_id.end())) {
          printf("          bank%d confilct size:%d\n", itr.first, itr.second);
          confict_size += itr.second;
        }
      }

      for (int i = 0; i < size; i++) {
        lmem_buf[free_addr+i] = true;
      }

      printf("%s\n", llvm::formatv("      alloc ok for {0}, free_addr:{1}, bank_id:{2}", name, free_addr, vector_to_string(ret_bank_id)).str().c_str());
      mem_struct mem_s;
      mem_s.addr = free_addr;
      mem_s.bank_id.assign(ret_bank_id.begin(), ret_bank_id.end());
      mem_s.size = size;
      mem_s.value = value;
      mem_s.slice_idx = slice_idx;
      mem_s.type = 0;
      std::string key = llvm::formatv("{0}_slice{1}", name, slice_idx).str();
      mem_dict[key] = mem_s;
      if (!rehearsal) {
        reload_mem_struct tmp_mem_s;
        tmp_mem_s.addr = free_addr;
        tmp_mem_s.bank_id.assign(ret_bank_id.begin(), ret_bank_id.end());
        if (vec_mem_alloc_his.find(key) != vec_mem_alloc_his.end()) {
          vec_mem_alloc_his[key].vec_reload_addr.push_back(std::make_pair(0xFFFFBBBB, tmp_mem_s));
        } else {
          his_mem_struct his_mem_s;
          his_mem_s.size = size;
          his_mem_s.value = value;
          his_mem_s.slice_idx = slice_idx;
          his_mem_s.vec_reload_addr.push_back(std::make_pair(0xFFFFBBBB, tmp_mem_s));
          vec_mem_alloc_his[key] = his_mem_s;
        }
      }
      return true;
    }
    if (!care_bank) {
      break;
    }
  }

  return false;
}

bool lmem_alloc::alloc_multi(int ts_idx, std::vector<mem_alloc_req_info>& vec_mem_req, bool sort_by_size) {
  int total_confict_size = 0, confict_size = 0, idx = 0, free_addr = -1;
  std::vector<int> new_vec_move_mem_min, ret_bank_id;
  if (sort_by_size) {
    printf("    alloc_multi sort_by_size\n");
    std::sort(vec_mem_req.begin(), vec_mem_req.end(), SortByMemSize);
    for (auto it : vec_mem_req) {
      printf("      name:%s, size:%d\n", it.name.c_str(), it.size);
    }
    printf("    start alloc:\n");
    for (auto it : vec_mem_req) {
      printf("      name:%s, size:%d\n", it.name.c_str(), it.size);
      // if (it.name == "weight.6") {
      //   int unused;
      //   // (void)lmem_alloc_ptr->show_mem(unused, unused, unused);
      //   show_mem(unused, unused, unused);
      // }

      if (!alloc(ts_idx, it.slice_idx, it.name, it.value, it.size)) {
        printf("_alloc fail\n");
        return false;
      }
    }
    return true;
  }

  std::vector<int> new_vec_move_mem;
  for (int i = 0; i < vec_mem_req.size(); i++) {
    new_vec_move_mem.push_back(i);
  }
  rehearsal = true;
  int min_confict_size = total_size;

  do {
    printf("\ntest permutation:%d\n", idx);
    for (auto i: new_vec_move_mem) {
      free(convert_name_to_key(vec_mem_req[i].name, vec_mem_req[i].slice_idx));
    }
    total_confict_size = 0;
    bool success = true;
    for (int i : new_vec_move_mem) {
      printf("  alloc i:%d tensor\n", i);
      if (!_alloc(vec_mem_req[i].slice_idx, vec_mem_req[i].name, vec_mem_req[i].value, vec_mem_req[i].size,
                  ret_bank_id, free_addr, confict_size)) {
        printf("_alloc fail\n");
        success = false;
        break;
      }
      printf("  free_addr:%d, confict_size:%d\n", free_addr, confict_size);
      total_confict_size += confict_size;
    }

    if (success && total_confict_size < min_confict_size) {
      min_confict_size = total_confict_size;
      new_vec_move_mem_min.assign(new_vec_move_mem.begin(), new_vec_move_mem.end());
      if (total_confict_size == 0) {
        printf("   no confilct\n");
        break;
      }
    }
    idx++;
  } while (std::next_permutation(new_vec_move_mem.begin(), new_vec_move_mem.end()));

  rehearsal = false;
  total_confict_size = 0;
  std::map<int, int> new_vec_move_mem_new_addr;
  printf("actually alloc the moved tensor again:\n");
  for (auto i: new_vec_move_mem) {
    free(convert_name_to_key(vec_mem_req[i].name, vec_mem_req[i].slice_idx));
  }
  for (int i : new_vec_move_mem_min) {
    printf("  i:%d\n", i);
    if (!_alloc(vec_mem_req[i].slice_idx, vec_mem_req[i].name, vec_mem_req[i].value, vec_mem_req[i].size,
                  ret_bank_id, free_addr, confict_size)) {
      printf("_alloc fail\n");
      return false;
    }
    printf("  confict_size:%d\n", confict_size);
    total_confict_size += confict_size;
  }
  printf("  total_confict_size:%d, min_confict_size:%d\n",total_confict_size, min_confict_size);
  assert(total_confict_size == min_confict_size);
  return true;
}


//refc：暂时只使用默认值1，考虑弃用
bool lmem_alloc::alloc(int ts_idx, int slice_idx, const std::string& name, Value value, int size) {
  printf("%s\n", llvm::formatv("      start alloc for {0}, size:{1}, slice_idx:{2}", name, size, slice_idx).str().c_str());
  assert(size > 0);
  int free_addr = -1, confict_size = -1;
  std::vector<int> ret_bank_id;
  if (!_alloc(slice_idx, name, value, size, ret_bank_id, free_addr, confict_size)) {
    printf("      alloc fail, current mem status:\n");
    // return false;
    int total_free_size = 0, max_free_mem_idx = 0, max_free_mem_size = 0;
    auto vec_mem_struct2 = *show_mem(total_free_size, max_free_mem_idx, max_free_mem_size);
    if (total_free_size < size) {
      printf("error! alloc total_free_size < size\n");
      return false;
    }

    int step = 0, tmp_size = 0, align_num = 0;
    std::vector<int> vec_merge_mem;
    std::vector<int> vec_move_mem;
    vec_merge_mem.push_back(max_free_mem_idx);
    for(int i = 1; i <= 5; i++) { //以最大空闲区域为中心，均匀的在两侧寻找最多5块空闲区域,以便能满足本次内存分配空间
      step = 0, tmp_size = max_free_mem_size;
      align_num = 0;
      for(int j = max_free_mem_idx + 1; j < vec_mem_struct2.size(); j++) {
        if (vec_mem_struct2[j].second.type == 1) {
          if (find(vec_merge_mem.begin(), vec_merge_mem.end(), j) == vec_merge_mem.end()) {
            vec_merge_mem.push_back(j);
          }

          tmp_size += vec_mem_struct2[j].second.size;
          if (tmp_size > size) {
            int addr = vec_mem_struct2[vec_merge_mem[0]].second.addr;
              //去掉最前面(vec_merge_mem[0])的空闲起始地址的非64字节对齐部分，保证移动后的tensor起始地址也满足64字节对齐
            align_num = 0;
            while (addr % 64 != 0) {
              addr++;
              align_num++;
            }
            if (tmp_size - align_num > size) {
              break;
            }
          }
          if (++step >= i) {
            align_num = 0;
            break;
          }
        } else {
          if (vec_mem_struct2[j].second.type == 2) {
            break;
          } else {
            if (find(vec_move_mem.begin(), vec_move_mem.end(), j) == vec_move_mem.end()) {
              vec_move_mem.push_back(j);
            }
          }
        }
      }
      if (tmp_size - align_num > size) {
        break;
      }
      step = 0;
      align_num = 0;
      for(int j = max_free_mem_idx - 1; j >= 0; j--) {
        if (vec_mem_struct2[j].second.type == 1) {
          if (find(vec_merge_mem.begin(), vec_merge_mem.end(), j) == vec_merge_mem.end()) {
            vec_merge_mem.insert(vec_merge_mem.begin(), j);
          }
          tmp_size += vec_mem_struct2[j].second.size;
          if (tmp_size > size) {
            int addr = vec_mem_struct2[vec_merge_mem[0]].second.addr;
            align_num = 0;
            while (addr % 64 != 0) {
              addr++;
              align_num++;
            }
            if (tmp_size - align_num > size) {
              break;
            }
          }
          if (++step >= i) {
            align_num = 0;
            break;
          }
        } else {
          if (find(vec_move_mem.begin(), vec_move_mem.end(), j) == vec_move_mem.end()) {
            vec_move_mem.insert(vec_move_mem.begin(), j);
          }
        }
      }
      if (tmp_size - align_num > size) {
        break;
      }
    }
    if (tmp_size - align_num < size) {
      printf("error! can not find enough free mem block\n");
      return false;
    }
    int min = 10000, max = -1;
    printf("vec_merge_mem:\n");
    for (auto i: vec_merge_mem) {
      printf("  i:%d\n", i);
      if (i < min) {
        min = i;
      }
      if (i > max) {
        max = i;
      }
    }
    std::vector<int> new_vec_move_mem;
    std::map<int, Value> new_vec_move_value;
    std::map<int, int> new_vec_move_mem_old_addr;
    std::map<int, int> new_vec_move_size;
    std::map<int, int> new_vec_slice_idx;
    printf("vec_move_mem:\n");
    for (auto i: vec_move_mem) {
      if (i > min && i < max) { //只对最上和最下空闲区域间的tensor进行移动
        new_vec_move_mem.push_back(i);
        new_vec_move_mem_old_addr[i] = vec_mem_struct2[i].second.addr;
        new_vec_move_size[i] =vec_mem_struct2[i].second.size;
        new_vec_move_value[i] = vec_mem_struct2[i].second.value;
        new_vec_slice_idx[i] = vec_mem_struct2[i].second.slice_idx;
        printf("  i:%d\n", i);
      }
    }

    rehearsal = true;
    new_vec_move_mem.push_back(-1); //表示对最新需分配的tensor
    std::vector<int> new_vec_move_mem_min;
    int min_confict_size = total_size;
    int total_confict_size = 0;
    bool force_not_care_bank = false;
    for (int j = 0; j < 2; j++) {
      int idx = 0;
      std::vector<int> order_idx;
      for (int i = 0; i < new_vec_move_mem.size(); i++) {
        order_idx.push_back(i);
      }

      do {
        printf("\ntest permutation:%d\n", idx);
        for (auto i: new_vec_move_mem) {
          if (i != -1)
              free(vec_mem_struct2[i].first);
          else
              free(convert_name_to_key(name, slice_idx));
        }
        int unused;
        // (void)lmem_alloc_ptr->show_mem(unused, unused, unused);
        show_mem(unused, unused, unused);
        total_confict_size = 0;
        bool success = true;
        for (int it : order_idx) {
          auto i = new_vec_move_mem[it];
          printf("  start alloc %dth tensor\n", i);
          if (i == -1) {
            if (!_alloc(slice_idx, name, value, size, ret_bank_id, free_addr, confict_size, force_not_care_bank)) {
              printf("_alloc fail 1\n");
              int unused;
              // (void)lmem_alloc_ptr->show_mem(unused, unused, unused);
              show_mem(unused, unused, unused);
              success = false;
              break;
            }
          } else {
            auto mem_s = vec_mem_struct2[i].second;
            auto name = module::getName(vec_mem_struct2[i].second.value).str();
            if (!_alloc(mem_s.slice_idx, name, mem_s.value, mem_s.size, ret_bank_id, free_addr, confict_size, force_not_care_bank)) {
              printf("_alloc fail 2\n");
              int unused;
              // (void)lmem_alloc_ptr->show_mem(unused, unused, unused);
              show_mem(unused, unused, unused);
              success = false;
              break;
            }
          }
          printf("  free_addr:%d, confict_size:%d\n", free_addr, confict_size);
          total_confict_size += confict_size;
        }
        if (success && total_confict_size < min_confict_size) {
          min_confict_size = total_confict_size;
          new_vec_move_mem_min.assign(order_idx.begin(), order_idx.end());
          if (total_confict_size == 0) {
            printf("   no confilct\n");
            break;
          }
        }
        idx++;
      } while (std::next_permutation(order_idx.begin(), order_idx.end()));
      if (min_confict_size != total_size) {
        break;
      }
      force_not_care_bank = true;
      printf("   enable force_not_care_bank\n");
    }

    rehearsal = false;
    total_confict_size = 0;
    std::map<int, int> new_vec_move_mem_new_addr;
    printf("actually alloc the moved tensor again:\n");
    for (auto i: new_vec_move_mem) {
      if (i != -1)
          free(vec_mem_struct2[i].first);
      else
          free(convert_name_to_key(name, slice_idx));
    }
    for (int it : new_vec_move_mem_min) {
      auto i = new_vec_move_mem[it];
      printf("  i:%d\n", i);
      // mem_struct mem_s;
      // std::string key;
      if (i == -1) {
        if (!_alloc(slice_idx, name, value, size, ret_bank_id, free_addr, confict_size, force_not_care_bank)) {
          return false;
        }
        // mem_s.size = size;
        // mem_s.value = value;
        // mem_s.slice_idx = slice_idx;
        // key = llvm::formatv("{0}_slice{1}", name, slice_idx).str();
        // reload_mem_struct tmp_mem_s;
        // tmp_mem_s.addr = free_addr;
        // tmp_mem_s.bank_id.assign(ret_bank_id.begin(), ret_bank_id.end());
        // if (vec_mem_alloc_his.find(key) != vec_mem_alloc_his.end()) {
        //   vec_mem_alloc_his[key].vec_reload_addr.push_back(std::make_pair(0xFFFFBBBB, tmp_mem_s));
        // } else {
        //   his_mem_struct his_mem_s;
        //   his_mem_s.size = size;
        //   his_mem_s.value = value;
        //   his_mem_s.slice_idx = slice_idx;
        //   his_mem_s.vec_reload_addr.push_back(std::make_pair(0xFFFFBBBB, tmp_mem_s));
        //   vec_mem_alloc_his[key] = his_mem_s;
        // }
      } else {
        auto op_name = module::getName(vec_mem_struct2[i].second.value).str();
        // key = vec_mem_struct2[i].first;
        auto mem_s = vec_mem_struct2[i].second;
        if (!_alloc(mem_s.slice_idx, op_name, mem_s.value, mem_s.size, ret_bank_id, free_addr, confict_size, force_not_care_bank)) {
          return false;
        }
        new_vec_move_mem_new_addr[i] = free_addr;
      }
      // mem_s.addr = free_addr;
      // mem_s.bank_id.assign(ret_bank_id.begin(), ret_bank_id.end());
      // mem_dict[key] = mem_s;
      printf("  confict_size:%d\n", confict_size);
      total_confict_size += confict_size;
    }
    printf("  total_confict_size:%d, min_confict_size:%d\n",total_confict_size, min_confict_size);
    assert(total_confict_size == min_confict_size);
    ts_move_info tmp;
    for (int it : new_vec_move_mem_min) {
      auto i = new_vec_move_mem[it];
      if (i != -1 && new_vec_move_mem_old_addr[i]!=new_vec_move_mem_new_addr[i]) {
       tmp.move_value.push_back(new_vec_move_value[i]);
       tmp.move_src_add.push_back(new_vec_move_mem_old_addr[i]);
       tmp.move_dest_add.push_back(new_vec_move_mem_new_addr[i]);
       tmp.move_size.push_back(new_vec_move_size[i]);
       tmp.slice_idx.push_back(new_vec_slice_idx[i]);
      }
    }
    tmp.name = "lmem_tensor_move_at_ts"+std::to_string(ts_idx);
    m_pILPTimeStep->inserted_timestep_table_[ts_idx] = tmp;
    printf("%s\n", llvm::formatv("      alloc ok for {0}, size:{1}",
                                  name, size).str().c_str());
  }

  return true;
}

bool lmem_alloc::alloc2(int ts_idx, int slice_idx, const std::string& name, Value value, int addr, int size) {
  printf("%s\n", llvm::formatv("      start alloc for {0}, size:{1}, slice_idx:{2}, addr:{3}", name, size, slice_idx, addr).str().c_str());
  int bidx = addr / (total_size/16);
  int end_bidx = (addr + size) / (total_size/16);
  std::vector<int> tmp_bank_id;
  for (int i = bidx; i <= end_bidx; i++) {
    tmp_bank_id.push_back(i);
  }
  for (int i = 0; i < size; i++) {
    lmem_buf[addr+i] = true;
  }

  reload_mem_struct tmp_mem_s;
  tmp_mem_s.addr = addr;
  tmp_mem_s.bank_id.assign(tmp_bank_id.begin(), tmp_bank_id.end());
  mem_struct mem_s;
  mem_s.addr = addr;
  mem_s.bank_id.assign(tmp_bank_id.begin(), tmp_bank_id.end());
  mem_s.size = size;
  mem_s.value = value;
  mem_s.slice_idx = slice_idx;
  mem_s.type = 2;
  std::string key = llvm::formatv("{0}_slice{1}", name, slice_idx).str();
  mem_dict[key] = mem_s;

  his_mem_struct his_mem_s;
  his_mem_s.size = size;
  his_mem_s.value = value;
  his_mem_s.slice_idx = slice_idx;
  his_mem_s.vec_reload_addr.push_back(std::make_pair(0xFFFFBBBB, tmp_mem_s));
  vec_mem_alloc_his[key] = his_mem_s;
  return true;
}


void lmem_alloc::free(const std::string& key, std::vector<std::pair<int,int>>* vec_pre_ts_free_mem) {
  if (mem_dict.find(key) != mem_dict.end()) {
    auto mem_s = mem_dict[key];
    printf("      free %s, addr:%d, size:%d\n", key.c_str(), mem_s.addr, mem_s.size);
    for (int i = 0; i < mem_s.size; i++) {
      assert(lmem_buf[mem_s.addr + i]);
      lmem_buf[mem_s.addr + i] = false;
    }
    mem_dict.erase(key);
    if (vec_pre_ts_free_mem) {
      (*vec_pre_ts_free_mem).push_back(std::make_pair(mem_s.addr, mem_s.addr + mem_s.size));
    }
  }
}

bool lmem_alloc::get_mem_struct(const std::string& key, mem_struct& mem_s) {
  if (mem_dict.find(key) != mem_dict.end()) {
    mem_s = mem_dict[key];
    return true;
  }
  return false;
}

std::vector<int> lmem_alloc::get_bank(const std::string& name) {
  std::vector<int> tmp;
  auto iter = mem_dict.find(name);
  if (iter != mem_dict.end()) {
    auto bank_id = iter->second.bank_id;
    tmp.assign(bank_id.begin(), bank_id.end());
    return tmp;
  }
  return tmp;
}

ILPTimeStep::ILPTimeStep(const LgInfo& group_info, int sec_per_core)
  :_group_info(group_info), solver(MPSolver::CreateSolver("SCIP")) {
 // Create the mip solver with the SCIP backend.
  if (!solver) {
    llvm::errs() << "SCIP solver unavailable.\n";;
  }
  ts_count = sec_per_core*_group_info.group_ops.size() + 2;
  slice_num = sec_per_core;
  for(int i = 0; i < ts_count; i++) {
    cycle_contrains.push_back(std::vector<std::pair<int, std::string>>());
  }
  for(int i = 0; i < ts_count; i++) {
    mem_contrains.push_back(std::vector<std::pair<int, std::string>>());
  }
  for(int i = 0; i < ts_count; i++) {
    timestep_table_.push_back(TimestepRow2());
  }

  objective = solver->MutableObjective();
  objective->SetMinimization();
  if (module::isDebugCmdEnable("ILPTimeStep_detail_log")) {
    detail_log = true;
  }
}

ILPTimeStep::~ILPTimeStep()
{
}

void ILPTimeStep::addValueInfo(int slice_idx, Value value, std::string varName) {
  if (mapValueInfo.find(value) == mapValueInfo.end()) {
    std::map<int, std::vector<std::string>> tmp;
    mapValueInfo[value] = tmp;
  } else {
    if (mapValueInfo[value].find(slice_idx) == mapValueInfo[value].end()) {
      std::vector<std::string> tmp;
      mapValueInfo[value][slice_idx] = tmp;
    }
  }
  mapValueInfo[value][slice_idx].push_back(varName); //确保时隙后面的变量放在最后面
}

void ILPTimeStep::addBinaryVar(int ts_idx, int slice_idx, int mode, std::string varName, Value value, tensor_info_t& info, int64_t lmem_bytes) {
  assert(solver != nullptr);
  MPVariable* x = solver->MakeIntVar(0, 1, varName);
  ilp_var_info var_info;
  var_info.ts_idx = ts_idx;
  var_info.slice_idx = slice_idx;
  var_info.mode = mode;
  var_info.ilp_var = x;
  var_info.tensor_info = info;
  mapILPVarInfo[varName] = var_info;

  ts_var_t tmp;
  tmp.varName = varName;
  tmp.value = value;
  tmp.info = info;
  tmp.lmem_bytes = align(lmem_bytes, 64);
  tmp.slice_idx = slice_idx;
  timestep_table_[ts_idx].vec_ts_var.push_back(tmp);
  addValueInfo(slice_idx, value, varName);
}

void ILPTimeStep::addTensorSize(int ts_idx, Value value, int lmem_size) {
  assert (timestep_table_[ts_idx].vec_op_infos.size() == 1);
  timestep_table_[ts_idx].vec_op_infos[0].tensor_size[value] = align(lmem_size, 64);
}

void ILPTimeStep::addTensorCycle(int ts_idx, Value value, int cycle) {
  assert (timestep_table_[ts_idx].vec_op_infos.size() == 1);
  timestep_table_[ts_idx].vec_op_infos[0].load_tensor_cycles[value] = cycle;
}

void ILPTimeStep::addTimestepGdmaCycle(int ts_idx, int cycle, std::string varName) {
  llvm::errs() << "addTimestepGdmaCycle, ts_idx:"<<ts_idx<< ", cycle:"<<cycle<< ", varName: "<<varName<<"\n";
  cycle_contrains[ts_idx].push_back(std::make_pair(cycle, varName));
}

void ILPTimeStep::addOpInfo(int ts_idx, Operation* op, int buffer_size, int mem_size_for_load, int bdc_cycle) {
  assert (timestep_table_[ts_idx].vec_op_infos.size() == 0);
  op_related_info_t tmp;
  tmp.op = op;
  tmp.slice_idx = (ts_idx -1)/_group_info.group_ops.size();
  tmp.mem_size_for_load = mem_size_for_load;
  tmp.buffer_size = align(buffer_size, 64);
  tmp.bdc_cycle = bdc_cycle;
  timestep_table_[ts_idx].vec_op_infos.push_back(tmp);
}

void ILPTimeStep::addTimestepMemUse(int ts_idx, int mem_size, std::vector<std::string>& varNames) {
  llvm::errs() << "addTimestepMemUse, ts_idx:"<<ts_idx<< ", mem_size:"<<mem_size<<"\n";
  for (auto varName: varNames) {
    llvm::errs() << "      varName: "<<varName<<"\n";
    mem_contrains[ts_idx].push_back(std::make_pair(mem_size, varName));
  }
}

MPVariable* ILPTimeStep::getMPVarByName(std::string varName) {
  if (mapILPVarInfo.find(varName) != mapILPVarInfo.end()) {
    return mapILPVarInfo[varName].ilp_var;
  }
  assert(false);
}

void ILPTimeStep::resideOpInValue(Operation* op, Value value) {
  if (reside_in_tensor.find(op) == reside_in_tensor.end()) {
    std::vector<Value> tmp;
    reside_in_tensor[op] = tmp;
  }
  reside_in_tensor[op].push_back(value);
}

void ILPTimeStep::addNewOutIntoReturnOp(std::vector<std::string> var_names, Value value) {
  if (values_need_store_to_grpout.find(value) == values_need_store_to_grpout.end()) {
    values_need_store_to_grpout[value] = var_names;
  }
}

void ILPTimeStep::addConstraint(double lb, double ub, std::vector<std::pair<int, MPVariable*>> coeff_var_items, bool test) {
  assert(coeff_var_items.size() > 0);
  MPConstraint* c0 = solver->MakeRowConstraint(lb, ub, "");
  constraint_info tmp;
  tmp.lb = lb;
  tmp.ub = ub;
  tmp.cons_var = c0;
  for (auto it: coeff_var_items) {
    c0->SetCoefficient(it.second, it.first);
    tmp.coeff_var_items.push_back(it);
  }
  vec_constraints.push_back(tmp);
  llvm::errs() <<"m_constraint_idx:"<<m_constraint_idx++<<"\n";
  // if (test) {
  if (false) {
    MPSolver::ResultStatus result_status = solver->Solve();
    if (result_status != MPSolver::OPTIMAL && result_status != MPSolver::FEASIBLE) {
      llvm::errs() << "after addConstraint, the problem does not have an optimal or feasible solution!, result_status:"<<(int)result_status<<"\n";
      // int idx = 0;
      // vec_constraints.pop_back();
      // for (auto it: vec_constraints) {
      //   it.cons_var->SetBounds(-MPSolver::infinity(), MPSolver::infinity());
      //   result_status = solver->Solve();
      //   if (result_status == MPSolver::OPTIMAL || result_status == MPSolver::FEASIBLE) {
      //     llvm::errs() << "success, maybe error m_constraint_idx:"<<idx<<"\n";
      //     showConstraintInfo(vec_constraints[idx]);
      //     showAllConstraint();
      //     break;
      //   }
      //   it.cons_var->SetBounds(it.lb, it.ub);
      //   idx++;
      // }
      // exit(0);
    } else {
      llvm::errs() <<"test pass\n";
    }
  }
}

void ILPTimeStep::showAllConstraint() {
  llvm::errs() <<"showAllConstraint:\n";

  std::map<MPVariable*, std::string> only_one_var_warning;
  for (auto it: vec_constraints) {
    if (it.coeff_var_items.size() == 1) {
      if (it.lb == it.ub) {
        if (it.lb == 1.0) {
          only_one_var_warning[it.coeff_var_items[0].second] = " const_1";
        } else if (it.lb == .0) {
          only_one_var_warning[it.coeff_var_items[0].second] = " const_0";
        }
      }
    }
  }

  for (auto& it: vec_constraints) {
    showConstraintInfo(it, only_one_var_warning);
  }
}


void ILPTimeStep::showConstraintInfo(constraint_info& cons_info, std::map<MPVariable*, std::string>& only_one_var_warning) {
  int i = -1;
  auto itr = std::find(vec_constraints.begin(), vec_constraints.end(), cons_info);
  if (itr != vec_constraints.end()) {
      i = std::distance(vec_constraints.begin(), itr);
  }
  double max_int = MPSolver::infinity();
  double min_int = -MPSolver::infinity();
  std::string str;
  auto& it = cons_info;
  llvm::errs() <<"MakeRowConstraint, lb:"<<(int)it.lb<<" ub:"<<(int)it.ub<<", m_constraint_idx:"<<i
                 <<", var num:"<<it.coeff_var_items.size()<<", coeff and var:\n";
  if (it.lb == it.ub) {
    for (auto it2: it.coeff_var_items) {
      str = it2.second->name();
      if (only_one_var_warning.find(it2.second) != only_one_var_warning.end()) {
        str += only_one_var_warning[it2.second];
      }
      llvm::errs() <<"  "<<it2.first<<" * "<<str<<"\n";
    }
    llvm::errs() <<"  == "<<(int)it.lb<<"\n";
  } else {
    if (it.lb != min_int && it.ub == max_int) {
      llvm::errs() <<"  "<<(int)it.lb<<"  <\n";
      for (auto it2: it.coeff_var_items) {
        str = it2.second->name();
        if (only_one_var_warning.find(it2.second) != only_one_var_warning.end()) {
          str += only_one_var_warning[it2.second];
        }
        llvm::errs() <<"  "<<it2.first<<" * "<<str<<"\n";
      }
    }
    if (it.lb == min_int && it.ub != max_int) {
      for (auto it2: it.coeff_var_items) {
        str = it2.second->name();
        if (only_one_var_warning.find(it2.second) != only_one_var_warning.end()) {
          str += only_one_var_warning[it2.second];
        }
        llvm::errs() <<"  "<<it2.first<<" * "<<str<<"\n";
      }
      llvm::errs() <<"  < "<<(int)it.ub<<"\n";
    }
    if (it.lb != min_int && it.ub != max_int) {
      llvm::errs() <<"  "<<(int)it.lb<<"  <\n";
      for (auto it2: it.coeff_var_items) {
        str = it2.second->name();
        if (only_one_var_warning.find(it2.second) != only_one_var_warning.end()) {
          str += only_one_var_warning[it2.second];
        }
        llvm::errs() <<"  "<<it2.first<<" * "<<str<<"\n";
      }
      llvm::errs() <<"  < "<<(int)it.ub<<"\n";
    }
  }
}

void ILPTimeStep::showRunInfo() {
  int i = 0;
  llvm::errs() << "showRunInfo:\n";;
  for (auto &itr: timestep_table_new) {
    llvm::errs() << "-------------------ts"<<i<<"--------------------\n";;
    int cycle = 0, total_cycle = 0;
    for (auto &itr2: itr.vec_ts_var) {
      if (mapILPVarInfo[itr2.varName].ilp_var->solution_value() == 1) {
        for (auto itr3: cycle_contrains_new[i]) {
          if (itr3.second == itr2.varName) {
            cycle = itr3.first;
            break;
          }
        }
        llvm::errs() <<"  dma var, name: " << itr2.varName <<", cycle:"<<cycle<<"\n";
        if (map_reside_value_info.find(itr2.value) == map_reside_value_info.end()) {
          total_cycle += cycle;
        }
      }
    }

    for (auto itr2: itr.vec_op_infos) {
      auto outs = get_output_values(itr2.op);
      llvm::errs() <<"  op name: " << module::getName(outs[0]).str()
                <<" , cycle:"<<itr2.bdc_cycle<<", free mem_size:"<<itr2.mem_size_for_load<<"\n";
    }
    i++;
  }
}

void ILPTimeStep::addRowConstraint(int ts_idx, Value load_tensor, std::vector<std::string> var_names) {
  assert(solver != nullptr);
  assert(var_names.size() > 0);

  llvm::errs() <<"ts_idx:"<<ts_idx<<", addRowConstraint:\n";
  std::vector<std::pair<int, MPVariable*>> coeff_var_items;
  for (auto var_name: var_names) {
    llvm::errs() << "      var_name: "<<var_name<<"\n";
    coeff_var_items.push_back(std::make_pair(1, mapILPVarInfo[var_name].ilp_var));
  }
  // if (var_names.size() > 1) {
  //   // addConstraint(1, 1, coeff_var_items);
  // }
  addConstraint(1, 1, coeff_var_items);

  if (ts_idx >= 0) {
    auto& need_load_var = timestep_table_[ts_idx].vec_op_infos[0].need_load_var;
    if (need_load_var.find(load_tensor) == need_load_var.end()) {
      std::vector<std::string> null_names;
      need_load_var[load_tensor] = null_names;
    }
    need_load_var[load_tensor].assign(var_names.begin(), var_names.end());
  }
}

void ILPTimeStep::setVarExpectValue(std::string var_name, int expect_value) {
  std::vector<std::pair<int, MPVariable*>> coeff_var_items;
  coeff_var_items.push_back(std::make_pair(1, mapILPVarInfo[var_name].ilp_var));
  addConstraint(expect_value, expect_value, coeff_var_items);
}

bool ILPTimeStep::run() {
  assert(solver != nullptr);
  // int max_int = (int)MPSolver::infinity();
  // int min_int = (int)-MPSolver::infinity();
  if (detail_log) {
    showAllConstraint();
    solver->EnableOutput();
  }
  llvm::errs() << "solve start\n";
  MPSolver::ResultStatus result_status = solver->Solve();

  // Check that the problem has an optimal solution.
  if (result_status != MPSolver::OPTIMAL && result_status != MPSolver::FEASIBLE) {
    llvm::errs() << "The problem does not have an optimal or feasible solution!, result_status:"<<(int)result_status<<"\n";

    // int idx = 0;
    // llvm::errs() << "start checking1\n";
    // for (auto it: vec_constraints) {
    //   it.cons_var->SetBounds(-MPSolver::infinity(), MPSolver::infinity());
    //   result_status = solver->Solve();
    //   if (result_status == MPSolver::OPTIMAL || result_status == MPSolver::FEASIBLE) {
    //     llvm::errs() << "success, error m_constraint_idx:"<<idx++<<"\n";
    //     showRunInfo();
    //     return true;
    //   }
    //   it.cons_var->SetBounds(it.lb, it.ub);
    // }

    // llvm::errs() << "start checking2\n";
    // idx = 0;
    // for (auto it: vec_constraints) {
    //   // if (it.lb != it.ub) {
    //   //   it.cons_var->SetBounds(-MPSolver::infinity(), MPSolver::infinity());
    //   // }
    //   it.cons_var->SetBounds(-MPSolver::infinity(), MPSolver::infinity());
    // }
    // for (auto it: vec_constraints) {
    //   it.cons_var->SetBounds(it.lb, it.ub);
    //   result_status = solver->Solve();
    //   if (result_status != MPSolver::OPTIMAL && result_status != MPSolver::FEASIBLE) {
    //     llvm::errs() << "find error, m_constraint_idx:"<<idx++<<"\n";
    //     break;
    //   }
    // }
    return false;
  }

  llvm::errs() << "Solution:\n";;
  llvm::errs() << "Objective value = " << objective->Value()<<"\n";

  // for (auto itr = mapILPVarInfo.begin(); itr != mapILPVarInfo.end(); ++itr) {
  //   llvm::errs() <<"var name: " << itr->first << ", value: " << itr->second.ilp_var->solution_value()<<"\n";
  // }

  llvm::errs() << "\nAdvanced usage:\n";;
  llvm::errs() << "Problem solved in " << solver->wall_time() << " milliseconds\n";;
  llvm::errs() << "Problem solved in " << solver->iterations() << " iterations\n";;
  llvm::errs() << "Problem solved in " << solver->nodes()
            << " branch-and-bound nodes\n";;
  return true;
}


void ILPTimeStep::addSliceNcdhwSteps(int core_id, std::vector<int64_t> ncdhw) {
  if (ncdhw_steps.find(core_id) == ncdhw_steps.end()) {
    std::vector<std::vector<int64_t>> tmp;
    ncdhw_steps[core_id] = tmp;
  }
  ncdhw_steps[core_id].push_back(ncdhw);
}

void ILPTimeStep::showTimeStepInfo(int debug_cmd) {
  llvm::errs() << "-------------------mem_contrains_info, after merge--------------------\n";;
  for(int i = 0; i < ts_count; i++) {
    llvm::errs() << "-------------------ts"<<i<<"--------------------\n";;
    if (!(i == 0 || i == ts_count - 1)) {
      for (auto it:timestep_table_new[i].vec_op_infos)
        llvm::errs() <<"op_name:"<<module::getName(it.op)<<"\n";
    }
    for (auto it: mem_contrains_new[i]) {
      llvm::errs() <<"  "<<it.first<<" * "<<it.second<<"\n";
    }
  }

  llvm::errs() << "-------------------cycle_contrains_info, after merge--------------------\n";;
  for(int i = 0; i < ts_count; i++) {
    llvm::errs() << "-------------------ts"<<i<<"--------------------\n";;
    if (!(i == 0 || i == ts_count - 1)) {
      for (auto it:timestep_table_new[i].vec_op_infos)
        llvm::errs() <<"op_name:"<<module::getName(it.op)<<"\n";
    }
    for (auto it: cycle_contrains_new[i]) {
      llvm::errs() <<"  "<<it.first<<" * "<<it.second<<"\n";
    }
  }
}

bool ILPTimeStep::merge_small_cycle_op(TensorInfo& tensor_infos) {
  return true;
  // llvm::errs() << "-------------------mem_contrains_info, before merge--------------------\n";;
  // for(int i = 0; i < ts_count; i++) {
  //   llvm::errs() << "-------------------ts"<<i<<"--------------------\n";;
  //   if (!(i == 0 || i == ts_count - 1)) {
  //     for (auto it:timestep_table_[i].vec_op_infos)
  //       llvm::errs() <<"op_name:"<<module::getName(it.op)<<"\n";
  //   }
  //   for (auto it: mem_contrains[i]) {
  //     llvm::errs() <<"  "<<it.first<<" * "<<it.second<<"\n";
  //   }
  // }

  // llvm::errs() << "\n-------------------cycle_contrains_info, before merge--------------------\n";;
  // for(int i = 0; i < ts_count; i++) {
  //   llvm::errs() << "-------------------ts"<<i<<"--------------------\n";;
  //   if (!(i == 0 || i == ts_count - 1)) {
  //     for (auto it:timestep_table_[i].vec_op_infos)
  //       llvm::errs() <<"op_name:"<<module::getName(it.op)<<"\n";
  //   }
  //   for (auto it: cycle_contrains[i]) {
  //     llvm::errs() <<"  "<<it.first<<" * "<<it.second<<"\n";
  //   }
  // }

  int dma_cycle = -1, merge_start = 0, min_pos = 0;
  std::vector<Value> big_load_tensor;
  llvm::errs() << "merge_small_cycle_op starting\n";
  for(int i = ts_count - 1; i >= 0; i--) {
    if (!(i == 0 || i == ts_count - 1)) {
      // int slice_idx = timestep_table_[i].vec_op_infos[0].slice_idx;
      auto op = timestep_table_[i].vec_op_infos[0].op;
      llvm::errs() << "ts:"<<i<< ", op:"<<module::getName(op).str()
                   << ", type:"<<op->getName().getStringRef().str()<<"\n";
      do {
        if (dma_cycle == -1) {
          if (isa<tpu::Conv2DOp>(op)) {
            auto in = op->getOperand(1);
            big_load_tensor.push_back(in);
            dma_cycle = timestep_table_[i].vec_op_infos[0].load_tensor_cycles[in];
            llvm::errs() <<"weight load cycle:"<<dma_cycle<<"\n";
            merge_start = i - 1;
          } else {
            break;
          }
        } else {
          merge_start = i;
          llvm::errs() << "ts:"<<i<< " achor dma_cycle:"<<dma_cycle<<" for merged op\n";
        }

        std::vector<int> cycle_sum;
        int sum = 0;
        for (int k = 0; k < 4; k++) { //最多连续4个op合并在一起
          if (merge_start - k < 1) //ts0没有op
            break;
          auto bdc_cycle = timestep_table_[merge_start - k].vec_op_infos[0].bdc_cycle;
          sum += bdc_cycle;
          llvm::errs() << "merge op"<<merge_start - k<<", bdc_cycle:"<<bdc_cycle<<", sum:"<<sum<<"\n";
          cycle_sum.push_back(std::abs(dma_cycle - sum));
        }
        min_pos = 0;
        if (cycle_sum.size() > 1) {
          auto min_cycle = *std::min_element(cycle_sum.begin(), cycle_sum.end()); //找到相差最小的
          auto it2 = std::find(cycle_sum.begin(), cycle_sum.end(), min_cycle);
          if (it2 != cycle_sum.end()) {
            min_pos = std::distance(cycle_sum.begin(), it2);
            if (min_pos > 0) {
              llvm::errs() << "find min_pos:"<<min_pos<<"\n";
              auto op_name = replaceChars_for_dot(module::getName(op).str());
              for (int m = 1; m <= min_pos; m++) { //load类型变量，保留融合op中最后一个ts(merge_start)的变量设置，其他ts的变量固定为0
                for (auto it3: cycle_contrains[merge_start - m]) {
                  int64_t mode2 = mapILPVarInfo[it3.second].tensor_info.mode2;
                  if (mode2&TIMESTEP2_LOAD || (mode2&TIMESTEP2_STORE_AND_LOAD && mapILPVarInfo[it3.second].mode == 1)) {
                    llvm::errs() << "for loadVar, setVarExpectValue:"<<it3.second<<" to const_0\n";
                    setVarExpectValue(it3.second, 0);
                  }
                }
              }
              for (int m = 0; m < min_pos; m++) { //store类型变量，保留融合op中第1个ts的变量设置，其他ts的变量固定为0
                for (auto it3: cycle_contrains[merge_start - m]) {
                  int64_t mode2 = mapILPVarInfo[it3.second].tensor_info.mode2;
                  if (mode2&TIMESTEP2_STORE || (mode2&TIMESTEP2_STORE_AND_LOAD && mapILPVarInfo[it3.second].mode == 0)) {
                    llvm::errs() << "for storeVar, setVarExpectValue:"<<it3.second<<" to const_0\n";
                    setVarExpectValue(it3.second, 0);
                  }
                }
              }
              // for (auto t: big_load_tensor) {
              //   if (mapValueInfo.find(t) != mapValueInfo.end()) {
              //     for (auto it: mapValueInfo[t][slice_idx]) {
              //       if (mapILPVarInfo[it].ts_idx == merge_start) {
              //         setVarExpectValue(it, 1);
              //         llvm::errs() << "setVarExpectValue:"<<it<<" to const_1\n";
              //         break;
              //       }
              //     }
              //   }
              // }
              big_load_tensor.clear();
              TimestepRow2 tmp;
              std::vector<Value> searched_value;
              dma_cycle = 0;
              for (int m = 0; m <= min_pos; m++) {
                auto timestep_row = timestep_table_[merge_start - m];
                auto op = timestep_row.vec_op_infos[0].op;
                auto load_tensor_cycles = timestep_row.vec_op_infos[0].load_tensor_cycles;
                for (const auto &res : llvm::enumerate(op->getOperands())) {
                  // if (reside_in_tensor.find(op) != reside_in_tensor.end()) {
                  //   auto tensors = reside_in_tensor[op];
                  //   if (find(tensors.begin(), tensors.end(), res.value()) != tensors.end()) {
                  //     llvm::errs() <<module::getName(res.value()).str()<<" will_reside\n";
                  //     continue;
                  //   }
                  // }
                  if (load_tensor_cycles.find(res.value()) != load_tensor_cycles.end()) {
                    dma_cycle += load_tensor_cycles[res.value()];
                    big_load_tensor.push_back(res.value());
                  }
                }

                tmp.vec_op_infos.push_back(timestep_row.vec_op_infos[0]);
                for (auto it: timestep_row.vec_ts_var) {
                  auto it3 = std::find(searched_value.begin(), searched_value.end(), it.value);
                  if (it3 == searched_value.end()) {
                    // llvm::errs() << "add "<<it.varName<<" to new vec_ts_var\n";
                    tmp.vec_ts_var.push_back(it);
                    searched_value.push_back(it.value);
                  }
                }
              }
              reverse(tmp.vec_op_infos.begin(), tmp.vec_op_infos.end());
              if (merge_start == i - 1) {
                timestep_table_new.push_back(timestep_table_[i]);
                llvm::errs() << "i:"<<i<< ", add "<<module::getName(timestep_table_[i].vec_op_infos[0].op).str()<<" to timestep_table_new\n";
              }
              timestep_table_new.push_back(tmp);

              //更新cycle约束、内存约束
              std::vector<std::pair<int, std::string>> new_cycle;
              std::vector<std::pair<int, std::string>> new_mem;
              for (auto it2: cycle_contrains[merge_start]) {
                // llvm::errs() << "add "<<it2.second<<" to new_cycle\n";
                new_cycle.push_back(it2);
              }

              for (auto it2: mem_contrains[merge_start]) {
                // llvm::errs() << "add "<<it2.second<<" to new_mem\n";
                new_mem.push_back(it2);
              }

              if (merge_start == i - 1) {
                cycle_contrains_new.push_back(cycle_contrains[i]);
                mem_contrains_new.push_back(mem_contrains[i]);
              }
              cycle_contrains_new.push_back(new_cycle);
              mem_contrains_new.push_back(new_mem);
              if (merge_start == i - 1) {
                i -= min_pos + 1;
              } else {
                i -= min_pos;
              }
              break; //合并完成
            }
          }
        }
        dma_cycle = -1;
      } while(false);
      if (min_pos > 0) {
        //合并发生，在前面添加合并ts配置信息
        continue;
      }
      llvm::errs() << "i:"<<i<< ", add2 "<<module::getName(timestep_table_[i].vec_op_infos[0].op).str()<<" to timestep_table_new\n";
    }
    timestep_table_new.push_back(timestep_table_[i]);
    cycle_contrains_new.push_back(cycle_contrains[i]);
    mem_contrains_new.push_back(mem_contrains[i]);
  }
  reverse(timestep_table_new.begin(), timestep_table_new.end());
  reverse(cycle_contrains_new.begin(), cycle_contrains_new.end());
  reverse(mem_contrains_new.begin(), mem_contrains_new.end());
  ts_count = timestep_table_new.size();
  llvm::errs() << "new ts_count:"<<ts_count<<"\n";
  return true;
}

bool ILPTimeStep::prepare(TensorInfo& tensor_infos) {
  if (timestep_table_new.size() == 0) {
    timestep_table_new.clear();
    cycle_contrains_new.clear();
    mem_contrains_new.clear();
    timestep_table_new.assign(timestep_table_.begin(), timestep_table_.end());
    cycle_contrains_new.assign(cycle_contrains.begin(), cycle_contrains.end());
    mem_contrains_new.assign(mem_contrains.begin(), mem_contrains.end());
  }
  // showTimeStepInfo();
  assert(solver != nullptr && objective != nullptr);
  std::vector<std::pair<std::string, MPVariable*>> objective_var;
  for(int i = 0; i < ts_count; i++) {
    std::string var_name = llvm::formatv("sum_var_ts{0}", i);
    MPVariable* x = solver->MakeIntVar(-MPSolver::infinity(), MPSolver::infinity(), var_name);
    objective_var.push_back(std::make_pair(var_name, x));

    std::string abs_var_name = llvm::formatv("sum_var_abs_ts{0}", i);
    MPVariable* x_abs = solver->MakeIntVar(-MPSolver::infinity(), MPSolver::infinity(), abs_var_name);
    objective->SetCoefficient(x_abs, 1);

    std::vector<std::pair<int, MPVariable*>> coeff_var_items;
    coeff_var_items.push_back(std::make_pair(1, x_abs));
    coeff_var_items.push_back(std::make_pair(1, x));
    addConstraint(0, MPSolver::infinity(), coeff_var_items);

    coeff_var_items.clear();
    coeff_var_items.push_back(std::make_pair(1, x_abs));
    coeff_var_items.push_back(std::make_pair(-1, x));
    addConstraint(0, MPSolver::infinity(), coeff_var_items);
  }

  llvm::errs() << ">>>>> add cycle_contrains:\n";
  std::string op_name;
  for(int i = 0; i < ts_count; i++) {
    double bdc_cycle = 0;
    op_name = "null";
    if (!(i == 0 || i == ts_count - 1)) {
      for (auto it: timestep_table_new[i].vec_op_infos) {
        op_name = module::getName(it.op).str() + "__" + op_name;
        bdc_cycle += it.bdc_cycle;
      }
      if (timestep_table_new[i].vec_op_infos.size() > 0) {
        op_name = op_name.substr(0, op_name.size()-6);
      }
      if (timestep_table_new[i].vec_op_infos.size() > 1) {
        op_name = "super_op_" + op_name;
      }
    }
    std::vector<std::pair<int, MPVariable*>> coeff_var_items;
    coeff_var_items.push_back(std::make_pair(1, objective_var[i].second));
    llvm::errs() <<"  i:"<<i<<", op_name:"<<op_name<<", bdc_cycle:"<<(int)bdc_cycle<<"\n";
    for (auto it: cycle_contrains_new[i]) {
      coeff_var_items.push_back(std::make_pair(it.first, mapILPVarInfo[it.second].ilp_var));
    }
    addConstraint(bdc_cycle, bdc_cycle, coeff_var_items);
  }

  llvm::errs() << ">>>>> add mem_contrains:\n";
  for(int i = 0; i < ts_count; i++) {
    if (i == 0 || i == ts_count - 1) {
      continue;
    }
    llvm::errs() <<" ts:"<<i<<"\n";
    if (mem_contrains_new[i].size() > 0) {
      op_name = "null";
      std::vector<int> op_free_size;
      int ts_reside_value_size = 0;
      for (auto it: timestep_table_new[i].vec_op_infos) {
        auto name = module::getName(it.op).str();
        llvm::errs() <<"  op:"<<name<<"\n";
        op_name = name + "__" + op_name;
        int op_reside_value_size = 0;
        for (auto in : get_input_values(it.op)) {
          if (reside_in_tensor.find(it.op) != reside_in_tensor.end()) {
            auto tensors = reside_in_tensor[it.op];
            if (find(tensors.begin(), tensors.end(), in) != tensors.end()) {
              if (detail_log)
                printf("   reside_in_tensor\n");
              op_reside_value_size += it.tensor_size[in];
              ts_reside_value_size += it.tensor_size[in];
            }
          }

          if (mapValueInfo.find(in) != mapValueInfo.end()) {
            for (auto it2:mapValueInfo[in][it.slice_idx]) {
              int64_t mode2 = mapILPVarInfo[it2].tensor_info.mode2; //back确保取到stay_free_var？？
              if (mode2&TIMESTEP2_STORE || mode2&TIMESTEP2_STORE_ONLY_FREE || //store应该是可以和op并行的？bank冲突吗? todo
                (mode2&TIMESTEP2_STORE_AND_LOAD && mapILPVarInfo[it2].mode == 0)) {
                op_reside_value_size += it.tensor_size[in];
                if (detail_log)
                  llvm::errs() <<"   in:"<<module::getName(in).str()<<" have store, tensor_size:"<<it.tensor_size[in]<<"\n";
                ts_reside_value_size += it.tensor_size[in];
                break;
              }
            }
          }
        }
        if (detail_log)
          llvm::errs() <<"  op_reside_value_size:"<<op_reside_value_size<<"\n";
        op_free_size.push_back(it.mem_size_for_load - op_reside_value_size);
      }
      // llvm::errs() <<"  ts_reside_value_size:"<<ts_reside_value_size<<"\n";
      auto max_free_size = *std::max_element(op_free_size.begin(), op_free_size.end()) + ts_reside_value_size;
      assert(max_free_size > 0);

      if (timestep_table_new[i].vec_op_infos.size() > 0) {
        op_name = op_name.substr(0, op_name.size()-6);
      }
      if (timestep_table_new[i].vec_op_infos.size() > 1) {
        op_name = "super_op_" + op_name;
      }
      llvm::errs() <<"  op_name:"<<op_name<<":\n";
      std::vector<std::pair<int, MPVariable*>> coeff_var_items;
      for (auto it: mem_contrains_new[i]) {
        coeff_var_items.push_back(std::make_pair(it.first, mapILPVarInfo[it.second].ilp_var));
      }
      addConstraint(0, max_free_size, coeff_var_items, true);
    }
  }
  return true;
}

bool ILPTimeStep::mem_alloc(mem_alloc_status& alloc_status, std::vector<std::pair<Value, int64_t>>& value_size, TensorInfo& tensor_infos) {
  llvm::errs() << "mem_alloc start:\n";
  AutoIndent auto_indent;
  lmem_alloc_ptr = std::make_shared<lmem_alloc>(_group_info.group_banked_tensors, this, ts_count);
  // value_size.clear();
  if (value_size.size() > 0) {
    int min_always_free_mem_size = lmem_alloc_ptr->total_size;
    for(int i = 0; i < ts_count; i++) {
      if (i == 0 || i == ts_count - 1) {
        continue;
      }
      if (mem_contrains_new[i].size() > 0) {
        int size = 0;
        for (auto it: mem_contrains_new[i]) {
          if (mapILPVarInfo[it.second].ilp_var->solution_value() == 1) {
            size += it.first; //累加当前时隙存在的tensor的内存占用量
          }
        }
        for (auto it: timestep_table_new[i].vec_op_infos) {
          double free_mem_size = it.mem_size_for_load - size; //每个op的可加载内存减去size
          if (free_mem_size < min_always_free_mem_size) {
            min_always_free_mem_size = free_mem_size;
          }
        }
      }
    }
    llvm::errs() << "min_always_free_mem_size:"<<min_always_free_mem_size<<"\n";

    llvm::errs() << "add reside_value:\n";
    if (min_always_free_mem_size > 0) {
      int addr = lmem_alloc_ptr->total_size - 16;
      for (auto itr: value_size) {
        if (min_always_free_mem_size > itr.second) {
          reside_value_info tmp;
          addr -= itr.second;
          addr = (addr/64)*64;
          tmp.addr = addr;
          tmp.size = itr.second;
          map_reside_value_info[itr.first] = tmp;
          min_always_free_mem_size -= itr.second;
          llvm::errs() << "  name:"<<module::getName(itr.first).str()<< ", addr:"<<addr<<"\n";
        } else {
          break;
        }
      }
    }
  }

  int i = 0;
  std::string name;
  bool ret = false;

  llvm::errs() << "show var value:\n";
  for (auto &itr: timestep_table_new) {
    llvm::errs() << "-------------------ts"<<i<<"--------------------\n";;
    int cycle = 0, total_cycle = 0;
    for (auto &itr2: itr.vec_ts_var) {
      if (mapILPVarInfo[itr2.varName].ilp_var->solution_value() == 1) {
        itr2.var_value = 1;
        for (auto itr3: cycle_contrains_new[i]) {
          if (itr3.second == itr2.varName) {
            cycle = itr3.first;
            break;
          }
        }
        llvm::errs() <<"  dma var, name: " << itr2.varName <<", cycle:"<<cycle<<"\n";
        if (map_reside_value_info.find(itr2.value) == map_reside_value_info.end()) {
          total_cycle += cycle;
        }
      }
    }

    for (auto itr2: itr.vec_op_infos) {
      auto outs = get_output_values(itr2.op);
      llvm::errs() <<"  op name: " << module::getName(outs[0]).str()
                <<" , cycle:"<<itr2.bdc_cycle<<", free mem_size:"<<itr2.mem_size_for_load<<"\n";
    }
    i++;
  }

  i = 0;
  llvm::errs() << "gen map_l2m_load2:\n";
  for (auto &itr: timestep_table_new) {
    llvm::errs() << "-------------------ts"<<i<<"--------------------\n";;
    for (auto itr2: itr.vec_op_infos) {
      for (auto itr3 = itr2.need_load_var.begin(); itr3 != itr2.need_load_var.end(); ++itr3) {
        if (map_reside_value_info.find(itr3->first) == map_reside_value_info.end()) {
          for (auto var: itr3->second) {
            if (mapILPVarInfo[var].ilp_var->solution_value() == 1) {
              name = module::getName(itr3->first).str();
              map_l2m_load2.insert(std::make_pair(std::make_pair(itr3->first, i), mapILPVarInfo[var].ts_idx));
              llvm::errs() <<"tensor name:"<<name<<", compute at pos"<<i<<", load at ts" << mapILPVarInfo[var].ts_idx<<"\n";
              break;
            }
          }
        }
      }
    }
    i++;
  }

  if (map_reside_value_info.size() > 0) {
    llvm::errs() << "cancel reside_value load:\n";;
    for(int i = 0; i < ts_count; i++) {
      //第1个slice的驻留权重加载不取消，后面的均取消，这样即能保证加载时的隐藏，又取消后面slice的加载，减少功耗
      if (i < ts_count - 2) {
        for (auto& it: timestep_table_new[i].vec_ts_var) {
          if (it.slice_idx > 0 && it.var_value == 1 && map_reside_value_info.find(it.value) != map_reside_value_info.end()) {
            it.var_value = 0;
            llvm::errs() << "  name:"<<module::getName(it.value).str()<< ", ts:"<<i<<"\n";
          }
        }
      }
    }
  }

  printf("deal weight pre dma load:\n");
  for(int i = 0; i < ts_count; i++) {
    for (auto it: timestep_table_new[i].vec_ts_var) {
      name = module::getName(it.value).str();
      if (it.info.mode2&TIMESTEP2_LOAD && it.var_value == 1) {
        if (map_reside_value_info.find(it.value) != map_reside_value_info.end()) {
          int addr = map_reside_value_info[it.value].addr;
          int size = map_reside_value_info[it.value].size;
          lmem_alloc_ptr->alloc2(i, it.slice_idx, name, it.value, addr, size);
        }
      }
    }
  }
  // int unused;
  // lmem_alloc_ptr->show_mem(unused, unused, unused);

  std::vector<mem_alloc_req_info> vec_mem_req;
  std::vector<std::pair<int,int>> vec_pre_ts_free_mem, vec_pre_ts_free_mem_pre;
  printf("start analog allocation\n");
  for(int i = 0; i < ts_count; i++) {
    printf(">>>ts:%d\n", i);
    // printf("  deal dma store for TIMESTEP2_STORE_ONLY_FREE:\n"); //为配合小op融合功能，改为在op执行后理解释放
    // if (i >= 2) {
    //   for (auto it: timestep_table_new[i].vec_ts_var) {//无需真的store搬出去，后面的其他load或op输出直接分配使用这些区域即可
    //     name = module::getName(it.value).str();
    //     if (it.info.mode2&TIMESTEP2_STORE_ONLY_FREE && it.var_value == 1) {
    //       lmem_alloc_ptr->free(convert_name_to_key(name, it.slice_idx));
    //     }
    //   }
    // }

    printf("  deal dma load:\n");
    if (i < ts_count - 2) {
      for (auto it: timestep_table_new[i].vec_ts_var) {
        name = module::getName(it.value).str();
        if (it.info.mode2&TIMESTEP2_LOAD && it.var_value == 1) {
          if (map_reside_value_info.find(it.value) == map_reside_value_info.end()) {
            mem_alloc_req_info tmp;
            tmp.slice_idx = it.slice_idx;
            tmp.name = name;
            tmp.value = it.value;
            tmp.size = it.lmem_bytes;
            vec_mem_req.push_back(tmp);
          }
        }
        if (it.info.mode2&TIMESTEP2_STORE_AND_LOAD && it.var_value == 1) {
          if (mapILPVarInfo[it.varName].mode == 1) {
            mem_alloc_req_info tmp;
            tmp.slice_idx = it.slice_idx;
            tmp.name = name;
            tmp.value = it.value;
            tmp.size = it.lmem_bytes;
            vec_mem_req.push_back(tmp);
          }
        }
      }
    }

    if (i > 0 && i < ts_count - 1) {
      printf("  deal tpu:\n");
      bool have_mem_dependent = false;
      for (auto it2: timestep_table_new[i].vec_op_infos) {
        auto outs = get_output_values(it2.op);
        name = module::getName(outs[0]).str();
        llvm::errs() << "    op name: "<<name<<"\n";
        int buffer_size = it2.buffer_size;
        if (buffer_size > 0) {
          Value tmp_value;
          mem_alloc_req_info tmp;
          tmp.slice_idx = it2.slice_idx;
          tmp.name = name + "_buffer";
          tmp.value = tmp_value;
          tmp.size = buffer_size;
          vec_mem_req.push_back(tmp);
        }

        for (auto out : outs) {
          mem_alloc_req_info tmp;
          tmp.slice_idx = it2.slice_idx;
          tmp.name = module::getName(out).str();
          tmp.value = out;
          tmp.size = it2.tensor_size[out];
          // if (tmp.size > 0) //训练图中maxpool的mask输出有时无需处理,why
          vec_mem_req.push_back(tmp);
        }

        bool sort_by_size = true;
        ret = lmem_alloc_ptr->alloc_multi(i, vec_mem_req, sort_by_size);
        if (!ret) {
          printf("      alloc_multi fail\n");
          return false;
        }
        if (i > 1 && !have_mem_dependent) { //复合op中某子op已与上一时戳有依赖，则不再重复检查
          for (auto it1: vec_mem_req) { //当前的请求不能与上一个时戳的任一释放有依赖
            mem_struct mem_s;
            auto key = convert_name_to_key(it1.name, it1.slice_idx);
            lmem_alloc_ptr->get_mem_struct(key, mem_s);
            for (auto it2: vec_pre_ts_free_mem_pre) {
              if (is_range_overlap(it2.first, it2.second, mem_s.addr, mem_s.addr + mem_s.size)) {
                have_mem_dependent = true;
                llvm::errs() << "         vec_mem_req, key:"<<key
                            << ", req start addr:"<<mem_s.addr<< ", end addr:"<<mem_s.addr + mem_s.size
                            << ", pre_ts start addr:"<<it2.first<< ", end addr:"<<it2.second<<"\n";
                // if (key == "826_buffer_slice0") {
                //   int unused;
                //   (void)lmem_alloc_ptr->show_mem(unused, unused, unused);
                //   // show_mem(unused, unused, unused);
                // }
                break;
              }
            }
          }
        }
        vec_mem_req.clear();

        printf("    deal input free:\n");
        for (auto in : get_input_values(it2.op)) {
          bool to_be_used = false;
          if (reside_in_tensor.find(it2.op) != reside_in_tensor.end()) {
            auto tensors = reside_in_tensor[it2.op];
            if (find(tensors.begin(), tensors.end(), in) != tensors.end()) {
              to_be_used = true;
              printf("       reside_in_tensor\n");
            }
          }
          if (mapValueInfo.find(in) != mapValueInfo.end()) {
            int max_ts = 0;
            for (auto it: mapValueInfo[in][it2.slice_idx]) {
              if (mapILPVarInfo[it].ilp_var->solution_value() == 1) {
                if (mapILPVarInfo[it].ts_idx > max_ts) {
                  max_ts = mapILPVarInfo[it].ts_idx;
                }
              }
            }
            bool valid = i < max_ts;
            for (auto it2:mapValueInfo[in][it2.slice_idx]) {
              int64_t mode2 = mapILPVarInfo[it2].tensor_info.mode2; //back确保取到stay_free_var？？
              if (valid && (mode2&TIMESTEP2_STORE || mode2&TIMESTEP2_STORE_ONLY_FREE || //store应该是可以和op并行的？bank冲突吗? todo
                (mode2&TIMESTEP2_STORE_AND_LOAD && mapILPVarInfo[it2].mode == 0))) {
                to_be_used = true;
                printf("       have store\n");
                break;
              }
            }
          }
          if (map_reside_value_info.find(in) != map_reside_value_info.end()) {
            to_be_used = true;
            printf("       reside_value\n");
          }
          name = module::getName(in).str();
          if (!to_be_used) {
            lmem_alloc_ptr->free(convert_name_to_key(name, it2.slice_idx), &vec_pre_ts_free_mem);
          } else {
            printf("        not need to free:%s\n", name.c_str());
          }
        }

        for (auto it3: it2.ada_var_for_free_mem) {//无需真的store搬出去，后面的其他load或op输出直接分配使用这些区域即可
          name = module::getName(it3.value).str();
          if (it3.info.mode2&TIMESTEP2_STORE_ONLY_FREE && it3.var_value == 1) {
            lmem_alloc_ptr->free(convert_name_to_key(name, it3.slice_idx));
          }
        }

        if (buffer_size > 0) {
          name = module::getName(outs[0]).str();
          lmem_alloc_ptr->free(convert_name_to_key(name + "_buffer", it2.slice_idx), &vec_pre_ts_free_mem);
        }
      }

      if (vec_pre_ts_free_mem_pre.size() > 0 && !have_mem_dependent) {
        printf("        ts:%d and ts:%d no mem dependent\n", i, i-1);
        // timestep_table_new[i].can_merge = true; //todo fix me
      }
    }

    printf("  deal dma store:\n");
    if (i > 1) {
      for (auto it: timestep_table_new[i].vec_ts_var) {//先store，后load //应该在最后store，本ts store时也不能给本时隙的load用
        name = module::getName(it.value).str();
        if (it.info.mode2&TIMESTEP2_STORE && it.var_value == 1) {
          lmem_alloc_ptr->free(convert_name_to_key(name, it.slice_idx), &vec_pre_ts_free_mem);
        }
        if (it.info.mode2&TIMESTEP2_STORE_AND_LOAD && it.var_value == 1) {
          if (mapILPVarInfo[it.varName].mode == 0) {
            lmem_alloc_ptr->free(convert_name_to_key(name, it.slice_idx), &vec_pre_ts_free_mem);
          }
        }
      }
    }
    vec_pre_ts_free_mem_pre.clear();
    vec_pre_ts_free_mem_pre.assign(vec_pre_ts_free_mem.begin(), vec_pre_ts_free_mem.end());
    vec_pre_ts_free_mem.clear();
  }
  return true;
}

} // namespace tpu
} // namespace tpu_mlir
