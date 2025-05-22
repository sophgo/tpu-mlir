//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/IlpTimeStep.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/TimeStepMethod.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>

namespace tpu_mlir {
namespace tpu {

bool SortByMemStruct(const std::pair<std::string, mem_struct> &v1,
                     const std::pair<std::string, mem_struct> &v2) {
  return v1.second.addr < v2.second.addr; //升序排列
}

bool SortByMemSize(const mem_alloc_req_info &v1, const mem_alloc_req_info &v2) {
  return v1.size > v2.size; //升序排列
}

inline std::string convert_name_to_key(const std::string &name, int slice_idx) {
  return llvm::formatv("{0}_slice{1}", name, slice_idx).str();
}

bool is_range_overlap(int start1, int end1, int start2, int end2) {
  if (std::max(start1, start2) < std::min(end1, end2)) {
    return true;
  }
  return false;
}

std::string vector_to_string(std::vector<int> vec_data) {
  std::string tmp = "[";
  for (auto it : vec_data) {
    tmp = tmp + std::to_string(it) + ",";
  }
  if (vec_data.size() > 0)
    tmp.pop_back();
  tmp = tmp + "]";
  return tmp;
}

l2mem_alloc::l2mem_alloc() {
  total_size = backend::BM168x::L2_SRAM_SIZE;
  lmem_buf = new bool[total_size];
  for (int i = 0; i < total_size; i++) {
    lmem_buf[i] = false;
  }
}

l2mem_alloc::~l2mem_alloc() { delete[] lmem_buf; }

bool l2mem_alloc::alloc(int slice_idx, const std::string &name, Value value,
                        int size) {
  int count = 0;
  int free_addr = -1;
  for (int i = 0; i < total_size; i++) {
    if (lmem_buf[i]) {
      free_addr = -1;
      count = 0;
    } else {
      if (free_addr == -1) {
        if (i % 64 == 0) {
          free_addr = i;
        }
      }
      if (free_addr == -1)
        continue;
      count++;
      if (count == size) {
        break;
      }
    }
  }
  if (count == size) {
    for (int i = 0; i < size; i++) {
      lmem_buf[free_addr + i] = true;
    }
    // fprintf(stderr, "%s\n", llvm::formatv("        alloc ok for {0},
    // free_addr:{1}", name, free_addr).str().c_str());
    mem_struct mem_s;
    mem_s.addr = free_addr;
    mem_s.size = size;
    mem_s.value = value;
    mem_s.slice_idx = slice_idx;
    mem_s.type = 0;
    std::string key = convert_name_to_key(name, slice_idx);
    mem_dict[key] = mem_s;

    reload_mem_struct tmp_mem_s;
    tmp_mem_s.addr = free_addr;
    his_mem_struct his_mem_s;
    his_mem_s.size = size;
    his_mem_s.value = value;
    his_mem_s.slice_idx = slice_idx;
    his_mem_s.vec_reload_addr.push_back(std::make_pair(0xFFFFBBBB, tmp_mem_s));
    vec_mem_alloc_his[key] = his_mem_s;
    return true;
  }

  return false;
}

bool l2mem_alloc::free(int slice_idx, const std::string &name) {
  std::string key = convert_name_to_key(name, slice_idx);
  if (mem_dict.find(key) != mem_dict.end()) {
    auto mem_s = mem_dict[key];
    // fprintf(stderr, "      free %s, addr:%d, size:%d\n", key.c_str(),
    // mem_s.addr, mem_s.size);
    for (int i = 0; i < mem_s.size; i++) {
      assert(lmem_buf[mem_s.addr + i]);
      lmem_buf[mem_s.addr + i] = false;
    }
    mem_dict.erase(key);
    return true;
  }
  return false;
}

void l2mem_alloc::clear() {
  for (int i = 0; i < total_size; i++) {
    lmem_buf[i] = false;
  }
  mem_dict.clear();
  vec_mem_alloc_his.clear();
}

lmem_alloc::lmem_alloc(
    std::map<std::string, std::vector<std::string>> &banked_tensors,
    ILPTimeStep *pILPTimeStep, int ts_count)
    : banked_tensors_(banked_tensors), m_pILPTimeStep(pILPTimeStep) {
  total_size = backend::Arch::LMEM_BYTES;
  lmem_buf = new bool[total_size];
  for (int i = 0; i < total_size; i++) {
    lmem_buf[i] = false;
  }

  for (int i = 0; i < backend::Arch::LMEM_BANKS; i++) {
    bank_area_start_addr[i] = i * backend::Arch::LMEM_BANK_BYTES;
  }
  bank_area_start_addr[backend::Arch::LMEM_BANKS] = total_size;
  m_ts_count = ts_count;
  if (module::isDebugCmdEnable("ILPTimeStep_detail_log")) {
    detail_log = true;
  }
}

lmem_alloc::~lmem_alloc() { delete[] lmem_buf; }

std::shared_ptr<std::vector<std::pair<std::string, mem_struct>>>
lmem_alloc::show_mem(int &total_free_size, int &max_free_mem_idx,
                     int &max_free_mem_size) {
  int free_start_addr = -1, free_count = 0, total_free_count = 0,
      start_bank = -1, end_bank = -1;
  for (int i = 0; i < total_size; i++) {
    if (!lmem_buf[i]) {
      if (free_start_addr == -1) {
        free_start_addr = i;
        start_bank = i / (total_size / 16);
      }
      free_count++;
      total_free_count++;
    } else {
      if (free_start_addr >= 0) {
        end_bank = (i - 1) / (total_size / 16);
        if (detail_log) {
          fprintf(stderr,
                  "        >>>free_start_addr:%d, end_addr:%d, size:%d, "
                  "start_bank:%d, end_bank:%d\n",
                  free_start_addr, free_start_addr + free_count - 1, free_count,
                  start_bank, end_bank);
        }
        free_start_addr = -1;
        free_count = 0;
      }
    }
  }
  std::vector<std::pair<std::string, mem_struct>> vec_mem_struct;
  for (auto itr = mem_dict.begin(); itr != mem_dict.end(); ++itr) {
    vec_mem_struct.push_back(std::make_pair(itr->first.c_str(), itr->second));
  }
  std::sort(vec_mem_struct.begin(), vec_mem_struct.end(), SortByMemStruct);
  if (detail_log) {
    if (free_start_addr >= 0) {
      fprintf(stderr,
              "        >>>free_start_addr:%d, end_addr:%d, size:%d, "
              "start_bank:%d, end_bank:15\n",
              free_start_addr, free_start_addr + free_count - 1, free_count,
              start_bank);
    }
    fprintf(stderr, "        >>>total_free_count:%d\n", total_free_count);

    fprintf(stderr, "        >>>mem_dict:\n");
    for (auto itr : vec_mem_struct) {
      fprintf(stderr, "        name:%s, addr:%d, size:%d\n", itr.first.c_str(),
              itr.second.addr, itr.second.size);
    }
  }
  int pre_s_addr = 0, free_mem_idx = 0;
  auto vec_mem_struct2 =
      std::make_shared<std::vector<std::pair<std::string, mem_struct>>>();
  int idx = 0;
  max_free_mem_idx = 0;
  max_free_mem_size = 0;
  total_free_size = 0;
  for (auto itr : vec_mem_struct) {
    if (itr.second.addr - pre_s_addr > 0) {
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
      vec_mem_struct2->push_back(
          std::make_pair("free_mem " + std::to_string(free_mem_idx++), mem_s));
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
    vec_mem_struct2->push_back(
        std::make_pair("free_mem " + std::to_string(free_mem_idx++), mem_s));
  }
  if (detail_log) {
    fprintf(stderr, "        >>>vec_mem_struct2:\n");
    int idx2 = 0;
    for (auto itr : *vec_mem_struct2) {
      if (itr.second.type == 2)
        continue;
      start_bank = itr.second.addr / (total_size / 16);
      int e_addr = itr.second.addr + itr.second.size - 1;
      end_bank = e_addr / (total_size / 16);
      fprintf(stderr,
              "           idx:%3d, key:%40s, s_addr:%8d, e_addr:%8d, size:%8d, "
              "bank_id:%d, type:%d, start_bank:%d, end_bank:%d\n",
              idx2++, itr.first.c_str(), itr.second.addr, e_addr,
              itr.second.size,
              itr.second.bank_id.size() > 0 ? itr.second.bank_id[0] : -1,
              itr.second.type, start_bank, end_bank);
    }
    fprintf(stderr, "max_free_mem_idx:%d, max_free_mem_size:%d\n",
            max_free_mem_idx, max_free_mem_size);
  }
  return std::move(vec_mem_struct2);
}

bool lmem_alloc::_alloc(int slice_idx, const std::string &name, Value value,
                        int size, std::vector<int> &ret_bank_id, int &free_addr,
                        int &conflict_size, bool force_not_care_bank) {
  std::vector<int> vec_bank_id, all_conf_bank_id;
  bool care_bank = false;
  if (!force_not_care_bank) {
    for (auto it : banked_tensors_[name]) {
      std::string key = convert_name_to_key(it, slice_idx);
      std::vector<int> bank_id = get_bank(key);
      all_conf_bank_id.insert(all_conf_bank_id.end(), bank_id.begin(),
                              bank_id.end());
      if (bank_id.size() > 0) {
        // fprintf(stderr, "        conflict to %s, bank_id:%s\n", key.c_str(),
        // vector_to_string(bank_id).c_str());
        vec_bank_id.insert(vec_bank_id.end(), bank_id.begin(), bank_id.end());
        care_bank = true;
      }
    }
  }

  std::vector<int> searched_bank;
  int bidx = 0;
  while (true) {
    if (care_bank && !force_not_care_bank) {
      bidx = -1;
      for (int i = 0; i < 16; i++) {
        if ((std::find(vec_bank_id.begin(), vec_bank_id.end(), i) ==
             vec_bank_id.end()) &&
            (std::find(searched_bank.begin(), searched_bank.end(), i) ==
             searched_bank.end())) {
          searched_bank.push_back(i);
          bidx = i;
          break;
        }
      }
      if (bidx == -1) {
        fprintf(stderr, "warning: not find valid bank, force no bank\n");
        care_bank = false;
      }
    }
    free_addr = -1;
    int saddr = care_bank ? bank_area_start_addr[bidx] : 0;
    // fprintf(stderr, "saddr:%d\n", saddr);
    int count = 0;
    for (int i = saddr; i < total_size; i++) {
      if (lmem_buf[i]) {
        free_addr = -1;
        count = 0;
      } else {
        if (free_addr == -1) {
          if (i % 64 == 0) {
            free_addr = i;
            // fprintf(stderr, "find free_addr:%d\n", free_addr); //todo, have a
            // bug
          }
        }
        if (free_addr == -1)
          continue;
        count++;
        if (count == size) {
          break;
        }
      }
    }
    if (count == size) {
      int s_bidx = free_addr / (total_size / 16);
      int e_bidx = (free_addr + size - 1) / (total_size / 16);
      ret_bank_id.clear();
      std::map<int, int> bank_size;
      for (int i = s_bidx; i <= e_bidx; i++) {
        ret_bank_id.push_back(i);
        if (i == s_bidx) {
          if (free_addr + size >
              bank_area_start_addr[i + 1]) //分配区域横跨2个bank
            bank_size[i] = bank_area_start_addr[i + 1] - free_addr;
          else
            bank_size[i] = size; //分配区域在s_bidx这个bank内
        } else if (i == e_bidx) {
          bank_size[i] = free_addr + size - bank_area_start_addr[e_bidx];
        } else {
          //中间的bank不可能冲突
        }
      }

      conflict_size = 0;
      for (auto itr : bank_size) {
        if ((std::find(all_conf_bank_id.begin(), all_conf_bank_id.end(),
                       itr.first) != all_conf_bank_id.end())) {
          fprintf(stderr, "          bank%d confilct size:%d\n", itr.first,
                  itr.second);
          conflict_size += itr.second;
        }
      }

      for (int i = 0; i < size; i++) {
        lmem_buf[free_addr + i] = true;
      }

      fprintf(stderr, "%s\n",
              llvm::formatv("        alloc ok for {0}, free_addr:{1}, "
                            "bank_id:{2}, conflict_size:{3}",
                            name, free_addr, vector_to_string(ret_bank_id),
                            conflict_size)
                  .str()
                  .c_str());
      mem_struct mem_s;
      mem_s.addr = free_addr;
      mem_s.bank_id.assign(ret_bank_id.begin(), ret_bank_id.end());
      mem_s.size = size;
      mem_s.value = value;
      mem_s.slice_idx = slice_idx;
      std::string key = convert_name_to_key(name, slice_idx);
      if (mem_dict.find(key) == mem_dict.end()) {
        mem_dict[key] = mem_s;
      } else {
        m_pILPTimeStep->dot_graph_log->export_dot("mem_key_conflict");
        fprintf(stderr, "%s\n",
                llvm::formatv("        alloc key conflict for {0}, "
                              "free_addr:{1}, bank_id:{2}",
                              name, free_addr, vector_to_string(ret_bank_id))
                    .str()
                    .c_str());
        assert(false);
      }
      if (!rehearsal) {
        reload_mem_struct tmp_mem_s;
        tmp_mem_s.addr = free_addr;
        tmp_mem_s.bank_id.assign(ret_bank_id.begin(), ret_bank_id.end());
        if (vec_mem_alloc_his.find(key) != vec_mem_alloc_his.end()) {
          vec_mem_alloc_his[key].vec_reload_addr.push_back(
              std::make_pair(0xFFFFBBBB, tmp_mem_s));
        } else {
          his_mem_struct his_mem_s;
          his_mem_s.size = size;
          his_mem_s.value = value;
          his_mem_s.slice_idx = slice_idx;
          his_mem_s.vec_reload_addr.push_back(
              std::make_pair(0xFFFFBBBB, tmp_mem_s));
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

bool lmem_alloc::alloc_multi(int ts_idx, int op_idx,
                             std::vector<mem_alloc_req_info> &vec_mem_req,
                             int &lack_mem_szie, bool sort_by_size) {
  if (detail_log) {
    fprintf(stderr, "    alloc_multi:\n");
    for (auto it : vec_mem_req) {
      fprintf(stderr, "      name:%s, size:%d\n", it.name.c_str(), it.size);
    }
    fprintf(stderr, "\n");
  }

  std::map<int, bool> alloc_ret;
  std::vector<int> order_idx, min_conflict_order_idx, ret_bank_id;
  int total_conflict_size = 0, conflict_size = 0, idx = 0, free_addr = -1,
      total_alloc_size = 0;
  for (int i = 0; i < vec_mem_req.size(); i++) {
    auto key = -1 * (i + 1);
    alloc_ret[key] = false;
    order_idx.push_back(key);
    total_alloc_size += vec_mem_req[i].size;
  }

  int total_free_size = 0, max_free_mem_idx = 0, max_free_mem_size = 0;
  auto vec_mem_struct2 =
      *show_mem(total_free_size, max_free_mem_idx, max_free_mem_size);
  if (total_free_size < total_alloc_size) {
    lack_mem_szie = total_alloc_size - total_free_size;
    fprintf(stderr, "error! alloc total_free_size(%d) < size(%d)\n",
            total_free_size, total_alloc_size);
    return false;
  }

  rehearsal = true;
  int min_conflict_size = total_size;
  do {
    fprintf(stderr, "test permutation:%d\n", idx);
    for (auto i : order_idx) {
      if (alloc_ret[i]) {
        auto k = -1 * i - 1;
        free(
            convert_name_to_key(vec_mem_req[k].name, vec_mem_req[k].slice_idx));
        alloc_ret[i] = false;
      }
    }
    total_conflict_size = 0;
    bool success = true;
    for (auto i : order_idx) {
      auto k = -1 * i - 1;
      fprintf(stderr, "  alloc %dth tensor\n", k);
      if (!_alloc(vec_mem_req[k].slice_idx, vec_mem_req[k].name,
                  vec_mem_req[k].value, vec_mem_req[k].size, ret_bank_id,
                  free_addr, conflict_size)) {
        fprintf(stderr, "_alloc fail\n");
        success = false;

        int unused;
        show_mem(unused, unused, unused);
        break;
      }
      total_conflict_size += conflict_size;
      alloc_ret[i] = true;
    }

    if (success && total_conflict_size < min_conflict_size) {
      min_conflict_size = total_conflict_size;
      min_conflict_order_idx.assign(order_idx.begin(), order_idx.end());
      if (total_conflict_size == 0) {
        fprintf(stderr, "  find no confilct alloc\n");
        break;
      }
    }
    idx++;
  } while (std::next_permutation(order_idx.begin(), order_idx.end()));

  if (min_conflict_order_idx.size() == 0) {
    if (module::isDebugCmdEnable("disble_lmem_move_op")) {
      return false;
    }
    int step = 0, tmp_size = 0, align_num = 0;
    std::vector<int> vec_merge_mem;
    std::vector<int> vec_move_mem;
    vec_merge_mem.push_back(max_free_mem_idx);
    // Take the maximum free area as the center, evenly find a maximum of 5
    // free areas on both sides, so as to meet the memory allocation space
    for (int i = 1; i <= 8; i++) { //
      step = 0, tmp_size = max_free_mem_size;
      align_num = 0;
      for (int j = max_free_mem_idx + 1; j < vec_mem_struct2.size(); j++) {
        if (vec_mem_struct2[j].second.type == 1) {
          if (find(vec_merge_mem.begin(), vec_merge_mem.end(), j) ==
              vec_merge_mem.end()) {
            vec_merge_mem.push_back(j);
          }

          tmp_size += vec_mem_struct2[j].second.size;
          if (tmp_size > total_alloc_size) {
            int addr = vec_mem_struct2[vec_merge_mem[0]].second.addr;
            // Remove the non-64-byte alignment of the first (vec_merge_mem[0])
            // free start address to ensure that the moved tensor start address
            // is also 64-byte alignment
            align_num = 0;
            while (addr % 64 != 0) {
              addr++;
              align_num++;
            }
            if (tmp_size - align_num > total_alloc_size) {
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
            if (find(vec_move_mem.begin(), vec_move_mem.end(), j) ==
                vec_move_mem.end()) {
              vec_move_mem.push_back(j);
            }
          }
        }
      }
      if (tmp_size - align_num > total_alloc_size) {
        break;
      }
      step = 0;
      align_num = 0;
      for (int j = max_free_mem_idx - 1; j >= 0; j--) {
        if (vec_mem_struct2[j].second.type == 1) {
          if (find(vec_merge_mem.begin(), vec_merge_mem.end(), j) ==
              vec_merge_mem.end()) {
            vec_merge_mem.insert(vec_merge_mem.begin(), j);
          }
          tmp_size += vec_mem_struct2[j].second.size;
          if (tmp_size > total_alloc_size) {
            int addr = vec_mem_struct2[vec_merge_mem[0]].second.addr;
            align_num = 0;
            while (addr % 64 != 0) {
              addr++;
              align_num++;
            }
            if (tmp_size - align_num > total_alloc_size) {
              break;
            }
          }
          if (++step >= i) {
            align_num = 0;
            break;
          }
        } else {
          if (find(vec_move_mem.begin(), vec_move_mem.end(), j) ==
              vec_move_mem.end()) {
            vec_move_mem.insert(vec_move_mem.begin(), j);
          }
        }
      }
      if (tmp_size - align_num > total_alloc_size) {
        break;
      }
    }
    if (tmp_size - align_num < total_alloc_size) {
      fprintf(stderr, "error! can not find enough free mem block\n");
      return false;
    }
    int min = 10000, max = -1;
    fprintf(stderr, "vec_merge_mem:\n");
    for (auto i : vec_merge_mem) {
      fprintf(stderr, "  i:%d\n", i);
      if (i < min) {
        min = i;
      }
      if (i > max) {
        max = i;
      }
    }

    fprintf(stderr, "vec_move_mem:\n");
    for (auto i : vec_move_mem) {
      if (i > min && i < max) { // Only move the tensor between the uppermost
                                // and lowermost free regions
        order_idx.push_back(i);
        alloc_ret[i] = true;
        fprintf(stderr, "  i:%d\n", i);
      }
    }

    rehearsal = true;
    std::vector<int> min_conflict_order_idx;
    min_conflict_size = total_size;
    bool force_not_care_bank = false;
    for (int j = 0; j < 2; j++) {
      idx = 0;
      do {
        fprintf(stderr, "\ntest permutation:%d\n", idx);
        for (auto i : order_idx) {
          if (alloc_ret[i]) {
            if (i >= 0) {
              free(vec_mem_struct2[i].first);
            } else {
              auto m = -1 * i - 1;
              free(convert_name_to_key(vec_mem_req[m].name,
                                       vec_mem_req[m].slice_idx));
            }
            alloc_ret[i] = false;
          }
        }

        total_conflict_size = 0;
        bool success = true;
        for (int i : order_idx) {
          fprintf(stderr, "  start alloc %dth tensor\n", i);
          if (i < 0) {
            auto req = vec_mem_req[-1 * i - 1];
            fprintf(stderr, "  alloc size:%d\n", req.size);
            if (!_alloc(req.slice_idx, req.name, req.value, req.size,
                        ret_bank_id, free_addr, conflict_size,
                        force_not_care_bank)) {
              fprintf(stderr, "_alloc fail 1\n");
              int unused;
              // (void)lmem_alloc_ptr->show_mem(unused, unused, unused);
              show_mem(unused, unused, unused);
              success = false;
              break;
            }
          } else {
            auto mem_s = vec_mem_struct2[i].second;
            std::string tmpStr = "slice_" + std::to_string(mem_s.slice_idx);
            auto name = vec_mem_struct2[i].first.substr(
                0, vec_mem_struct2[i].first.size() - tmpStr.size());
            if (!_alloc(mem_s.slice_idx, name, mem_s.value, mem_s.size,
                        ret_bank_id, free_addr, conflict_size,
                        force_not_care_bank)) {
              fprintf(stderr, "_alloc fail 2\n");
              int unused;
              // (void)lmem_alloc_ptr->show_mem(unused, unused, unused);
              show_mem(unused, unused, unused);
              success = false;
              break;
            }
          }
          alloc_ret[i] = true;
          total_conflict_size += conflict_size;
        }
        if (success && total_conflict_size < min_conflict_size) {
          min_conflict_size = total_conflict_size;
          min_conflict_order_idx.assign(order_idx.begin(), order_idx.end());
          if (total_conflict_size == 0) {
            fprintf(stderr, "   no confilct\n");
            break;
          }
        }
        idx++;
      } while (std::next_permutation(order_idx.begin(), order_idx.end()));
      if (min_conflict_size != total_size) {
        break;
      }
      force_not_care_bank = true;
      fprintf(stderr, "   enable force_not_care_bank\n");
    }

    rehearsal = false;
    total_conflict_size = 0;
    std::map<int, int> new_vec_move_mem_new_addr;
    fprintf(stderr, "actually alloc the moved tensor again:\n");
    for (auto i : min_conflict_order_idx) {
      if (alloc_ret[i]) {
        if (i >= 0) {
          free(vec_mem_struct2[i].first);
        } else {
          auto req = vec_mem_req[-1 * i - 1];
          free(convert_name_to_key(req.name, req.slice_idx));
        }
      }
    }
    for (int i : min_conflict_order_idx) {
      fprintf(stderr, "  i:%d\n", i);
      if (i >= 0) {
        auto mem_s = vec_mem_struct2[i].second;
        std::string tmpStr = "slice_" + std::to_string(mem_s.slice_idx);
        auto op_name = vec_mem_struct2[i].first.substr(
            0, vec_mem_struct2[i].first.size() - tmpStr.size());
        if (!_alloc(mem_s.slice_idx, op_name, mem_s.value, mem_s.size,
                    ret_bank_id, free_addr, conflict_size,
                    force_not_care_bank)) {
          return false;
        }
        new_vec_move_mem_new_addr[i] = free_addr;
      } else {
        auto req = vec_mem_req[-1 * i - 1];
        if (!_alloc(req.slice_idx, req.name, req.value, req.size, ret_bank_id,
                    free_addr, conflict_size, force_not_care_bank)) {
          return false;
        }
      }
      fprintf(stderr, "  conflict_size:%d\n", conflict_size);
      total_conflict_size += conflict_size;
    }
    fprintf(stderr, "  total_conflict_size:%d, min_conflict_size:%d\n",
            total_conflict_size, min_conflict_size);
    assert(total_conflict_size == min_conflict_size);
    ts_move_info tmp;
    for (auto i : min_conflict_order_idx) {
      if (i >= 0 && vec_mem_struct2[i].second.type != 3) {
        if (vec_mem_struct2[i].second.addr != new_vec_move_mem_new_addr[i]) {
          tmp.move_value.push_back(vec_mem_struct2[i].second.value);
          tmp.move_src_add.push_back(vec_mem_struct2[i].second.addr);
          tmp.move_dest_add.push_back(new_vec_move_mem_new_addr[i]);
          tmp.move_size.push_back(vec_mem_struct2[i].second.size);
          tmp.slice_idx.push_back(vec_mem_struct2[i].second.slice_idx);
        }
      }
    }
    tmp.combine_ts_op_idx = op_idx;
    tmp.name = "lmem_tensor_move_at_ts" + std::to_string(ts_idx) + "_op" +
               std::to_string(op_idx);
    if (m_pILPTimeStep->inserted_timestep_table_.find(ts_idx) ==
        m_pILPTimeStep->inserted_timestep_table_.end()) {
      m_pILPTimeStep->inserted_timestep_table_[ts_idx] =
          std::vector<ts_move_info>();
    }
    m_pILPTimeStep->inserted_timestep_table_[ts_idx].push_back(tmp);
  } else {
    rehearsal = false;
    fprintf(stderr, "actually alloc tensor:\n");
    for (auto i : min_conflict_order_idx) {
      if (alloc_ret[i]) {
        auto req = vec_mem_req[-1 * i - 1];
        free(convert_name_to_key(req.name, req.slice_idx));
      }
    }

    for (int i : min_conflict_order_idx) {
      auto req = vec_mem_req[-1 * i - 1];
      fprintf(stderr, "      start alloc %dth tensor\n", i);
      if (!_alloc(req.slice_idx, req.name, req.value, req.size, ret_bank_id,
                  free_addr, conflict_size, false)) {
        fprintf(stderr, "_alloc fail\n");
        return false;
      }
    }
  }

  return true;
}

bool lmem_alloc::alloc2(int slice_idx, const std::string &name, Value value,
                        int addr, int size) {
  if (detail_log) {
    fprintf(stderr, "%s\n",
            llvm::formatv(
                "      start alloc for {0}, size:{1}, slice_idx:{2}, addr:{3}",
                name, size, slice_idx, addr)
                .str()
                .c_str());
  }
  int bidx = addr / (total_size / 16);
  int end_bidx = (addr + size - 1) / (total_size / 16);
  std::vector<int> tmp_bank_id;
  for (int i = bidx; i <= end_bidx; i++) {
    tmp_bank_id.push_back(i);
  }
  for (int i = 0; i < size; i++) {
    lmem_buf[addr + i] = true;
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
  std::string key = convert_name_to_key(name, slice_idx);
  mem_dict[key] = mem_s;

  his_mem_struct his_mem_s;
  his_mem_s.size = size;
  his_mem_s.value = value;
  his_mem_s.slice_idx = slice_idx;
  his_mem_s.vec_reload_addr.push_back(std::make_pair(0xFFFFBBBB, tmp_mem_s));
  // llvm::errs() <<"alloc2 vec_mem_alloc_his key:"<<key<<"\n";
  vec_mem_alloc_his[key] = his_mem_s;
  return true;
}

bool lmem_alloc::free(const std::string &key,
                      std::vector<std::pair<int, int>> *vec_pre_ts_free_mem) {
  if (mem_dict.find(key) != mem_dict.end()) {
    auto mem_s = mem_dict[key];
    if (detail_log) {
      fprintf(stderr, "      free %s, addr:%d, size:%d\n", key.c_str(),
              mem_s.addr, mem_s.size);
    }
    for (int i = 0; i < mem_s.size; i++) {
      assert(lmem_buf[mem_s.addr + i]);
      lmem_buf[mem_s.addr + i] = false;
    }
    mem_dict.erase(key);
    if (vec_pre_ts_free_mem) {
      (*vec_pre_ts_free_mem)
          .push_back(std::make_pair(mem_s.addr, mem_s.addr + mem_s.size));
    }
    return true;
  } else {
    fprintf(stderr, "      free %s failed\n", key.c_str());
    assert(false);
  }
  return false;
}

bool lmem_alloc::get_mem_struct(const std::string &key, mem_struct &mem_s) {
  if (mem_dict.find(key) != mem_dict.end()) {
    mem_s = mem_dict[key];
    return true;
  }
  return false;
}

std::vector<int> lmem_alloc::get_bank(const std::string &name) {
  std::vector<int> tmp;
  auto iter = mem_dict.find(name);
  if (iter != mem_dict.end()) {
    auto bank_id = iter->second.bank_id;
    tmp.assign(bank_id.begin(), bank_id.end());
    return tmp;
  }
  return tmp;
}

ILPTimeStep::ILPTimeStep(const LgInfo &group_info,
                         std::shared_ptr<dot_graph> tmp_dot_graph_log,
                         int sec_per_core)
    : _group_info(group_info), solver(MPSolver::CreateSolver("SCIP")),
      dot_graph_log(tmp_dot_graph_log) {
  if (!solver) {
    llvm::errs() << "SCIP solver unavailable.\n";
    ;
  }
  ts_count = sec_per_core * _group_info.group_ops.size() + 2;
  slice_num = sec_per_core;
  for (int i = 0; i < ts_count; i++) {
    cycle_contrains.push_back(std::vector<std::pair<int, std::string>>());
  }
  for (int i = 0; i < ts_count; i++) {
    mem_contrains.push_back(std::vector<std::pair<int, std::string>>());
  }
  for (int i = 0; i < ts_count; i++) {
    timestep_table_.push_back(TimestepRow2());
  }

  objective = solver->MutableObjective();
  objective->SetMinimization();
  if (module::isDebugCmdEnable("ILPTimeStep_detail_log")) {
    detail_log = true;
  }
}

ILPTimeStep::~ILPTimeStep() {}

std::shared_ptr<ILPTimeStep> ILPTimeStep::clone() {
  auto ilp_timeStep =
      std::make_shared<ILPTimeStep>(_group_info, dot_graph_log, slice_num);
  ilp_timeStep->timestep_table_.assign(timestep_table_.begin(),
                                       timestep_table_.end());
  ilp_timeStep->cycle_contrains.assign(cycle_contrains.begin(),
                                       cycle_contrains.end());
  ilp_timeStep->mem_contrains.assign(mem_contrains.begin(),
                                     mem_contrains.end());
  ilp_timeStep->mapILPVarInfo.insert(mapILPVarInfo.begin(),
                                     mapILPVarInfo.end());
  ilp_timeStep->mapValueInfo.insert(mapValueInfo.begin(), mapValueInfo.end());
  ilp_timeStep->mapConsInfo.insert(mapConsInfo.begin(), mapConsInfo.end());
  ilp_timeStep->dam_tensor_size.insert(dam_tensor_size.begin(),
                                       dam_tensor_size.end());
  ilp_timeStep->load_tensor_cycles.insert(load_tensor_cycles.begin(),
                                          load_tensor_cycles.end());
  return ilp_timeStep;
}

void ILPTimeStep::addValueInfo(int slice_idx, Value value,
                               std::string varName) {
  if (mapValueInfo.find(value) == mapValueInfo.end()) {
    std::map<int, std::vector<std::string>> tmp;
    mapValueInfo[value] = tmp;
  } else {
    if (mapValueInfo[value].find(slice_idx) == mapValueInfo[value].end()) {
      std::vector<std::string> tmp;
      mapValueInfo[value][slice_idx] = tmp;
    }
  }
  mapValueInfo[value][slice_idx].push_back(
      varName); // Make sure that the variable after the time slot is placed at
                // the end
}

void ILPTimeStep::addBinaryVar(int ts_idx, int slice_idx, int mode,
                               std::string varName, Value value,
                               tensor_info_t &info, int64_t lmem_bytes) {
  assert(solver != nullptr);
  MPVariable *x = solver->MakeIntVar(0, 1, varName);
  ilp_var_info var_info;
  var_info.ts_idx = ts_idx;
  var_info.slice_idx = slice_idx;
  var_info.store_load_mode = mode;
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

void ILPTimeStep::addTensorSize(Value value, int slice_idx, int lmem_size) {
  dam_tensor_size[std::make_pair(value, slice_idx)] = align(lmem_size, 64);
}

void ILPTimeStep::addTensorCycle(Value value, int slice_idx, int cycle) {
  auto key = std::make_pair(value, slice_idx);
  if (load_tensor_cycles.find(key) == load_tensor_cycles.end()) {
    load_tensor_cycles[key] = cycle;
  }
}

void ILPTimeStep::addTimestepGdmaCycle(int ts_idx, int cycle,
                                       std::string varName) {
  // llvm::errs() << "addTimestepGdmaCycle, ts_idx:"<<ts_idx<< ",
  // cycle:"<<cycle<< ", varName: "<<varName<<"\n";
  cycle_contrains[ts_idx].push_back(std::make_pair(cycle, varName));
}

void ILPTimeStep::addOpInfo(int ts_idx, Operation *op, int buffer_size,
                            int mem_size_for_load, int bdc_cycle) {
  assert(timestep_table_[ts_idx].vec_op_infos.size() == 0);
  op_related_info_t tmp;
  tmp.op = op;
  tmp.slice_idx = (ts_idx - 1) / _group_info.group_ops.size();
  tmp.mem_size_for_load = mem_size_for_load;
  tmp.buffer_size = align(buffer_size, 64);
  tmp.bdc_cycle = bdc_cycle;
  timestep_table_[ts_idx].vec_op_infos.push_back(tmp);
}

void ILPTimeStep::addTimestepMemUse(int ts_idx, int mem_size,
                                    std::vector<std::string> &varNames) {
  // llvm::errs() << "addTimestepMemUse, ts_idx:"<<ts_idx<< ",
  // mem_size:"<<mem_size<<"\n";
  for (auto varName : varNames) {
    // llvm::errs() << "      varName: "<<varName<<"\n";
    mem_contrains[ts_idx].push_back(std::make_pair(mem_size, varName));
  }
}

MPVariable *ILPTimeStep::getMPVarByName(std::string varName) {
  if (mapILPVarInfo.find(varName) != mapILPVarInfo.end()) {
    return mapILPVarInfo[varName].ilp_var;
  }
  assert(false);
}

void ILPTimeStep::resideOpInValue(Operation *op, Value value) {
  if (reside_in_tensor.find(op) == reside_in_tensor.end()) {
    std::vector<Value> tmp;
    reside_in_tensor[op] = tmp;
  }
  reside_in_tensor[op].push_back(value);
}

void ILPTimeStep::addNewOutIntoReturnOp(std::vector<std::string> var_names,
                                        Value value) {
  if (values_need_store_to_grpout.find(value) ==
      values_need_store_to_grpout.end()) {
    values_need_store_to_grpout[value] = var_names;
  }
}

int ILPTimeStep::addConstraint(
    double lb, double ub,
    std::vector<std::pair<int, MPVariable *>> coeff_var_items,
    std::string info_for_tips, bool test) {
  assert(coeff_var_items.size() > 0);
  MPConstraint *c0 = solver->MakeRowConstraint(lb, ub, "");
  constraint_info tmp;
  tmp.lb = lb;
  tmp.ub = ub;
  tmp.cons_var = c0;
  tmp.info_for_tips = info_for_tips;
  for (auto it : coeff_var_items) {
    c0->SetCoefficient(it.second, it.first);
    tmp.coeff_var_items.push_back(it);
  }

  int cons_idx = vec_constraints.size();
  vec_constraints.push_back(tmp);
  if (module::isDebugCmdEnable("solve_per_constraint")) {
    if (test) {
      std::cerr << std::fixed << std::setprecision(2) << "lb:" << lb
                << ", ub:" << ub << "\n";
      for (auto it : coeff_var_items) {
        llvm::errs() << "a:" << it.first << ", var name:" << it.second->name()
                     << "\n";
      }
      MPSolver::ResultStatus result_status = solver->Solve();
      if (result_status == MPSolver::OPTIMAL ||
          result_status == MPSolver::FEASIBLE) {
        llvm::errs() << "Solve success, info_for_tips:" << info_for_tips
                     << ", idx:" << vec_constraints.size() - 1 << "\n";
      } else {
        llvm::errs() << "Solve fail, info_for_tips:" << info_for_tips
                     << ", idx:" << vec_constraints.size() - 1 << "\n";
      }
    }
  }
  return cons_idx;
}

void ILPTimeStep::showAllConstraint() {
  llvm::errs() << "showAllConstraint:\n";

  std::map<MPVariable *, std::string> only_one_var_warning;
  for (auto it : vec_constraints) {
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

  for (auto &it : vec_constraints) {
    showConstraintInfo(it, only_one_var_warning);
  }
}

void ILPTimeStep::showConstraintInfo(
    constraint_info &cons_info,
    std::map<MPVariable *, std::string> &only_one_var_warning) {
  int i = -1;
  auto itr =
      std::find(vec_constraints.begin(), vec_constraints.end(), cons_info);
  if (itr != vec_constraints.end()) {
    i = std::distance(vec_constraints.begin(), itr);
  }
  double max_int = MPSolver::infinity();
  double min_int = -MPSolver::infinity();
  std::string str;
  auto &it = cons_info;
  std::cerr << std::fixed << std::setprecision(2)
            << "MakeRowConstraint, lb:" << it.lb << " ub:" << it.ub
            << ", m_constraint_idx:" << i
            << ", info_for_tips:" << it.info_for_tips
            << ", var num:" << it.coeff_var_items.size()
            << ", coeff and var:\n";
  if (it.lb == it.ub) {
    for (auto it2 : it.coeff_var_items) {
      str = it2.second->name();
      if (only_one_var_warning.find(it2.second) != only_one_var_warning.end()) {
        str += only_one_var_warning[it2.second];
      }
      llvm::errs() << "  " << it2.first << " * " << str << "\n";
    }
    llvm::errs() << "  == " << it.lb << "\n";
  } else {
    if (it.lb != min_int && it.ub == max_int) {
      llvm::errs() << "  " << it.lb << "  <\n";
      for (auto it2 : it.coeff_var_items) {
        str = it2.second->name();
        if (only_one_var_warning.find(it2.second) !=
            only_one_var_warning.end()) {
          str += only_one_var_warning[it2.second];
        }
        llvm::errs() << "  " << it2.first << " * " << str << "\n";
      }
    }
    if (it.lb == min_int && it.ub != max_int) {
      for (auto it2 : it.coeff_var_items) {
        str = it2.second->name();
        if (only_one_var_warning.find(it2.second) !=
            only_one_var_warning.end()) {
          str += only_one_var_warning[it2.second];
        }
        llvm::errs() << "  " << it2.first << " * " << str << "\n";
      }
      llvm::errs() << "  < " << it.ub << "\n";
    }
    if (it.lb != min_int && it.ub != max_int) {
      llvm::errs() << "  " << (int)it.lb << "  <\n";
      for (auto it2 : it.coeff_var_items) {
        str = it2.second->name();
        if (only_one_var_warning.find(it2.second) !=
            only_one_var_warning.end()) {
          str += only_one_var_warning[it2.second];
        }
        llvm::errs() << "  " << it2.first << " * " << str << "\n";
      }
      llvm::errs() << "  < " << it.ub << "\n";
    }
  }
}

void ILPTimeStep::showRunInfo() {
  int i = 0;
  llvm::errs() << "showRunInfo:\n";
  ;
  for (auto &itr : timestep_table_new) {
    llvm::errs() << "-------------------ts" << i << "--------------------\n";
    ;
    int cycle = 0, total_cycle = 0;
    for (auto &itr2 : itr.vec_ts_var) {
      if (mapILPVarInfo[itr2.varName].ilp_var->solution_value() == 1) {
        for (auto itr3 : cycle_contrains_new[i]) {
          if (itr3.second == itr2.varName) {
            cycle = itr3.first;
            break;
          }
        }
        llvm::errs() << "  dma var, name: " << itr2.varName
                     << ", cycle:" << cycle << "\n";
        if (map_reside_value_info.find(itr2.value) ==
            map_reside_value_info.end()) {
          total_cycle += cycle;
        }
      }
    }

    for (auto itr2 : itr.vec_op_infos) {
      auto outs = get_output_values(itr2.op);
      llvm::errs() << "  op name: " << module::getName(outs[0]).str()
                   << " , cycle:" << itr2.bdc_cycle
                   << ", free mem_size:" << itr2.mem_size_for_load << "\n";
    }
    i++;
  }
}

void ILPTimeStep::addRowConstraint(int ts_idx, Value value,
                                   std::vector<std::string> var_names,
                                   bool store, bool load_to_l2m) {
  assert(solver != nullptr);
  assert(var_names.size() > 0);
  std::vector<std::pair<int, MPVariable *>> coeff_var_items;
  for (auto var_name : var_names) {
    coeff_var_items.push_back(
        std::make_pair(1, mapILPVarInfo[var_name].ilp_var));
  }

  int cons_idx = addConstraint(1, 1, coeff_var_items);
  int slice_idx = (ts_idx - 1) / _group_info.group_ops.size();
  if (mapConsInfo.find(value) == mapConsInfo.end()) {
    cons_info tmp;
    std::map<int, cons_info> tmp_cons_info;
    tmp_cons_info[slice_idx] = tmp;
    mapConsInfo[value] = tmp_cons_info;
  } else {
    if (mapConsInfo[value].find(slice_idx) == mapConsInfo[value].end()) {
      cons_info tmp;
      mapConsInfo[value][slice_idx] = tmp;
    }
  }
  auto &tmp2 = mapConsInfo[value][slice_idx];
  // llvm::errs() <<"add into mapConsInfo, value:
  // "<<module::getName(value).str()<<", slice_idx:"<< slice_idx<<", store:"<<
  // store
  //              <<", var_names[0]: "<< var_names[0]<<"\n";
  if (store) {
    tmp2.store_cons_idx = cons_idx;
    tmp2.store_var_names.assign(var_names.begin(), var_names.end());
  } else {
    tmp2.load_cons_idx = cons_idx;
    tmp2.load_var_names.assign(var_names.begin(), var_names.end());
  }

  if (load_to_l2m) {
    auto &vars_need_load_to_l2m =
        timestep_table_[ts_idx].vec_op_infos[0].vars_need_load_to_l2m;
    if (vars_need_load_to_l2m.find(value) == vars_need_load_to_l2m.end()) {
      std::vector<std::string> null_names;
      vars_need_load_to_l2m[value] = null_names;
    }
    vars_need_load_to_l2m[value].assign(var_names.begin(), var_names.end());
  }
}

void ILPTimeStep::setVarExpectValue(std::string var_name, int expect_value) {
  std::vector<std::pair<int, MPVariable *>> coeff_var_items;
  coeff_var_items.push_back(std::make_pair(1, mapILPVarInfo[var_name].ilp_var));
  addConstraint(expect_value, expect_value, coeff_var_items);
}

bool ILPTimeStep::run(Operation *&fail_op) {
  assert(solver != nullptr);
  fail_op = nullptr;
  if (detail_log) {
    showAllConstraint();
    // solver->EnableOutput();
  }
  LAYER_GROUP_LOG_DEBUG_BLOCK({ llvm::errs() << "solve start\n"; });
  MPSolver::ResultStatus result_status = solver->Solve();
  // Check that the problem has an optimal solution.
  if (result_status != MPSolver::OPTIMAL &&
      result_status != MPSolver::FEASIBLE) {
    LAYER_GROUP_LOG_DEBUG_BLOCK({
      llvm::errs() << "The problem does not have an optimal or feasible "
                      "solution!, result_status:"
                   << (int)result_status << "\n";
    });
    for (auto ts_mem_con : ts_mem_contrains) {
      auto cons = vec_constraints[ts_mem_con.cons_idx];
      cons.cons_var->SetBounds(-MPSolver::infinity(), MPSolver::infinity());
      result_status = solver->Solve();
      if (result_status == MPSolver::OPTIMAL ||
          result_status == MPSolver::FEASIBLE) {
        fail_op = ts_mem_con.op;
        if (fail_op) {
          // The check on whether there is enough memory in
          // backward_gen_ilp_var2::load_bytes_for_next_ts is removed, it is
          // completely checked here, and it can be more complete/simple check
          // here!
          LAYER_GROUP_LOG_DEBUG_BLOCK({
            llvm::errs() << "success, mem constraint fail at op:"
                         << module::getName(fail_op).str() << "\n";
          });
          break;
        }
      }
      cons.cons_var->SetBounds(cons.lb, cons.ub);
    }
    return false;
  }

  LAYER_GROUP_LOG_DEBUG_BLOCK({ llvm::errs() << "Solution:\n"; });
  LAYER_GROUP_LOG_DEBUG_BLOCK(
      { llvm::errs() << "Objective value = " << objective->Value() << "\n"; });
  LAYER_GROUP_LOG_DEBUG_BLOCK({ llvm::errs() << "\nAdvanced usage:\n"; });
  LAYER_GROUP_LOG_DEBUG_BLOCK({
    llvm::errs() << "Problem solved in " << solver->wall_time()
                 << " milliseconds\n";
  });
  LAYER_GROUP_LOG_DEBUG_BLOCK({
    llvm::errs() << "Problem solved in " << solver->iterations()
                 << " iterations\n";
  });
  LAYER_GROUP_LOG_DEBUG_BLOCK({
    llvm::errs() << "Problem solved in " << solver->nodes()
                 << " branch-and-bound nodes\n";
  });
  solved = true;
  return true;
}

bool ILPTimeStep::IsSameWith(const std::shared_ptr<ILPTimeStep> other) {
  if (!solved || !other->solved) {
    llvm::errs() << "failed, ILPTimeStep must be solved\n";
    ;
    exit(0);
  }
  if (ts_count != other->ts_count) {
    return false;
  }
  for (auto itr1 : mapValueInfo) {
    for (auto itr2 : itr1.second) {
      for (auto itr3 : other->mapValueInfo) {
        for (auto itr4 : itr3.second) {
          if (itr1.first == itr3.first && itr2.first == itr4.first) {
            int ts_idx1 = -1, ts_idx2 = -1;
            for (auto itr5 : itr2.second) {
              if (mapILPVarInfo[itr5].ilp_var->solution_value() == 1) {
                ts_idx1 = mapILPVarInfo[itr5].ts_idx;
                break;
              }
            }
            for (auto itr5 : itr4.second) {
              if (other->mapILPVarInfo[itr5].ilp_var->solution_value() == 1) {
                ts_idx2 = other->mapILPVarInfo[itr5].ts_idx;
                break;
              }
            }
            if (ts_idx1 == -1 || ts_idx2 == -1 || ts_idx1 != ts_idx2) {
              return false;
            }
          }
        }
      }
    }
  }

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
  llvm::errs() << "-------------------mem_contrains_info, after "
                  "merge--------------------\n";
  ;
  for (int i = 0; i < ts_count; i++) {
    llvm::errs() << "-------------------ts" << i << "--------------------\n";
    ;
    if (!(i == 0 || i == ts_count - 1)) {
      for (auto it : timestep_table_new[i].vec_op_infos)
        llvm::errs() << "op_name:" << module::getName(it.op) << "\n";
    }
    for (auto it : mem_contrains_new[i]) {
      llvm::errs() << "  " << it.first << " * " << it.second << "\n";
    }
  }

  llvm::errs() << "-------------------cycle_contrains_info, after "
                  "merge--------------------\n";
  ;
  for (int i = 0; i < ts_count; i++) {
    llvm::errs() << "-------------------ts" << i << "--------------------\n";
    ;
    if (!(i == 0 || i == ts_count - 1)) {
      for (auto it : timestep_table_new[i].vec_op_infos)
        llvm::errs() << "op_name:" << module::getName(it.op) << "\n";
    }
    for (auto it : cycle_contrains_new[i]) {
      llvm::errs() << "  " << it.first << " * " << it.second << "\n";
    }
  }
}

bool ILPTimeStep::merge_small_cycle_op(
    TensorInfo &tensor_infos, bool &merged,
    std::shared_ptr<dot_graph> dot_graph_log) {
  if (module::isDebugCmdEnable("disable_small_cycle_op_merge"))
    return true;

  // llvm::errs() << "-------------------mem_contrains_info, before
  // merge--------------------\n";; for(int i = 0; i < ts_count; i++) {
  //   llvm::errs() << "-------------------ts"<<i<<"--------------------\n";;
  //   if (!(i == 0 || i == ts_count - 1)) {
  //     for (auto it:timestep_table_[i].vec_op_infos)
  //       llvm::errs() <<"op_name:"<<module::getName(it.op)<<"\n";
  //   }
  //   for (auto it: mem_contrains[i]) {
  //     llvm::errs() <<"  "<<it.first<<" * "<<it.second<<"\n";
  //   }
  // }

  // llvm::errs() << "\n-------------------cycle_contrains_info, before
  // merge--------------------\n";; for(int i = 0; i < ts_count; i++) {
  //   llvm::errs() << "-------------------ts"<<i<<"--------------------\n";;
  //   if (!(i == 0 || i == ts_count - 1)) {
  //     for (auto it:timestep_table_[i].vec_op_infos)
  //       llvm::errs() <<"op_name:"<<module::getName(it.op)<<"\n";
  //   }
  //   for (auto it: cycle_contrains[i]) {
  //     llvm::errs() <<"  "<<it.first<<" * "<<it.second<<"\n";
  //   }
  // }

  int dma_cycle = -1, merge_start = 0, min_pos = 0, max_merge_op_num = 5;
  std::vector<Value> big_load_tensor;
  LAYER_GROUP_LOG_DEBUG_BLOCK(
      { llvm::errs() << "merge_small_cycle_op starting\n"; });
  for (int ts_idx = ts_count - 1; ts_idx >= 0; ts_idx--) {
    if (!(ts_idx == 0 || ts_idx == ts_count - 1)) {
      auto slice_idx = timestep_table_[ts_idx].vec_op_infos[0].slice_idx;
      auto op = timestep_table_[ts_idx].vec_op_infos[0].op;
      if (!op) {
        continue;
      }
      if (detail_log)
        LAYER_GROUP_LOG_DEBUG_BLOCK({
          llvm::errs() << "begin, ts_idx:" << ts_idx
                       << ", op:" << module::getName(op).str()
                       << ", type:" << op->getName().getStringRef().str()
                       << "\n";
        });
      if (dma_cycle == -1) {
        if (isa<tpu::Conv2DOp>(op)) {
          auto in = op->getOperand(1);
          dma_cycle = load_tensor_cycles[std::make_pair(in, slice_idx)];
          LAYER_GROUP_LOG_DEBUG_BLOCK(
              { llvm::errs() << "weight load cycle:" << dma_cycle << "\n"; });
          merge_start = ts_idx - 1;
        } else {
          continue;
        }
      } else {
        merge_start = ts_idx;
        LAYER_GROUP_LOG_DEBUG_BLOCK({
          llvm::errs() << "ts_idx:" << ts_idx
                       << " achor dma_cycle:" << dma_cycle
                       << " for merged op\n";
        });
      }

      std::vector<int> cycle_sum;
      int sum = 0;
      bool meet_nullOp = false;
      for (int k = 0; k < max_merge_op_num;
           k++) { // A maximum of four consecutive ops can be combined
        if (merge_start - k < 1) // ts0 has no op
          break;
        auto vec_op_info = timestep_table_[merge_start - k].vec_op_infos[0];
        if (!vec_op_info.op) {
          meet_nullOp = true;
          break;
        }
        auto bdc_cycle = vec_op_info.bdc_cycle;
        auto type = vec_op_info.op->getName().getStringRef().str();
        sum += bdc_cycle;
        LAYER_GROUP_LOG_DEBUG_BLOCK({
          llvm::errs() << "merge op" << merge_start - k
                       << ", name:" << module::getName(vec_op_info.op).str()
                       << ", type:" << type << ", bdc_cycle:" << bdc_cycle
                       << ", sum:" << sum << "\n";
        });
        cycle_sum.push_back(std::abs(dma_cycle - sum));
      }
      if (meet_nullOp) {
        dma_cycle = -1;
        llvm::errs() << "meet null op\n";
        continue;
      }
      min_pos = 0;
      if (cycle_sum.size() > 1) {
        // Find the one with the smallest difference
        auto min_cycle = *std::min_element(cycle_sum.begin(), cycle_sum.end());
        auto it2 = std::find(cycle_sum.begin(), cycle_sum.end(), min_cycle);
        if (it2 != cycle_sum.end()) {
          min_pos = std::distance(cycle_sum.begin(), it2);
          if (min_pos > 0) {
            map_merge_start_to_merge_len[merge_start] = min_pos;
            LAYER_GROUP_LOG_DEBUG_BLOCK(
                { llvm::errs() << "find min_pos:" << min_pos << "\n"; });
            std::string tmp_str =
                merge_start == ts_idx - 1 ? "merge next " : "merge ";
            dot_graph_log->add_node_label(
                module::getName(op).str(),
                tmp_str + std::to_string(min_pos + 1) + "ops");

            big_load_tensor.clear();
            dma_cycle = 0;
            for (int m = 0; m <= min_pos; m++) {
              auto timestep_row = timestep_table_[merge_start - m];
              auto op = timestep_row.vec_op_infos[0].op;
              auto slice_idx2 = timestep_row.vec_op_infos[0].slice_idx;
              for (const auto &res : llvm::enumerate(op->getOperands())) {
                if (slice_idx2 > 0 && map_reside_value_info.find(res.value()) !=
                                          map_reside_value_info.end()) {
                  continue;
                }
                auto key2 = std::make_pair(res.value(), slice_idx2);
                if (load_tensor_cycles.find(key2) != load_tensor_cycles.end()) {
                  dma_cycle += load_tensor_cycles[key2];
                  big_load_tensor.push_back(res.value());
                }
              }
            }

            if (merge_start == ts_idx - 1) {
              ts_idx -= min_pos + 1;
            } else {
              ts_idx -= min_pos;
            }
            continue;
          }
        }
      }
      dma_cycle = -1;
    }
  }
  merged = map_merge_start_to_merge_len.size() > 0;

  std::string tmp_var;
  for (int merge_start = ts_count - 1; merge_start >= 0; merge_start--) {
    if (!(merge_start == 0 || merge_start == ts_count - 1)) {
      int min_pos = 0;
      if (map_merge_start_to_merge_len.find(merge_start) !=
          map_merge_start_to_merge_len.end()) {
        min_pos = map_merge_start_to_merge_len[merge_start];
      } else {
        continue;
      }
      for (int m = 0; m <= min_pos; m++) {
        int slice_idx =
            timestep_table_[merge_start - m].vec_op_infos[0].slice_idx;
        auto op = timestep_table_[merge_start - m].vec_op_infos[0].op;
        auto name = module::getName(op).str();
        if (detail_log)
          LAYER_GROUP_LOG_DEBUG_BLOCK({
            llvm::errs() << "idx: " << merge_start - m << ", name: " << name
                         << ", slice_idx: " << slice_idx << "\n";
          });
        for (auto in : get_input_values(op)) {
          if (mapConsInfo.find(in) != mapConsInfo.end()) {
            int64_t min_merge_start = llvm::maxIntN(64);
            if (mapConsInfo[in][slice_idx].load_var_names.size() > 0) {
              for (auto var_name : mapConsInfo[in][slice_idx].load_var_names) {
                if (min_merge_start > mapILPVarInfo[var_name].ts_idx) {
                  min_merge_start = mapILPVarInfo[var_name].ts_idx;
                  tmp_var = var_name;
                }
              }
              LAYER_GROUP_LOG_DEBUG_BLOCK({
                llvm::errs()
                    << "min_merge_start:" << min_merge_start
                    << ", low_merge_merge_start:" << merge_start - min_pos
                    << "\n";
              });
              if (min_merge_start < llvm::maxIntN(64) &&
                  min_merge_start >= merge_start - min_pos) {
                int tmp_merge_start = merge_start - min_pos - 1;
                auto var_name2 =
                    tmp_var + "_extend_to_ts" + std::to_string(tmp_merge_start);
                LAYER_GROUP_LOG_DEBUG_BLOCK(
                    { llvm::errs() << "define " << var_name2 << "\n"; });
                auto key = std::make_pair(in, slice_idx);
                if (dam_tensor_size.find(key) == dam_tensor_size.end()) {
                  assert(false);
                }
                auto full_slice_bytes = dam_tensor_size[key];
                for (int idx = tmp_merge_start; idx <= merge_start - m - 1;
                     idx++) {
                  mem_contrains[idx].push_back(
                      std::make_pair(full_slice_bytes, var_name2));
                }

                addBinaryVar(tmp_merge_start, slice_idx, 0, var_name2, in,
                             mapILPVarInfo[tmp_var].tensor_info,
                             full_slice_bytes);
                auto new_item =
                    std::make_pair(1, mapILPVarInfo[var_name2].ilp_var);
                auto &cons =
                    vec_constraints[mapConsInfo[in][slice_idx].load_cons_idx];
                cons.cons_var->SetCoefficient(new_item.second, new_item.first);
                cons.coeff_var_items.push_back(new_item);
                addTimestepGdmaCycle(tmp_merge_start, load_tensor_cycles[key],
                                     var_name2);
                setVarExpectValue(var_name2, 1);
              }
            }
          }
        }

        for (auto out : get_output_values(op)) {
          if (mapConsInfo.find(out) != mapConsInfo.end()) {
            if (mapConsInfo[out][slice_idx].store_var_names.size() > 0) {
              int64_t max_merge_start = -1;
              for (auto var_name :
                   mapConsInfo[out][slice_idx].store_var_names) {
                if (max_merge_start < mapILPVarInfo[var_name].ts_idx) {
                  max_merge_start = mapILPVarInfo[var_name].ts_idx;
                  tmp_var = var_name;
                }
              }
              LAYER_GROUP_LOG_DEBUG_BLOCK({
                llvm::errs()
                    << "max_merge_start " << max_merge_start
                    << ", high_merge_merge_start: " << merge_start << "\n";
              });
              if (max_merge_start > 0 && max_merge_start <= merge_start) {
                int tmp_merge_start = merge_start + 1;
                auto var_name2 =
                    tmp_var + "_extend_to_ts" + std::to_string(tmp_merge_start);
                LAYER_GROUP_LOG_DEBUG_BLOCK(
                    { llvm::errs() << "define " << var_name2 << "\n"; });
                auto key = std::make_pair(out, slice_idx);
                auto full_slice_bytes = dam_tensor_size[key];
                for (int idx = merge_start - m + 1; idx <= tmp_merge_start;
                     idx++) {
                  mem_contrains[idx].push_back(
                      std::make_pair(full_slice_bytes, var_name2));
                }
                addBinaryVar(tmp_merge_start, slice_idx, 0, var_name2, out,
                             mapILPVarInfo[tmp_var].tensor_info,
                             full_slice_bytes);
                auto new_item =
                    std::make_pair(1, mapILPVarInfo[var_name2].ilp_var);
                auto &cons =
                    vec_constraints[mapConsInfo[out][slice_idx].store_cons_idx];
                cons.cons_var->SetCoefficient(new_item.second, new_item.first);
                cons.coeff_var_items.push_back(new_item);
                addTimestepGdmaCycle(tmp_merge_start, load_tensor_cycles[key],
                                     var_name2);
                setVarExpectValue(var_name2, 1);
              }
            }
          }
        }
      }
    }
  }

  for (int merge_start = ts_count - 1; merge_start >= 0; merge_start--) {
    int min_pos = 0;
    if (map_merge_start_to_merge_len.find(merge_start) !=
        map_merge_start_to_merge_len.end()) {
      min_pos = map_merge_start_to_merge_len[merge_start];
    }

    if ((merge_start == 0 || merge_start == ts_count - 1) || min_pos == 0) {
      timestep_table_new.push_back(timestep_table_[merge_start]);
      cycle_contrains_new.push_back(cycle_contrains[merge_start]);
      mem_contrains_new.push_back(mem_contrains[merge_start]);
      continue;
    }

    TimestepRow2 tmp;
    for (auto it3 : timestep_table_[merge_start].vec_ts_var) {
      int64_t mode2 = mapILPVarInfo[it3.varName].tensor_info.mode2;
      if (mode2 & TIMESTEP2_LOAD ||
          (mode2 & TIMESTEP2_STORE_AND_LOAD &&
           mapILPVarInfo[it3.varName].store_load_mode == 1)) {
        auto op2 = timestep_table_[merge_start].vec_op_infos[0].op;
        dot_graph_log->add_node_label(module::getName(op2).str(),
                                      "add loadVar " + it3.varName +
                                          " to new vec_ts_var\n");
        tmp.vec_ts_var.push_back(it3);
      }
    }
    // load type variable, the last ts(merge_start) variable in the fusion op is
    // retained, and other ts variables are fixed to 0
    for (int m = 1; m <= min_pos; m++) {
      for (auto it3 : cycle_contrains[merge_start - m]) {
        int64_t mode2 = mapILPVarInfo[it3.second].tensor_info.mode2;
        if (mode2 & TIMESTEP2_LOAD ||
            (mode2 & TIMESTEP2_STORE_AND_LOAD &&
             mapILPVarInfo[it3.second].store_load_mode == 1)) {
          if (detail_log)
            LAYER_GROUP_LOG_DEBUG_BLOCK({
              llvm::errs() << "for loadVar, setVarExpectValue:" << it3.second
                           << " to const_0\n";
            });
          auto op2 = timestep_table_[merge_start - m].vec_op_infos[0].op;
          dot_graph_log->add_node_label(module::getName(op2).str(),
                                        "set loadVar:" + it3.second +
                                            " to 0\n");
          setVarExpectValue(it3.second, 0);
        }
      }
    }
    for (auto it3 : timestep_table_[merge_start - min_pos].vec_ts_var) {
      int64_t mode2 = mapILPVarInfo[it3.varName].tensor_info.mode2;
      if (mode2 & TIMESTEP2_STORE ||
          (mode2 & TIMESTEP2_STORE_AND_LOAD &&
           mapILPVarInfo[it3.varName].store_load_mode == 0)) {
        auto op2 = timestep_table_[merge_start - min_pos].vec_op_infos[0].op;
        dot_graph_log->add_node_label(module::getName(op2).str(),
                                      "add storeVar " + it3.varName +
                                          " to new vec_ts_var\n");
        tmp.vec_ts_var.push_back(it3);
      }
    }
    // store type variable. The variable setting of the first ts in the fusion
    // op is retained, and the variables of other ts are fixed to 0
    for (int m = 0; m < min_pos; m++) {
      for (auto it3 : cycle_contrains[merge_start - m]) {
        int64_t mode2 = mapILPVarInfo[it3.second].tensor_info.mode2;
        if (mode2 & TIMESTEP2_STORE ||
            (mode2 & TIMESTEP2_STORE_AND_LOAD &&
             mapILPVarInfo[it3.second].store_load_mode == 0)) {
          if (detail_log)
            LAYER_GROUP_LOG_DEBUG_BLOCK({
              llvm::errs() << "for storeVar, setVarExpectValue:" << it3.second
                           << " to const_0\n";
            });
          auto op2 = timestep_table_[merge_start - m].vec_op_infos[0].op;
          dot_graph_log->add_node_label(module::getName(op2).str(),
                                        "set storeVar:" + it3.second +
                                            " to 0\n");
          setVarExpectValue(it3.second, 0);
        }
      }
    }

    for (auto ts_var : tmp.vec_ts_var) {
      for (auto ts_var2 : tmp.vec_ts_var) {
        if (ts_var.value == ts_var2.value &&
            ts_var.varName != ts_var2.varName) {
          dot_graph_log->add_node_label(
              module::getName(ts_var.value).str(),
              ts_var.varName + "," + ts_var2.varName +
                  " in a same large ts, set its to 0\n");
          setVarExpectValue(ts_var.varName, 0);
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

    for (int m = 0; m <= min_pos; m++) {
      tmp.vec_op_infos.push_back(
          timestep_table_[merge_start - m].vec_op_infos[0]);
    }
    reverse(tmp.vec_op_infos.begin(), tmp.vec_op_infos.end());

    std::vector<std::pair<int, std::string>> new_cycle;
    for (auto it2 : cycle_contrains[merge_start]) {
      int64_t mode2 = mapILPVarInfo[it2.second].tensor_info.mode2;
      if (mode2 & TIMESTEP2_LOAD ||
          (mode2 & TIMESTEP2_STORE_AND_LOAD &&
           mapILPVarInfo[it2.second].store_load_mode == 1)) {
        new_cycle.push_back(it2);
      }
    }
    for (auto it2 : cycle_contrains[merge_start - min_pos]) {
      int64_t mode2 = mapILPVarInfo[it2.second].tensor_info.mode2;
      if (mode2 & TIMESTEP2_STORE ||
          (mode2 & TIMESTEP2_STORE_AND_LOAD &&
           mapILPVarInfo[it2.second].store_load_mode == 0)) {
        new_cycle.push_back(it2);
      }
    }

    std::vector<std::pair<int, std::string>> new_mem;
    for (auto it2 : mem_contrains[merge_start]) {
      if (isLoadVar(it2.second)) {
        new_mem.push_back(it2);
      }
    }
    for (auto it2 : mem_contrains[merge_start - min_pos]) {
      if (isStoreVar(it2.second)) {
        new_mem.push_back(it2);
      }
    }

    cycle_contrains_new.push_back(new_cycle);
    mem_contrains_new.push_back(new_mem);
    timestep_table_new.push_back(tmp);
    merge_start -= min_pos;
  }

  reverse(timestep_table_new.begin(), timestep_table_new.end());
  reverse(cycle_contrains_new.begin(), cycle_contrains_new.end());
  reverse(mem_contrains_new.begin(), mem_contrains_new.end());
  ts_count = timestep_table_new.size();
  LAYER_GROUP_LOG_DEBUG_BLOCK(
      { llvm::errs() << "new ts_count:" << ts_count << "\n"; });
  return true;
}

bool ILPTimeStep::isStoreVar(std::string var_name) {
  if (mapILPVarInfo.find(var_name) != mapILPVarInfo.end()) {
    int64_t mode2 = mapILPVarInfo[var_name].tensor_info.mode2;
    if (mode2 & TIMESTEP2_STORE ||
        (mode2 & TIMESTEP2_STORE_AND_LOAD &&
         mapILPVarInfo[var_name].store_load_mode == 0)) {
      return true;
    }
  }
  return false;
}

bool ILPTimeStep::isLoadVar(std::string var_name) {
  if (mapILPVarInfo.find(var_name) != mapILPVarInfo.end()) {
    int64_t mode2 = mapILPVarInfo[var_name].tensor_info.mode2;
    if (mode2 & TIMESTEP2_LOAD ||
        (mode2 & TIMESTEP2_STORE_AND_LOAD &&
         mapILPVarInfo[var_name].store_load_mode == 1)) {
      return true;
    }
  }
  return false;
}

bool ILPTimeStep::prepare(TensorInfo &tensor_infos) {
  if (timestep_table_new.size() == 0) {
    timestep_table_new.assign(timestep_table_.begin(), timestep_table_.end());
    cycle_contrains_new.assign(cycle_contrains.begin(), cycle_contrains.end());
    mem_contrains_new.assign(mem_contrains.begin(), mem_contrains.end());
  }

  // showTimeStepInfo();
  assert(solver != nullptr && objective != nullptr);
  std::vector<std::pair<std::string, MPVariable *>> objective_var;
  for (int i = 0; i < ts_count; i++) {
    std::string var_name = llvm::formatv("sum_var_ts{0}", i);
    MPVariable *x = solver->MakeIntVar(-MPSolver::infinity(),
                                       MPSolver::infinity(), var_name);
    objective_var.push_back(std::make_pair(var_name, x));

    std::string abs_var_name = llvm::formatv("sum_var_abs_ts{0}", i);
    MPVariable *x_abs = solver->MakeIntVar(-MPSolver::infinity(),
                                           MPSolver::infinity(), abs_var_name);
    objective->SetCoefficient(x_abs, 1);

    std::vector<std::pair<int, MPVariable *>> coeff_var_items;
    coeff_var_items.push_back(std::make_pair(1, x_abs));
    coeff_var_items.push_back(std::make_pair(1, x));
    addConstraint(0, MPSolver::infinity(), coeff_var_items, "", false);

    coeff_var_items.clear();
    coeff_var_items.push_back(std::make_pair(1, x_abs));
    coeff_var_items.push_back(std::make_pair(-1, x));
    addConstraint(0, MPSolver::infinity(), coeff_var_items);
  }

  LAYER_GROUP_LOG_DEBUG_BLOCK(
      { llvm::errs() << ">>>>> add cycle_contrains:\n"; });
  std::string op_name;
  for (int i = 0; i < ts_count; i++) {
    double bdc_cycle = 0;
    op_name = "null";
    if (!(i == 0 || i == ts_count - 1)) {
      for (auto it : timestep_table_new[i].vec_op_infos) {
        if (!it.op) {
          continue;
        }
        op_name = module::getName(it.op).str() + "__" + op_name;
        bdc_cycle += it.bdc_cycle;
      }
      if (timestep_table_new[i].vec_op_infos.size() > 0) {
        op_name = op_name.substr(0, op_name.size() - 6);
      }
      if (timestep_table_new[i].vec_op_infos.size() > 1) {
        op_name = "super_op_" + op_name;
      }
    }
    std::vector<std::pair<int, MPVariable *>> coeff_var_items;
    coeff_var_items.push_back(std::make_pair(1, objective_var[i].second));
    if (detail_log)
      LAYER_GROUP_LOG_DEBUG_BLOCK({
        llvm::errs() << "  i:" << i << ", op_name:" << op_name
                     << ", bdc_cycle:" << (int)bdc_cycle << "\n";
      });
    for (auto it : cycle_contrains_new[i]) {
      coeff_var_items.push_back(
          std::make_pair(it.first, mapILPVarInfo[it.second].ilp_var));
    }
    addConstraint(bdc_cycle, bdc_cycle, coeff_var_items);
  }

  LAYER_GROUP_LOG_DEBUG_BLOCK(
      { llvm::errs() << ">>>>> add mem_contrains:\n"; });
  for (int i = 0; i < ts_count; i++) {
    if (i == 0 || i == ts_count - 1) {
      continue;
    }
    // llvm::errs() <<" ts:"<<i<<"\n";
    if (mem_contrains_new[i].size() > 0) {
      op_name = "null";
      Operation *min_free_mem_op = nullptr;
      int ts_reside_value_size = 0, min_free_mem = -1;
      for (auto it : timestep_table_new[i].vec_op_infos) {
        if (!it.op) {
          assert(timestep_table_new[i].vec_op_infos.size() == 1);
          min_free_mem = backend::Arch::LMEM_BYTES;
          continue;
        }
        auto cur_free_mem = it.mem_size_for_load - ts_reside_value_size;
        if (min_free_mem == -1 || cur_free_mem < min_free_mem) {
          min_free_mem = cur_free_mem;
          min_free_mem_op = it.op;
        }
        auto name = module::getName(it.op).str();
        // llvm::errs() <<"  op:"<<name<<"\n";
        op_name = name + "__" + op_name;
        for (auto in : get_input_values(it.op)) {
          auto key = std::make_pair(in, it.slice_idx);
          if (reside_in_tensor.find(it.op) != reside_in_tensor.end()) {
            auto tensors = reside_in_tensor[it.op];
            if (find(tensors.begin(), tensors.end(), in) != tensors.end()) {
              if (detail_log)
                fprintf(stderr, "   reside_in_tensor\n");
              ts_reside_value_size += dam_tensor_size[key];
            }
          }

          if (is_value_stored(in, i) || is_value_stored(in, i - 1)) {
            ts_reside_value_size += dam_tensor_size[key];
          }
        }
      }

      if (timestep_table_new[i].vec_op_infos.size() > 0) {
        op_name = op_name.substr(0, op_name.size() - 6);
      }
      if (timestep_table_new[i].vec_op_infos.size() > 1) {
        op_name = "super_op_" + op_name;
      }
      if (detail_log)
        LAYER_GROUP_LOG_DEBUG_BLOCK({
          llvm::errs() << "  ts:" << i << "  op_name:" << op_name << ":\n";
        });
      std::vector<std::pair<int, MPVariable *>> coeff_var_items;
      for (auto it : mem_contrains_new[i]) {
        coeff_var_items.push_back(
            std::make_pair(it.first, mapILPVarInfo[it.second].ilp_var));
      }
      std::string info = llvm::formatv("for_ts{0}", i).str();
      int cons_idx =
          addConstraint(0, min_free_mem, coeff_var_items, info, true);
      ts_mem_cons tmp;
      tmp.cons_idx = cons_idx;
      tmp.op = min_free_mem_op;
      ts_mem_contrains.push_back(tmp);
    }
  }
  return true;
}

void ILPTimeStep::get_group_cycle_info(int &total_cycle, int &total_diff,
                                       std::vector<ts_cycle_info> &ts_cycle) {
  int i = 0, diff = 0;
  total_cycle = 0;
  total_diff = 0;
  ts_cycle.clear();
  for (auto &itr : timestep_table_new) {
    std::vector<Operation *> tmp_ops;
    for (auto itr2 : itr.vec_op_infos) {
      tmp_ops.push_back(itr2.op);
    }
    Operation *max_cycle_dma_op = nullptr;
    bool max_cycle_is_load = true;
    int dma_cycle = 0, bdc_cycle = 0, store_cycle = 0, load_cycle = 0;
    float max_cycle_dma = 0;
    for (auto &itr2 : itr.vec_ts_var) {
      if (itr2.var_value) {
        for (auto itr3 : cycle_contrains_new[i]) {
          if (itr3.second == itr2.varName) {
            if (detail_log) {
              llvm::errs() << "i:" << i << ", varName:" << itr2.varName
                           << ", cycle:" << itr3.first << "\n";
            }
            dma_cycle += itr3.first;
            if (isLoadVar(itr2.varName)) {
              load_cycle += itr3.first;
              if (itr3.first > max_cycle_dma) {
                max_cycle_dma = itr3.first;
                max_cycle_dma_op = *(itr2.value.getUsers().begin());
                max_cycle_is_load = true;
              }
            }
            if (isStoreVar(itr2.varName)) {
              store_cycle += itr3.first;
              if (itr3.first > max_cycle_dma) {
                max_cycle_dma = itr3.first;
                max_cycle_dma_op = itr2.value.getDefiningOp();
                max_cycle_is_load = false;
              }
            }
            break;
          }
        }
      }
    }

    for (auto itr2 : itr.vec_op_infos) {
      bdc_cycle += itr2.bdc_cycle;
    }

    float max_dma_ratio = max_cycle_dma / dma_cycle;
    ts_cycle_info cycle_info;
    int cycle = std::max(dma_cycle, bdc_cycle);
    cycle_info.cycle = cycle;
    total_cycle += cycle;
    diff = std::abs(dma_cycle - bdc_cycle);
    auto op_name = tmp_ops.size() > 0 ? show_op_info(tmp_ops[0]) : "null op";
    if (detail_log) {
      llvm::errs() << "i:" << i << ", dma_cycle:" << dma_cycle
                   << ", bdc_cycle:" << bdc_cycle << ", diff:" << diff
                   << ", cycle:" << cycle << ", max_dma_ratio:" << max_dma_ratio
                   << ", " << op_name << "\n";
    }
    cycle_info.cycle_diff = diff;
    total_diff += diff;
    cycle_info.ts_idx = i++;
    if (bdc_cycle > dma_cycle || max_dma_ratio < 0.5) {
      cycle_info.cut_op = tmp_ops.size() ? tmp_ops.front() : nullptr;
      cycle_info.mode = 0;
    } else {
      cycle_info.cut_op = max_cycle_dma_op;
      cycle_info.mode = max_cycle_is_load ? 1 : 2;
    }
    cycle_info.load_cycle_is_bigger = load_cycle > store_cycle;
    if (itr.vec_op_infos.size() && itr.vec_op_infos.back().slice_idx == 0) {
      ts_cycle.push_back(cycle_info);
    }
  }
}

bool ILPTimeStep::is_value_stored(Value value, int ts_idx) {
  for (auto it3 : timestep_table_new[ts_idx + 1].vec_ts_var) {
    if ((it3.value == value) &&
        (it3.info.mode2 & TIMESTEP2_STORE ||
         it3.info.mode2 == TIMESTEP2_ONLY_RESIDE ||
         (it3.info.mode2 & TIMESTEP2_STORE_AND_LOAD &&
          mapILPVarInfo[it3.varName].store_load_mode == 0))) {
      return true;
    }
  }
  return false;
}

bool ILPTimeStep::mem_alloc(mem_alloc_status &alloc_status,
                            std::vector<std::pair<Value, int64_t>> &value_size,
                            TensorInfo &tensor_infos, Operation *&fail_op,
                            int &nonOp_insert_mode) {
  LAYER_GROUP_LOG_DEBUG_BLOCK({ llvm::errs() << "mem_alloc start:\n"; });
  fail_op = nullptr;
  lmem_alloc_ptr = std::make_shared<lmem_alloc>(
      _group_info.group_banked_tensors, this, ts_count);
  // value_size.clear();
  if (value_size.size() > 0) {
    int min_always_free_mem_size = lmem_alloc_ptr->total_size;
    for (int i = 0; i < ts_count; i++) {
      if (i == 0 || i == ts_count - 1) {
        continue;
      }
      if (mem_contrains_new[i].size() > 0) {
        int size = 0;
        for (auto it : mem_contrains_new[i]) {
          if (mapILPVarInfo[it.second].ilp_var->solution_value() == 1) {
            size += it.first; // Adds up the memory footprint of the tensor
                              // present in the current time slot
          }
        }
        for (auto it : timestep_table_new[i].vec_op_infos) {
          double free_mem_size =
              it.mem_size_for_load -
              size; // The loadable memory for each op is subtracted by size
          if (free_mem_size < min_always_free_mem_size) {
            min_always_free_mem_size = free_mem_size;
          }
        }
      }
    }
    LAYER_GROUP_LOG_DEBUG_BLOCK({
      llvm::errs() << "min_always_free_mem_size:" << min_always_free_mem_size
                   << "\n";
    });

    LAYER_GROUP_LOG_DEBUG_BLOCK({ llvm::errs() << "add reside_value:\n"; });
    if (min_always_free_mem_size > 0) {
      int addr = lmem_alloc_ptr->total_size - 16, new_addr = 0; // todo why 16?
      for (auto itr : value_size) {
        if (min_always_free_mem_size > itr.second) {
          reside_value_info tmp;
          addr -= itr.second;
          new_addr = addr / 64 *
                     64; // The actual space used after the alignment is larger
          tmp.addr = new_addr;
          tmp.size = itr.second;
          map_reside_value_info[itr.first] = tmp;
          min_always_free_mem_size -= itr.second + addr - new_addr;
          LAYER_GROUP_LOG_DEBUG_BLOCK({
            llvm::errs() << "  name:" << module::getName(itr.first).str()
                         << ", addr:" << addr << "\n";
          });
        } else {
          LAYER_GROUP_LOG_DEBUG_BLOCK({
            llvm::errs() << "residual min_always_free_mem_size:"
                         << min_always_free_mem_size << "\n";
          });
          break;
        }
      }
    }
  }
  int i = 0, lack_mem_szie;
  std::string name;
  bool ret = false;

  LAYER_GROUP_LOG_DEBUG_BLOCK({ llvm::errs() << "show var value:\n"; });
  for (auto &itr : timestep_table_new) {
    LAYER_GROUP_LOG_DEBUG_BLOCK({
      llvm::errs() << "-------------------ts" << i << "--------------------\n";
    });
    int cycle = 0, total_cycle = 0;
    for (auto &itr2 : itr.vec_ts_var) {
      if (mapILPVarInfo[itr2.varName].ilp_var->solution_value() == 1) {
        itr2.var_value = 1;
        for (auto itr3 : cycle_contrains_new[i]) {
          if (itr3.second == itr2.varName) {
            cycle = itr3.first;
            break;
          }
        }
        LAYER_GROUP_LOG_DEBUG_BLOCK({
          llvm::errs() << "  dma var, name: " << itr2.varName
                       << ", cycle:" << cycle << "\n";
        });
        if (itr2.slice_idx == 0 || map_reside_value_info.find(itr2.value) ==
                                       map_reside_value_info.end()) {
          total_cycle += cycle;
        }
      }
    }

    for (auto itr2 : itr.vec_op_infos) {
      if (itr2.op) {
        auto outs = get_output_values(itr2.op);
        LAYER_GROUP_LOG_DEBUG_BLOCK({
          llvm::errs() << "  op name: " << module::getName(outs[0]).str()
                       << " , cycle:" << itr2.bdc_cycle
                       << ", free mem_size:" << itr2.mem_size_for_load << "\n";
        });
      }
    }
    i++;
  }

  i = 0;
  llvm::errs() << "gen vec_l2m_value_info:\n";
  for (auto &itr : timestep_table_new) {
    llvm::errs() << "-------------------ts" << i << "--------------------\n";
    ;
    for (auto itr2 : itr.vec_op_infos) {
      for (auto itr3 = itr2.vars_need_load_to_l2m.begin();
           itr3 != itr2.vars_need_load_to_l2m.end(); ++itr3) {
        // The resident weight of the first slice is also pre-loaded to l2m
        if (itr2.slice_idx == 0 || map_reside_value_info.find(itr3->first) ==
                                       map_reside_value_info.end()) {
          for (auto var : itr3->second) {
            if (mapILPVarInfo[var].ilp_var->solution_value() == 1) {
              name = module::getName(itr3->first).str();
              l2m_value_info info;
              info.slice_idx = itr2.slice_idx;
              info.value = itr3->first;
              // info.size = dam_tensor_size[std::make_pair(info.value,
              // itr2.slice_idx)];
              info.size = 0;
              info.free_ts = i + 1;
              info.load_ts = mapILPVarInfo[var].ts_idx - 1;
              vec_l2m_value_info.push_back(info);
              llvm::errs() << "tensor name:" << name << ", compute at pos:" << i
                           << ", load at ts:" << info.load_ts << "\n";
              break;
            }
          }
        }
      }
    }
    i++;
  }

  if (map_reside_value_info.size() > 0) {
    llvm::errs() << "cancel reside_value load:\n";
    ;
    for (int i = 0; i < ts_count; i++) {
      // The resident weight loading of the first slice is not cancelled, and
      // the following are cancelled, which can ensure the hiding during
      // loading, and cancel the loading of the following slice to reduce power
      // consumption
      if (i < ts_count - 2) {
        for (auto &it : timestep_table_new[i].vec_ts_var) {
          if (it.slice_idx > 0 && it.var_value == 1 &&
              map_reside_value_info.find(it.value) !=
                  map_reside_value_info.end()) {
            it.var_value = 0;
            llvm::errs() << "  name:" << module::getName(it.value).str()
                         << ", ts:" << i << "\n";
          }
        }
      }
    }
  }

  fprintf(stderr, "deal weight pre dma load:\n");
  for (int i = 0; i < ts_count; i++) {
    for (auto it : timestep_table_new[i].vec_ts_var) {
      name = module::getName(it.value).str();
      if (it.info.mode2 & TIMESTEP2_LOAD && it.var_value == 1) {
        if (map_reside_value_info.find(it.value) !=
            map_reside_value_info.end()) {
          int addr = map_reside_value_info[it.value].addr;
          int size = map_reside_value_info[it.value].size;
          lmem_alloc_ptr->alloc2(it.slice_idx, name, it.value, addr, size);
        }
      }
    }
  }
  // int unused;
  // lmem_alloc_ptr->show_mem(unused, unused, unused);

  std::vector<mem_alloc_req_info> vec_mem_req;
  std::vector<std::pair<int, int>> vec_pre_ts_free_mem, vec_pre_ts_free_mem_pre;
  std::set<std::string> used_op_name_splice;
  fprintf(stderr, "start analog allocation\n");
  for (int ts_idx = 0; ts_idx < ts_count; ts_idx++) {
    int total_store_size = 0, total_load_size = 0;
    fprintf(stderr, ">>>ts:%d\n", ts_idx);
    fprintf(stderr, "  deal dma load:\n");
    if (ts_idx < ts_count - 2) {
      for (auto it : timestep_table_new[ts_idx].vec_ts_var) {
        name = module::getName(it.value).str();
        if (it.info.mode2 & TIMESTEP2_LOAD && it.var_value == 1) {
          if (map_reside_value_info.find(it.value) ==
              map_reside_value_info.end()) {
            mem_alloc_req_info tmp;
            tmp.slice_idx = it.slice_idx;
            tmp.name = name;
            tmp.value = it.value;
            tmp.size = it.lmem_bytes;
            auto key = convert_name_to_key(tmp.name, tmp.slice_idx);
            if (used_op_name_splice.find(key) != used_op_name_splice.end()) {
              continue;
            }
            used_op_name_splice.insert(key);
            vec_mem_req.push_back(tmp);
            total_load_size += it.lmem_bytes;
            llvm::errs() << "    load value: "
                         << module::getName(it.value).str()
                         << ", slice_idx: " << it.slice_idx
                         << " for varName: " << it.varName << "\n";
          }
        }
        if (it.info.mode2 & TIMESTEP2_STORE_AND_LOAD && it.var_value == 1) {
          if (mapILPVarInfo[it.varName].store_load_mode == 1) {
            mem_alloc_req_info tmp;
            tmp.slice_idx = it.slice_idx;
            tmp.name = name;
            tmp.value = it.value;
            tmp.size = it.lmem_bytes;
            vec_mem_req.push_back(tmp);
            total_load_size += it.lmem_bytes;
            llvm::errs() << "    load value: "
                         << module::getName(it.value).str()
                         << ", slice_idx: " << it.slice_idx
                         << " for varName: " << it.varName << "\n";
          }
        }
      }
    }

    if (ts_idx > 0 && ts_idx < ts_count - 1) {
      fprintf(stderr, "  deal op compute:\n");
      bool have_mem_dependent = false;
      for (auto [op_idx, it2] :
           llvm::enumerate(timestep_table_new[ts_idx].vec_op_infos)) {
        if (!it2.op) {
          continue;
        }
        auto outs = get_output_values(it2.op);
        name = module::getName(outs[0]).str();
        llvm::errs() << "    op name: " << name << "\n";
        for (auto out : outs) {
          mem_alloc_req_info tmp;
          tmp.slice_idx = it2.slice_idx;
          tmp.name = module::getName(out).str();
          tmp.value = out;
          auto key2 = std::make_pair(out, it2.slice_idx);
          if (dam_tensor_size.find(key2) == dam_tensor_size.end()) {
            assert(false);
          }
          auto size = dam_tensor_size[key2];
          llvm::errs() << "      size: " << size << "\n";
          tmp.size = size;
          // if (tmp.size > 0) //The mask output of maxpool in the training
          // diagram sometimes does not need to be processed,why
          vec_mem_req.push_back(tmp);
        }

        int buffer_size = it2.buffer_size;
        // Placing the op buffer allocation after the outs allocation prevents
        // fragmentation from becoming complicated
        if (buffer_size > 0) {
          Value tmp_value;
          mem_alloc_req_info tmp;
          tmp.slice_idx = it2.slice_idx;
          tmp.name = name + "_buffer";
          tmp.value = tmp_value;
          tmp.size = buffer_size;
          vec_mem_req.push_back(tmp);
        }

        bool sort_by_size = false;
        // if (module::isDebugCmdEnable("disable_alloc_multi_sort_by_size")) {
        //   sort_by_size = false;
        // }
        ret = lmem_alloc_ptr->alloc_multi(ts_idx, op_idx, vec_mem_req,
                                          lack_mem_szie, sort_by_size);
        if (!ret) {
          fprintf(stderr, "      alloc_multi fail\n");
          if (ts_idx > 1) {
            for (auto it3 : timestep_table_new[ts_idx].vec_ts_var) {
              if (it3.var_value == 1 &&
                  (it3.info.mode2 & TIMESTEP2_STORE ||
                   (it3.info.mode2 & TIMESTEP2_STORE_AND_LOAD &&
                    mapILPVarInfo[it3.varName].store_load_mode == 0))) {
                total_store_size += it3.lmem_bytes;
              }
            }
          }
          if (total_load_size > lack_mem_szie) {
            nonOp_insert_mode = 0;
          } else if (total_store_size + total_load_size > lack_mem_szie) {
            nonOp_insert_mode = 1;
          } else {
            nonOp_insert_mode = 2;
          }
          fail_op = it2.op;
          return false;
        }
        // If a sub-OP in the composite op is already dependent on the previous
        // timestamp, the check is not repeated
        if (ts_idx > 1 && !have_mem_dependent) {
          // The current request cannot be dependent on any release of the
          // previous timestamp
          for (auto it1 : vec_mem_req) {
            mem_struct mem_s;
            auto key = convert_name_to_key(it1.name, it1.slice_idx);
            lmem_alloc_ptr->get_mem_struct(key, mem_s);
            for (auto it2 : vec_pre_ts_free_mem_pre) {
              if (is_range_overlap(it2.first, it2.second, mem_s.addr,
                                   mem_s.addr + mem_s.size)) {
                have_mem_dependent = true;
                llvm::errs() << "         vec_mem_req, key:" << key
                             << ", req start addr:" << mem_s.addr
                             << ", end addr:" << mem_s.addr + mem_s.size
                             << ", pre_ts start addr:" << it2.first
                             << ", end addr:" << it2.second << "\n";
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

        fprintf(stderr, "    deal input free:\n");
        auto ins = get_input_values(it2.op);
        auto last = std::unique(ins.begin(), ins.end());
        ins.erase(last, ins.end());
        for (auto in : ins) {
          // if (mapValueUserCount.find(in) == mapValueUserCount.end()) {
          //   mapValueUserCount[in] = 0;
          // }
          // mapValueUserCount[in] += 1; //
          // mapValueUserCount[in] == get_user_count_in_group(in,
          // _lgInfo.group_ops)

          bool to_be_used = false;
          // As long as the store or adaReside variable exists in the next time
          // slot, the memory
          //  release right is waived here. TIMESTEP2_LDST_UNKNOWN Specifies the
          //  resident value
          // As long as the store or adaReside variable exists in the next time
          // slot, the memory free right is waived here
          if (is_value_stored(in, ts_idx) || is_value_stored(in, ts_idx - 1)) {
            to_be_used = true;
            fprintf(stderr, "       value will be stored\n");
          }

          if (reside_in_tensor.find(it2.op) != reside_in_tensor.end()) {
            auto tensors = reside_in_tensor[it2.op];
            if (find(tensors.begin(), tensors.end(), in) != tensors.end()) {
              to_be_used = true;
              fprintf(stderr, "       reside_in_tensor\n");
            }
          }

          if (map_reside_value_info.find(in) != map_reside_value_info.end()) {
            to_be_used = true;
            fprintf(stderr, "       reside_value\n");
          }

          name = module::getName(in).str();
          if (!to_be_used) {
            lmem_alloc_ptr->free(convert_name_to_key(name, it2.slice_idx),
                                 &vec_pre_ts_free_mem);
          } else {
            fprintf(stderr, "        not need to free:%s\n", name.c_str());
          }
        }

        if (buffer_size > 0) {
          name = module::getName(outs[0]).str();
          lmem_alloc_ptr->free(
              convert_name_to_key(name + "_buffer", it2.slice_idx),
              &vec_pre_ts_free_mem);
        }
      }

      if (vec_pre_ts_free_mem_pre.size() > 0 && !have_mem_dependent) {
        fprintf(stderr, "        ts:%d and ts:%d no mem dependent\n", i, i - 1);
        // timestep_table_new[ts_idx].can_merge = true; //todo fix me
      }
    }

    fprintf(stderr, "  deal dma store:\n");
    if (ts_idx > 1) {
      // In the last store, this ts store cannot be used for the load of the
      // current time slot
      for (auto it : timestep_table_new[ts_idx].vec_ts_var) {
        name = module::getName(it.value).str();
        if (it.var_value == 1 &&
            (it.info.mode2 & TIMESTEP2_STORE ||
             (it.info.mode2 & TIMESTEP2_STORE_AND_LOAD &&
              mapILPVarInfo[it.varName].store_load_mode == 0))) {
          lmem_alloc_ptr->free(convert_name_to_key(name, it.slice_idx),
                               &vec_pre_ts_free_mem);
          fprintf(stderr, "    store value for varName:%s\n",
                  it.varName.c_str());
        }
      }
    }
    vec_pre_ts_free_mem_pre.clear();
    vec_pre_ts_free_mem_pre.assign(vec_pre_ts_free_mem.begin(),
                                   vec_pre_ts_free_mem.end());
    vec_pre_ts_free_mem.clear();
  }
  return true;
}

} // namespace tpu
} // namespace tpu_mlir
