#ifndef __PPL_MEM_H__
#define __PPL_MEM_H__
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <stdint.h>
#include <vector>
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"

struct __PplMemNode {
  int idx;
  int64_t live_s;
  int64_t live_e;
  std::set<int> bank_conf;
  __PplMemNode() {
    idx = -1;
    live_s = -1;
    live_e = -1;
  }
  __PplMemNode(int _idx, int64_t _live_s, int64_t _live_e,
               const std::set<int> &&_bank_conf)
      : idx(_idx), live_s(_live_s), live_e(_live_e), bank_conf(_bank_conf) {}
  __PplMemNode(__PplMemNode &&other)
      : idx(other.idx), live_s(other.live_s), live_e(other.live_e),
        bank_conf(other.bank_conf) {}
  __PplMemNode(const __PplMemNode &other)
      : idx(other.idx), live_s(other.live_s), live_e(other.live_e),
        bank_conf(other.bank_conf) {}
};

struct __PplPos {
  int idx;
  int64_t start = 0;
  int64_t end = 0;
  int64_t size = 0;
  int64_t first_pos = 0;
  int64_t end_pos = 0;
  __PplPos(int _idx, int64_t _size, int64_t _first_pos, int64_t _end_pos)
      : idx(_idx), size(_size), first_pos(_first_pos), end_pos(_end_pos) {}
};

static int64_t align_up(int64_t val, int64_t align) {
  return (val + align - 1) & ~(align - 1);
}
static std::shared_ptr<__PplPos>
searchAddr(int mem, std::map<int, __PplMemNode> &nodes,
           std::map<int, int64_t> &nodeSizes, int64_t offset,
           int64_t end_offset,
           std::list<std::shared_ptr<__PplPos>> &allocated_mem_list) {
  int64_t mem_size = nodeSizes[mem];
  std::shared_ptr<__PplPos> mem_addr = std::make_shared<__PplPos>(
      mem, mem_size, nodes[mem].live_s, nodes[mem].live_e);
  int64_t prev_offset = offset;
  int64_t best_offset = -1;
  int64_t smallest_gap = std::numeric_limits<int64_t>::max();
  for (auto &allocated_addr : allocated_mem_list) {
    if (allocated_addr->start > end_offset) {
      break;
    }
    int64_t max_first_pos =
        std::max(mem_addr->first_pos, allocated_addr->first_pos);
    int64_t min_last_pos = std::min(mem_addr->end_pos, allocated_addr->end_pos);
    if (max_first_pos < min_last_pos) {
      int64_t gap = allocated_addr->start - prev_offset;
      if (gap >= mem_addr->size && gap < smallest_gap) {
        smallest_gap = gap;
        best_offset = prev_offset;
      }
      prev_offset = std::max(prev_offset, allocated_addr->end);
    }
  }
  if (best_offset == -1) {
    best_offset = prev_offset;
  }
  mem_addr->start = best_offset;
  mem_addr->end = mem_addr->start + mem_size;
  return mem_addr;
}

static int getConflictCount(std::shared_ptr<__PplPos> mem_addr, int bankSize,
                            std::map<int, __PplMemNode> &nodes,
                            std::vector<std::vector<int>> &bank_mems) {
  int bank_start = mem_addr->start / bankSize;
  int bank_end = mem_addr->end / bankSize;
  int count = 0;
  for (auto i = bank_start; i <= bank_end; ++i) {
    for (auto mem : bank_mems[i]) {
      if (nodes[mem_addr->idx].bank_conf.count(mem)) {
        count++;
      }
    }
  }
  return count;
}

static void insertAddr(std::shared_ptr<__PplPos> mem_addr,
                       std::vector<std::vector<int>> &bank_mems,
                       std::list<std::shared_ptr<__PplPos>> &allocated_mem_list,
                       int bankSize, int64_t &total_consumer) {
  total_consumer = std::max(total_consumer, mem_addr->end);
  auto iter = std::find_if(allocated_mem_list.begin(), allocated_mem_list.end(),
                           [&mem_addr](std::shared_ptr<__PplPos> &p) {
                             return p->start >= mem_addr->start;
                           });
  allocated_mem_list.emplace(iter, mem_addr);

  int bank_start = mem_addr->start / bankSize;
  int bank_end = mem_addr->end / bankSize;
  for (auto i = bank_start; i <= bank_end; ++i) {
    bank_mems[i].push_back(mem_addr->idx);
  }
}

static int assignAddr(int bankSize, int bankNum,
                      std::map<int, __PplMemNode> &nodes,
                      std::map<int, int64_t> &nodeSizes,
                      std::vector<int64_t> &addrs, int64_t &totalSize) {
  std::list<std::shared_ptr<__PplPos>> allocated_mem_list;
  std::vector<std::vector<int>> bank_conf_list;
  std::list<int> mem_list;
  int64_t total_consumer = 0;

  bank_conf_list.resize(bankNum);
  int64_t max_mem_size = bankSize * bankNum;
  for (auto &node : nodes) {
    mem_list.push_back(node.first);
  }

  mem_list.sort([&](int a, int b) { return nodeSizes[a] >= nodeSizes[b]; });
  for (auto &mem : mem_list) {
    std::shared_ptr<__PplPos> best_addr;
    int min_conflict_count = std::numeric_limits<int>::max();
    for (auto i = 0; i < bankNum; ++i) {
      int64_t offset = i * bankSize;
      int mem_cross_bank_num =
          std::ceil(static_cast<float>(nodeSizes[mem]) / bankSize);
      int64_t end_offset = offset + (mem_cross_bank_num + 1) * bankSize;
      if (i + mem_cross_bank_num >= bankNum) {
        break;
      }
      auto mem_addr = searchAddr(mem, nodes, nodeSizes, offset, end_offset,
                                 allocated_mem_list);
      if (mem_addr->start + mem_addr->size <
          std::min(end_offset, max_mem_size)) {
        int conf_count =
            getConflictCount(mem_addr, bankSize, nodes, bank_conf_list);
        if (conf_count < min_conflict_count) {
          min_conflict_count = conf_count;
          best_addr = mem_addr;
        }
      }
    }
    if (!best_addr) {
#ifndef NDEBUG
      printf("Assign addr with Bank Conflict failed\n");
#endif
      return -1;
    }
    insertAddr(best_addr, bank_conf_list, allocated_mem_list, bankSize,
               total_consumer);
    addrs[mem] = best_addr->start;
  }
  totalSize = total_consumer;
  return 0;
}

static int assignAddrFast(int bankSize, int bankNum,
                          std::map<int, __PplMemNode> &nodes,
                          std::map<int, int64_t> &nodeSizes,
                          std::vector<int64_t> &addrs, int64_t &totalSize,
                          std::vector<int> &newOpIds,
                          std::map<int, __PplMemNode> &newNodes,
                          std::vector<std::vector<int>> &BankConf) {
  std::vector<std::vector<int>> bank_conf_list;
  int64_t total_consumer = 0;
  bank_conf_list.resize(bankNum);
  // int64_t max_mem_size = bankSize * bankNum;

  std::list<int> new_mem_list(newOpIds.begin(), newOpIds.end());
  std::map<int, int64_t> newNodeSizes;
  std::vector<std::vector<int64_t>> maxSizePerGroup(new_mem_list.size());
  std::vector<std::vector<std::vector<int32_t>>> opsPerGroup(
      new_mem_list.size());
  for (uint32_t i = 0; i < new_mem_list.size(); i++) {
    // reuse the buffer inside the group
    std::vector<std::tuple<int32_t, int32_t, int32_t>> nodesPerGroup;
    for (uint32_t j = 0; j < BankConf[i].size(); j++) {
      nodesPerGroup.emplace_back(nodes[BankConf[i][j]].idx,
                                 nodes[BankConf[i][j]].live_s,
                                 nodes[BankConf[i][j]].live_e);
    }

    std::sort(nodesPerGroup.begin(), nodesPerGroup.end(),
              [&](const std::tuple<int32_t, int32_t, int32_t> &a,
                  const std::tuple<int32_t, int32_t, int32_t> &b) {
                return nodeSizes[std::get<0>(a)] > nodeSizes[std::get<0>(b)];
              });

    for (auto &node : nodesPerGroup) {
      bool canGroup = false;
      for (uint32_t k = 0; k < opsPerGroup[i].size(); k++) {
        bool confict = false;
        for (uint32_t j = 0; j < opsPerGroup[i][k].size(); j++) {
          if (nodes[opsPerGroup[i][k][j]].live_s < std::get<2>(node) &&
              std::get<1>(node) < nodes[opsPerGroup[i][k][j]].live_e) {
            confict = true;
            break;
          }
        }

        if (confict)
          continue;
        else {
          opsPerGroup[i][k].emplace_back(std::get<0>(node));
          canGroup = true;
          break;
        }
      }

      if (!canGroup) {
        opsPerGroup[i].emplace_back(std::vector<int32_t>());
        opsPerGroup[i].back().emplace_back(std::get<0>(node));
      }
    }

    for (uint32_t j = 0; j < opsPerGroup[i].size(); j++) {
      int64_t maxSize = 0;
      for (uint32_t k = 0; k < opsPerGroup[i][j].size(); k++) {
        maxSize = std::max(maxSize, nodeSizes[opsPerGroup[i][j][k]]);
      }
      maxSizePerGroup[i].emplace_back(maxSize);
    }
    newNodeSizes[i] = std::accumulate(maxSizePerGroup[i].begin(),
                                      maxSizePerGroup[i].end(), 0);
  }

  new_mem_list.sort([&](int a, int b) {
    return newNodeSizes[a] % bankSize < newNodeSizes[b] % bankSize;
  });
  int64_t offset_bank = 0;
  for (auto &mem : new_mem_list) {
    int mem_cross_bank_num =
        std::ceil(static_cast<float>(newNodeSizes[mem]) / bankSize);
    if (offset_bank + mem_cross_bank_num > bankNum) {
#ifndef NDEBUG
      printf("Assign addr with Bank Conflict failed\n");
#endif
      return -1;
    }

    int64_t new_offset = offset_bank * bankSize;
    for (uint32_t i = 0; i < opsPerGroup[mem].size(); i++) {
      for (uint32_t j = 0; j < opsPerGroup[mem][i].size(); j++) {
        addrs[opsPerGroup[mem][i][j]] =
            new_offset + std::accumulate(maxSizePerGroup[mem].begin(),
                                         maxSizePerGroup[mem].begin() + i, 0);
      }
    }

    offset_bank += mem_cross_bank_num;
  }
  totalSize = total_consumer;
  return 0;
}

static int64_t assignAddr(std::map<int, __PplMemNode> &nodes,
                          std::map<int, int64_t> &nodeSizes,
                          std::vector<int64_t> &addrs) {
  std::list<std::shared_ptr<__PplPos>> mem_list;
  std::list<std::shared_ptr<__PplPos>> allocated_mem_list;
  for (auto &node : nodes) {
    std::shared_ptr<__PplPos> mem_addr =
        std::make_shared<__PplPos>(node.first, nodeSizes[node.first],
                                   node.second.live_s, node.second.live_e);
    mem_list.push_back(mem_addr);
    addrs[node.first] = -1;
  }

  mem_list.sort(
      [&](std::shared_ptr<__PplPos> &a, std::shared_ptr<__PplPos> &b) {
        return a->size >= b->size;
      });

  int64_t total_consumer = 0;
  for (auto &mem_addr : mem_list) {
    int64_t prev_offset = 0;
    int64_t best_offset = -1;
    int64_t smallest_gap = std::numeric_limits<int64_t>::max();
    for (auto &allocated_addr : allocated_mem_list) {
      int64_t max_first_pos =
          std::max(mem_addr->first_pos, allocated_addr->first_pos);
      int64_t min_last_pos =
          std::min(mem_addr->end_pos, allocated_addr->end_pos);
      if (max_first_pos < min_last_pos) {
        int64_t gap = allocated_addr->start - prev_offset;
        if (gap >= mem_addr->size && gap < smallest_gap) {
          smallest_gap = gap;
          best_offset = prev_offset;
        }
        prev_offset = std::max(prev_offset, allocated_addr->end);
      }
    }
    if (best_offset == -1) {
      best_offset = prev_offset;
    }
    mem_addr->start = best_offset;
    mem_addr->end = mem_addr->start + mem_addr->size;
    total_consumer = std::max(total_consumer, mem_addr->end);
    auto iter =
        std::find_if(allocated_mem_list.begin(), allocated_mem_list.end(),
                     [&mem_addr](std::shared_ptr<__PplPos> &p) {
                       return p->start >= mem_addr->start;
                     });
    allocated_mem_list.emplace(iter, mem_addr);
  }

  for (auto &mem_addr : allocated_mem_list) {
    addrs[mem_addr->idx] = mem_addr->start;
  }

  return total_consumer;
}

static int check_mem_valid(
    int64_t localMemSize, int64_t l2MemSize, int bankSize, int bankNum,
    std::map<int, __PplMemNode> &localMems, std::map<int, __PplMemNode> &l2Mems,
    std::map<int, int64_t> &memSizes, std::vector<int64_t> &addrs,
    std::vector<int> &newOpIds, std::map<int, __PplMemNode> &newNodes,
    std::vector<std::vector<int>> &BankConf, int memory_alloc_method = 0) {
  addrs.resize(memSizes.size());
  if (!localMems.empty()) {
    if (memory_alloc_method == 0) {
      int64_t total_consumer = 0;
      // total_consumer = assignAddr(localMems, memSizes, addrs);
      int ret = assignAddr(bankSize, bankNum, localMems, memSizes, addrs,
                           total_consumer);
      if (ret || total_consumer > localMemSize) {
#ifndef NDEBUG
        printf("Failed to allocate local mem! Not enough memory\n");
#endif
        return 0x11;
      }
    } else {
      int64_t total_consumer = 0;
      total_consumer = assignAddr(localMems, memSizes, addrs);
      if (total_consumer > localMemSize) {
#ifndef NDEBUG
        printf("Failed to allocate local mem! Not enough memory\n");
#endif
        return 0x11;
      }
      std::vector<int64_t> addrs_bak(memSizes.size());
      int ret = assignAddr(bankSize, bankNum, localMems, memSizes, addrs_bak,
                           total_consumer);
      if (!ret && (total_consumer <= localMemSize)) {
        addrs.swap(addrs_bak);
      }
    }
    // int ret = assignAddrFast(bankSize, bankNum, localMems, memSizes,
    // addrs_bak,
    //                          total_consumer, newOpIds, newNodes, BankConf);
  }
  if (!l2Mems.empty()) {
    int64_t total_consumer = assignAddr(l2Mems, memSizes, addrs);
    if (total_consumer > l2MemSize) {
#ifndef NDEBUG
      printf("Assign l2 mem failed\n");
#endif
      return 0x1A;
    }
  }
  return 0;
}

static int64_t calc_mem_size(const std::vector<int> &&shape, int npu_num,
                             int eu_bytes, int dsize, int type) {
  int eu_num = eu_bytes / dsize;
  int64_t stride_hw;
  switch (type) {
  case 0: { // CONTINUOUS
    return shape[0] * shape[1] * shape[2] * shape[3] * dsize;
  }
  case 1: { // TPU_ALIGN
    stride_hw = align_up(shape[2] * shape[3], eu_num);
    break;
  }
  case 2: { // TPU_COMPACT
    stride_hw = shape[2] * shape[3];
    break;
  }
  case 3: { // TPU_ROW_ALIGN
    stride_hw = shape[2] * align_up(shape[3], eu_num);
    break;
  }
  default: {
    assert(0 && "unsupport align mode!!!");
    break;
  }
  }
  int64_t aligned_c = std::ceil(static_cast<float>(shape[1]) / npu_num);
  return shape[0] * aligned_c * stride_hw * dsize;
}

#endif
