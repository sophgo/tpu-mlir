// TopoSorter.h

#pragma once
#include <queue>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tpu_mlir {
class TopoSorter {
private:
  std::unordered_map<std::string, std::vector<std::string>> graph;
  std::unordered_map<std::string, int> descendants;
  std::unordered_map<std::string, std::string> parent;
  std::vector<std::string> topologicalOrder;
  int _cost;

  int countDescendants(const std::string &node,
                       std::unordered_map<std::string, int> &memo) {
    if (memo.find(node) != memo.end()) {
      return memo[node];
    }

    int total_descendants = 0;
    for (const auto &neighbor : graph[node]) {
      total_descendants += 1 + countDescendants(neighbor, memo);
    }

    memo[node] = total_descendants;

    return total_descendants;
  }

  void
  makeParent(const std::vector<std::pair<std::string, std::string>> &edges) {
    for (const auto &edge : edges) {
      if (parent.find(edge.first) == parent.end()) {
        parent[edge.first] = edge.first;
      }
      parent[edge.second] = edge.first;
      // std::cout << "[" << '"' << edge.first << '"' << "," << '"' << edge.second
      //           << '"' << "]," << "\n";
    }
  }

public:
  std::string getParent(const std::string &key) { return parent[key]; }
  int getCost() { return _cost; }
  int getTime() { return topologicalOrder.size(); }

  std::unordered_map<std::string, int> topologicalSortWithPriority(
      const std::vector<std::pair<std::string, std::string>> &edges) {
    for (const auto &edge : edges) {
      graph[edge.first].push_back(edge.second);
    }

    makeParent(edges);

    std::unordered_map<std::string, int> inDegree;
    for (const auto &edge : edges) {
      inDegree[edge.first] = 0;
      inDegree[edge.second] = 0;
    }

    for (const auto &edge : edges) {
      inDegree[edge.second]++;
      // std::cout << edge.first << " =asd " << edge.second << "\n";
    }

    std::unordered_map<std::string, int> memo;
    for (const auto &kv : graph) {
      descendants[kv.first] = countDescendants(kv.first, memo);
    }

    std::priority_queue<std::pair<int, std::string>,
                        std::vector<std::pair<int, std::string>>,
                        std::greater<>>
        pq;
    for (const auto &node : inDegree) {
      if (node.second == 0) {
        pq.push({descendants[node.first], node.first});
      }
      // std::cout << node.first << " degree = " << node.second << "\n";
    }

    while (!pq.empty()) {
      auto front = pq.top();
      // std::cout << front.first << " priority= " << front.second << "\n";
      pq.pop();
      std::string node = front.second;
      topologicalOrder.push_back(node);

      for (const std::string &neighbor : graph[node]) {
        inDegree[neighbor]--;
        if (inDegree[neighbor] == 0) {
          pq.push({descendants[neighbor], neighbor});
        }
      }
    }

    if (topologicalOrder.size() != graph.size()) {
      throw std::runtime_error("The graph has cycles");
    }

    std::unordered_map<std::string, int> ret;

    int cost = 0;
    for (int i = 0; i < topologicalOrder.size(); i++) {
      // std::cout << i << " final= " << topologicalOrder[i] << "\n";
      ret[topologicalOrder[i]] = i;
    }
    for (int i = 0; i < topologicalOrder.size(); i++) {
      cost += i - ret[parent[topologicalOrder[i]]];
    }

    _cost = cost;

    return std::move(ret);
  }
};

} // namespace tpu_mlir
