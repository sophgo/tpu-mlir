# -*- coding: utf-8 -*-
import torch
import pdb
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition
from torch.fx.graph_module import GraphModule
from torch.fx.node import _get_qualified_name
from torch.fx.passes.operator_support import OperatorSupport
from typing import Dict, List, Optional, Sequence
MIN_BLOCK_SIZE = 5

class TpuMlirPartitioner(CapabilityBasedPartitioner):
    """Partitioner to split an FX graph into subgraphs based on operator support

    Args:
        graph_module: FX GraphModule to partition
        operator_support: OperatorSupport class describing allowed operators
        non_compute_ops: Operators which are not considered computational (e.g. getattr)
        allowed_single_node_partition_ops: Nodes which can be included in single-node partitons.
            Generally useful for module-level exclusion ops which are intensive despite being single functions
        min_block_size: Minimum number of computational operators per block
    Returns:
        torch.fx.GraphModule
    """
    def __init__(
        self,
        graph_module: GraphModule,
        operator_support: OperatorSupport,
        *,
        non_compute_ops: Optional[Sequence[str]] = None,
        allowed_single_node_partition_ops: Optional[Sequence[str]] = None,
        min_block_size=MIN_BLOCK_SIZE,
    ) -> None:
        super().__init__(
            graph_module,
            operator_support,
            allows_single_node_partition=True,
            non_compute_ops=non_compute_ops,
            allowed_single_node_partition_ops=allowed_single_node_partition_ops,
        )

        self.min_block_size = min_block_size

    def propose_partitions(self) -> List[Partition]:
        # Propose partitions using the default, then refine the results
        initial_proposed_partitions = super().propose_partitions()
        partitions = {i: part for i, part in enumerate(initial_proposed_partitions)}
        print(f'propose_partitions initial_partitions: ')
        for i in partitions:
            print(f'{i}th partitions: {partitions[i]}')

        partitions_to_remove = {}
        if len(initial_proposed_partitions) == 1: #整个图只有1个划分
            invalid = False
            for node in self.graph_module.graph.nodes:
                if node not in initial_proposed_partitions[0].nodes:
                    invalid = True #原始图有节点不在这唯一的划分中，该划分无效，原始问题是?
                    break
            if invalid:
                partitions_to_remove[0] = 0

        # For each partition, determine whether or not the number of computational operators
        # exceeds the threshold, and if not, remove that partition
        for id, partition in partitions.items():
            default_non_compute_ops = {"torch.ops.aten.view", "_operator.getitem"}
            non_compute_ops = default_non_compute_ops.union(set(self.non_compute_ops))
            exempted_partition = False

            compute_node_count = 0
            for node in partition.nodes:
                # Partitions are exempted from min_block_size if they contain an allowed single-node op
                if (
                    node.op == "call_function"
                    and _get_qualified_name(node.target)
                    in self.allowed_single_node_partition_ops
                ):
                    exempted_partition = True
                    break
                elif (
                    node.op == "call_function"
                    and _get_qualified_name(node.target) not in non_compute_ops
                ):
                    compute_node_count += 1

            if compute_node_count < self.min_block_size and not exempted_partition:
                partitions_to_remove[id] = compute_node_count

        # Remove any nodes violating the criteria specified by the user
        for id, count in partitions_to_remove.items():
            print(
                f"Removing {id} partition which has {count} < {self.min_block_size} computational operators"
            )
            del partitions[id]

        print(f'propose_partitions updated_partitions: ')
        for i in partitions:
            print(f'{i}th partitions: {partitions[i]}')

        return [partitions[k] for k in sorted(partitions.keys())]

    def partition_and_fuse(self) -> GraphModule:
        partitions = self.propose_partitions()
        fused_gm = self.fuse_partitions(partitions)
        return fused_gm

class TpuMlirOperatorSupport(OperatorSupport):
    """Class to determine whether operators within a module are supported"""

    def __init__(self, support_dict=None, torch_executed_ops=set()):
        super().__init__(support_dict)

        # Initialize sets of supported/unsupported operators
        self.supported_operators = set()
        self.unsupported_operators = set()
        self.torch_executed_ops = torch_executed_ops

    def is_node_supported(
        self, submodules: Dict[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        return True
        node_name = (
            _get_qualified_name(node.target)
            if not isinstance(node.target, str)
            else node.target
        )

        if node_name == 'torch.ops.aten.nll_loss_forward':
            print('aten.nll_loss_forward not support')
            return False
        if node_name == 'torch.ops.aten.expand':
            print('aten.expand not support because of bug')
            return False
        # if node_name == 'torch.ops.aten.relu': #仅仅用于分割测试
        #     print('aten.relu not support')
        #     return False
        # if node_name == 'torch.ops.aten.sum':
        #     print('aten.sum not support')
        #     return False
        if node_name == 'torch.ops.aten.max_pool2d_with_indices':
            print('aten.max_pool2d_with_indices not support by backend')
            return False
        if node_name == 'torch.ops.aten.max_pool2d_with_indices_backward':
            print('aten.max_pool2d_with_indices_backward not support by backend')
            return False
        if node_name == 'output':
            print('output not support')
            return False
        if node_name == 'torch.ops.aten.add' and list(node.meta['val'].size()) == []:
            print('aten.add.default for scalar not support')
            return False
        return True

    def print_support_overview(self, num_trt_blocks: Optional[int] = None):
        print("\nAll Nodes Supported\n")


def partition(
    gm: torch.fx.GraphModule,
    verbose: bool = True,
    min_block_size: int = MIN_BLOCK_SIZE,
    torch_executed_ops: Sequence[str] = set(),
) -> torch.fx.GraphModule:
    """Partition an FX GraphModule with aten ops into TRT engines
    Partitioning is based on converter operator support

    Args:
        gm: FX GraphModule to partition
        verbose: Bool representing whether to print operator support
        min_block_size: Minimum number of operators per TRT-Engine Block
        torch_executed_ops: Sequence of operations to run in Torch, regardless of converter coverage
    Returns:
        torch.fx.GraphModule
    """
    supported_ops = TpuMlirOperatorSupport(torch_executed_ops=torch_executed_ops)
    partitioner = TpuMlirPartitioner(gm, supported_ops, min_block_size=min_block_size)

    # Determine partitions based on user specifications and operator support
    # Then, fuse partitions and display overview of supported/unsupported operators
    partitions = partitioner.propose_partitions()
    fused_graph = partitioner.fuse_partitions(partitions)
    # partitioner.remove_bookend_non_compute_ops(partitions)

    # if verbose:
    #     supported_ops.print_support_overview(len(partitions))

    return fused_graph
