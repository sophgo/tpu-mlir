from typing import Dict, Callable, Any
from collections import deque
from mlir.ir import *
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
import time
import pymlir
from utils.mlir_parser import MlirParser
from calibration.data_selector import DataSelector
from utils.preprocess import preprocess
import copy
import datetime
from .cali_math import POOL_THREADS

import threading
import queue

pymlir.set_mem_mode("force_value_mem")


class DataDrivenDAGRunner:

    def __init__(self,
                 mlir_file: str = None,
                 module: pymlir.module = None,
                 parser: MlirParser = None,
                 ds: DataSelector = None,
                 img_list: list = None,
                 ref_act: dict = None,
                 op_callback: Callable = None):
        self.value_registry: Dict[Value, Any] = {}  # SSA value → computed result
        self.dependency_graph: Dict[Operation, set] = {}  # Op → dependent ops
        self.reverse_deps: Dict[Operation, set] = {}  # Op → ops it depends on
        self.ready_queue = deque()  # Ops ready for execution
        if module is not None:
            assert parser is not None, "Parser must be provided if module is given"
            self.module = module
            self.parser = parser
        else:
            assert mlir_file is not None, "MLIR file must be provided"
            self.module = pymlir.module()
            self.module.load(mlir_file)
            self.parser = MlirParser(mlir_file)
        self.batch_size = self.parser.get_batch_size()
        self.input_num = self.parser.get_input_num()
        self.ppa_list = []
        for i in range(self.input_num):
            tmp = preprocess()
            tmp.load_config(self.parser.get_input_op_by_idx(i))
            self.ppa_list.append(tmp)
        self.build_dependency_graph()
        self.op_callback = op_callback

        self.ds = ds if ds is not None else None
        self.data_list = self.ds.data_list if self.ds is not None else img_list
        self.net_inputs = ref_act if ref_act is not None else {}

        self.samples = len(self.data_list) if self.data_list is not None else len(
            self.net_inputs) if self.net_inputs else 0
        assert self.samples > 0, "No input data provided"
        if ref_act is not None and (ds is not None or img_list is not None):
            print(
                "Warning: Both reference activations and input data provided; using reference activations."
            )
            sys.exit(1)
        if ds is not None and img_list is not None:
            print("Warning: Both DataSelector and image list provided; using DataSelector.")
            sys.exit(1)
        if self.ds is not None and self.ds.all_image:
            n = self.samples % self.batch_size
            if n != 0:
                for i in range(self.batch_size - n):
                    self.data_list.append(self.data_list[-1])
                    samples += 1
            self.samples = self.samples // self.batch_size
        if self.ds is not None:
            self.load_net_input_ds()
        elif self.data_list is not None and len(self.data_list) > 0:
            self.load_net_input_img()

    def load_net_input_ds(self):
        inp_ref_dict = {}
        for input in self.module.input_names:
            inp_ref_dict[input] = self.parser.get_use_count_by_op_name(input)

        if self.ds.all_image:
            batched_inputs = self.input_num * ['']
        else:
            batched_inputs = {}
        idx = 0
        self.net_inputs[idx] = {}
        only_one = len(self.module.input_names) == 1
        for data in self.data_list:
            if self.ds.all_npz:
                x = np.load(data)
                if only_one:
                    assert (len(x.files) == 1)
                    n0 = self.module.input_names[0]
                    n1 = x.files[0]
                    if x[n1].shape[0] > 1:
                        self.net_inputs[idx][n0] = [x[n1], inp_ref_dict[n0]]
                    else:
                        batched_inputs[n1] = (np.concatenate(
                            [batched_inputs[n1], x[n1].astype(np.float32)], axis=0)
                                              if n1 in batched_inputs else x[n1].astype(np.float32))
                        if batched_inputs[n1].shape[0] >= self.batch_size:
                            self.net_inputs[idx][n0] = [
                                batched_inputs[n1][:self.batch_size], inp_ref_dict[n0]
                            ]
                            batched_inputs.pop(n1)
                        else:
                            continue
                else:
                    for input in self.module.input_names:
                        assert (input in x)
                        if x[input].shape[0] > 1:
                            self.net_inputs[idx][input] = [x[input], inp_ref_dict[input]]
                            batch_size = self.batch_size
                        else:
                            batched_inputs[input] = (np.concatenate(
                                [batched_inputs[input], x[input].astype(np.float32)],
                                axis=0) if input in batched_inputs else x[input].astype(np.float32))
                            batch_size = batched_inputs[input].shape[0]
                            if batched_inputs[input].shape[0] >= self.batch_size:
                                real_batch_size = self.parser.get_op_by_op_name(input).shape[0]
                                self.net_inputs[idx][input] = [
                                    batched_inputs[input][:real_batch_size], inp_ref_dict[input]
                                ]
                                batched_inputs.pop(input)

                    if batch_size < self.batch_size:
                        continue

            elif self.ds.all_image:
                inputs = data.split(',')
                inputs = [s.strip() for s in inputs]
                assert (self.input_num == len(inputs))
                for i in range(self.input_num):
                    batched_inputs[i] += '{},'.format(inputs[i])
                    if (idx + 1) % self.batch_size == 0:
                        x = self.ppa_list[i].run(batched_inputs[i][:-1])
                        name = self.ppa_list[i].input_name
                        self.net_inputs[idx][name] = [x, inp_ref_dict[name]]
                        batched_inputs = self.input_num * ['']
            else:
                self.net_inputs[idx] = {}
                inputs = data.split(',')
                inputs = [s.strip() for s in inputs]
                assert (self.input_num == len(inputs))
                for name, input in zip(self.module.input_names, inputs):
                    x = np.load(input)
                    self.net_inputs[idx][name] = [x, inp_ref_dict[name]]
            idx += 1
            self.net_inputs[idx] = {}

        if len(self.net_inputs[idx]) == 0:
            print(f'last input data (idx={idx}) not valid, droped')
            self.net_inputs.pop(idx)
        self.samples = min(self.samples, len(self.net_inputs))
        print(f"input_num = {self.samples}, ref = {len(self.net_inputs)}")

    def load_net_input_img(self):
        self.net_inputs = {}
        inp_ref_dict = {}
        for input in self.module.input_names:
            inp_ref_dict[input] = self.parser.get_use_count_by_op_name(input)

        batched_inputs = self.input_num * ['']
        idx = 0
        self.net_inputs[idx] = {}
        only_one = len(self.module.input_names) == 1
        if not only_one:
            print(f'Warning: multiple inputs detected, not support for image input')
            sys.exit(1)
        for data in self.data_list:
            inputs = data.split(',')
            inputs = [s.strip() for s in inputs]
            assert (self.input_num == len(inputs))
            for i in range(self.input_num):
                batched_inputs[i] += '{},'.format(inputs[i])
                if (idx + 1) % self.batch_size == 0:
                    x = self.ppa_list[i].run(batched_inputs[i][:-1])
                    name = self.ppa_list[i].input_name
                    self.net_inputs[idx][name] = [x, inp_ref_dict[name]]
                    batched_inputs = self.input_num * ['']
            idx += 1
            self.net_inputs[idx] = {}

        if len(self.net_inputs[idx]) == 0:
            print(f'last input data (idx={idx}) not valid, droped')
            self.net_inputs.pop(idx)
        self.samples = min(self.samples, len(self.net_inputs))
        print(f"input_num = {self.samples}, ref = {len(self.net_inputs)}")

    def register_operation_runner(self, op_name: str, runner: Callable):
        """Register a runner for a specific operation type"""
        self.operation_runners[op_name] = runner

    def build_dependency_graph(self):
        """Analyze the MLIR module to build execution dependencies"""
        # Clear previous state
        self.dependency_graph.clear()
        self.reverse_deps.clear()
        self.value_registry.clear()

        # Process all operations in all blocks
        for op_name in self.parser.get_op_name_list():
            self._process_operation(op_name)

    def _process_operation(self, op_name: str):
        """Process a single operation and its dependencies"""
        # if op_name not in self.operation_runners:
        #     raise ValueError(f"No runner registered for operation: {op_name}")

        # Initialize dependency tracking
        self.dependency_graph[op_name] = set()
        self.reverse_deps[op_name] = set()

        # Check all operands
        ready = True
        for operand in self.parser.get_pre_op_by_op_name(op_name):
            # Record dependency on defining operation
            self.dependency_graph[operand].add(op_name)
            self.reverse_deps[op_name].add(operand)
            ready = False

        if ready:
            self.ready_queue.append(op_name)

    def execute(self, idx):
        """Execute the DAG in topological order"""
        assert (idx < self.samples), f"Index {idx} out of range for samples {self.samples}"
        r_q = copy.deepcopy(self.ready_queue)
        d_g = copy.deepcopy(self.dependency_graph)
        r_d = copy.deepcopy(self.reverse_deps)
        while r_q:
            op_name = r_q.popleft()
            if op_name in self.module.input_names:
                self.module.set_tensor(op_name, self.net_inputs[idx][op_name][0],
                                       self.net_inputs[idx][op_name][0].shape)
            else:
                # print(f'start {op_name} at {datetime.datetime.now()}')
                self.module.invoke_at(op_name)
                # print(f'end {op_name} at {datetime.datetime.now()}')
            if self.op_callback is not None:
                self.op_callback(op_name, idx)

            # Notify dependents and check if they're ready
            for dependent in d_g[op_name]:
                r_d[dependent].remove(op_name)
                if not r_d[dependent]:
                    r_q.append(dependent)
        results = []
        for name in self.module.output_names:
            t = self.module.get_tensor(name)
            results.append((name, t))
        return results

    def _execute_operation(self, op_name: str, idx: int):
        """Execute a single operation using registered runner"""
        # Get operand values
        # operands = [self.value_registry.get(operand) for operand in self.parser.get_pre_op_by_op_name(op_name)]

        # Run the operation
        if op_name in self.module.input_names:
            self.module.set_tensor(op_name, self.net_inputs[idx][op_name][0],
                                   self.net_inputs[idx][op_name][0].shape)
        else:
            # print(f'start {op_name} at {datetime.datetime.now()}')
            self.module.invoke_at(op_name)
            # print(f'end {op_name} at {datetime.datetime.now()}')
        return op_name

    def parallel_execute(self, idx, max_workers=36):
        """Execute ready operations in parallel"""
        assert (idx < self.samples), f"Index {idx} out of range for samples {self.samples}"
        r_q = copy.deepcopy(self.ready_queue)
        d_g = copy.deepcopy(self.dependency_graph)
        r_d = copy.deepcopy(self.reverse_deps)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while r_q:
                # Execute all ready ops in parallel
                futures = []
                # print(f'Executing operations for index {idx}, ready queue size: {len(r_q)}')
                while r_q:
                    op_name = r_q.popleft()
                    futures.append(executor.submit(self._execute_operation, op_name, idx))
                for future in as_completed(futures):
                    if future.exception() is not None:
                        print(f"Error: {future.exception()}")
                    op_name = future.result()
                    for dependent in d_g[op_name]:
                        r_d[dependent].remove(op_name)
                        if not r_d[dependent]:
                            r_q.append(dependent)
        results = []
        for name in self.module.output_names:
            t = self.module.get_tensor(name)
            results.append((name, t))
        return results

    def parallel_thread_execute(self, idx, max_workers=5):
        """Execute ready operations in parallel"""
        assert (idx < self.samples), f"Index {idx} out of range for samples {self.samples}"
        r_q = copy.deepcopy(self.ready_queue)
        d_g = copy.deepcopy(self.dependency_graph)
        r_d = copy.deepcopy(self.reverse_deps)

        def execute_thread(queue: queue.Queue):
            while True:
                op_name, idx = queue.get()
                if op_name is None or idx < 0:
                    break
                if op_name in self.module.input_names:
                    self.module.set_tensor(op_name, self.net_inputs[idx][op_name][0],
                                           self.net_inputs[idx][op_name][0].shape)
                else:
                    self.module.invoke_at(op_name)
                if self.op_callback is not None:
                    self.op_callback(op_name, idx)
                for dependent in d_g[op_name]:
                    r_d[dependent].remove(op_name)
                    if not r_d[dependent]:
                        ready_q.put(dependent)

        work_queue = []
        for i in range(max_workers):
            work_queue.append(queue.Queue(maxsize=5))
        worker = []
        for i in range(max_workers):
            worker.append(threading.Thread(target=execute_thread, args=(work_queue[i], )))
        ready_q = queue.Queue(maxsize=20)
        while r_q:
            ready_q.put(r_q.popleft())
        done_op = len(self.parser.ops) - len(r_q)

        for i in range(max_workers):
            worker[i].start()
        while done_op > 0:
            dispatched = False
            op_name = ready_q.get()
            for i in range(max_workers):
                if work_queue[i].empty():
                    work_queue[i].put((op_name, idx))
                    dispatched = True
                    break
            if not dispatched:
                for i in range(max_workers):
                    if not work_queue[i].full():
                        work_queue[i].put((op_name, idx))
                        dispatched = True
                        break
            if not dispatched:
                print("all work queue full, fault")
                sys.exit(1)
            done_op -= 1
        for i in range(max_workers):
            work_queue[i].put((None, -1))
        for i in range(max_workers):
            worker[i].join()
        results = []
        for name in self.module.output_names:
            t = self.module.get_tensor(name)
            results.append((name, t))
        return results

    def set_scheduling_policy(self, policy: str):
        """Set execution scheduling policy"""
        if policy == "fifo":
            self.ready_queue = deque()  # Default FIFO
        elif policy == "lifo":
            self.ready_queue = []  # Stack behavior
        elif policy == "priority":
            self.ready_queue = PriorityQueue()  # Needs custom comparator
