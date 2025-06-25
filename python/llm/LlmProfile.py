# Copyright (C) 2025 Sophgo Technologies In"  All" rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import os
import math


class FAttention:
    # prefill:
    # [B, NUM_Q_HEAD, L, HEAD_DIM] @ [B, NUM_Q_HEAD, HEAD_DIM, L] => [B, NUM_Q_HEAD, L, L]
    # [B, NUM_Q_HEAD, L, L] @ [B, NUM_Q_HEAD, L, HEAD_DIM] => [B, NUM_Q_HEAD, L, HEAD_DIM]
    # decode:
    # [B, NUM_Q_HEAD, 1, HEAD_DIM] @ [B, NUM_Q_HEAD, HEAD_DIM, L+1] => [B, NUM_Q_HEAD, 1, L+1]
    # [B, NUM_Q_HEAD, 1, L+1] @ [B, NUM_Q_HEAD, L+1, HEAD_DIM] => [B, NUM_Q_HEAD, 1, HEAD_DIM]
    def __init__(self, NUM_Q_HEAD, SEQ, HEAD_DIM, prefill: bool = True):
        self.B = 1 * NUM_Q_HEAD
        self.M0 = SEQ if prefill else 1
        self.K0 = HEAD_DIM
        self.N0 = SEQ if prefill else SEQ + 1
        self.M1 = SEQ if prefill else 1
        self.K1 = SEQ if prefill else SEQ + 1
        self.N1 = HEAD_DIM

    def get_flops(self):
        self.flops = 2 * self.B * (self.M0 * self.K0 * self.N0 + self.M1 * self.K1 * self.N1)
        return self.flops

    def get_bytes(self, quantize_type, group_size):
        self.bytes = 2 * self.B * (self.M0 * self.K0 + self.K0 * self.N0 + self.K1 * self.N1 +
                                   self.M1 * self.N1) + 2 * self.M0 * self.N0
        return self.bytes


class MatMul:

    def __init__(self, L: list, R: list):
        assert (len(L) >= len(R))
        self.B = 1
        if len(L) > 2:
            self.B = math.prod(L[:-2])
        self.M = L[-2]
        self.K = L[-1]
        self.N = R[-1]
        self.io_bytes = 2 * self.B * (self.M * self.K + self.M * self.N)

    def get_flops(self):
        self.flops = 2 * self.B * self.M * self.K * self.N
        return self.flops

    def get_bytes(self, quantize_type="f16", group_size=64):
        if quantize_type == "f16" or quantize_type == "bf16":
            self.bytes_a16 = self.io_bytes + 2 * self.K * self.N
            return self.bytes_a16
        elif quantize_type == "w8f16" or quantize_type == "w8bf16":
            self.bytes_w8a16 = self.io_bytes + self.K * self.N + 3 * self.K * self.N / group_size
            return self.bytes_w8a16
        elif quantize_type == "w4f16" or quantize_type == "w4bf16":
            self.bytes_w4a16 = self.io_bytes + 0.5 * self.K * self.N + 3 * self.K * self.N / group_size
            return self.bytes_w4a16
        else:
            raise ValueError(f"Unsupported quantize type: {quantize_type}")


class LlmProfiler:

    def __init__(self, args, config):
        # user config
        self.seq_length = args.seq_length
        self.quantize = args.quantize
        self.group_size = args.group_size
        self.chip = args.chip
        self.num_device = args.num_device
        self.num_core = args.num_core
        self.lmhead_quantize = self.quantize if args.quant_lmhead else 'f16'
        self.model_name = os.path.basename(args.model_path.rstrip('/')).lower()
        self.profile_name = f"{self.model_name}_{self.quantize}_seq{self.seq_length}_{self.chip}"
        # chip config
        if self.chip == "bm1684x":
            self.tflops = 16.
            self.dma_bw = 64.
            self.p2p_bw = 3.
            self.num_core = 1
            self.prefill_mac_util = 0.55
            self.prefill_ddr_util = 0.3
            self.decode_mac_util = 0.2
            self.decode_ddr_util = 0.85
            self.tpu_freq = 950
            self.profile_name += f"_{self.num_device}dev"
        elif self.chip == "bm1688":
            self.tflops = 3.6
            self.dma_bw = 32.
            self.num_device = 1
            self.prefill_mac_util = 0.5
            self.prefill_ddr_util = 0.1
            self.decode_mac_util = 0.1
            self.decode_ddr_util = 0.8
            self.tpu_freq = 900
            self.profile_name += f"_{self.num_core}core"
        elif self.chip == "cv186x":
            self.tflops = 1.8
            self.dma_bw = 24.
            self.num_core = 1
            self.num_device = 1
            self.prefill_mac_util = 0.5
            self.prefill_ddr_util = 0.1
            self.decode_mac_util = 0.1
            self.decode_ddr_util = 0.8
            self.tpu_freq = 750
            self.profile_name += f"_{self.num_core}core"
        else:
            raise ValueError(f"Unsupported chip type: {args.chip}")
        self.tpu_freq = args.tpu_freq if args.tpu_freq is not None else self.tpu_freq
        self.profile_name += f"_{self.tpu_freq}MHz"
        # model config
        self.hidden_size = config.hidden_size
        self.interm_size = config.intermediate_size
        self.num_attn_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_attn_heads)
        self.q_dim = self.num_attn_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.kv_tile = self.num_attn_heads // self.num_kv_heads
        self.tie_word_embeddings = config.tie_word_embeddings
        self.quantization_config = getattr(config, "quantization_config", None)
        if self.quantization_config:
            self.group_size = self.quantization_config["group_size"]

    def get_tiu_time(self, total_flops, mac_util, tflops=0.):
        tflops = self.tflops if tflops == 0. else tflops
        tpu_peak_flops = tflops * 1024 * 1e9
        tiu_time_ms = total_flops / tpu_peak_flops / (self.tpu_freq / 1000) / mac_util * 1000
        return tiu_time_ms

    def get_dma_time(self, total_bytes, ddr_util):
        gbytes_per_dev = total_bytes / 2**30
        dma_time_ms = gbytes_per_dev / self.dma_bw / (self.tpu_freq / 1000) / ddr_util * 1000
        return dma_time_ms

    def get_allreduce_time(self, prefill: bool = True):
        allreduce_num = self.num_layers * 2
        if prefill:
            bf16_size = 2
            ring_data_ratio = (self.num_device - 1) * 2 / self.num_device
            gbytes = self.seq_length * self.hidden_size * bf16_size * ring_data_ratio * allreduce_num / 2**30
            p2p_time_ms = gbytes / self.p2p_bw * self.tpu_freq
            add_time_ms = (gbytes * 3 / self.dma_bw) / self.prefill_ddr_util * self.tpu_freq
            allreduce_time_ms = p2p_time_ms + add_time_ms
            return allreduce_time_ms
        else:
            time_per_allreduce = 0
            if self.num_device == 2:
                time_per_allreduce = 0.12
            elif self.num_device == 4:
                time_per_allreduce = 0.15
            elif self.num_device == 8:
                time_per_allreduce = 0.2
            elif self.num_device == 1:
                return 0
            allreduce_time_ms = time_per_allreduce * allreduce_num
            return allreduce_time_ms

    def get_pcie_interrupt_time(self, pcie_avg_ms):
        pcie_time_ms = pcie_avg_ms * self.num_layers * 2
        return pcie_time_ms

    def _analyze_prefill(self):
        self.prefill_stage = [
            # attn
            MatMul([1, self.seq_length, self.hidden_size],
                   [self.hidden_size, self.q_dim / self.num_device]),
            MatMul([1, self.seq_length, self.hidden_size],
                   [self.hidden_size, self.kv_dim / self.num_device]),
            MatMul([1, self.seq_length, self.hidden_size],
                   [self.hidden_size, self.kv_dim / self.num_device]),
            FAttention(self.num_attn_heads / self.num_device,
                       self.seq_length,
                       self.head_dim,
                       prefill=True),
            MatMul([1, self.seq_length, self.q_dim / self.num_device],
                   [self.q_dim / self.num_device, self.hidden_size]),
            # mlp
            MatMul([1, self.seq_length, self.hidden_size],
                   [self.hidden_size, self.interm_size / self.num_device]),
            MatMul([1, self.seq_length, self.hidden_size],
                   [self.hidden_size, self.interm_size / self.num_device]),
            MatMul([1, self.seq_length, self.interm_size / self.num_device],
                   [self.interm_size / self.num_device, self.hidden_size])
        ]
        lm_head = MatMul([1, self.hidden_size / self.num_device],
                         [self.hidden_size / self.num_device, self.vocab_size])
        lmhead_time = max(
            self.get_dma_time(lm_head.get_bytes(self.lmhead_quantize), self.prefill_ddr_util),
            self.get_tiu_time(lm_head.get_flops(), self.prefill_mac_util, tflops=self.tflops / 4))
        block_flops = sum([op.get_flops() for op in self.prefill_stage]) * self.num_layers
        block_bytes = sum(
            [op.get_bytes(self.quantize, self.group_size)
             for op in self.prefill_stage]) * self.num_layers
        block_time = sum([
            max(
                self.get_dma_time(op.get_bytes(self.quantize, self.group_size),
                                  self.prefill_ddr_util),
                self.get_tiu_time(op.get_flops(), self.prefill_mac_util))
            for op in self.prefill_stage
        ]) * self.num_layers

        self.prefill_flops = block_flops + lm_head.get_flops()
        self.prefill_bytes = block_bytes + lm_head.get_bytes(self.lmhead_quantize)
        self.prefill_tiu_theo_time = self.get_tiu_time(self.prefill_flops, mac_util=1.)
        self.prefill_dma_theo_time = self.get_dma_time(self.prefill_bytes, ddr_util=1.)
        self.prefill_tiu_time = self.get_tiu_time(self.prefill_flops, self.prefill_mac_util)
        self.prefill_dma_time = self.get_dma_time(self.prefill_bytes, self.prefill_ddr_util)
        self.prefill_allreduce_time = self.get_allreduce_time(prefill=True)
        self.prefill_total_time = block_time + lmhead_time + self.prefill_allreduce_time

    def _analyze_decode(self):
        self.decode_stage = [
            # attn
            MatMul([1, 1, self.hidden_size], [self.hidden_size, self.q_dim / self.num_device]),
            MatMul([1, 1, self.hidden_size], [self.hidden_size, self.kv_dim / self.num_device]),
            MatMul([1, 1, self.hidden_size], [self.hidden_size, self.kv_dim / self.num_device]),
            FAttention(self.num_attn_heads / self.num_device,
                       self.seq_length,
                       self.head_dim,
                       prefill=False),
            MatMul([1, 1, self.q_dim / self.num_device],
                   [self.q_dim / self.num_device, self.hidden_size]),
            # mlp
            MatMul([1, 1, self.hidden_size],
                   [self.hidden_size, self.interm_size / self.num_device]),
            MatMul([1, 1, self.hidden_size],
                   [self.hidden_size, self.interm_size / self.num_device]),
            MatMul([1, 1, self.interm_size / self.num_device],
                   [self.interm_size / self.num_device, self.hidden_size])
        ]
        lm_head = MatMul([1, self.hidden_size / self.num_device],
                         [self.hidden_size / self.num_device, self.vocab_size])
        lmhead_time = max(
            self.get_dma_time(lm_head.get_bytes(self.lmhead_quantize), self.decode_ddr_util),
            self.get_tiu_time(lm_head.get_flops(), self.decode_mac_util, tflops=self.tflops / 4))
        block_flops = sum([op.get_flops() for op in self.decode_stage]) * self.num_layers
        block_bytes = sum(
            [op.get_bytes(self.quantize, self.group_size)
             for op in self.decode_stage]) * self.num_layers
        block_time = sum([
            max(
                self.get_dma_time(op.get_bytes(self.quantize, self.group_size),
                                  self.decode_ddr_util),
                self.get_tiu_time(op.get_flops(), self.decode_mac_util, tflops=self.tflops / 4))
            for op in self.decode_stage
        ]) * self.num_layers

        self.decode_flops = block_flops + lm_head.get_flops()
        self.decode_bytes = block_bytes + lm_head.get_bytes(self.lmhead_quantize)
        self.decode_tiu_theo_time = self.get_tiu_time(self.decode_flops, mac_util=1.)
        self.decode_dma_theo_time = self.get_dma_time(self.decode_bytes, ddr_util=1.)
        self.decode_tiu_time = self.get_tiu_time(self.decode_flops, self.decode_mac_util)
        self.decode_dma_time = self.get_dma_time(self.decode_bytes, self.decode_ddr_util)
        self.decode_allreduce_time = self.get_allreduce_time(prefill=False)
        self.decode_pcie_time = self.get_pcie_interrupt_time(pcie_avg_ms=0.05)
        self.decode_total_time = block_time + lmhead_time + self.decode_allreduce_time + self.decode_pcie_time

    def _analyze_mem_usage(self):
        lm_head = MatMul([1, self.hidden_size / self.num_device],
                         [self.hidden_size / self.num_device, self.vocab_size])
        weight_bytes = sum([
                           op.get_bytes(self.quantize, self.group_size) - op.io_bytes \
                           if isinstance(op, MatMul) else 0 for op in self.decode_stage
                       ]) * self.num_layers + lm_head.get_bytes() - lm_head.io_bytes
        if not self.tie_word_embeddings:
            weight_bytes += lm_head.get_bytes(self.lmhead_quantize) - lm_head.io_bytes
        kv_cache_bytes = self.seq_length * self.kv_dim * 2 * 2
        instruct_bytes = 0.
        runtime_bytes = 0.
        sys_usage = 78 * 2**20
        self.memory_usage = weight_bytes + instruct_bytes + runtime_bytes + kv_cache_bytes + sys_usage

    def analyze(self):
        self._analyze_prefill()
        self._analyze_decode()
        self._analyze_mem_usage()

        print(f"\n=== {self.profile_name} ===")
        print(f"Model Config:")
        print(f'  hidden_size: {self.hidden_size}')
        print(f'  num_layers: {self.num_layers}')
        print(f'  num_attn_heads: {self.num_attn_heads}')
        print(f'  num_kv_heads: {self.num_kv_heads}')
        print(f'  intermediate_size: {self.interm_size}'),
        print(f'  vocab_size: {self.vocab_size}\n')

        print("Prefill:")
        print(f"  Total Flops: {self.prefill_flops / 1e9:.3f} GFLOPs")
        print(f"  Total Bytes: {self.prefill_bytes / 2**20:.3f} MiB")
        print(f"  Total Time: {self.prefill_total_time:.3f} ms")
        print(f"  TPU Theo Time: {self.prefill_tiu_theo_time:.3f} ms")
        print(f"  DDR Theo Time: {self.prefill_dma_theo_time:.3f} ms")
        print(f"  TPU Time: {self.prefill_tiu_time:.3f} ms")
        print(f"  DDR Time: {self.prefill_dma_time:.3f} ms\n")

        print("Decode:")
        print(f"  Total Flops: {self.decode_flops / 1e9:.3f} GFLOPs")
        print(f"  Total Bytes: {self.decode_bytes / 2**20:.3f} MiB")
        print(f"  Total Time: {self.decode_total_time:.3f} ms")
        print(f"  TPU Theo Time: {self.decode_tiu_theo_time:.3f} ms")
        print(f"  DDR Theo Time: {self.decode_dma_theo_time:.3f} ms")
        print(f"  TPU Time: {self.decode_tiu_time:.3f} ms")
        print(f"  DDR Time: {self.decode_dma_time:.3f} ms\n")

        print(f"FTL: {self.prefill_total_time / 1000:.3} s")
        print(f"TPS: {1000 / self.decode_total_time:.3f} token/s")
        print(f"Mem: {self.memory_usage / 2**20:.3f} MiB\n")
