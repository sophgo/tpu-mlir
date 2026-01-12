# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import torch
import os
from transform.MLIRImporter import MLIRImporter, Platform
from transform.BaseConverter import BaseConverter
from .LlmInfo import *
from .LlmLoad import *
import numpy as np
from tqdm import tqdm
from datetime import datetime

import concurrent.futures
import subprocess
import sys
from mlir.ir import *
import mlir.dialects.top as top


class LlmConverter(BaseConverter):

    def __init__(self, args, config):
        super().__init__()
        self.model_path = os.path.normpath(args.model_path)
        self.seq_length = args.seq_length
        self.max_input_length = args.max_input_length if (
            args.max_input_length > 0
            and args.max_input_length < self.seq_length) else self.seq_length
        self.max_prefill_kv_length = args.max_prefill_kv_length
        self.share_prompt = args.share_prompt
        self.quantize = args.quantize
        self.num_device = args.num_device
        self.q_group_size = args.q_group_size
        self.high_precision = True
        self.symmetric = args.symmetric
        self.chip = args.chip
        self.embedding_disk = args.embedding_disk
        self.dynamic = args.dynamic
        self.dynamic_vit = args.dynamic
        self.use_block_with_kv = args.use_block_with_kv
        self.same_addr = "0:0" if args.use_same_addr else ""
        self.debug = args.debug
        self.lora_rank = args.lora_max_rank
        self.do_lora = self.lora_rank > 0
        self.lmhead_with_topk = False if args.do_sample or self.do_lora else True
        self.position_shape = [1, self.max_input_length]
        self.num_core = args.num_core
        if self.num_core == 0:
            if args.chip == "bm1688":
                self.num_core = 2
            elif args.chip == "bm1690":
                self.num_core = 8
            else:
                self.num_core = 1
        self.half_precision_quantize = "bf16" if "bf16" in self.quantize else "f16"
        self.quant_mode = None
        self.quant_bits = 0
        self.vit_f16_out_bf16 = False  # force vit f16, output bf16
        # init config
        self.load_pretrained(config)
        self.llm_config.max_position_embeddings = self.seq_length
        if not hasattr(self.llm_config, "rope_scaling"):
            self.llm_config.rope_scaling = None  # no need rope scaling
        # get attributes
        self.init_config()
        self.do_vit = False
        self.again = args.again
        self.cos, self.sin = self.rotary_embedding()
        cpu_count = os.cpu_count()
        self.max_workers = max(cpu_count, 4)
        # check dtype
        self.check_dtype()
        # get file path
        self.out_dir = os.path.abspath(args.out_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = os.path.basename(self.model_path).lower()
        if args.chip == "bm1684x":
            folder_name = f"{self.model_name}_{self.quantize}_seq{self.seq_length}_{self.chip}_{self.num_device}dev"
        else:
            folder_name = f"{self.model_name}_{self.quantize}_seq{self.seq_length}_{self.chip}_{self.num_core}core"
        folder_name += "_dynamic" if args.dynamic else "_static"
        self.out_bmodel = os.path.join(self.out_dir, f"{folder_name}_{timestamp}.bmodel")
        self.bmodel_dir = os.path.join(self.out_dir, folder_name)
        self.config_dir = os.path.join(self.out_dir, "config")
        self.commands = []
        self.all_gen_mlirs = []
        self.all_compiles = []
        self.all_bmodels = []
        self.all_bmodels_without_bytes = []
        self.extern_block_weights = {}
        # store all weights name because some weights like qkv.weights may be splitted
        self.weights = []

    def run(self):
        os.makedirs(self.bmodel_dir, exist_ok=True)
        self.gen_config()
        ori_path = os.getcwd()
        os.chdir(self.bmodel_dir)
        if not self.again:
            self.gen_all_mlir()
        del self.model
        self.compile_all()
        os.chdir(ori_path)
        print(f"Success: {self.model_path} has converted to {self.out_dir}")

    def check_dtype(self):
        if hasattr(self.llm_config, "dtype"):
            dtype = self.llm_config.dtype
        elif hasattr(self.llm_config, "torch_dtype"):
            dtype = self.llm_config.torch_dtype
        else:
            return
        if self.quant_mode == "awq" and self.quantize == "w4f16":
            return  # skip check for awq w4f16
        if (dtype == torch.bfloat16 and self.half_precision_quantize
                == "f16") or (dtype == torch.float16 and self.half_precision_quantize == "bf16"):
            print(
                f"Warning: Please make sure your type({self.quantize}) is correct. Your torch dtype is {dtype}."
            )
            choice = input("Continue ? (Y/n):").strip().lower()
            if choice != "y":
                sys.exit(0)

    def gen_config(self):
        import shutil
        # copy model json file to config dir
        if self.config_dir.startswith(os.path.abspath(self.model_path)):
            os.rmdir(self.bmodel_dir)
            os.rmdir(self.out_dir)
            raise RuntimeError("Can't run under original model path!")
        shutil.copytree(self.model_path,
                        self.config_dir,
                        ignore=shutil.ignore_patterns("*.safetensors", ".*", "*.pth", "*.pt",
                                                      "*.py", "*.bin", "*.bin.index.json",
                                                      "model.safetensors.index.json"),
                        dirs_exist_ok=True)

    def gen_all_mlir(self):
        if self.do_vit:
            self.all_gen_mlirs.append(self.gen_vit_mlir)
        self.all_gen_mlirs.append(self.gen_embedding_lmhead_mlir)
        if not self.lmhead_with_topk:
            self.all_gen_mlirs.append(self.gen_sample_head_mlir)
        for i in range(self.num_layers):
            self.all_gen_mlirs.append(lambda i=i: self.gen_block_mlir(i))

        if self.debug:
            for func in self.all_gen_mlirs:
                func()
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []

            for func in self.all_gen_mlirs:
                futures.append(executor.submit(func))

            # Wait for all threads to complete
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures),
                               desc="generate mlir"):
                try:
                    # This will raise exceptions if any occurred during thread execution
                    future.result()
                except Exception as e:
                    for future in futures:
                        if not future.done():
                            future.cancel()
                    print(f"Error:gen mlir failed: {e}")
                    sys.exit(1)

    def load_pretrained(self, config):
        self.config = config
        self.model = LlmLoad(self.model_path)
        self.model_type = self.config.model_type
        self.model_info = COMMON_INFO
        # default llm_config is model config; but in vlm, maybe it is not the same
        if hasattr(self.config, "text_config"):
            self.llm_config = self.config.text_config
        else:
            self.llm_config = config
        self.llm_type = self.llm_config.model_type

    def rotary_embedding(self):
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        rotary_embed = LlamaRotaryEmbedding(config=self.llm_config)
        position_ids = torch.arange(self.seq_length, dtype=torch.long).reshape(1, self.seq_length)
        x = torch.zeros([1, self.seq_length, self.hidden_size], dtype=torch.float32)
        cos, sin = rotary_embed(x, position_ids)
        cos = cos.reshape(self.seq_length, 1, -1)
        sin = sin.reshape(self.seq_length, 1, -1)
        return cos.numpy(), sin.numpy()  #[seq, 1, 64]

    def rms_norm(self, mlir_gen, in_op, norm_path: str, name: str = "", eps=None):
        if not self.model.is_exist(norm_path + ".weight"):
            return in_op
        input_shape = list(in_op.type.shape)
        norm_shape = [1] * (len(input_shape) - 1) + [input_shape[-1]]
        weight_op = mlir_gen.create_weight_op(norm_path + ".weight", norm_shape)
        loc_name = name if name else norm_path
        eps = self.rms_norm_eps if eps is None else eps
        weight_keep_f32 = True if self.llm_type in [LlmType.GEMMA3] else False
        return top.RMSNormOp(mlir_gen.get_tensor_type(input_shape),
                             in_op,
                             weight_op,
                             eps=eps,
                             weight_keep_f32=weight_keep_f32,
                             loc=self.get_loc(loc_name, mlir_gen),
                             ip=mlir_gen.insert_point).output

    def layer_norm(self, mlir_gen, in_op, norm_path: str, eps, name: str = ""):
        if not self.model.is_exist(norm_path + ".weight"):
            return in_op
        input_shape = list(in_op.type.shape)
        norm_shape = [1] * (len(input_shape) - 1) + [input_shape[-1]]
        weight_op = mlir_gen.create_weight_op(norm_path + ".weight", norm_shape)
        bias_op = mlir_gen.create_weight_op(norm_path + ".bias", norm_shape)
        loc_name = name if name else norm_path
        return top.LayerNormOp(mlir_gen.get_tensor_type(input_shape),
                               in_op,
                               weight_op,
                               bias_op,
                               normalized_shape=[input_shape[-1]],
                               axis=len(input_shape) - 1,
                               eps=eps,
                               loc=self.get_loc(loc_name, mlir_gen),
                               ip=mlir_gen.insert_point).output

    def activate(self, mlir_gen, in_op, act_type: ActType, path: str):
        input_shape = list(in_op.type.shape)
        if act_type == ActType.SILU:
            return top.SiLUOp(mlir_gen.get_tensor_type(input_shape),
                              in_op,
                              loc=self.get_loc(path + ".silu", mlir_gen),
                              ip=mlir_gen.insert_point).output
        elif act_type == ActType.GELU_PYTORCH_TANH:
            return top.GELUOp(mlir_gen.get_tensor_type(input_shape),
                              in_op,
                              approx_mode=StringAttr.get("tanh"),
                              loc=self.get_loc(path + ".gelu", mlir_gen),
                              ip=mlir_gen.insert_point).output
        elif act_type == ActType.QUICK_GELU:
            return top.SwishOp(mlir_gen.get_tensor_type(input_shape),
                               in_op,
                               beta=1.702,
                               loc=self.get_loc(path + ".swish", mlir_gen),
                               ip=mlir_gen.insert_point).output
        elif act_type == ActType.GELU:
            return top.GELUOp(mlir_gen.get_tensor_type(input_shape),
                              in_op,
                              loc=self.get_loc(path + ".gelu", mlir_gen),
                              ip=mlir_gen.insert_point).output
        else:
            raise NotImplementedError(f"Unsupported activation type: {act_type}")

    def unpack_weights(self, qweight, qzeros, bits, quant_mode):
        dtype = np.int32
        compress_ratio = 32 // bits
        mask = 0xF if bits == 4 else 0xFF
        K, N = qweight.shape
        Kz, Nz = qzeros.shape
        unpacked_zeros = np.zeros((Kz, Nz * compress_ratio), dtype=np.uint8)

        if quant_mode == "gptq":
            unpacked_weights = np.zeros((K * compress_ratio, N), dtype=dtype)
            pack_int8_weights = np.zeros((K * compress_ratio // 2, N), dtype=np.uint8)
            order_map = [i for i in range(compress_ratio)]
            for row in range(unpacked_weights.shape[0]):
                i = order_map[row % compress_ratio]
                unpacked_weights[row, :] = (qweight[row // compress_ratio, :] >> (bits * i)) & mask
                if bits == 4:
                    if row % 2 == 0:
                        pack_int8_weights[row // 2, :] = unpacked_weights[row, :]
                    else:
                        pack_int8_weights[
                            row //
                            2, :] = unpacked_weights[row, :] << 4 | pack_int8_weights[row // 2, :]

        elif quant_mode == "awq":
            unpacked_weights = np.zeros((K, N * compress_ratio), dtype=dtype)
            pack_int8_weights = np.zeros((K // 2, N * compress_ratio), dtype=np.uint8)
            order_map = [0, 4, 1, 5, 2, 6, 3, 7]
            for col in range(unpacked_weights.shape[1]):
                i = order_map[col % compress_ratio]
                unpacked_weights[:, col] = (qweight[:, col // compress_ratio] >> (bits * i)) & mask
            if bits == 4:
                for row in range(unpacked_weights.shape[0]):
                    if row % 2 == 0:
                        pack_int8_weights[row // 2, :] = unpacked_weights[row, :]
                    else:
                        pack_int8_weights[
                            row //
                            2, :] = unpacked_weights[row, :] << 4 | pack_int8_weights[row // 2, :]
        else:
            raise NotImplementedError(f"Not support now: {quant_mode}")

        for col in range(unpacked_zeros.shape[1]):
            i = order_map[col % compress_ratio]
            unpacked_zeros[:, col] = (qzeros[:, col // compress_ratio] >> (bits * i)) & mask

        if bits == 8:
            pack_int8_weights = unpacked_weights.astype("uint8")

        if quant_mode == "gptq":
            return unpacked_weights, pack_int8_weights, unpacked_zeros + 1
        else:
            return unpacked_weights, pack_int8_weights, unpacked_zeros

    def decompressed_weights(self, weight_packed, weight_scale, qzeros):
        N, K = weight_packed.shape
        Ns, Ks = weight_scale.shape
        assert (N == Ns)
        bits = self.quant_bits
        compress_ratio = 32 // bits
        mask = 0xF if bits == 4 else 0xFF
        unpacked_weights = np.zeros((N, K * compress_ratio), dtype=np.int32)
        pack_int8_weights = np.zeros((N, K * compress_ratio // 2), dtype=np.uint8)
        unpacked_zeros = np.zeros((Ns, Ks), dtype=np.uint8)
        order_map = [i for i in range(compress_ratio)]
        for row in range(unpacked_weights.shape[1]):
            i = order_map[row % compress_ratio]
            unpacked_weights[:, row] = (weight_packed[:, row // compress_ratio] >>
                                        (bits * i)) & mask
            if bits == 4:
                if row % 2 == 0:
                    pack_int8_weights[:, row // 2] = unpacked_weights[:, row]
                else:
                    pack_int8_weights[:, row //
                                      2] = unpacked_weights[:, row] << 4 | pack_int8_weights[:,
                                                                                             row //
                                                                                             2]
        if qzeros is not None:
            for col in range(unpacked_zeros.shape[0]):
                i = order_map[col % compress_ratio]
                unpacked_zeros[col, :] = (qzeros[col // compress_ratio, :] >> (bits * i)) & mask
        else:
            # fill for zero points
            unpacked_zeros.fill((1 << (bits - 1)))

        if bits == 8:
            pack_int8_weights = unpacked_weights.astype("uint8")
        return unpacked_weights, pack_int8_weights, unpacked_zeros

    def init_config(self):
        c = self.model_info.config
        self.num_layers = getattr(self.llm_config, c.num_hidden_layers)
        self.rope_theta = getattr(self.llm_config, c.rope_theta, 10000.0)
        self.num_attention_heads = getattr(self.llm_config, c.num_attention_heads)
        self.num_key_value_heads = getattr(self.llm_config, c.num_key_value_heads,
                                           self.num_attention_heads)
        self.hidden_size = getattr(self.llm_config, c.hidden_size)
        self.vocab_size = getattr(self.llm_config, c.vocab_size)
        self.intermediate_size = getattr(self.llm_config, c.intermediate_size)
        self.rms_norm_eps = getattr(self.llm_config, c.rms_norm_eps)
        self.head_dim = getattr(self.llm_config, "head_dim",
                                self.hidden_size // self.num_attention_heads)
        self.hidden_act = getattr(self.llm_config, c.hidden_act, ActType.SILU)
        self.kv_dim = self.num_key_value_heads * self.head_dim
        self.kv_tile = self.num_attention_heads // self.num_key_value_heads
        # for minicpm4
        self.scale_emb = getattr(self.llm_config, "scale_emb", 1.)
        if self.llm_type == LlmType.GEMMA3:
            self.scale_emb = self.hidden_size**0.5
        self.scale_depth = getattr(self.llm_config, "scale_depth", 1.)
        self.dim_model_base = getattr(self.llm_config, "dim_model_base", 1.)
        # whether llm head and embedding share weight
        self.tie_word_embeddings = getattr(self.llm_config, 'tie_word_embeddings', False)
        self.init_quantization()

    def init_quantization(self):
        c = self.model_info.config
        self.quantization_config = getattr(self.llm_config, c.quantization_config, None)
        if self.quantization_config is None:
            self.quantization_config = getattr(self.config, c.quantization_config, None)
        if self.quantization_config:
            self.quant_mode = self.quantization_config["quant_method"]
            if self.quant_mode not in ["gptq", "awq", "compressed-tensors", "auto-round"]:
                raise NotImplementedError(f"Not support quantization method: {self.quant_mode}")
            if self.quant_mode != "compressed-tensors":
                self.q_group_size = self.quantization_config["group_size"]
                self.quant_bits = self.quantization_config["bits"]
            if self.quant_mode == "awq":
                assert self.quantization_config["version"] == "gemm", (
                    "AWQ only support gemm version for now")
                assert self.quant_bits == 4, ("AWQ only support quant bits == 4 for now")
                if self.quantize != "w4f16":
                    print("Warning: AWQ only support w4f16 quantize, change quantize to w4f16")
                    self.quantize = "w4f16"
            if self.quant_mode == "auto-round":
                packing_format = self.quantization_config.get("packing_format",
                                                              "auto_round:auto_gptq")
                if packing_format == "auto_round:auto_gptq":
                    self.quant_mode = "gptq"
                elif packing_format == "auto_round:auto_awq":
                    self.quant_mode = "awq"
                else:
                    raise NotImplementedError(f"Not support packing_format: {packing_format}")
            if self.quant_mode == "compressed-tensors":
                format = self.quantization_config.get("format", "pack-quantized")
                quantization_status = self.quantization_config.get("quantization_status",
                                                                   "compressed")
                if format != "pack-quantized" and quantization_status != "compressed":
                    raise NotImplementedError("Only support compressed pack-quantized now")
                config_groups = self.quantization_config.get("config_groups", {})
                assert len(config_groups) == 1, "Only support one group config now"
                group_0 = config_groups.get("group_0", {})
                weights_config = group_0.get("weights", {})
                self.quant_bits = weights_config.get("num_bits")
                self.q_group_size = weights_config.get("group_size")
                self.compressed_with_zp = weights_config.get("symmetric", True) is False
                weight_type = weights_config.get("type")
                assert (weight_type == "int")

        if self.q_group_size < 0:
            self.q_group_size = 0

    def get_loc(self, names, mlir):
        if isinstance(names, str):
            return Location.fused([Location.name(names)], context=mlir.ctx)
        elif isinstance(names, list):
            return Location.fused([Location.name(n) for n in names], context=mlir.ctx)
        else:
            raise RuntimeError("Unknown names:{}".format(names))

    def lora_path(self, weight_path: str, dim: int = 0):
        # dim: which dim the lora rank is
        return f"lora.{self.lora_rank}.{dim}.{weight_path}"

    def gen_embedding_bin(self, embedding_data):
        embedding_file = os.path.join(self.config_dir, 'embedding.bin')
        if os.path.exists(embedding_file):
            print(f"{embedding_file} already exists. Skipping export.")
            return
        import ctypes
        weight = torch.from_numpy(embedding_data)
        if 'bf16' in self.quantize:
            tensor_data = weight.to(torch.bfloat16)
        elif 'f16' in self.quantize:
            tensor_data = weight.to(torch.float16)
        else:
            raise NotImplementedError("Not support now")
        data_ptr = tensor_data.untyped_storage().data_ptr()
        buffer = (ctypes.c_byte * (tensor_data.numel() * 2)).from_address(data_ptr)
        with open(embedding_file, 'wb') as f:
            f.write(buffer)

    def gen_embedding_lmhead_mlir(self):
        tqdm.write("generate embedding and lm_head mlir ...")
        embedding = self.model_info.weights[LlmList.EMBEDING]
        embedding_data = self.model.read(embedding + ".weight")
        if self.embedding_disk:
            self.gen_embedding_bin(embedding_data)
        else:
            # read embedding weights
            embedding_weights = {embedding + ".weight": embedding_data}
            embedding_npz = "embedding_top_weights.npz"
            np.savez(embedding_npz, **embedding_weights)

        # read lm_head weights
        lmhead = self.model_info.weights[LlmList.LMHEAD]
        lmhead_path = lmhead + ".weight"
        norm = self.model_info.weights[LlmList.NORM]
        norm_path = norm + ".weight"
        if self.tie_word_embeddings:
            lmhead_data = embedding_data
        else:
            lmhead_data = self.model.read(lmhead_path)
        lmhead_weights = {lmhead_path: lmhead_data}
        lmhead_npz = "lm_head_top_weights.npz"
        np.savez(lmhead_npz, **lmhead_weights)

        if self.do_lora:
            embedding_lora_weights = {}
            self.set_lora_weight(embedding_lora_weights, embedding,
                                 (self.vocab_size, self.lora_rank),
                                 (self.lora_rank, self.hidden_size))
            np.savez("embedding_lora_top_weights.npz", **embedding_lora_weights)
            lmhead_lora_weights = {}
            self.set_linear_lora_weight(lmhead_lora_weights, lmhead, self.hidden_size,
                                        self.vocab_size)
            # walkaround: avoid lmhead_lora merged with embedding_lora
            lmhead_lora_weights[f"{lmhead}.lora_A.weight"].fill(1.0)
            np.savez("lm_head_lora_top_weights.npz", **lmhead_lora_weights)

        # gen embedding mlir
        def gen_embedding_by_length(name: str, seq_length: int):
            out_shape = [1, seq_length, self.hidden_size]
            embedding_mlir = MLIRImporter([[1, seq_length]], [out_shape],
                                          name,
                                          Platform.LLM,
                                          input_types=["INT32"],
                                          weight_file=f"../{embedding_npz}")
            input_op = embedding_mlir.create_input_op(self.get_loc("input_ids", embedding_mlir), 0)
            weight_op = embedding_mlir.create_weight_op(embedding + ".weight",
                                                        [self.vocab_size, self.hidden_size])
            new_op = top.GatherOp(embedding_mlir.get_tensor_type(out_shape),
                                  weight_op,
                                  input_op,
                                  axis=0,
                                  loc=self.get_loc(name, embedding_mlir),
                                  ip=embedding_mlir.insert_point).output
            if self.scale_emb != 1.0:
                new_op = top.MulConstOp(embedding_mlir.get_tensor_type(out_shape),
                                        new_op,
                                        const_val=self.scale_emb,
                                        loc=self.get_loc(name + ".scale", embedding_mlir),
                                        ip=embedding_mlir.insert_point).output
            embedding_mlir.create_return_op([new_op])
            mlir_txt = embedding_mlir.print_module()
            if not os.path.exists(name):
                os.makedirs(name)
            with open(f"{name}/{name}.mlir", "w") as f:
                f.write(mlir_txt)

        def gen_embedding_lora_by_length(name: str, seq_length: int):
            hidden_shape = [1, seq_length, self.hidden_size]
            lora_mlir = MLIRImporter([[1, seq_length], hidden_shape], [hidden_shape],
                                     name,
                                     Platform.LLM,
                                     input_types=["INT32", "F32"],
                                     weight_file="../embedding_lora_top_weights.npz")
            input_op = lora_mlir.create_input_op(self.get_loc("input_ids", lora_mlir), 0)
            state_op = lora_mlir.create_input_op(self.get_loc("input_states", lora_mlir), 1)
            lora_a_weight = f"{embedding}.lora_A.weight"
            weight_op = lora_mlir.create_weight_op(lora_a_weight, [self.vocab_size, self.lora_rank],
                                                   path=self.lora_path(lora_a_weight, 1))
            a_op = top.GatherOp(lora_mlir.get_tensor_type([1, seq_length, self.lora_rank]),
                                weight_op,
                                input_op,
                                axis=0,
                                loc=self.get_loc(f"{name}.lora_A", lora_mlir),
                                ip=lora_mlir.insert_point).output
            lora_b_weight = f"{embedding}.lora_B.weight"
            weight_op = lora_mlir.create_weight_op(lora_b_weight,
                                                   [self.lora_rank, self.hidden_size],
                                                   path=self.lora_path(lora_b_weight))
            b_op = top.MatMulOp(lora_mlir.get_tensor_type(hidden_shape),
                                a_op,
                                weight_op,
                                lora_mlir.none_op,
                                do_relu=False,
                                is_lora=True,
                                loc=self.get_loc(f"{name}.lora_B", lora_mlir),
                                ip=lora_mlir.insert_point).output
            if self.scale_emb != 1.0:
                b_op = top.MulConstOp(lora_mlir.get_tensor_type(hidden_shape),
                                      b_op,
                                      const_val=self.scale_emb,
                                      loc=self.get_loc(name + ".scale", lora_mlir),
                                      ip=lora_mlir.insert_point).output
            new_op = top.AddOp(lora_mlir.get_tensor_type(hidden_shape), [state_op, b_op],
                               loc=self.get_loc(f"{name}.lora_add", lora_mlir),
                               ip=lora_mlir.insert_point).output
            lora_mlir.create_return_op([new_op])
            mlir_txt = lora_mlir.print_module()
            if not os.path.exists(name):
                os.makedirs(name)
            with open(f"{name}/{name}.mlir", "w") as f:
                f.write(mlir_txt)

        # gen lm_head mlir
        def gen_lm_head():
            name = "lm_head"
            out_shape = [[1, self.vocab_size]]
            if self.lmhead_with_topk:
                out_shape = [[1, 1]]
            lmhead_mlir = MLIRImporter([[1, self.hidden_size]],
                                       out_shape,
                                       name,
                                       Platform.LLM,
                                       weight_file=f"../{lmhead_npz}")
            input_op = lmhead_mlir.create_input_op(self.get_loc("hidden_states", lmhead_mlir), 0)
            if self.llm_type == LlmType.MINICPM4:
                input_op = top.MulConstOp(lmhead_mlir.get_tensor_type([1, self.hidden_size]),
                                          input_op,
                                          const_val=self.dim_model_base / self.hidden_size,
                                          loc=self.get_loc(lmhead + ".scale", lmhead_mlir),
                                          ip=lmhead_mlir.insert_point).output
            if self.num_device > 1:
                weight_op = lmhead_mlir.create_weight_op(norm_path, [1, self.hidden_size])
                input_op = self.rms_norm(lmhead_mlir, input_op, norm)

            w_shape = [self.vocab_size, self.hidden_size]
            weight_op = lmhead_mlir.create_weight_op(lmhead + ".weight", w_shape)
            lmhead_op = top.MatMulOp(lmhead_mlir.get_tensor_type([self.vocab_size, 1]),
                                     weight_op,
                                     input_op,
                                     lmhead_mlir.none_op,
                                     do_relu=False,
                                     right_transpose=True,
                                     loc=self.get_loc(lmhead, lmhead_mlir),
                                     ip=lmhead_mlir.insert_point).output
            lmhead_op = top.ReshapeOp(lmhead_mlir.get_tensor_type([1, self.vocab_size]),
                                      lmhead_op,
                                      loc=self.get_loc(lmhead + ".reshape", lmhead_mlir),
                                      ip=lmhead_mlir.insert_point).output
            if self.lmhead_with_topk:
                topk_op = top.TopKOp(*lmhead_mlir.get_tensor_type([[1, 1], [1, 1]]),
                                     lmhead_op,
                                     axis=1,
                                     K=1,
                                     loc=self.get_loc(["token_value", "token_id"], lmhead_mlir),
                                     ip=lmhead_mlir.insert_point)
                # topk_op.values, topk_op.indices
                lmhead_mlir.create_return_op([topk_op.indices])
            else:
                lmhead_mlir.create_return_op([lmhead_op])

            mlir_txt = lmhead_mlir.print_module()
            if not os.path.exists(name):
                os.makedirs(name)
            with open(f"{name}/{name}.mlir", "w") as f:
                f.write(mlir_txt)

        def gen_lm_head_lora():
            name = "lm_head_lora"
            out_shape = [1, self.vocab_size]
            lmhead_mlir = MLIRImporter([[1, self.hidden_size], out_shape], [out_shape],
                                       name,
                                       Platform.LLM,
                                       input_types=["F32", "F32"],
                                       weight_file="../lm_head_lora_top_weights.npz")
            input_op = lmhead_mlir.create_input_op(self.get_loc("input_states", lmhead_mlir), 0)
            state_op = lmhead_mlir.create_input_op(self.get_loc("input_logits", lmhead_mlir), 1)
            if self.llm_type == LlmType.MINICPM4:
                input_op = top.MulConstOp(lmhead_mlir.get_tensor_type([1, self.hidden_size]),
                                          input_op,
                                          const_val=self.dim_model_base / self.hidden_size,
                                          loc=self.get_loc(lmhead + ".scale", lmhead_mlir),
                                          ip=lmhead_mlir.insert_point).output
            lora_a_weight = f"{lmhead}.lora_A.weight"
            a_weight_op = lmhead_mlir.create_weight_op(lora_a_weight,
                                                       [self.lora_rank, self.hidden_size],
                                                       path=self.lora_path(lora_a_weight))
            a_op = top.MatMulOp(lmhead_mlir.get_tensor_type([1, self.lora_rank]),
                                input_op,
                                a_weight_op,
                                lmhead_mlir.none_op,
                                do_relu=False,
                                right_transpose=True,
                                is_lora=True,
                                loc=self.get_loc(f"{name}.lora_A", lmhead_mlir),
                                ip=lmhead_mlir.insert_point).output
            lora_b_weight = f"{lmhead}.lora_B.weight"
            b_weight_op = lmhead_mlir.create_weight_op(lora_b_weight,
                                                       [self.lora_rank, self.vocab_size],
                                                       path=self.lora_path(lora_b_weight))
            b_op = top.MatMulOp(lmhead_mlir.get_tensor_type([1, self.vocab_size]),
                                a_op,
                                b_weight_op,
                                lmhead_mlir.none_op,
                                do_relu=False,
                                is_lora=True,
                                loc=self.get_loc(f"{name}.lora_B", lmhead_mlir),
                                ip=lmhead_mlir.insert_point).output
            new_op = top.AddOp(lmhead_mlir.get_tensor_type([1, self.vocab_size]), [state_op, b_op],
                               loc=self.get_loc(f"{name}.lora_add", lmhead_mlir),
                               ip=lmhead_mlir.insert_point).output
            lmhead_mlir.create_return_op([new_op])
            mlir_txt = lmhead_mlir.print_module()
            if not os.path.exists(name):
                os.makedirs(name)
            with open(f"{name}/{name}.mlir", "w") as f:
                f.write(mlir_txt)

        if not self.embedding_disk:
            input_len = self.max_input_length
            if self.share_prompt:
                input_len = max(self.max_prefill_kv_length, self.max_input_length)
            gen_embedding_by_length("embedding", input_len)
            gen_embedding_by_length("embedding_cache", 1)
        gen_lm_head()
        if self.do_lora:
            input_len = self.max_input_length
            if self.share_prompt:
                input_len = max(self.max_prefill_kv_length, self.max_input_length)
            gen_embedding_lora_by_length("embedding_lora", input_len)
            gen_embedding_lora_by_length("embedding_cache_lora", 1)
            gen_lm_head_lora()

    def gen_sample_head_mlir(self, max_top_k=50, min_tokens_to_keep=5):
        tqdm.write("generate greedy head and sample head mlir ...")
        name = "greedy_head"
        # greedy head
        greedy_head_mlir = MLIRImporter([[1, self.vocab_size]], [[1, 1]],
                                        name,
                                        Platform.LLM,
                                        weight_file=None)
        input_op = greedy_head_mlir.create_input_op(self.get_loc("m_logits", greedy_head_mlir), 0)
        topk_op = top.TopKOp(*greedy_head_mlir.get_tensor_type([[1, 1], [1, 1]]),
                             input_op,
                             axis=1,
                             K=1,
                             loc=self.get_loc(["token_value", "token_id"], greedy_head_mlir),
                             ip=greedy_head_mlir.insert_point)
        greedy_head_mlir.create_return_op([topk_op.indices])
        mlir_txt = greedy_head_mlir.print_module()
        if not os.path.exists(name):
            os.makedirs(name)
        with open(f"{name}/{name}.mlir", "w") as f:
            f.write(mlir_txt)

        # sample head
        constant0 = []
        constant1 = []
        for i in range(min_tokens_to_keep):
            constant0.append([0, i])
            constant1.append(1)
        sample_head_weights = {}
        sample_head_weights["Constant0"] = np.array([1.]).astype(np.float32)
        sample_head_weights["Constant1"] = np.array([constant0]).astype(np.float32)
        sample_head_weights["Constant2"] = np.array([constant1]).astype(np.float32)
        np.savez("sample_head_top_weights.npz", **sample_head_weights)

        name = "sample_head"
        sample_head_mlir = MLIRImporter(
            [[1, self.vocab_size], [1, self.seq_length], [1], [1], [1], [1]],
            [[1, max_top_k], [1, max_top_k]],
            name,
            Platform.LLM,
            input_types=['F32', 'INT32', 'F32', 'F32', 'INT32', 'F32'],
            weight_file="../sample_head_top_weights.npz")
        ip = sample_head_mlir.insert_point

        def T(shape: list):
            return sample_head_mlir.get_tensor_type(shape)

        def L(name: str):
            return self.get_loc(name, sample_head_mlir)

        kwargs = {}
        kwargs['shape_tensor'] = [max_top_k]
        in0_op = sample_head_mlir.create_input_op(L("m_logits"), 0)
        in1_op = sample_head_mlir.create_input_op(L("input_ids"), 1)
        in2_op = sample_head_mlir.create_input_op(L("penalty"), 2)
        in3_op = sample_head_mlir.create_input_op(L("temperature"), 3)
        in4_op = sample_head_mlir.create_input_op(L("top_k"), 4, kwargs)
        in5_op = sample_head_mlir.create_input_op(L("top_p"), 5)
        gather_op = top.GatherElementsOp(T([1, self.seq_length]),
                                         in0_op,
                                         in1_op,
                                         axis=1,
                                         loc=L("GatherElements"),
                                         ip=ip).output
        cmpconst_op = top.CompareConstOp(T([1, self.seq_length]),
                                         gather_op,
                                         mode=StringAttr.get("Less"),
                                         const_val=0.,
                                         inversed=False,
                                         loc=L("CompareConst"),
                                         ip=ip).output
        mul_op = top.MulOp(T([1, self.seq_length]), [gather_op, in2_op], loc=L("Mul"), ip=ip).output
        div0_op = top.DivOp(T([1, self.seq_length]), [gather_op, in2_op], loc=L("Div0"),
                            ip=ip).output
        where0_op = top.WhereOp(T([1, self.seq_length]),
                                cmpconst_op,
                                mul_op,
                                div0_op,
                                loc=L("Where0"),
                                ip=ip).output
        scatter_op = top.ScatterElementsOp(T([1, self.vocab_size]),
                                           in0_op,
                                           in1_op,
                                           where0_op,
                                           axis=1,
                                           loc=L("ScatterElements"),
                                           ip=ip).output
        topk_op = top.TopKOp(*T([[1, max_top_k], [1, max_top_k]]),
                             scatter_op,
                             kT=in4_op,
                             axis=1,
                             K=max_top_k,
                             loc=L(["token_value", "token_idx"]),
                             ip=ip)
        div1_op = top.DivOp(T([1, max_top_k]), [topk_op.values, in3_op], loc=L("Div1"),
                            ip=ip).output
        softmax0_op = top.SoftmaxOp(T([1, max_top_k]), div1_op, axis=1, loc=L("Softmax0"),
                                    ip=ip).output
        weight0_op = sample_head_mlir.create_weight_op("Constant0", [1])
        cumsum_op = top.CumSumOp(T([1, max_top_k]),
                                 softmax0_op,
                                 weight0_op,
                                 axis=1,
                                 loc=L("CumSum"),
                                 ip=ip).output
        compare_op = top.CompareOp(T([1, max_top_k]),
                                   cumsum_op,
                                   in5_op,
                                   mode=StringAttr.get("Less"),
                                   loc=L("Compare"),
                                   ip=ip).output
        weight1_op = sample_head_mlir.create_weight_op("Constant1", [1, min_tokens_to_keep, 2])
        weight2_op = sample_head_mlir.create_weight_op("Constant2", [1, min_tokens_to_keep])
        scatternd_op = top.ScatterNDOp(T([1, max_top_k]),
                                       compare_op,
                                       weight1_op,
                                       weight2_op,
                                       reduction=0,
                                       loc=L("ScatterND"),
                                       ip=ip).output
        where1_op = top.WhereOp(T([1, max_top_k]),
                                scatternd_op,
                                div1_op,
                                sample_head_mlir.none_op,
                                y_is_const=True,
                                y_const_val=-1000.,
                                loc=L("Where1"),
                                ip=ip).output
        softmax1_op = top.SoftmaxOp(T([1, max_top_k]), where1_op, axis=1, loc=L("Softmax1"),
                                    ip=ip).output
        sample_head_mlir.create_return_op([softmax1_op, topk_op.indices])
        mlir_txt = sample_head_mlir.print_module()

        if not os.path.exists(name):
            os.makedirs(name)
        with open(f"{name}/{name}.mlir", "w") as f:
            f.write(mlir_txt)

    def repeat_kv(self, mlir_gen, kv_op, len: int, prefix: str):
        unsqueeze = top.UnsqueezeOp(mlir_gen.get_tensor_type(
            [1, len, self.num_key_value_heads, 1, self.head_dim]),
                                    kv_op,
                                    loc=self.get_loc(prefix + ".unsqueeze", mlir_gen),
                                    ip=mlir_gen.insert_point,
                                    axes=[3]).output
        tile = top.TileOp(mlir_gen.get_tensor_type(
            [1, len, self.num_key_value_heads, self.kv_tile, self.head_dim]),
                          unsqueeze,
                          tile=[1, 1, 1, self.kv_tile, 1],
                          loc=self.get_loc(prefix + ".tile", mlir_gen),
                          ip=mlir_gen.insert_point).output
        rs = top.ReshapeOp(mlir_gen.get_tensor_type(
            [1, len, self.num_attention_heads, self.head_dim]),
                           tile,
                           shape=[1, -1, self.num_attention_heads, self.head_dim],
                           loc=self.get_loc(prefix + ".tile.reshape", mlir_gen),
                           ip=mlir_gen.insert_point).output
        return rs

    def linear(self,
               mlir_gen,
               proj: str,
               input_op,
               weight_shape: list,
               out_shape: list,
               force_bias: bool = False,
               do_lora: bool = False):
        if self.model.is_exist(proj + ".bias") or force_bias:
            bias_shape = [1] * (len(out_shape) - 1) + [out_shape[-1]]
            bias_op = mlir_gen.create_weight_op(proj + ".bias", bias_shape)
        else:
            bias_op = mlir_gen.none_op
        if self.quant_mode and (self.model.is_exist(proj + ".qweight") or
                                (proj + ".qweight" in self.weights)):
            qweight_op = mlir_gen.create_weight_op(
                proj + ".qweight", [weight_shape[1], weight_shape[0] // (8 // self.quant_bits)],
                'UINT8')
            scale_shape = [weight_shape[1], weight_shape[0] //
                           self.q_group_size] if self.q_group_size > 0 else [weight_shape[1], 1]
            scale_op = mlir_gen.create_weight_op(proj + ".scales", scale_shape)
            zp_op = mlir_gen.create_weight_op(proj + ".qzeros", scale_shape, 'UINT8')
            new_op = top.A16MatMulOp(mlir_gen.get_tensor_type(out_shape),
                                     input_op,
                                     qweight_op,
                                     scale_op,
                                     zp_op,
                                     bias_op,
                                     right_transpose=True,
                                     q_group_size=self.q_group_size,
                                     weight_bits=self.quant_bits,
                                     loc=self.get_loc(proj, mlir_gen),
                                     ip=mlir_gen.insert_point).output
        else:
            weight_op = mlir_gen.create_weight_op(proj + ".weight", weight_shape)
            new_op = top.MatMulOp(mlir_gen.get_tensor_type(out_shape),
                                  input_op,
                                  weight_op,
                                  bias_op,
                                  do_relu=False,
                                  loc=self.get_loc(proj, mlir_gen),
                                  ip=mlir_gen.insert_point).output
        if not do_lora:
            return new_op
        if bias_op is not mlir_gen.none_op:
            raise NotImplementedError("Lora with bias is not supported yet.")
        # add lora
        lora_a_weight = f"{proj}.lora_A.weight"
        lora_b_weight = f"{proj}.lora_B.weight"
        weight_op = mlir_gen.create_weight_op(lora_a_weight, [self.lora_rank, weight_shape[0]],
                                              path=self.lora_path(lora_a_weight))
        lora_a_shape = list(out_shape)
        lora_a_shape[-1] = self.lora_rank
        lora_op = top.MatMulOp(mlir_gen.get_tensor_type(lora_a_shape),
                               input_op,
                               weight_op,
                               mlir_gen.none_op,
                               do_relu=False,
                               right_transpose=True,
                               is_lora=True,
                               loc=self.get_loc(f"{proj}.lora_A", mlir_gen),
                               ip=mlir_gen.insert_point).output
        weight_op = mlir_gen.create_weight_op(lora_b_weight, [self.lora_rank, weight_shape[1]],
                                              path=self.lora_path(lora_b_weight))
        lora_op = top.MatMulOp(mlir_gen.get_tensor_type(out_shape),
                               lora_op,
                               weight_op,
                               mlir_gen.none_op,
                               do_relu=False,
                               is_lora=True,
                               loc=self.get_loc(f"{proj}.lora_B", mlir_gen),
                               ip=mlir_gen.insert_point).output
        new_op = top.AddOp(mlir_gen.get_tensor_type(out_shape), [new_op, lora_op],
                           loc=self.get_loc(proj + ".lora_add", mlir_gen),
                           ip=mlir_gen.insert_point).output
        return new_op

    # q_embed = (q * cos) + (rotate_half(q) * sin)
    # k_embed = (k * cos) + (rotate_half(k) * sin)
    def rotary_pos(self, mlir_gen, in_op, cos_op, sin_op, out_name: str):
        in_shape = in_op.type.shape
        prefix = f"{out_name}.rotary_pos"
        half_shape = list(in_shape)
        half_shape[-1] = half_shape[-1] // 2
        mul_q_proj = top.MulOp(mlir_gen.get_tensor_type(in_shape), [in_op, cos_op],
                               loc=self.get_loc(prefix + ".mul0", mlir_gen),
                               ip=mlir_gen.insert_point).output
        half_q0 = top.SliceOp(mlir_gen.get_tensor_type(half_shape),
                              in_op,
                              mlir_gen.none_op,
                              mlir_gen.none_op,
                              mlir_gen.none_op,
                              offset=[0, 0, 0, 0],
                              steps=[1, 1, 1, 1],
                              ends=half_shape,
                              axes=[],
                              loc=self.get_loc(prefix + ".slice1", mlir_gen),
                              ip=mlir_gen.insert_point).output

        half_q1 = top.SliceOp(mlir_gen.get_tensor_type(half_shape),
                              in_op,
                              mlir_gen.none_op,
                              mlir_gen.none_op,
                              mlir_gen.none_op,
                              offset=[0, 0, 0, half_shape[-1]],
                              steps=[1, 1, 1, 1],
                              ends=in_shape,
                              axes=[],
                              loc=self.get_loc(prefix + ".slice2", mlir_gen),
                              ip=mlir_gen.insert_point).output

        neg_half_q1 = top.MulConstOp(mlir_gen.get_tensor_type(half_shape),
                                     half_q1,
                                     const_val=-1.0,
                                     loc=self.get_loc(prefix + ".neg3", mlir_gen),
                                     ip=mlir_gen.insert_point).output
        new_q = top.ConcatOp(mlir_gen.get_tensor_type(in_shape), [neg_half_q1, half_q0],
                             axis=3,
                             loc=self.get_loc(prefix + ".concat4", mlir_gen),
                             ip=mlir_gen.insert_point).output
        new_q = top.MulOp(mlir_gen.get_tensor_type(in_shape), [new_q, sin_op],
                          loc=self.get_loc(prefix + ".mul5", mlir_gen),
                          ip=mlir_gen.insert_point).output
        new_q = top.AddOp(mlir_gen.get_tensor_type(in_shape), [mul_q_proj, new_q],
                          loc=self.get_loc(out_name, mlir_gen),
                          ip=mlir_gen.insert_point).output
        return new_q

    def apply_rotary_pos(self, mlir_gen, pos_op, q_op, k_op, rotary_cos: str, rotary_sin: str):
        dim = pos_op.type.shape[-1]
        weight_op = mlir_gen.create_weight_op(rotary_cos + ".weight",
                                              [self.seq_length, 1, self.head_dim])
        cos_op = top.GatherOp(mlir_gen.get_tensor_type([1, dim, 1, self.head_dim]),
                              weight_op,
                              pos_op,
                              axis=0,
                              loc=self.get_loc(rotary_cos, mlir_gen),
                              ip=mlir_gen.insert_point).output
        weight_op = mlir_gen.create_weight_op(rotary_sin + ".weight",
                                              [self.seq_length, 1, self.head_dim])
        sin_op = top.GatherOp(mlir_gen.get_tensor_type([1, dim, 1, self.head_dim]),
                              weight_op,
                              pos_op,
                              axis=0,
                              loc=self.get_loc(rotary_sin, mlir_gen),
                              ip=mlir_gen.insert_point).output
        q_op_shape = q_op.type.shape
        q_op = top.RopeOp(mlir_gen.get_tensor_type(q_op_shape),
                          q_op,
                          sin_op,
                          cos_op,
                          rope_mode=StringAttr.get("contiguous_halves"),
                          loc=self.get_loc("q_proj", mlir_gen),
                          ip=mlir_gen.insert_point).output
        k_op_shape = k_op.type.shape
        k_op = top.RopeOp(mlir_gen.get_tensor_type(k_op_shape),
                          k_op,
                          sin_op,
                          cos_op,
                          rope_mode=StringAttr.get("contiguous_halves"),
                          loc=self.get_loc("k_cache", mlir_gen),
                          ip=mlir_gen.insert_point).output
        return q_op, k_op

    def set_linear_lora_weight(self, weight_dict: dict, path: str, K: int, N: int):
        self.set_lora_weight(weight_dict, path, (self.lora_rank, K), (self.lora_rank, N))

    def set_lora_weight(self, weight_dict: dict, path: str, A_shape: tuple[int],
                        B_shape: tuple[int]):
        lora_a_path = path + ".lora_A.weight"
        lora_b_path = path + ".lora_B.weight"
        weight_dict[lora_a_path] = np.zeros(A_shape, dtype=np.float32)
        weight_dict[lora_b_path] = np.zeros(B_shape, dtype=np.float32)

    def set_linear_weight(self, path: str, weight_dict: dict, do_lora: bool = False):
        is_quant = False
        K, N = 0, 0
        if self.quant_mode is not None:
            if self.model.is_exist(path + ".qweight") or self.model.is_exist(path +
                                                                             ".weight_packed"):
                is_quant = True
        if not is_quant:
            weight_path = path + ".weight"
            if self.model.is_exist(weight_path):
                data = self.model.read(weight_path)
                weight_dict[weight_path] = np.ascontiguousarray(np.transpose(data, (1, 0)))
                K = data.shape[1]
                N = data.shape[0]
            else:
                raise RuntimeError("Can't find key: {}".format(weight_path))
        elif self.quant_mode in ["gptq", "awq"]:
            qweight_path = path + ".qweight"
            scale_path = path + ".scales"
            zp_path = path + ".qzeros"
            qweight_data = self.model.read(qweight_path)
            scale_data = self.model.read(scale_path)
            zp_data = self.model.read(zp_path)
            _, pack_int8_weights, unpacked_zeros = self.unpack_weights(
                qweight_data, zp_data, self.quant_bits, self.quant_mode)
            weight_dict[qweight_path] = np.ascontiguousarray(np.transpose(
                pack_int8_weights, (1, 0)))
            weight_dict[scale_path] = np.ascontiguousarray(np.transpose(scale_data, (1, 0)))
            weight_dict[zp_path] = np.ascontiguousarray(np.transpose(unpacked_zeros, (1, 0)))
            K = pack_int8_weights.shape[0] * (8 // self.quant_bits)
            N = pack_int8_weights.shape[1]
        elif self.quant_mode == "compressed-tensors":
            qweight_path = path + ".weight_packed"
            scale_path = path + ".weight_scale"
            zp_path = path + ".weight_zero_point"
            qweight_data = self.model.read(qweight_path)
            scale_data = self.model.read(scale_path)
            if self.compressed_with_zp:
                zp_data = self.model.read(zp_path)
            else:
                zp_data = None
            _, pack_int8_weights, unpacked_zeros = self.decompressed_weights(
                qweight_data, scale_data, zp_data)
            weight_dict[path + ".qweight"] = pack_int8_weights
            weight_dict[path + ".scales"] = scale_data
            weight_dict[path + ".qzeros"] = unpacked_zeros
            K = pack_int8_weights.shape[1] * (8 // self.quant_bits)
            N = pack_int8_weights.shape[0]

        bias_path = path + ".bias"
        if self.model.is_exist(bias_path):
            weight_dict[bias_path] = self.model.read(bias_path)
        if do_lora:
            self.set_linear_lora_weight(weight_dict, path, K, N)

    def set_common_weight(self, path: str, weight_dict: dict, type=WeightType.NORMAL):
        weight_path = path + ".weight"
        bias_path = path + ".bias"
        has_weight = self.model.is_exist(weight_path)
        has_bias = self.model.is_exist(bias_path)
        has_path = self.model.is_exist(path)
        if not has_weight and not has_bias and not has_path:
            raise RuntimeError("Can't find key: {}".format(path))
        if has_weight:
            data = self.model.read(weight_path)
            if type == WeightType.RMS_NORM and self.llm_type in [LlmType.GEMMA3]:
                data = data + 1.0  # GEMMA3 RMSNorm weight is not same as others
            weight_dict[weight_path] = data
        if has_bias:
            weight_dict[bias_path] = self.model.read(bias_path)
        if has_path:
            weight_dict[path] = self.model.read(path)

    def gen_block_mlir(self, idx: int):
        tqdm.write(f"generate block_{idx} mlir ...")
        # torch path
        TOP_PATH = f'{self.model_info.weights[LlmList.LAYERS]}.{idx}.'
        input_ln = TOP_PATH + self.model_info.weights[LlmList.INPUT_LN]
        q_proj = TOP_PATH + self.model_info.weights[LlmList.Q_PROJ]
        q_norm = TOP_PATH + self.model_info.weights[LlmList.Q_NORM]
        k_proj = TOP_PATH + self.model_info.weights[LlmList.K_PROJ]
        k_norm = TOP_PATH + self.model_info.weights[LlmList.K_NORM]
        v_proj = TOP_PATH + self.model_info.weights[LlmList.V_PROJ]
        o_proj = TOP_PATH + self.model_info.weights[LlmList.O_PROJ]
        post_attn_ln = TOP_PATH + self.model_info.weights[LlmList.POST_ATTN_LN]
        mlp_gate = TOP_PATH + self.model_info.weights[LlmList.MLP_GATE]
        mlp_up = TOP_PATH + self.model_info.weights[LlmList.MLP_UP]
        mlp_down = TOP_PATH + self.model_info.weights[LlmList.MLP_DOWN]
        if self.llm_type in [LlmType.GEMMA3]:
            pre_mlp_ln = TOP_PATH + self.model_info.weights[LlmList.PRE_MLP_LN]
            post_mlp_ln = TOP_PATH + self.model_info.weights[LlmList.POST_MLP_LN]
        norm = self.model_info.weights[LlmList.NORM]
        do_norm = self.num_device < 2 and idx == self.num_layers - 1
        rotary_cos = "rotary_cos"
        rotary_sin = "rotary_sin"

        # save weight
        weight_file = f"block_{idx}_top_weights.npz"
        weight_dict = {
            rotary_cos + ".weight": self.cos,
            rotary_sin + ".weight": self.sin,
        }
        self.set_common_weight(input_ln, weight_dict, WeightType.RMS_NORM)
        self.set_linear_weight(q_proj, weight_dict, do_lora=self.do_lora)
        self.set_linear_weight(k_proj, weight_dict, do_lora=self.do_lora)
        self.set_linear_weight(v_proj, weight_dict, do_lora=self.do_lora)
        self.set_linear_weight(o_proj, weight_dict, do_lora=self.do_lora)
        if self.llm_type in [LlmType.QWEN3, LlmType.GEMMA3]:
            self.set_common_weight(q_norm, weight_dict, WeightType.RMS_NORM)
            self.set_common_weight(k_norm, weight_dict, WeightType.RMS_NORM)
        if self.llm_type in [LlmType.GEMMA3]:
            self.set_common_weight(pre_mlp_ln, weight_dict, WeightType.RMS_NORM)
            self.set_common_weight(post_mlp_ln, weight_dict, WeightType.RMS_NORM)
        self.set_common_weight(post_attn_ln, weight_dict, WeightType.RMS_NORM)
        self.set_linear_weight(mlp_gate, weight_dict, do_lora=self.do_lora)
        self.set_linear_weight(mlp_up, weight_dict, do_lora=self.do_lora)
        self.set_linear_weight(mlp_down, weight_dict, do_lora=self.do_lora)
        if do_norm:
            self.set_common_weight(norm, weight_dict, WeightType.RMS_NORM)
        if self.extern_block_weights:
            weight_dict.update(self.extern_block_weights)
        self.weights.extend(list(weight_dict.keys()))
        np.savez(weight_file, **weight_dict)

        def gen_mlp(mlir_gen, input_shape, in_op):
            ip = mlir_gen.insert_point
            len = input_shape[1]
            new_op = in_op
            if self.llm_type in [LlmType.GEMMA3]:
                new_op = self.rms_norm(mlir_gen, in_op, pre_mlp_ln)
            else:
                new_op = self.rms_norm(mlir_gen, in_op, post_attn_ln)

            gate_op = self.linear(mlir_gen,
                                  mlp_gate,
                                  new_op, [self.hidden_size, self.intermediate_size],
                                  [1, len, self.intermediate_size],
                                  do_lora=self.do_lora)
            act_op = self.activate(mlir_gen, gate_op, self.hidden_act, mlp_gate)
            up_op = self.linear(mlir_gen,
                                mlp_up,
                                new_op, [self.hidden_size, self.intermediate_size],
                                [1, len, self.intermediate_size],
                                do_lora=self.do_lora)
            new_op = top.MulOp(mlir_gen.get_tensor_type([1, len, self.intermediate_size]),
                               [act_op, up_op],
                               loc=self.get_loc(mlp_up + ".mul", mlir_gen),
                               ip=ip).output
            down_op = self.linear(mlir_gen,
                                  mlp_down,
                                  new_op, [self.intermediate_size, self.hidden_size],
                                  input_shape,
                                  do_lora=self.do_lora)
            if self.llm_type in [LlmType.GEMMA3]:
                down_op = self.rms_norm(mlir_gen, down_op, post_mlp_ln)
            if self.llm_type == LlmType.MINICPM4:
                down_op = top.MulConstOp(mlir_gen.get_tensor_type(input_shape),
                                         down_op,
                                         const_val=self.scale_depth / np.sqrt(self.num_layers),
                                         loc=self.get_loc(mlp_down + ".scale", mlir_gen),
                                         ip=ip).output
            last_name = "output_states"
            new_name = last_name if idx != self.num_layers - 1 else f"{mlp_down}.add"
            new_op = top.AddOp(mlir_gen.get_tensor_type(input_shape), [in_op, down_op],
                               loc=self.get_loc(new_name, mlir_gen),
                               ip=ip).output
            if do_norm:
                new_op = self.rms_norm(mlir_gen, new_op, norm, last_name)

            return new_op

        # create block mlir
        def gen_block_by_length(name: str, input_len: int):
            input_shape = [1, input_len, self.hidden_size]
            id_shape = list(self.position_shape)
            id_shape[-1] = input_len
            mask_shape = [1, 1, input_len, input_len]

            q_shape = [1, input_len, self.num_attention_heads, self.head_dim]
            kv_shape = [1, input_len, self.num_key_value_heads, self.head_dim]
            block_mlir = MLIRImporter([input_shape, id_shape, mask_shape],
                                      [input_shape, kv_shape, kv_shape],
                                      name,
                                      Platform.LLM, ["F32", "INT32", "F32"],
                                      weight_file=f"../{weight_file}")

            def T(shape: list):
                return block_mlir.get_tensor_type(shape)

            def L(name: str):
                return self.get_loc(name, block_mlir)

            ip = block_mlir.insert_point

            in0_op = block_mlir.create_input_op(L("input_states"), 0)
            in1_op = block_mlir.create_input_op(L("position_ids"), 1)
            in2_op = block_mlir.create_input_op(L("attention_mask"), 2)
            return_ops = []
            ln_op = self.rms_norm(block_mlir, in0_op, input_ln)

            # q_proj
            q_dim = self.num_attention_heads * self.head_dim
            q_op = self.linear(block_mlir,
                               q_proj,
                               ln_op, [self.hidden_size, q_dim], [1, input_len, q_dim],
                               do_lora=self.do_lora)
            # k_proj
            k_op = self.linear(block_mlir,
                               k_proj,
                               ln_op, [self.hidden_size, self.kv_dim], [1, input_len, self.kv_dim],
                               do_lora=self.do_lora)

            # v_proj
            v_op = self.linear(block_mlir,
                               v_proj,
                               ln_op, [self.hidden_size, self.kv_dim], [1, input_len, self.kv_dim],
                               do_lora=self.do_lora)
            # reshape q,k,v
            q_op = top.ReshapeOp(T(q_shape),
                                 q_op,
                                 shape=[1, -1, self.num_attention_heads, self.head_dim],
                                 loc=L(q_proj + ".reshape"),
                                 ip=ip).output
            k_op = top.ReshapeOp(T(kv_shape),
                                 k_op,
                                 shape=[1, -1, self.num_key_value_heads, self.head_dim],
                                 loc=L(k_proj + ".reshape"),
                                 ip=ip).output
            v_op = top.ReshapeOp(T(kv_shape),
                                 v_op,
                                 shape=[1, -1, self.num_key_value_heads, self.head_dim],
                                 loc=L("v_cache"),
                                 ip=ip).output
            if self.llm_type in [LlmType.QWEN3, LlmType.GEMMA3]:
                q_op = self.rms_norm(block_mlir, q_op, q_norm)
                k_op = self.rms_norm(block_mlir, k_op, k_norm)

            # rotary cos/sin
            q_op, k_op = self.apply_rotary_pos(block_mlir, in1_op, q_op, k_op, rotary_cos,
                                               rotary_sin)
            return_ops.append(k_op)
            return_ops.append(v_op)
            # ======= fattention =========
            fa_op = top.FAttentionOp(T([1, input_len, q_dim]),
                                     q_op,
                                     k_op,
                                     v_op,
                                     in2_op,
                                     block_mlir.none_op,
                                     scale=self.head_dim**-0.5,
                                     batch=1,
                                     q_head=self.num_attention_heads,
                                     kv_head=self.num_key_value_heads,
                                     dim=self.head_dim,
                                     mq=input_len,
                                     mk=input_len,
                                     keep_dims=False,
                                     loc=L(TOP_PATH + "fattention"),
                                     ip=ip).output
            o_op = self.linear(block_mlir,
                               o_proj,
                               fa_op, [q_dim, self.hidden_size],
                               input_shape,
                               do_lora=self.do_lora)
            if self.llm_type == LlmType.GEMMA3:
                o_op = self.rms_norm(block_mlir, o_op, post_attn_ln)
            if self.llm_type == LlmType.MINICPM4:
                o_op = top.MulConstOp(T(input_shape),
                                      o_op,
                                      const_val=self.scale_depth / np.sqrt(self.num_layers),
                                      loc=L(o_proj + ".scale"),
                                      ip=ip).output
            o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(o_proj + ".add"), ip=ip).output
            # ========== mlp =============
            new_op = gen_mlp(block_mlir, input_shape, o_op)
            block_mlir.create_return_op([new_op] + return_ops)
            mlir_txt = block_mlir.print_module()
            if not os.path.exists(name):
                os.makedirs(name)
            target = os.path.join(name, f"{name}.mlir")
            with open(target, "w") as f:
                f.write(mlir_txt)

        def gen_block():
            name = f"block_{idx}"
            if self.share_prompt:
                name = f"block_prompt_{idx}"
                gen_block_by_length(name, self.max_prefill_kv_length)
                return

            gen_block_by_length(name, self.max_input_length)
            return

        def gen_block_cache():
            name = f"block_cache_{idx}"
            input_shape = [1, 1, self.hidden_size]
            id_shape = list(self.position_shape)
            id_shape[-1] = 1
            mask_shape = [1, 1, 1, self.seq_length + 1]
            history_shape = [1, self.seq_length, self.num_key_value_heads, self.head_dim]

            q_shape = [1, 1, self.num_attention_heads, self.head_dim]
            kv_shape = [1, 1, self.num_key_value_heads, self.head_dim]

            block_mlir = MLIRImporter(
                [input_shape, id_shape, mask_shape, history_shape, history_shape],
                [input_shape, kv_shape, kv_shape],
                name,
                Platform.LLM, ["F32", "INT32", "F32", "F32", "F32"],
                weight_file=f"../{weight_file}")

            def T(shape: list):
                return block_mlir.get_tensor_type(shape)

            def L(name: str):
                return self.get_loc(name, block_mlir)

            ip = block_mlir.insert_point

            in0_op = block_mlir.create_input_op(L("input_states"), 0)
            in1_op = block_mlir.create_input_op(L("position_ids"), 1)
            in2_op = block_mlir.create_input_op(L("attention_mask"), 2)
            in3_op = block_mlir.create_input_op(L("history_k"), 3)
            in4_op = block_mlir.create_input_op(L("history_v"), 4)
            return_ops = []
            ln_op = self.rms_norm(block_mlir, in0_op, input_ln)

            # q_proj
            q_dim = self.num_attention_heads * self.head_dim
            q_op = self.linear(block_mlir,
                               q_proj,
                               ln_op, [self.hidden_size, q_dim], [1, 1, q_dim],
                               do_lora=self.do_lora)
            # k_proj
            k_op = self.linear(block_mlir,
                               k_proj,
                               ln_op, [self.hidden_size, self.kv_dim], [1, 1, self.kv_dim],
                               do_lora=self.do_lora)
            # v_proj
            v_op = self.linear(block_mlir,
                               v_proj,
                               ln_op, [self.hidden_size, self.kv_dim], [1, 1, self.kv_dim],
                               do_lora=self.do_lora)
            # reshape q,k,v
            q_op = top.ReshapeOp(T(q_shape), q_op, loc=L(q_proj + ".reshape"), ip=ip).output
            k_op = top.ReshapeOp(T(kv_shape), k_op, loc=L(k_proj + ".reshape"), ip=ip).output
            v_op = top.ReshapeOp(T(kv_shape), v_op, loc=L("v_cache"), ip=ip).output
            if self.llm_type in [LlmType.QWEN3, LlmType.GEMMA3]:
                q_op = self.rms_norm(block_mlir, q_op, q_norm)
                k_op = self.rms_norm(block_mlir, k_op, k_norm)
            # rotary cos/sin
            q_op, k_op = self.apply_rotary_pos(block_mlir, in1_op, q_op, k_op, rotary_cos,
                                               rotary_sin)
            return_ops.append(k_op)
            return_ops.append(v_op)
            # ====== kv concat ========
            k_op = top.ConcatOp(T([1, self.seq_length + 1, self.num_key_value_heads,
                                   self.head_dim]), [in3_op, k_op],
                                axis=1,
                                only_merge=True,
                                loc=L(k_proj + ".concat"),
                                ip=ip).output
            v_op = top.ConcatOp(T([1, self.seq_length + 1, self.num_key_value_heads,
                                   self.head_dim]), [in4_op, v_op],
                                axis=1,
                                only_merge=True,
                                loc=L(v_proj + ".concat"),
                                ip=ip).output
            # ======= fattention =========
            fa_op = top.FAttentionOp(T([1, 1, q_dim]),
                                     q_op,
                                     k_op,
                                     v_op,
                                     in2_op,
                                     block_mlir.none_op,
                                     scale=self.head_dim**-0.5,
                                     batch=1,
                                     q_head=self.num_attention_heads,
                                     kv_head=self.num_key_value_heads,
                                     dim=self.head_dim,
                                     mq=1,
                                     mk=self.seq_length + 1,
                                     keep_dims=False,
                                     loc=L(TOP_PATH + "fattention"),
                                     ip=ip).output
            o_op = self.linear(block_mlir,
                               o_proj,
                               fa_op, [q_dim, self.hidden_size],
                               input_shape,
                               do_lora=self.do_lora)
            if self.llm_type == LlmType.GEMMA3:
                o_op = self.rms_norm(block_mlir, o_op, post_attn_ln)
            if self.llm_type == LlmType.MINICPM4:
                o_op = top.MulConstOp(T(input_shape),
                                      o_op,
                                      const_val=self.scale_depth / np.sqrt(self.num_layers),
                                      loc=L(o_proj + ".scale0"),
                                      ip=ip).output
            o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(o_proj + ".add"), ip=ip).output
            # ========== mlp =============
            new_op = gen_mlp(block_mlir, input_shape, o_op)
            block_mlir.create_return_op([new_op] + return_ops)
            mlir_txt = block_mlir.print_module()
            if not os.path.exists(name):
                os.makedirs(name)
            with open(f"{name}/{name}.mlir", "w") as f:
                f.write(mlir_txt)

        def gen_block_with_kv():
            # Generate block with kv cache related operations
            name = f"block_{idx}"
            input_len = self.max_input_length
            input_shape = [1, input_len, self.hidden_size]
            id_shape = list(self.position_shape)
            max_kv_len = self.max_prefill_kv_length + input_len
            mask_shape = [1, 1, input_len, max_kv_len]
            history_shape = [1, self.max_prefill_kv_length, self.num_key_value_heads, self.head_dim]

            q_shape = [1, input_len, self.num_attention_heads, self.head_dim]
            kv_shape = [1, input_len, self.num_key_value_heads, self.head_dim]

            block_mlir = MLIRImporter(
                [input_shape, id_shape, mask_shape, history_shape, history_shape],
                [input_shape, kv_shape, kv_shape],
                name,
                Platform.LLM, ["F32", "INT32", "F32", "F32", "F32"],
                weight_file=f"../{weight_file}")

            def T(shape: list):
                return block_mlir.get_tensor_type(shape)

            def L(name: str):
                return self.get_loc(name, block_mlir)

            ip = block_mlir.insert_point

            in0_op = block_mlir.create_input_op(L("input_states"), 0)
            in1_op = block_mlir.create_input_op(L("position_ids"), 1)
            in2_op = block_mlir.create_input_op(L("attention_mask"), 2)
            in3_op = block_mlir.create_input_op(L("history_k"), 3)
            in4_op = block_mlir.create_input_op(L("history_v"), 4)
            return_ops = []
            ln_op = self.rms_norm(block_mlir, in0_op, input_ln)

            # q_proj
            q_dim = self.num_attention_heads * self.head_dim
            q_op = self.linear(block_mlir,
                               q_proj,
                               ln_op, [self.hidden_size, q_dim], [1, input_len, q_dim],
                               do_lora=self.do_lora)
            # k_proj
            k_op = self.linear(block_mlir,
                               k_proj,
                               ln_op, [self.hidden_size, self.kv_dim], [1, input_len, self.kv_dim],
                               do_lora=self.do_lora)
            # v_proj
            v_op = self.linear(block_mlir,
                               v_proj,
                               ln_op, [self.hidden_size, self.kv_dim], [1, input_len, self.kv_dim],
                               do_lora=self.do_lora)
            # reshape q,k,v
            q_op = top.ReshapeOp(T(q_shape),
                                 q_op,
                                 shape=[1, -1, self.num_attention_heads, self.head_dim],
                                 loc=L(q_proj + ".reshape"),
                                 ip=ip).output
            k_op = top.ReshapeOp(T(kv_shape),
                                 k_op,
                                 shape=[1, -1, self.num_key_value_heads, self.head_dim],
                                 loc=L(k_proj + ".reshape"),
                                 ip=ip).output
            v_op = top.ReshapeOp(T(kv_shape),
                                 v_op,
                                 shape=[1, -1, self.num_key_value_heads, self.head_dim],
                                 loc=L("v_cache"),
                                 ip=ip).output
            if self.llm_type in [LlmType.QWEN3, LlmType.GEMMA3]:
                q_op = self.rms_norm(block_mlir, q_op, q_norm)
                k_op = self.rms_norm(block_mlir, k_op, k_norm)
            # rotary cos/sin
            q_op, k_op = self.apply_rotary_pos(block_mlir, in1_op, q_op, k_op, rotary_cos,
                                               rotary_sin)
            return_ops.append(k_op)
            return_ops.append(v_op)
            # ====== kv concat ========
            k_op = top.ConcatOp(T([1, max_kv_len, self.num_key_value_heads, self.head_dim]),
                                [in3_op, k_op],
                                axis=1,
                                only_merge=True,
                                loc=L(k_proj + ".concat"),
                                ip=ip).output
            v_op = top.ConcatOp(T([1, max_kv_len, self.num_key_value_heads, self.head_dim]),
                                [in4_op, v_op],
                                axis=1,
                                only_merge=True,
                                loc=L(v_proj + ".concat"),
                                ip=ip).output
            # ======= fattention =========
            fa_op = top.FAttentionOp(T([1, input_len, q_dim]),
                                     q_op,
                                     k_op,
                                     v_op,
                                     in2_op,
                                     block_mlir.none_op,
                                     scale=self.head_dim**-0.5,
                                     batch=1,
                                     q_head=self.num_attention_heads,
                                     kv_head=self.num_key_value_heads,
                                     dim=self.head_dim,
                                     mq=input_len,
                                     mk=max_kv_len,
                                     keep_dims=False,
                                     loc=L(TOP_PATH + "fattention"),
                                     ip=ip).output
            o_op = self.linear(block_mlir,
                               o_proj,
                               fa_op, [q_dim, self.hidden_size],
                               input_shape,
                               do_lora=self.do_lora)
            if self.llm_type == LlmType.GEMMA3:
                o_op = self.rms_norm(block_mlir, o_op, post_attn_ln)
            if self.llm_type == LlmType.MINICPM4:
                o_op = top.MulConstOp(T(input_shape),
                                      o_op,
                                      const_val=self.scale_depth / np.sqrt(self.num_layers),
                                      loc=L(o_proj + ".scale0"),
                                      ip=ip).output
            o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(o_proj + ".add"), ip=ip).output
            # ========== mlp =============
            new_op = gen_mlp(block_mlir, input_shape, o_op)
            block_mlir.create_return_op([new_op] + return_ops)
            mlir_txt = block_mlir.print_module()
            if not os.path.exists(name):
                os.makedirs(name)
            with open(f"{name}/{name}.mlir", "w") as f:
                f.write(mlir_txt)

        if self.use_block_with_kv:
            gen_block_with_kv()
        else:
            gen_block()
        if self.share_prompt:
            gen_block()
        gen_block_cache()

    def gen_vit_mlir(self):
        pass

    # ============= compile all code =============
    def add_task(self, command: list[str], log_file: str):
        command.append(f") > {log_file}\n")
        cmd = "(" + ' '.join(command)
        self.commands.append(cmd)

    def run_command(self, command):
        GREEN_COLOR = "\033[92m"  # ANSI escape code for green text
        RED_COLOR = "\033[91m"
        RESET_COLOR = "\033[0m"
        try:
            print(f"{GREEN_COLOR}Executing command: \n{' '.join(command)}{RESET_COLOR}"
                  )  # Print the command in green
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            # Print the error message in red
            print(f"{RED_COLOR}Error: Command failed with return code {e.returncode}{RESET_COLOR}")
            print(f"{RED_COLOR}Failed command: {' '.join(command)}{RESET_COLOR}")
            # Exit the program with the same return code as the failed command
            sys.exit(e.returncode)

    def execute_tasks(self):
        task_file = "task.txt"
        with open(task_file, "w") as f:
            f.writelines(self.commands)
        self.commands.clear()
        parallel_cmd = [
            "parallel", f"-j {self.max_workers}", "--halt now,fail=1", "--progress",
            f"--joblog {task_file}.log", f"< {task_file}"
        ]
        self.run_command(['bash', '-c', ' '.join(parallel_cmd)])

    def compile_lm_head(self):
        name = "lm_head"
        model_path = f"{name}/{name}.bmodel"
        if self.tie_word_embeddings:
            # share the embedding weights
            self.all_bmodels_without_bytes.append(model_path)
        else:
            self.all_bmodels.append(model_path)
        if os.path.exists(model_path):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return
        deploy_args = [
            f'pushd {name} && ', 'model_deploy.py', f'--mlir {name}.mlir',
            f'--quantize {self.half_precision_quantize}', '--quant_input', f'--chip {self.chip}',
            f'--num_core {self.num_core}', f'--num_device {self.num_device}',
            f'--model {name}.bmodel'
        ]
        if self.debug:
            deploy_args.append('--debug')
        deploy_args.append('&& popd')
        self.add_task(deploy_args, f"{name}.log")

    def compile_greedy_head(self):
        name = "greedy_head"
        model_path = f"{name}/{name}.bmodel"
        self.all_bmodels.append(model_path)
        if os.path.exists(model_path):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return
        deploy_args = [
            f'pushd {name} && ', 'model_deploy.py', f'--mlir {name}.mlir', f'--chip {self.chip}',
            '--addr_mode io_alone', f'--model {name}.bmodel'
        ]
        if self.debug:
            deploy_args.append('--debug')
        deploy_args.append('&& popd')
        self.add_task(deploy_args, f"{name}.log")

    def compile_sample_head(self):
        name = "sample_head"
        model_path = f"{name}/{name}.bmodel"
        self.all_bmodels.append(model_path)
        if os.path.exists(model_path):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return
        deploy_args = [
            f'pushd {name} && ', 'model_deploy.py', f'--mlir {name}.mlir', f'--chip {self.chip}',
            '--dynamic', '--addr_mode io_alone', f'--model {name}.bmodel'
        ]
        if self.debug:
            deploy_args.append('--debug')
        deploy_args.append('&& popd')
        self.add_task(deploy_args, f"{name}.log")

    def compile_block(self, layer_id):
        name = f"block_{layer_id}"
        model_path = f"{name}/{name}.bmodel"
        self.all_bmodels_without_bytes.append(model_path)
        if os.path.exists(model_path):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return

        deploy_args = [
            f'pushd {name} && ', 'model_deploy.py', f'--mlir {name}.mlir',
            f'--quantize {self.quantize}', f'--q_group_size {self.q_group_size}', '--quant_input',
            '--quant_output', f'--chip {self.chip}', f'--num_core {self.num_core}',
            f'--num_device {self.num_device}', f'--model {name}.bmodel'
        ]
        if self.high_precision:
            deploy_args.append('--high_precision')
        if self.symmetric:
            deploy_args.append('--q_symmetric')
        if self.dynamic:
            deploy_args.append('--dynamic')
        if self.debug:
            deploy_args.append('--debug')
        if self.same_addr:
            deploy_args.append(f'--same_addr {self.same_addr}')
        deploy_args.append('&& popd')
        self.add_task(deploy_args, f"{name}.log")

    def compile_block_cache(self, layer_id):
        name = f"block_cache_{layer_id}"
        model_path = f"{name}/{name}.bmodel"
        self.all_bmodels.append(model_path)
        if os.path.exists(model_path):
            print(f"{model_path} already exists. Skipping compilation.")
            return

        deploy_args = [
            f'pushd {name} && ', 'model_deploy.py', f'--mlir {name}.mlir',
            f'--quantize {self.quantize}', f'--q_group_size {self.q_group_size}', '--quant_input',
            '--quant_output', f'--chip {self.chip}', '--addr_mode io_alone',
            f'--num_core {self.num_core}', f'--num_device {self.num_device}',
            f'--model {name}.bmodel'
        ]
        if self.high_precision:
            deploy_args.append('--high_precision')
        if self.symmetric:
            deploy_args.append('--q_symmetric')
        if self.debug:
            deploy_args.append('--debug')
        if self.same_addr:
            deploy_args.append(f'--same_addr {self.same_addr}')
        deploy_args.append('&& popd')
        self.add_task(deploy_args, f"{name}.log")

    def compile_block_prompt(self, layer_id):
        name = f"block_prompt_{layer_id}"
        model_path = f"{name}/{name}.bmodel"
        self.all_bmodels_without_bytes.append(model_path)
        if os.path.exists(model_path):
            print(f"{model_path} already exists. Skipping compilation.")
            return

        deploy_args = [
            f'pushd {name} && ', 'model_deploy.py', f'--mlir {name}.mlir',
            f'--quantize {self.quantize}', f'--q_group_size {self.q_group_size}', '--quant_input',
            '--quant_output', f'--chip {self.chip}', f'--num_core {self.num_core}',
            f'--num_device {self.num_device}', f'--model {name}.bmodel'
        ]
        if self.high_precision:
            deploy_args.append('--high_precision')
        if self.symmetric:
            deploy_args.append('--q_symmetric')
        if self.debug:
            deploy_args.append('--debug')
        deploy_args.append('&& popd')
        self.add_task(deploy_args, f"{name}.log")

    def compile_vit(self):
        if not self.do_vit:
            return
        name = "vit"
        model_path = f"{name}/{name}.bmodel"
        self.all_bmodels.append(model_path)
        if os.path.exists(model_path):
            print(f"{model_path} already exists. Skipping compilation.")
            return
        deploy_args = [
            f'pushd {name} && ', 'model_deploy.py', f'--mlir {name}.mlir', f'--chip {self.chip}',
            f'--num_core {self.num_core}', f'--num_device {self.num_device}',
            f'--model {name}.bmodel'
        ]
        if self.half_precision_quantize == 'bf16' and self.vit_f16_out_bf16:
            deploy_args.append('--quantize f16')
            deploy_args.append('--quant_output_bf16')
        else:
            deploy_args.append(f'--quantize {self.half_precision_quantize}')
            deploy_args.append('--quant_output')
        if self.high_precision:
            deploy_args.append('--high_precision')
        if self.debug:
            deploy_args.append('--debug')
        if self.dynamic_vit:
            deploy_args.append('--dynamic')
        deploy_args.append('&& popd')
        self.add_task(deploy_args, f"{name}.log")

    def compile_common(self, name, with_size=False, io_alone=False):
        model_path = f"{name}/{name}.bmodel"
        if not with_size:
            self.all_bmodels_without_bytes.append(model_path)
        else:
            self.all_bmodels.append(model_path)
        if os.path.exists(model_path):
            print(f"{model_path} already exists. Skipping compilation.")
            return
        deploy_args = [
            f'pushd {name} && ', 'model_deploy.py', f'--mlir {name}.mlir',
            f'--quantize {self.half_precision_quantize}', '--quant_input', '--quant_output',
            f'--chip {self.chip}', f'--num_core {self.num_core}', f'--num_device {self.num_device}',
            f'--model {name}.bmodel'
        ]
        if self.debug:
            deploy_args.append('--debug')
        if io_alone:
            deploy_args.append('--addr_mode io_alone')
        deploy_args.append('&& popd')
        self.add_task(deploy_args, f"{name}.log")

    def combine(self):
        bmodel_list = []
        total_bytes = 0
        for bmodel in self.all_bmodels:
            bmodel_list += [bmodel]
            total_bytes += os.path.getsize(bmodel)
        for bmodel in self.all_bmodels_without_bytes:
            bmodel_list += [bmodel]

        combine_args = ['model_tool', '--combine', ' '.join(bmodel_list), '-o', self.out_bmodel]
        self.run_command(['bash', '-c', ' '.join(combine_args)])
        # Get the size of the combined bmodel
        bmodel_size = os.path.getsize(self.out_bmodel)
        print(f"Combined bmodel size: {bmodel_size / (1024.0 ** 3)} GB")
        if bmodel_size > total_bytes * 1.2:
            raise RuntimeError("Combined bmodel size is too large, please check the model.")

        get_info_args = ['model_tool', '--info', self.out_bmodel, '> ../model.log']
        self.run_command(['bash', '-c', ' '.join(get_info_args)])

    def compile_all(self):

        if self.do_vit:
            self.all_compiles.append(self.compile_vit)

        if not self.embedding_disk:
            self.all_compiles.append(lambda: self.compile_common("embedding", with_size=True))
            self.all_compiles.append(
                lambda: self.compile_common("embedding_cache", with_size=False))

        if self.do_lora:
            self.all_compiles.append(
                lambda: self.compile_common("lm_head_lora", with_size=True, io_alone=True))
            self.all_compiles.append(
                lambda: self.compile_common("embedding_lora", with_size=True, io_alone=True))
            self.all_compiles.append(
                lambda: self.compile_common("embedding_cache_lora", with_size=False, io_alone=True))

        self.all_compiles.append(self.compile_lm_head)

        if not self.lmhead_with_topk:
            self.all_compiles.append(self.compile_greedy_head)
            self.all_compiles.append(self.compile_sample_head)

        for i in range(self.num_layers):
            self.all_compiles.append(lambda i=i: self.compile_block(i))
            self.all_compiles.append(lambda i=i: self.compile_block_cache(i))
            if self.share_prompt:
                self.all_compiles.append(lambda i=i: self.compile_block_prompt(i))

        for func in self.all_compiles:
            func()

        self.execute_tasks()

        # Combine all bmodel files
        self.combine()

        # Remove any .npz files
        if not self.debug:
            for dirpath, _, filenames in os.walk('.'):
                if dirpath.startswith("./config"):
                    continue
                for filename in filenames:
                    if filename.endswith('.npz'):
                        file_path = os.path.join(dirpath, filename)
                        os.remove(file_path)
