# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import torch
import os
from .MLIRImporter import MLIRImporter, Platform
from .BaseConverter import BaseConverter
from .LlmInfo import COMMON_INFO, LlmList
from transformers import AutoConfig
import numpy as np
from tqdm import tqdm
from datetime import datetime
from safetensors import safe_open
import concurrent.futures
import subprocess
import sys
from mlir.ir import *
import mlir.dialects.top as top


class LlmLoad:

    def __init__(self, model_path: str):
        self.st_files = []
        # get all safetensors
        for entry in os.listdir(model_path):
            file_path = os.path.join(model_path, entry)
            if os.path.isfile(file_path) and entry.lower().endswith('.safetensors'):
                f = safe_open(file_path, "pt")
                self.st_files.append(f)

    def read(self, key: str):
        for f in self.st_files:
            if key in f.keys():
                data = f.get_tensor(key)
                if data.dtype in [torch.float16, torch.bfloat16]:
                    return data.float().numpy()
                return data.numpy()
        raise RuntimeError(f"Can't find key: {key}")

    def is_exist(self, key: str):
        for f in self.st_files:
            if key in f.keys():
                return True
        return False


# support qwen2/llama
class LlmConverter(BaseConverter):

    def __init__(self, args):
        super().__init__()
        self.MODEL_SUPPORED = ["qwen2", "llama"]
        self.model_path = os.path.normpath(args.model_path)
        self.seq_length = args.seq_length
        self.quantize = args.quantize
        self.num_device = args.num_device
        self.q_group_size = args.q_group_size
        self.high_precision = True
        self.symmetric = args.symmetric
        self.lmhead_with_topk = True
        self.chip = args.chip
        self.num_device = args.num_device
        self.embedding_disk = args.embedding_disk
        self.debug = args.debug
        self.num_core = args.num_core
        if self.num_core == 0:
            self.num_core = 1 if args.chip != "bm1688" else 2
        self.half_precision_quantize = "bf16" if "bf16" in self.quantize else "f16"
        self.load_pretrained()
        # get attributes
        self.init_config()
        cos, sin = self.get_rotary_pos_emb(self.seq_length)
        self.cos = cos.numpy()
        self.sin = sin.numpy()
        cpu_count = os.cpu_count()
        self.max_workers = max(cpu_count, 4)
        # get file path
        self.out_dir = args.out_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = os.path.basename(self.model_path).lower()
        if args.chip == "bm1684x":
            folder_name = f"bmodel_seq{self.seq_length}_{self.quantize}_{self.chip}_{self.num_device}dev"
            self.out_bmodel = f"../{self.model_name}_{self.quantize}_seq{self.seq_length}_{self.chip}_{self.num_device}dev_{timestamp}.bmodel"
        else:
            folder_name = f"bmodel_seq{self.seq_length}_{self.quantize}_{self.chip}_{self.num_core}core"
            self.out_bmodel = f"../{self.model_name}_{self.quantize}_seq{self.seq_length}_{self.chip}_{self.num_core}core_{timestamp}.bmodel"

        self.bmodel_dir = os.path.join(self.out_dir, folder_name)
        self.commands = []

    def run(self):
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.bmodel_dir, exist_ok=True)
        ori_path = os.getcwd()
        os.chdir(self.bmodel_dir)
        self.gen_all_mlir()
        del self.model
        self.compile_all()
        os.chdir(ori_path)
        print(f"Success: {self.model_path} has converted to {self.out_dir}")

    def gen_all_mlir(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []

            futures.append(executor.submit(self.gen_embedding_lmhead_mlir))

            for i in range(self.num_layers):
                futures.append(executor.submit(self.gen_block_mlir, i))

            # Wait for all threads to complete
            for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="generate mlir",
            ):
                # This will raise exceptions if any occurred during thread execution
                future.result()

    def load_pretrained(self):
        self.config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = LlmLoad(self.model_path)
        self.model_type = self.config.model_type
        if self.model_type not in self.MODEL_SUPPORED:
            raise RuntimeError("Not Implemented")
        self.model_info = COMMON_INFO

    def get_rotary_pos_emb(self, seq_length):
        position_ids = torch.tensor([range(seq_length)], dtype=torch.long)
        theta = 1.0 / (self.rope_theta**(torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) /
                                         self.rotary_dim))
        position_ids = position_ids.float().reshape(-1, 1)
        idx_theta = position_ids * theta
        rotary_pos_emb = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)])
        if self.model_type != 'chatglm2':
            rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(2).unsqueeze(1)
        return rotary_pos_emb

    def init_config(self):
        c = self.model_info.config
        self.num_layers = getattr(self.config, c.num_hidden_layers)
        self.rope_theta = getattr(self.config, c.rope_theta, 10000.0)
        self.num_attention_heads = getattr(self.config, c.num_attention_heads)
        self.num_key_value_heads = getattr(self.config, c.num_key_value_heads,
                                           self.num_attention_heads)
        self.hidden_size = getattr(self.config, c.hidden_size)
        self.vocab_size = getattr(self.config, c.vocab_size)
        self.intermediate_size = getattr(self.config, c.intermediate_size)
        self.rms_norm_eps = getattr(self.config, c.rms_norm_eps)
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.rotary_dim = self.head_dim
        self.kv_dim = self.num_key_value_heads * self.head_dim
        self.kv_tile = self.num_attention_heads // self.num_key_value_heads
        if hasattr(self.config, 'rotary_dim'):
            self.rotary_dim = self.config.rotary_dim
        if self.model_type == 'chatglm':
            self.rotary_dim = self.config.head_dim // 2
        # whether llm head and embedding share weight
        self.tie_word_embeddings = getattr(self.config, 'tie_word_embeddings', False)
        # whether to merge lm_head and embedding in bmodel
        self.do_lmhead_merge = self.tie_word_embeddings and not self.embedding_disk and self.num_device < 2

    def get_loc(self, names, mlir):
        if isinstance(names, str):
            return Location.fused([Location.name(names)], context=mlir.ctx)
        elif isinstance(names, list):
            return Location.fused([Location.name(n) for n in names], context=mlir.ctx)
        else:
            raise RuntimeError("Unknown names:{}".format(names))

    def gen_embedding_bin(self, embedding_data):
        embedding_file = '../embedding.bin'
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
        embedding_path = self.model_info.weights[LlmList.EMBEDING] + ".weight"
        embedding_data = self.model.read(embedding_path)
        if self.embedding_disk:
            self.gen_embedding_bin(embedding_data)
        else:
            # read embedding weights
            embedding_weights = {embedding_path: embedding_data}
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
        if not self.do_lmhead_merge:
            lmhead_data = np.ascontiguousarray(np.transpose(lmhead_data, (1, 0)))
            norm_data = self.model.read(norm_path)
            lmhead_weights = {lmhead_path: lmhead_data, norm_path: norm_data}
        else:
            lmhead_weights = {lmhead_path: lmhead_data}

        lmhead_npz = "lm_head_top_weights.npz"
        np.savez(lmhead_npz, **lmhead_weights)

        # gen embedding mlir
        def gen_by_length(name: str, seq_length: int):
            embedding_mlir = MLIRImporter([[1, seq_length]], [[1, seq_length, self.hidden_size]],
                                          name,
                                          Platform.LLM,
                                          input_types=["INT32"],
                                          weight_file=embedding_npz)
            input_op = embedding_mlir.create_input_op(self.get_loc("input_ids", embedding_mlir), 0)
            weight_op = embedding_mlir.create_weight_op(embedding_path,
                                                        [self.vocab_size, self.hidden_size])
            new_op = top.GatherOp(embedding_mlir.get_tensor_type([1, seq_length, self.hidden_size]),
                                  weight_op,
                                  input_op,
                                  axis=0,
                                  loc=self.get_loc(name, embedding_mlir),
                                  ip=embedding_mlir.insert_point).output
            embedding_mlir.create_return_op([new_op])
            mlir_txt = embedding_mlir.print_module()
            with open(f"{name}.mlir", "w") as f:
                f.write(mlir_txt)

        # gen lm_head mlir
        def gen_lm_head():
            lmhead_mlir = MLIRImporter([[1, self.hidden_size]], [[1, 1]],
                                       "lm_head",
                                       Platform.LLM,
                                       weight_file=lmhead_npz)
            input_op = lmhead_mlir.create_input_op(self.get_loc("hidden_states", lmhead_mlir), 0)
            if not self.do_lmhead_merge:
                weight_op = lmhead_mlir.create_weight_op(norm_path, [1, self.hidden_size])
                input_op = top.RMSNormOp(lmhead_mlir.get_tensor_type([1, self.hidden_size]),
                                         input_op,
                                         weight_op,
                                         eps=self.rms_norm_eps,
                                         loc=self.get_loc(norm, lmhead_mlir),
                                         ip=lmhead_mlir.insert_point).output
                w_shape = [self.hidden_size, self.vocab_size]
                lmhead_op = self.linear(lmhead_mlir, lmhead, input_op, w_shape,
                                        [1, self.vocab_size])
            else:
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

            topk_op = top.TopKOp(*lmhead_mlir.get_tensor_type([[1, 1], [1, 1]]),
                                 lmhead_op,
                                 axis=1,
                                 K=1,
                                 loc=self.get_loc(["token_value", "token_id"], lmhead_mlir),
                                 ip=lmhead_mlir.insert_point)
            # topk_op.values, topk_op.indices
            lmhead_mlir.create_return_op([topk_op.indices])
            mlir_txt = lmhead_mlir.print_module()
            with open("lm_head.mlir", "w") as f:
                f.write(mlir_txt)

        if not self.embedding_disk:
            gen_by_length("embedding", self.seq_length)
            gen_by_length("embedding_cache", 1)
        gen_lm_head()

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
                           loc=self.get_loc(prefix + ".tile.reshape", mlir_gen),
                           ip=mlir_gen.insert_point).output
        return rs

    def linear(self, mlir_gen, proj: str, input_op, weight_shape: list, out_shape: list):
        weight_op = mlir_gen.create_weight_op(proj + ".weight", weight_shape)
        if self.model.is_exist(proj + ".bias"):
            bias_shape = [1] * (len(out_shape) - 1) + [out_shape[-1]]
            bias_op = mlir_gen.create_weight_op(proj + ".bias", bias_shape)
        else:
            bias_op = mlir_gen.none_op
        return top.MatMulOp(mlir_gen.get_tensor_type(out_shape),
                            input_op,
                            weight_op,
                            bias_op,
                            do_relu=False,
                            loc=self.get_loc(proj, mlir_gen),
                            ip=mlir_gen.insert_point).output

    def rotary_pos(self, mlir_gen, in_op, cos_op, sin_op, out_name: str, in_shape: list,
                   half_shape: list):
        prefix = f"{out_name}.rotary_pos"
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

    def set_linear_weight(self, path: str, weight_dict: dict):
        weight_path = path + ".weight"
        bias_path = path + ".bias"
        if self.model.is_exist(weight_path):
            data = self.model.read(weight_path)
            weight_dict[weight_path] = np.ascontiguousarray(np.transpose(data, (1, 0)))
        else:
            raise RuntimeError("Can't find key: {}".format(weight_path))
        if self.model.is_exist(bias_path):
            weight_dict[bias_path] = self.model.read(bias_path)

    def set_common_weight(self, path: str, weight_dict: dict):
        weight_path = path + ".weight"
        if self.model.is_exist(weight_path):
            weight_dict[weight_path] = self.model.read(weight_path)
        else:
            raise RuntimeError("Can't find key: {}".format(weight_path))

    def gen_block_mlir(self, idx: int):
        tqdm.write(f"generate block_{idx} mlir ...")
        # torch path
        TOP_PATH = f'{self.model_info.weights[LlmList.LAYERS]}.{idx}.'
        input_ln = TOP_PATH + self.model_info.weights[LlmList.INPUT_LN]
        q_proj = TOP_PATH + self.model_info.weights[LlmList.Q_PROJ]
        k_proj = TOP_PATH + self.model_info.weights[LlmList.K_PROJ]
        v_proj = TOP_PATH + self.model_info.weights[LlmList.V_PROJ]
        o_proj = TOP_PATH + self.model_info.weights[LlmList.O_PROJ]
        post_ln = TOP_PATH + self.model_info.weights[LlmList.POST_LN]
        mlp_gate = TOP_PATH + self.model_info.weights[LlmList.MLP_GATE]
        mlp_up = TOP_PATH + self.model_info.weights[LlmList.MLP_UP]
        mlp_down = TOP_PATH + self.model_info.weights[LlmList.MLP_DOWN]
        norm = self.model_info.weights[LlmList.NORM]
        do_norm = self.do_lmhead_merge and idx == self.num_layers - 1
        rotary_cos = "rotary_cos"
        rotary_sin = "rotary_sin"

        # save weight
        weight_file = f"block_{idx}_top_weights.npz"
        weight_dict = {
            rotary_cos + ".weight": self.cos,
            rotary_sin + ".weight": self.sin,
        }
        self.set_common_weight(input_ln, weight_dict)
        self.set_linear_weight(q_proj, weight_dict)
        self.set_linear_weight(k_proj, weight_dict)
        self.set_linear_weight(v_proj, weight_dict)
        self.set_linear_weight(o_proj, weight_dict)
        self.set_common_weight(post_ln, weight_dict)
        self.set_linear_weight(mlp_gate, weight_dict)
        self.set_linear_weight(mlp_up, weight_dict)
        self.set_linear_weight(mlp_down, weight_dict)
        if do_norm:
            self.set_common_weight(norm, weight_dict)

        np.savez(weight_file, **weight_dict)

        def gen_mlp(mlir_gen, input_shape, in_op):
            ip = mlir_gen.insert_point
            len = input_shape[1]
            if self.model.is_exist(post_ln + ".weight"):
                weight_op = mlir_gen.create_weight_op(post_ln + ".weight", [1, 1, self.hidden_size])
                new_op = top.RMSNormOp(mlir_gen.get_tensor_type(input_shape),
                                       in_op,
                                       weight_op,
                                       eps=self.rms_norm_eps,
                                       loc=self.get_loc(post_ln, mlir_gen),
                                       ip=ip).output
            else:
                new_op = in_op
            gate_op = self.linear(mlir_gen, mlp_gate, new_op,
                                  [self.hidden_size, self.intermediate_size],
                                  [1, len, self.intermediate_size])
            silu_op = top.SiLUOp(mlir_gen.get_tensor_type([1, len, self.intermediate_size]),
                                 gate_op,
                                 loc=self.get_loc(mlp_gate + ".silu", mlir_gen),
                                 ip=ip).output
            up_op = self.linear(mlir_gen, mlp_up, new_op,
                                [self.hidden_size, self.intermediate_size],
                                [1, len, self.intermediate_size])
            new_op = top.MulOp(mlir_gen.get_tensor_type([1, len, self.intermediate_size]),
                               [silu_op, up_op],
                               loc=self.get_loc(mlp_up + ".mul", mlir_gen),
                               ip=ip).output
            down_op = self.linear(mlir_gen, mlp_down, new_op,
                                  [self.intermediate_size, self.hidden_size], input_shape)
            last_name = "output_states"
            new_name = last_name if idx != self.num_layers - 1 else f"{mlp_down}.add"
            new_op = top.AddOp(mlir_gen.get_tensor_type(input_shape), [in_op, down_op],
                               loc=self.get_loc(new_name, mlir_gen),
                               ip=ip).output
            if do_norm:
                weight_op = mlir_gen.create_weight_op(norm + ".weight", [1, 1, self.hidden_size])
                new_op = top.RMSNormOp(mlir_gen.get_tensor_type(input_shape),
                                       new_op,
                                       weight_op,
                                       eps=self.rms_norm_eps,
                                       loc=self.get_loc(last_name, mlir_gen),
                                       ip=ip).output
            return new_op

        # create block mlir
        def gen_block():
            name = f"block_{idx}"
            input_shape = [1, self.seq_length, self.hidden_size]
            id_shape = [1, self.seq_length]
            mask_shape = [1, 1, self.seq_length, self.seq_length]

            q_shape = [1, self.seq_length, self.num_attention_heads, self.head_dim]
            q_half_shape = [1, self.seq_length, self.num_attention_heads, self.head_dim // 2]
            kv_shape = [1, self.seq_length, self.num_key_value_heads, self.head_dim]
            kv_half_shape = [1, self.seq_length, self.num_key_value_heads, self.head_dim // 2]
            block_mlir = MLIRImporter([input_shape, id_shape, mask_shape],
                                      [input_shape, kv_shape, kv_shape],
                                      name,
                                      Platform.LLM, ["F32", "INT32", "F32"],
                                      weight_file=weight_file)

            def T(shape: list):
                return block_mlir.get_tensor_type(shape)

            def L(name: str):
                return self.get_loc(name, block_mlir)

            ip = block_mlir.insert_point

            in0_op = block_mlir.create_input_op(L("input_states"), 0)
            in1_op = block_mlir.create_input_op(L("position_ids"), 1)
            in2_op = block_mlir.create_input_op(L("attention_mask"), 2)
            return_ops = []
            weight_op = block_mlir.create_weight_op(input_ln + ".weight", [1, 1, self.hidden_size])
            ln_op = top.RMSNormOp(T(input_shape),
                                  in0_op,
                                  weight_op,
                                  eps=self.rms_norm_eps,
                                  loc=L(input_ln),
                                  ip=ip).output
            # q_proj
            q_op = self.linear(block_mlir, q_proj, ln_op, [self.hidden_size, self.hidden_size],
                               input_shape)
            # k_proj
            k_op = self.linear(block_mlir, k_proj, ln_op, [self.hidden_size, self.kv_dim],
                               [1, self.seq_length, self.kv_dim])
            # v_proj
            v_op = self.linear(block_mlir, v_proj, ln_op, [self.hidden_size, self.kv_dim],
                               [1, self.seq_length, self.kv_dim])
            # reshape q,k,v
            q_op = top.ReshapeOp(T(q_shape), q_op, loc=L(q_proj + ".reshpae"), ip=ip).output
            k_op = top.ReshapeOp(T(kv_shape), k_op, loc=L(k_proj + ".reshpae"), ip=ip).output
            v_op = top.ReshapeOp(T(kv_shape), v_op, loc=L("v_cache"), ip=ip).output
            # rotary cos/sin
            weight_op = block_mlir.create_weight_op(rotary_cos + ".weight",
                                                    [self.seq_length, 1, self.head_dim])
            cos_op = top.GatherOp(T([1, self.seq_length, 1, self.head_dim]),
                                  weight_op,
                                  in1_op,
                                  axis=0,
                                  loc=L(rotary_cos),
                                  ip=ip).output
            weight_op = block_mlir.create_weight_op(rotary_sin + ".weight",
                                                    [self.seq_length, 1, self.head_dim])
            sin_op = top.GatherOp(T([1, self.seq_length, 1, self.head_dim]),
                                  weight_op,
                                  in1_op,
                                  axis=0,
                                  loc=L(rotary_sin),
                                  ip=ip).output
            # ===== q_proj rotary ========
            q_op = self.rotary_pos(block_mlir, q_op, cos_op, sin_op, "q_proj", q_shape,
                                   q_half_shape)

            # ===== k_proj rotary ========
            k_op = self.rotary_pos(block_mlir, k_op, cos_op, sin_op, "k_cache", kv_shape,
                                   kv_half_shape)
            return_ops.append(k_op)
            return_ops.append(v_op)
            # ======= fattention =========
            fa_op = top.FAttentionOp(T(input_shape),
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
                                     mq=self.seq_length,
                                     mk=self.seq_length,
                                     loc=L(TOP_PATH + "fattention"),
                                     ip=ip).output
            o_op = self.linear(block_mlir, o_proj, fa_op, [self.hidden_size, self.hidden_size],
                               input_shape)
            o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(o_proj + ".add"), ip=ip).output
            # ========== mlp =============
            new_op = gen_mlp(block_mlir, input_shape, o_op)
            block_mlir.create_return_op([new_op] + return_ops)
            mlir_txt = block_mlir.print_module()
            with open(f"{name}.mlir", "w") as f:
                f.write(mlir_txt)

        def gen_block_cache():
            name = f"block_cache_{idx}"
            input_shape = [1, 1, self.hidden_size]
            id_shape = [1, 1]
            mask_shape = [1, 1, 1, self.seq_length + 1]
            history_shape = [1, self.seq_length, self.num_key_value_heads, self.head_dim]

            q_shape = [1, 1, self.num_attention_heads, self.head_dim]
            q_half_shape = [1, 1, self.num_attention_heads, self.head_dim // 2]
            kv_shape = [1, 1, self.num_key_value_heads, self.head_dim]
            kv_half_shape = [1, 1, self.num_key_value_heads, self.head_dim // 2]

            block_mlir = MLIRImporter(
                [input_shape, id_shape, mask_shape, history_shape, history_shape],
                [input_shape, kv_shape, kv_shape],
                name,
                Platform.LLM, ["F32", "INT32", "F32", "F32", "F32"],
                weight_file=weight_file)

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
            weight_op = block_mlir.create_weight_op(input_ln + ".weight", [1, 1, self.hidden_size])
            ln_op = top.RMSNormOp(T(input_shape),
                                  in0_op,
                                  weight_op,
                                  eps=self.rms_norm_eps,
                                  loc=L(input_ln),
                                  ip=ip).output
            # q_proj
            q_op = self.linear(block_mlir, q_proj, ln_op, [self.hidden_size, self.hidden_size],
                               input_shape)
            # k_proj
            k_op = self.linear(block_mlir, k_proj, ln_op, [self.hidden_size, self.kv_dim],
                               [1, 1, self.kv_dim])
            # v_proj
            v_op = self.linear(block_mlir, v_proj, ln_op, [self.hidden_size, self.kv_dim],
                               [1, 1, self.kv_dim])
            # reshape q,k,v
            q_op = top.ReshapeOp(T(q_shape), q_op, loc=L(q_proj + ".reshpae"), ip=ip).output
            k_op = top.ReshapeOp(T(kv_shape), k_op, loc=L(k_proj + ".reshpae"), ip=ip).output
            v_op = top.ReshapeOp(T(kv_shape), v_op, loc=L("v_cache"), ip=ip).output
            # rotary cos/sin
            weight_op = block_mlir.create_weight_op(rotary_cos + ".weight",
                                                    [self.seq_length, 1, self.head_dim])
            cos_op = top.GatherOp(T([1, 1, 1, self.head_dim]),
                                  weight_op,
                                  in1_op,
                                  axis=0,
                                  loc=L(rotary_cos),
                                  ip=ip).output
            weight_op = block_mlir.create_weight_op(rotary_sin + ".weight",
                                                    [self.seq_length, 1, self.head_dim])
            sin_op = top.GatherOp(T([1, 1, 1, self.head_dim]),
                                  weight_op,
                                  in1_op,
                                  axis=0,
                                  loc=L(rotary_sin),
                                  ip=ip).output
            # ===== q_proj rotary ========
            q_op = self.rotary_pos(block_mlir, q_op, cos_op, sin_op, "q_rotary", q_shape,
                                   q_half_shape)

            # ===== k_proj rotary ========
            k_op = self.rotary_pos(block_mlir, k_op, cos_op, sin_op, "k_cache", kv_shape,
                                   kv_half_shape)
            return_ops.append(k_op)
            return_ops.append(v_op)
            # ====== kv concat ========
            k_op = top.ConcatOp(T([1, self.seq_length + 1, self.num_key_value_heads,
                                   self.head_dim]), [in3_op, k_op],
                                axis=1,
                                loc=L(k_proj + ".concat"),
                                ip=ip).output
            v_op = top.ConcatOp(T([1, self.seq_length + 1, self.num_key_value_heads,
                                   self.head_dim]), [in4_op, v_op],
                                axis=1,
                                loc=L(v_proj + ".concat"),
                                ip=ip).output
            # ======= fattention =========
            fa_op = top.FAttentionOp(T(input_shape),
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
                                     loc=L(TOP_PATH + "fattention"),
                                     ip=ip).output
            o_op = self.linear(block_mlir, o_proj, fa_op, [self.hidden_size, self.hidden_size],
                               input_shape)
            o_op = top.AddOp(T(input_shape), [in0_op, o_op], loc=L(o_proj + ".add"), ip=ip).output
            # ========== mlp =============
            new_op = gen_mlp(block_mlir, input_shape, o_op)
            block_mlir.create_return_op([new_op] + return_ops)
            mlir_txt = block_mlir.print_module()
            with open(f"{name}.mlir", "w") as f:
                f.write(mlir_txt)

        gen_block()
        gen_block_cache()

    # ============= compile all code =============
    def send_command(self, command: list[str], log_file: str):
        command.append(f"> {log_file}\n")
        cmd = ' '.join(command)
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

    def execute_commands(self):
        task_file = "task.txt"
        with open(task_file, "w") as f:
            f.writelines(self.commands)
        self.commands.clear()
        parallel_cmd = [
            "parallel", f"-j {self.max_workers}", "--progress", f"--joblog {task_file}.log",
            f"< {task_file}"
        ]
        self.run_command(['bash', '-c', ' '.join(parallel_cmd)])

    def compile_embedding(self):
        name = "embedding"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return
        deploy_args = [
            'model_deploy.py', f'--mlir {name}.mlir', f'--quantize {self.half_precision_quantize}',
            '--quant_input', '--quant_output', f'--chip {self.chip}', f'--num_core {self.num_core}',
            f'--num_device {self.num_device}', f'--model {name}.bmodel'
        ]
        if self.debug:
            deploy_args.append('--debug')
        self.send_command(deploy_args, f"{name}.log")

    def compile_embedding_cache(self):
        name = "embedding_cache"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return
        deploy_args = [
            'model_deploy.py', f'--mlir {name}.mlir', f'--quantize {self.half_precision_quantize}',
            '--quant_input', '--quant_output', f'--chip {self.chip}', f'--num_core {self.num_core}',
            f'--num_device {self.num_device}', f'--model {name}.bmodel'
        ]
        if self.debug:
            deploy_args.append('--debug')
        self.send_command(deploy_args, f"{name}.log")

    def compile_lm_head(self):
        name = "lm_head"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return
        deploy_args = [
            'model_deploy.py', f'--mlir {name}.mlir', f'--quantize {self.half_precision_quantize}',
            '--quant_input', f'--chip {self.chip}', f'--num_core {self.num_core}',
            f'--num_device {self.num_device}', f'--model {name}.bmodel'
        ]
        if self.debug:
            deploy_args.append('--debug')
        self.send_command(deploy_args, f"{name}.log")

    def compile_greedy_head(self):
        name = "greedy_head"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return
        deploy_args = [
            'model_deploy.py', f'--mlir {name}.mlir', f'--chip {self.chip}',
            f'--model {name}.bmodel'
        ]
        if self.debug:
            deploy_args.append('--debug')
        self.send_command(deploy_args, f"{name}.log")

    def compile_penalty_head(self):
        name = "penalty_sample_head"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return
        deploy_args = [
            'model_deploy.py', f'--mlir {name}.mlir', f'--chip {self.chip}',
            f'--model {name}.bmodel'
        ]
        if self.debug:
            deploy_args.append('--debug')
        self.send_command(deploy_args, f"{name}.log")

    def compile_block(self, layer_id):
        name = f"block_{layer_id}"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return

        deploy_args = [
            'model_deploy.py', f'--mlir {name}.mlir', f'--quantize {self.quantize}',
            f'--q_group_size {self.q_group_size}', '--quant_input', '--quant_output',
            f'--chip {self.chip}', f'--num_core {self.num_core}', f'--num_device {self.num_device}',
            f'--model {name}.bmodel'
        ]
        if self.high_precision:
            deploy_args.append('--high_precision')
        if self.symmetric:
            deploy_args.append('--q_symmetric')
        if self.debug:
            deploy_args.append('--debug')
        self.send_command(deploy_args, f"{name}.log")

    def compile_block_cache(self, layer_id):
        name = f"block_cache_{layer_id}"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return

        deploy_args = [
            'model_deploy.py', f'--mlir {name}.mlir', f'--quantize {self.quantize}',
            f'--q_group_size {self.q_group_size}', '--quant_input', '--quant_output',
            f'--chip {self.chip}', '--addr_mode io_alone', f'--num_core {self.num_core}',
            f'--num_device {self.num_device}', f'--model {name}.bmodel'
        ]
        if self.high_precision:
            deploy_args.append('--high_precision')
        if self.symmetric:
            deploy_args.append('--q_symmetric')
        if self.debug:
            deploy_args.append('--debug')
        self.send_command(deploy_args, f"{name}.log")

    def combine(self):
        bmodel_list = []
        total_bytes = 0
        for i in range(self.num_layers):
            bmodel_list = bmodel_list + [f"block_{i}.bmodel", f"block_cache_{i}.bmodel"]
            total_bytes += os.path.getsize("block_0.bmodel")
        if not self.embedding_disk:
            bmodel_list += ['embedding.bmodel', 'embedding_cache.bmodel']
            total_bytes += os.path.getsize("embedding.bmodel")
        if not self.lmhead_with_topk:
            bmodel_list += ["greedy_head.bmodel", "penalty_sample_head.bmodel"]
        bmodel_list += ["lm_head.bmodel"]
        total_bytes += os.path.getsize("lm_head.bmodel")

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

        if not self.embedding_disk:
            self.compile_embedding()
            self.compile_embedding_cache()

        self.compile_lm_head()

        if not self.lmhead_with_topk:
            self.compile_greedy_head()
            self.compile_penalty_head()

        for i in range(self.num_layers):
            self.compile_block(i)
            self.compile_block_cache(i)

        self.execute_commands()

        # Combine all bmodel files
        self.combine()

        # Remove any .npz files
        if not self.debug:
            for npz_file in os.listdir():
                if os.path.splitext(npz_file)[-1] == '.npz':
                    os.remove(npz_file)
