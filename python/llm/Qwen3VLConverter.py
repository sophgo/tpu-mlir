# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .LlmConverter import *
from typing_extensions import override


class Qwen3VLConverter(LlmConverter):

    def __init__(self, args, config):
        super().__init__(args, config)
        self.max_pixels = args.max_pixels
        if args.max_pixels == 0:
            raise RuntimeError("max_pixels is 0, please set max_pixels to a value greater than 0.")
        if args.max_pixels % (32 * 32) != 0:
            raise RuntimeError(
                "max_pixels is not a multiple of 32*32, please set max_pixels to a value that is a multiple of 32*32."
            )
        self.do_vit = True
        self.dynamic_vit = True  # force dynamic vit
        # vision config
        self.init_vconfig()
        self.vit_path = "model.visual"
        self.extern_gen_mlirs.append(self.gen_add_mlir)
        self.extern_compiles.append(self.compile_add_mlir)
        self.extern_bmodels.append("add.bmodel")
        self.extern_block_weights = {"mrope_interleave_idx": self.get_mrope_index()}

    def init_vconfig(self):
        self.vconfig = self.config.vision_config
        self.patch_size = self.vconfig.patch_size
        self.temporal_patch_size = self.vconfig.temporal_patch_size
        self.spatial_merge_size = self.vconfig.spatial_merge_size
        self.in_channels = self.vconfig.in_chans
        self.depth = self.vconfig.depth
        self.num_patches = self.max_pixels // (self.patch_size * self.patch_size)
        self.patch_dim = self.in_channels * self.patch_size * self.patch_size * self.temporal_patch_size
        self.embed_dim = self.vconfig.hidden_size
        self.vnum_heads = self.vconfig.num_heads
        self.vhead_dim = self.embed_dim // self.vnum_heads
        self.vintermediate_size = self.vconfig.intermediate_size
        self.position_shape = [3, self.max_input_length]
        self.deepstack_visual_indexes = self.vconfig.deepstack_visual_indexes
        self.num_position_embeddings = self.vconfig.num_position_embeddings
        self.mrope_section = getattr(self.config.rope_scaling, 'mrope_section', [24, 20, 20])

    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        self.llm_type = LlmType.QWEN3

    @override
    def rotary_embedding(self):
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLRotaryEmbedding
        rotary_embed = Qwen2VLRotaryEmbedding(self.llm_config)
        position_ids = torch.arange(self.seq_length, dtype=torch.long).reshape(
            1, 1, self.seq_length).expand(3, 1, self.seq_length)
        x = torch.zeros([1, self.seq_length, self.hidden_size], dtype=torch.float32)
        cos, sin = rotary_embed(x, position_ids)
        cos = cos[0].reshape(self.seq_length, 1, -1)
        sin = sin[0].reshape(self.seq_length, 1, -1)
        assert (cos.shape[-1] == self.head_dim)
        assert (sin.shape[-1] == self.head_dim)
        # half
        cos = cos[:, :, :self.head_dim // 2]
        sin = sin[:, :, :self.head_dim // 2]
        return cos.numpy(), sin.numpy()  #[seq, 1, 64]

    def apply_interleaved_mrope(freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHWTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def get_mrope_index(self):
        freqs = np.arange(0, 3 * self.head_dim // 2,
                          dtype=np.int32).reshape(3, 1, self.head_dim // 2)
        freqs_t = self.apply_interleaved_mrope(freqs, self.mrope_section)  # [1, 64]
        freqs_t = np.tile(freqs_t, (1, 2))  # [1, 128]
        return freqs_t

    def mrope(self, mlir_gen, in_op, name: str):
        dim = in_op.type.shape[-1]
        weight_op = mlir_gen.create_weight_op(name + ".weight",
                                              [self.seq_length, 1, self.head_dim // 2])
        in_op = top.GatherOp(mlir_gen.get_tensor_type([3, dim, 1, self.head_dim // 2]),
                             weight_op,
                             in_op,
                             axis=0,
                             loc=self.get_loc(name, mlir_gen),
                             ip=mlir_gen.insert_point).output
        new_op = top.PermuteOp(mlir_gen.get_tensor_type([3, self.head_dim // 2, 1, dim]),
                               in_op,
                               order=[0, 3, 2, 1],
                               loc=self.get_loc(name + ".permute", mlir_gen),
                               ip=mlir_gen.insert_point).output
        new_op = top.ReshapeOp(mlir_gen.get_tensor_type([3 * self.head_dim // 2, 1, dim]),
                               new_op,
                               loc=self.get_loc(name + ".reshape", mlir_gen),
                               ip=mlir_gen.insert_point).output
        weight_op = mlir_gen.create_weight_op("mrope_interleave_idx", [1, self.head_dim])
        new_op = top.GatherOp(mlir_gen.get_tensor_type([1, self.head_dim, 1, dim]),
                              new_op,
                              weight_op,
                              axis=0,
                              loc=self.get_loc(name + ".gather", mlir_gen),
                              ip=mlir_gen.insert_point).output
        new_op = top.PermuteOp(mlir_gen.get_tensor_type([1, dim, 1, self.head_dim]),
                               new_op,
                               order=[0, 3, 2, 1],
                               loc=self.get_loc(name + ".permute2", mlir_gen),
                               ip=mlir_gen.insert_point).output
        return new_op

    @override
    def apply_rotary_pos(self, mlir_gen, pos_op, q_op, k_op, rotary_cos: str, rotary_sin: str):
        # cos MROPE
        cos_op = self.mrope(mlir_gen, pos_op, rotary_cos)
        # sin MROPE
        sin_op = self.mrope(mlir_gen, pos_op, rotary_sin)
        # ===== q_proj rotary ========
        q_op = self.rotary_pos(mlir_gen, q_op, cos_op, sin_op, "q_proj")

        # ===== k_proj rotary ========
        k_op = self.rotary_pos(mlir_gen, k_op, cos_op, sin_op, "k_cache")
        return q_op, k_op

    def vision_rotary(self):
        from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionRotaryEmbedding
        head_dim = self.vconfig.hidden_size // self.vnum_heads
        rotary_embed = VisionRotaryEmbedding(head_dim // 2)
        freqs = rotary_embed(self.num_patches)
        return freqs.cos().numpy(), freqs.sin().numpy()

    def vision_block(self, vit_mlir, id: int, in_op, cos_op, sin_op):
        norm1 = f"{self.vit_path}.blocks.{id}.norm1"
        attn_q = f"{self.vit_path}.blocks.{id}.attn.q"
        attn_k = f"{self.vit_path}.blocks.{id}.attn.k"
        attn_v = f"{self.vit_path}.blocks.{id}.attn.v"
        attn_proj = f"{self.vit_path}.blocks.{id}.attn.proj"
        norm2 = f"{self.vit_path}.blocks.{id}.norm2"
        ip = vit_mlir.insert_point

        def T(shape: list):
            return vit_mlir.get_tensor_type(shape)

        def L(name: str):
            return self.get_loc(name, vit_mlir)

        def vision_attention(in_op):
            norm1_op = self.layer_norm(vit_mlir, in_op, norm1, eps=1e-6)
            hidden_shape = [self.num_patches, self.embed_dim]
            q_op = self.linear(vit_mlir,
                               attn_q,
                               norm1_op, [self.embed_dim, self.embed_dim],
                               hidden_shape,
                               force_bias=True)
            k_op = self.linear(vit_mlir,
                               attn_k,
                               norm1_op, [self.embed_dim, self.embed_dim],
                               hidden_shape,
                               force_bias=True)
            v_op = self.linear(vit_mlir,
                               attn_v,
                               norm1_op, [self.embed_dim, self.embed_dim],
                               hidden_shape,
                               force_bias=True)
            qk_shape = [1, self.num_patches, self.vnum_heads, self.vhead_dim]
            q_op = top.ReshapeOp(T(qk_shape), q_op, loc=L(attn_q + ".reshape"), ip=ip).output

            k_op = top.ReshapeOp(T(qk_shape), k_op, loc=L(attn_k + ".reshape"), ip=ip).output
            v_op = top.ReshapeOp(T(qk_shape), v_op, loc=L(attn_v + ".reshape"), ip=ip).output
            q_op = self.rotary_pos(vit_mlir, q_op, cos_op, sin_op, attn_q + ".rotary")
            k_op = self.rotary_pos(vit_mlir, k_op, cos_op, sin_op, attn_k + ".rotary")
            fa_op = top.FAttentionOp(T(qk_shape),
                                     q_op,
                                     k_op,
                                     v_op,
                                     vit_mlir.none_op,
                                     vit_mlir.none_op,
                                     scale=self.vhead_dim**-0.5,
                                     batch=1,
                                     q_head=self.vnum_heads,
                                     kv_head=self.vnum_heads,
                                     dim=self.vhead_dim,
                                     mq=self.num_patches,
                                     mk=self.num_patches,
                                     loc=L(f"{self.vit_path}.blocks.{id}.fattention"),
                                     ip=ip).output
            fa_op = top.ReshapeOp(T(hidden_shape),
                                  fa_op,
                                  loc=L(f"{self.vit_path}.blocks.{id}.fattention.reshape"),
                                  ip=ip).output
            out_op = self.linear(vit_mlir,
                                 attn_proj,
                                 fa_op, [self.embed_dim, self.embed_dim],
                                 [self.num_patches, self.embed_dim],
                                 force_bias=True)
            out_op = top.AddOp(T(hidden_shape), [in_op, out_op], loc=L(attn_proj + ".add"),
                               ip=ip).output
            return out_op

        def vision_mlp(in_op):
            in_shape = [self.num_patches, self.embed_dim]
            linear_fc1 = f"{self.vit_path}.blocks.{id}.mlp.linear_fc1"
            linear_fc2 = f"{self.vit_path}.blocks.{id}.mlp.linear_fc2"

            new_op = self.layer_norm(vit_mlir, in_op, norm2, eps=1e-6)

            fc1_op = self.linear(vit_mlir, linear_fc1, new_op,
                                 [self.embed_dim, self.vintermediate_size],
                                 [self.num_patches, self.vintermediate_size])
            act_op = self.activate(vit_mlir, fc1_op, self.vconfig.hidden_act, linear_fc1)
            fc2_op = self.linear(vit_mlir, linear_fc2, act_op,
                                 [self.vintermediate_size, self.embed_dim],
                                 [self.num_patches, self.embed_dim])
            new_op = top.AddOp(T(in_shape), [in_op, fc2_op], loc=L(linear_fc2 + ".add"),
                               ip=ip).output
            return new_op

        in_op = vision_attention(in_op)
        in_op = vision_mlp(in_op)
        return in_op

    @override
    def gen_vit_mlir(self):
        tqdm.write(f"generate vit mlir ...")
        # create weights file
        vit_npz = "vit_top_weights.npz"
        patch_embed = f"{self.vit_path}.patch_embed.proj"
        pos_embed = f"{self.vit_path}.pos_embed"
        rotary_cos = f"{self.vit_path}.rotary.cos"
        rotary_sin = f"{self.vit_path}.rotary.sin"
        merger_norm = f"{self.vit_path}.merger.norm"
        linear_fc1 = f"{self.vit_path}.merger.linear_fc1"
        linear_fc2 = f"{self.vit_path}.merger.linear_fc2"
        deepstack_norm_list = []
        deepstack_fc1_list = []
        deepstack_fc2_list = []
        for idx in range(len(self.deepstack_visual_indexes)):
            deepstack_norm_list.append(f"{self.vit_path}.deepstack_merger_list.{idx}.norm")
            deepstack_fc1_list.append(f"{self.vit_path}.deepstack_merger_list.{idx}.linear_fc1")
            deepstack_fc2_list.append(f"{self.vit_path}.deepstack_merger_list.{idx}.linear_fc2")

        def save_weights():
            cos, sin = self.vision_rotary()
            weights_dict = {
                rotary_cos + ".weight": cos,
                rotary_sin + ".weight": sin,
            }
            data = self.model.read(patch_embed + ".weight").reshape(self.embed_dim, self.patch_dim)
            data = np.ascontiguousarray(np.transpose(data, (1, 0)))
            weights_dict[patch_embed + ".weight"] = data
            weights_dict[patch_embed + ".bias"] = self.model.read(patch_embed + ".bias")
            self.set_common_weight(pos_embed, weights_dict)  # fast_pos_embed_interpolate
            # merger
            self.set_common_weight(merger_norm, weights_dict)
            self.set_linear_weight(linear_fc1, weights_dict)
            self.set_linear_weight(linear_fc2, weights_dict)
            for idx in range(len(self.deepstack_visual_indexes)):
                self.set_linear_weight(deepstack_fc1_list[idx], weights_dict)
                self.set_linear_weight(deepstack_fc2_list[idx], weights_dict)
            for i in range(self.depth):
                self.set_common_weight(f"{self.vit_path}.blocks.{i}.norm1", weights_dict)
                self.set_common_weight(f"{self.vit_path}.blocks.{i}.norm2", weights_dict)
                self.set_linear_weight(f"{self.vit_path}.blocks.{i}.attn.proj", weights_dict)
                self.set_linear_weight(f"{self.vit_path}.blocks.{i}.mlp.linear_fc1", weights_dict)
                self.set_linear_weight(f"{self.vit_path}.blocks.{i}.mlp.linear_fc2", weights_dict)

                # split qkv
                # self.set_linear_weight(f"visual.blocks.{i}.attn.qkv", weights_dict)
                weight = self.model.read(f"{self.vit_path}.blocks.{i}.attn.qkv.weight").reshape(
                    3 * self.embed_dim, self.embed_dim)
                bias = self.model.read(f"{self.vit_path}.blocks.{i}.attn.qkv.bias").reshape(
                    3 * self.embed_dim)
                q_w = weight[:self.embed_dim, :]
                k_w = weight[self.embed_dim:2 * self.embed_dim, :]
                v_w = weight[2 * self.embed_dim:, :]
                q_b = bias[:self.embed_dim]
                k_b = bias[self.embed_dim:2 * self.embed_dim]
                v_b = bias[2 * self.embed_dim:]
                q_w = np.ascontiguousarray(np.transpose(q_w, (1, 0)))
                k_w = np.ascontiguousarray(np.transpose(k_w, (1, 0)))
                v_w = np.ascontiguousarray(np.transpose(v_w, (1, 0)))
                weights_dict[f"{self.vit_path}.blocks.{i}.attn.q.weight"] = q_w
                weights_dict[f"{self.vit_path}.blocks.{i}.attn.k.weight"] = k_w
                weights_dict[f"{self.vit_path}.blocks.{i}.attn.v.weight"] = v_w
                weights_dict[f"{self.vit_path}.blocks.{i}.attn.q.bias"] = q_b
                weights_dict[f"{self.vit_path}.blocks.{i}.attn.k.bias"] = k_b
                weights_dict[f"{self.vit_path}.blocks.{i}.attn.v.bias"] = v_b
            # save weights
            np.savez(vit_npz, **weights_dict)

        # create mlir file
        in_shape = [self.num_patches, self.patch_dim]
        position_shape = [self.num_patches, 2]
        out_dim = self.num_patches // (self.spatial_merge_size**2)
        out_shape = [out_dim, self.hidden_size]
        input_shapes = [in_shape, position_shape, [4, self.num_patches], [4, self.num_patches, 1]]
        input_types = ['F32', 'INT32', 'INT32', 'F32']
        out_num = 1 + len(self.deepstack_visual_indexes)

        vit_mlir = MLIRImporter(input_shapes, [out_shape] * out_num,
                                "vit",
                                Platform.LLM,
                                input_types,
                                weight_file=vit_npz)
        ip = vit_mlir.insert_point

        def T(shape: list):
            return vit_mlir.get_tensor_type(shape)

        def L(name: str):
            return self.get_loc(name, vit_mlir)

        in0_op = vit_mlir.create_input_op(L('input_states'), 0)
        in1_op = vit_mlir.create_input_op(L('position_ids'), 1)
        in2_op = vit_mlir.create_input_op(L('pos_idx'),
                                          2)  # by fast_pos_embed_interpolate, need to be reordered
        in3_op = vit_mlir.create_input_op(L('pos_weight'),
                                          3)  # by fast_pos_embed_interpolate, need to be reordered
        new_weight = vit_mlir.create_weight_op(patch_embed + ".weight",
                                               [self.patch_dim, self.embed_dim])
        new_bias = vit_mlir.create_weight_op(patch_embed + ".bias", [self.embed_dim])
        new_op = top.MatMulOp(T([self.num_patches, self.embed_dim]),
                              in0_op,
                              new_weight,
                              new_bias,
                              loc=L(patch_embed),
                              ip=ip).output
        # fast_pos_embed_interpolate
        new_weight = vit_mlir.create_weight_op(pos_embed + ".weight",
                                               [self.num_position_embeddings, self.embed_dim])
        pos_idx_gather = top.GatherOp(T([4, self.num_patches, self.embed_dim]),
                                      new_weight,
                                      in2_op,
                                      axis=0,
                                      loc=L(pos_embed),
                                      ip=ip).output
        pos_mul = top.MulOp(T([4, self.num_patches, self.embed_dim]), [pos_idx_gather, in3_op],
                            loc=L(pos_embed + ".mul"),
                            ip=ip).output
        pos_sum = top.ReduceOp(T([self.num_patches, self.embed_dim]),
                               pos_mul,
                               axes=[0],
                               keepdims=0,
                               mode=StringAttr.get("ReduceSum"),
                               loc=L(pos_embed + ".sum"),
                               ip=ip).output
        new_op = top.AddOp(T([self.num_patches, self.embed_dim]), [new_op, pos_sum],
                           loc=L(pos_embed + ".add"),
                           ip=ip).output
        # rotary embedding
        new_weight = vit_mlir.create_weight_op(rotary_cos + ".weight",
                                               [self.num_patches, self.vhead_dim // 4])
        cos_op = top.GatherOp(T([self.num_patches, 2, self.vhead_dim // 4]),
                              new_weight,
                              in1_op,
                              axis=0,
                              loc=L(rotary_cos),
                              ip=ip).output
        cos_op = top.ReshapeOp(T([1, self.num_patches, 1, self.vhead_dim // 2]),
                               cos_op,
                               loc=L(rotary_cos + ".reshape"),
                               ip=ip).output
        cos_op = top.TileOp(T([1, self.num_patches, 1, self.vhead_dim]),
                            cos_op,
                            tile=[1, 1, 1, 2],
                            loc=L(rotary_cos + ".tile"),
                            ip=ip).output
        new_weight = vit_mlir.create_weight_op(rotary_sin + ".weight",
                                               [self.num_patches, self.vhead_dim // 4])
        sin_op = top.GatherOp(T([self.num_patches, 2, self.vhead_dim // 4]),
                              new_weight,
                              in1_op,
                              axis=0,
                              loc=L(rotary_sin),
                              ip=ip).output
        sin_op = top.ReshapeOp(T([1, self.num_patches, 1, self.vhead_dim // 2]),
                               sin_op,
                               loc=L(rotary_sin + ".reshape"),
                               ip=ip).output
        sin_op = top.TileOp(T([1, self.num_patches, 1, self.vhead_dim]),
                            sin_op,
                            tile=[1, 1, 1, 2],
                            loc=L(rotary_sin + ".tile"),
                            ip=ip).output
        deepstack_list = []
        for id in range(self.depth):
            new_op = self.vision_block(vit_mlir, id, new_op, cos_op, sin_op)
            if id in self.deepstack_visual_indexes:
                deepstack_list.append(new_op)

        def patch_merger(in_op, fc1_path: str, fc2_path: str, norm_path: str):
            in_op = self.layer_norm(vit_mlir, in_op, norm_path, eps=1e-6)
            out_dim = self.embed_dim * (self.spatial_merge_size**2)
            in_dim = self.num_patches // (self.spatial_merge_size**2)
            new_op = top.ReshapeOp(T([in_dim, out_dim]),
                                   new_op,
                                   loc=L(fc1_path + ".preshape"),
                                   ip=ip).output
            new_op = self.linear(vit_mlir, fc1_path, new_op, [out_dim, out_dim], [in_dim, out_dim])
            new_op = self.activate(vit_mlir, new_op, ActType.GELU, fc1_path)
            new_op = self.linear(vit_mlir, fc2_path, new_op, [out_dim, self.hidden_size],
                                 [in_dim, self.hidden_size])

        ret_ops = []
        new_op = patch_merger(new_op, linear_fc1, linear_fc2, merger_norm)
        ret_ops.append(new_op)
        for idx in range(len(deepstack_list)):
            deep_op = patch_merger(deepstack_list[idx], deepstack_fc1_list[idx],
                                   deepstack_fc2_list[idx], deepstack_norm_list[idx])
            ret_ops.append(deep_op)

        vit_mlir.create_return_op(ret_ops)
        mlir_txt = vit_mlir.print_module()
        with open(f"vit.mlir", "w") as f:
            f.write(mlir_txt)
        save_weights()

    def gen_add_mlir(self):
        input_shape = [self.seq_length * self.hidden_size]
        add_mlir = MLIRImporter([input_shape] * 2, [input_shape], "add", Platform.LLM,
                                ['F32', 'F32'])
        ip = add_mlir.insert_point
        in0_op = add_mlir.create_input_op(self.get_loc('input0', add_mlir), 0)
        in1_op = add_mlir.create_input_op(self.get_loc('input1', add_mlir), 1)
        out_op = top.AddOp(add_mlir.get_tensor_type(input_shape), [in0_op, in1_op],
                           loc=self.get_loc("add", add_mlir),
                           ip=ip).output
        add_mlir.create_return_op([out_op])
        mlir_txt = add_mlir.print_module()
        with open(f"add.mlir", "w") as f:
            f.write(mlir_txt)

    def compile_add_mlir(self):
        name = "add"
        if os.path.exists(f"{name}.bmodel"):
            print(f"{name}.bmodel already exists. Skipping compilation.")
            return
        deploy_args = [
            'model_deploy.py', f'--mlir {name}.mlir', f'--chip {self.chip}',
            f'--num_core {self.num_core}', f'--num_device {self.num_device}', f'--dynamic',
            f'--quant_input', f'--quant_output', f'--model {name}.bmodel'
        ]
        deploy_args.append(f'--quantize {self.half_precision_quantize}')
        self.add_task(deploy_args, f"{name}.log")
