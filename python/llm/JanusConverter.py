# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .LlmConverter import *
from typing_extensions import override
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Final,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)
from transformers import AutoConfig, LlamaConfig


@dataclass
class SigLIPVisionCfg:
    width: int = 1152
    layers: Union[Tuple[int, int, int, int], int] = 27
    heads: int = 16
    patch_size: int = 14
    image_size: Union[Tuple[int, int], int] = 336
    global_pool: str = "map"
    mlp_ratio: float = 3.7362
    class_token: bool = False
    num_classes: int = 0
    use_checkpoint: bool = False


SigLIP_MODEL_CONFIG = {
    "siglip_so400m_patch14_384": {
        "image_size": 336,
        "patch_size": 14,
        "width": 1152,
        "layers": 27,
        "heads": 16,
        "mlp_ratio": 3.7362,
        "global_pool": "map",
        "use_checkpoint": False,
    },
    "siglip_so400m_patch14_224": {
        "image_size": 224,
        "patch_size": 14,
        "width": 1152,
        "layers": 27,
        "heads": 16,
        "mlp_ratio": 3.7362,
        "global_pool": "map",
        "use_checkpoint": False,
    },
    "siglip_large_patch16_384": {
        "image_size": 384,
        "patch_size": 16,
        "width": 1024,
        "layers": 24,
        "heads": 16,
        "mlp_ratio": 4,
        "global_pool": "map",
        "use_checkpoint": False,
    },
}


class JanusConverter(LlmConverter):

    def __init__(self, args, config):
        super().__init__(args, config)

        self.do_vit = True
        self.debug = False
        # vision config
        self.init_vconfig()
        self.vit_path = "vision_model.vision_tower"

    def init_vconfig(self):
        self.vconfig = self.config.vision_config
        self.model_name = self.vconfig.params['model_name']
        assert (
            self.model_name
            in SigLIP_MODEL_CONFIG.keys()), f"model name should be in {SigLIP_MODEL_CONFIG.keys()}"
        self.siglip_cfg = SigLIPVisionCfg(**SigLIP_MODEL_CONFIG[self.model_name])
        self.patch_size = self.siglip_cfg.patch_size
        self.image_size = self.siglip_cfg.image_size
        self.depth = self.siglip_cfg.layers
        self.num_patches = (self.image_size // self.patch_size)**2
        self.patch_dim = self.siglip_cfg.patch_size
        self.embed_dim = self.siglip_cfg.width
        self.vnum_heads = self.siglip_cfg.heads
        self.vhead_dim = self.embed_dim // self.vnum_heads
        self.vintermediate_size = self.embed_dim * self.siglip_cfg.mlp_ratio
        self.llm_hidden_size = self.config.language_config['hidden_size']

    @override
    def load_pretrained(self, config):
        super().load_pretrained(config)
        self.model_info = JANUS_INFO
        self.llm_config = LlamaConfig.from_dict(self.config.language_config)

    def vision_block(self, vit_mlir, id: int, in_op, mask_op):
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
            hidden_shape = [1, self.num_patches, self.embed_dim]
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

            fa_op = top.FAttentionOp(T(qk_shape),
                                     q_op,
                                     k_op,
                                     v_op,
                                     mask_op,
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
                                 hidden_shape,
                                 force_bias=True)
            out_op = top.AddOp(T(hidden_shape), [in_op, out_op], loc=L(attn_proj + ".add"),
                               ip=ip).output
            return out_op

        def vision_mlp(in_op):
            in_shape = [1, self.num_patches, self.embed_dim]
            mlp_fc1 = f"{self.vit_path}.blocks.{id}.mlp.fc1"
            mlp_fc2 = f"{self.vit_path}.blocks.{id}.mlp.fc2"

            new_op = self.layer_norm(vit_mlir, in_op, norm2, eps=1e-6)

            gate_op = self.linear(vit_mlir, mlp_fc1, new_op,
                                  [self.embed_dim, self.vintermediate_size],
                                  [1, self.num_patches, self.vintermediate_size])
            act_op = self.activate(vit_mlir, gate_op, ActType.GELU, mlp_fc1)
            up_op = self.linear(vit_mlir, mlp_fc2, act_op,
                                [self.vintermediate_size, self.embed_dim],
                                [1, self.num_patches, self.embed_dim])

            new_op = top.AddOp(T(in_shape), [in_op, up_op], loc=L(mlp_fc2 + ".add"), ip=ip).output
            return new_op

        in_op = vision_attention(in_op)
        in_op = vision_mlp(in_op)
        return in_op

    @override
    def gen_vit_mlir(self):
        tqdm.write(f"generate vit mlir ...")
        # create weights file
        vit_npz = "vit_top_weights.npz"
        pos_embed = f"{self.vit_path}.pos_embed"
        patch_embed = f"{self.vit_path}.patch_embed.proj"
        norm = f"{self.vit_path}.norm"

        aligner_0 = f"aligner.layers.0"
        aligner_2 = f"aligner.layers.2"

        def save_weights():
            weights_dict = {}
            data = self.model.read(pos_embed).reshape(1, self.num_patches, self.embed_dim)
            weights_dict[pos_embed] = data

            self.set_common_weight(patch_embed, weights_dict)
            self.set_linear_weight(aligner_0, weights_dict)
            self.set_linear_weight(aligner_2, weights_dict)
            self.set_common_weight(f"{self.vit_path}.norm", weights_dict)
            for i in range(self.depth):
                self.set_common_weight(f"{self.vit_path}.blocks.{i}.norm1", weights_dict)
                self.set_common_weight(f"{self.vit_path}.blocks.{i}.norm2", weights_dict)
                self.set_linear_weight(f"{self.vit_path}.blocks.{i}.attn.proj", weights_dict)
                self.set_linear_weight(f"{self.vit_path}.blocks.{i}.mlp.fc1", weights_dict)
                self.set_linear_weight(f"{self.vit_path}.blocks.{i}.mlp.fc2", weights_dict)
                # split qkv
                # self.set_linear_weight(f"{self.vit_path}.blocks.{i}.attn.qkv", weights_dict)
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
        in_shape = [1, 3, self.vconfig.params['image_size'], self.vconfig.params['image_size']]
        out_shape = [1, self.num_patches, self.llm_hidden_size]
        input_shapes = [in_shape]
        input_types = ['F32']

        vit_mlir = MLIRImporter(input_shapes, [out_shape],
                                "vit",
                                Platform.LLM,
                                input_types,
                                weight_file=vit_npz)
        ip = vit_mlir.insert_point

        def T(shape: list):
            return vit_mlir.get_tensor_type(shape)

        def L(name: str):
            return self.get_loc(name, vit_mlir)

        in0_op = vit_mlir.create_input_op(L('pixel_values'), 0)

        patch_weight = vit_mlir.create_weight_op(
            patch_embed + ".weight", [self.embed_dim, 3, self.patch_dim, self.patch_dim])
        patch_bias = vit_mlir.create_weight_op(patch_embed + ".bias", [self.embed_dim])
        patch_conv_op = top.ConvOp(T([
            1, self.embed_dim, self.image_size // self.patch_size,
            self.image_size // self.patch_size
        ]),
                                   in0_op,
                                   patch_weight,
                                   patch_bias,
                                   kernel_shape=[self.patch_size, self.patch_size],
                                   strides=[self.patch_size, self.patch_size],
                                   pads=[0, 0, 0, 0],
                                   dilations=[1, 1],
                                   loc=L(patch_embed),
                                   ip=ip).output
        patch_reshape_op = top.ReshapeOp(T([1, self.embed_dim, self.num_patches]),
                                         patch_conv_op,
                                         loc=L(patch_embed + ".reshape"),
                                         ip=ip).output
        patch_perm_op = top.PermuteOp(T([1, self.num_patches, self.embed_dim]),
                                      patch_reshape_op,
                                      order=[0, 2, 1],
                                      loc=L(patch_embed + "_permute"),
                                      ip=ip).output
        pos_weight = vit_mlir.create_weight_op(pos_embed, [1, self.num_patches, self.embed_dim])
        new_op = top.AddOp(T([1, self.num_patches, self.embed_dim]), [patch_perm_op, pos_weight],
                           loc=L(pos_embed + ".add"),
                           ip=ip).output
        # new_op = top.SqueezeOp(T([self.num_patches, self.embed_dim]),
        #                       new_op,
        #                       axes=[0],
        #                       loc=L(pos_embed + "add_squeeze"),
        #                       ip=ip).output

        for id in range(self.depth):
            new_op = self.vision_block(vit_mlir, id, new_op, vit_mlir.none_op)

        # merge
        post_norm_op = self.layer_norm(vit_mlir, new_op, norm, eps=1e-6)
        post_aligner_op0 = self.linear(vit_mlir, aligner_0, post_norm_op,
                                       [self.embed_dim, self.llm_hidden_size],
                                       [1, self.num_patches, self.llm_hidden_size])
        post_gelu_op = self.activate(vit_mlir, post_aligner_op0, ActType.GELU, aligner_0)
        new_op = self.linear(vit_mlir, aligner_2, post_gelu_op,
                             [self.llm_hidden_size, self.llm_hidden_size],
                             [1, self.num_patches, self.llm_hidden_size])

        vit_mlir.create_return_op([new_op])
        mlir_txt = vit_mlir.print_module()
        with open(f"vit.mlir", "w") as f:
            f.write(mlir_txt)
        save_weights()
