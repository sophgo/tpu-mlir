#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
"""Combine LLM bmodels into pipeline-parallel (PP) groups.

Two ways to use:

1. Group by num_device and combine in one go. Pass the bmodel directory
   produced by llm_convert; it is searched recursively for bmodels (laid
   out as {name}/{name}.bmodel). A {out_base}_combine.json config is
   generated next to the output bmodel:

       llm_combine.py --num_device 7 \
           /path/to/qwen3.5-..._w4bf16_seq147456_bm1684x_7dev_history_dynamic

   --num_device is inferred from the "_Ndev" in the directory name, or
   from an existing combine config next to the output, when omitted; -o
   defaults to the combined bmodel sitting next to the directory (falling
   back to {dir_name}.bmodel). Explicit bmodel files are also accepted
   as inputs.

2. Combine from a (possibly hand-edited) config, e.g. after moving bmodels
   between groups or renaming group outputs:

       llm_combine.py --config /path/to/model_combine.json

Config format ({out_base}_combine.json):

    {
        "num_device": 7,
        "tar": "out_pp.tar",                // {out_base}[_timestamp]_pp.tar, null skips tarring
        "bmodel_dir": "model_dir",          // common root of the input bmodels
        "groups": [
            {"output": "out_pp/embed_vit.bmodel",
             "bmodels": ["embedding/embedding.bmodel", ...]},
            {"output": "out_pp/block_00.bmodel", "bmodels": [...]},
            {"output": "out_pp/lmhead.bmodel",   "bmodels": [...]}
        ]
    }

Entries in "bmodels" are relative to "bmodel_dir"; "bmodel_dir", "tar" and
the group outputs are relative to the config file's directory. Absolute
paths also work everywhere. Each group bmodel is written to disk under its
"output" name and removed again once it is in the tar. Group outputs sit
in a folder named after the tar (unique per model/timestamp) so models
sharing one output directory do not overwrite each other's intermediates.

The tar holds that same folder, with the group bmodels under their short
names, so `tar xvf out_pp.tar` extracts to:

    out_pp/embed_vit.bmodel
    out_pp/block_00.bmodel
    out_pp/lmhead.bmodel
"""

import argparse
import datetime
import json
import math
import os
import re
import subprocess
import sys
import tarfile


def group_bmodels(bmodel_list: list, num_device: int) -> list:
    """Classify LLM bmodels and split them into num_device PP groups.

    Layout: group 0 is embedding + vit, groups 1..num_device-2 are the
    transformer blocks (split evenly by layer id), the last group is the
    remaining models (lm_head, greedy_head, sample_head, ...). Residual
    "add" bmodels go to the first block group.
    """
    embedding_vit_group = []  # embedding_xxx and vit_xxx
    block_models = {}  # layer_id -> list of bmodels (block + block_kv + block_cache)
    add_group = []  # add operations for residual connections
    remaining_group = []  # lm_head, greedy_head, sample_head, etc.

    block_pattern = re.compile(r'block(?:_cache|_kv)?_(\d+)')
    embedding_vit_pattern = re.compile(r'(embedding|vit)')
    add_pattern = re.compile(r'add')
    for bmodel in bmodel_list:
        basename = os.path.basename(bmodel)
        m = block_pattern.match(basename)
        if m:
            layer_id = int(m.group(1))
            block_models.setdefault(layer_id, []).append(bmodel)
        elif embedding_vit_pattern.search(basename):
            embedding_vit_group.append(bmodel)
        elif add_pattern.search(basename):
            add_group.append(bmodel)
        else:
            remaining_group.append(bmodel)

    # Split block layers into (num_device - 2) groups
    num_block_groups = num_device - 2
    if num_block_groups <= 0 and block_models:
        raise ValueError(
            f"num_device={num_device} leaves no device for {len(block_models)} block layers, "
            "num_device must be at least 3 for PP combining.")
    sorted_layer_ids = sorted(block_models.keys())
    total_layers = len(sorted_layer_ids)
    layers_per_group = math.ceil(total_layers /
                                 num_block_groups) if num_block_groups > 0 else total_layers

    groups = []
    # Group 0: embedding + vit
    groups.append(embedding_vit_group)
    # Groups 1 to num_device-2: block layers
    for g in range(num_block_groups):
        start = g * layers_per_group
        end = min((g + 1) * layers_per_group, total_layers)
        group_bmodels = []
        for lid in sorted_layer_ids[start:end]:
            group_bmodels.extend(block_models[lid])
        groups.append(group_bmodels)
    if add_group:
        if num_block_groups > 0:
            groups[1].extend(add_group)  # Add residual connections to the first block group
        else:
            groups[0].extend(add_group)
    # Last group: remaining (lm_head, greedy_head, sample_head, etc.)
    groups.append(remaining_group)
    return groups


def gen_group_names(out_bmodel: str, num_block_groups: int) -> list:
    """Generate short per-group bmodel names.

    cpp_demo_pp distinguishes components by substring match on the
    filename: "embed_vit" -> embedding+vit, "block" -> transformer
    blocks, "lmhead" -> LM head (and friends). The block files are then
    loaded in lexicographic order, so we zero-pad the index. The same
    short names are used on disk and inside the tar:
        group 0                  -> embed_vit{ext}
        groups 1..num_block_grps -> block_{i}{ext}
        last group               -> lmhead{ext}
    """
    _, out_ext = os.path.splitext(out_bmodel)
    pad = max(2, len(str(max(num_block_groups - 1, 0))))
    group_names = [f"embed_vit{out_ext}"]
    for n in range(num_block_groups):
        group_names.append(f"block_{n:0{pad}d}{out_ext}")
    group_names.append(f"lmhead{out_ext}")
    return group_names


def gen_combine_config(bmodel_list: list,
                       num_device: int,
                       out_bmodel: str,
                       config_dir: str = '.') -> dict:
    """Build the combine config for grouping bmodel_list by num_device.

    Group outputs are short names (embed_vit.bmodel, block_00.bmodel, ...)
    under a folder named after the tar, resolved against the config file's
    directory; the tar is stored relative to config_dir. Input bmodels are
    stored relative to their common root, factored out into a "bmodel_dir"
    field, so each entry stays a short {name}/{name}.bmodel path.
    combine_by_config resolves everything back against the config file's
    directory.
    """
    groups = group_bmodels(bmodel_list, num_device)
    num_block_groups = len(groups) - 2
    group_names = gen_group_names(out_bmodel, num_block_groups)
    assert len(group_names) == len(groups), (
        f"PP group count mismatch: {len(group_names)} names vs {len(groups)} groups")
    out_base = os.path.splitext(out_bmodel)[0]
    config_dir = os.path.abspath(config_dir)

    def rel(path: str) -> str:
        return os.path.relpath(os.path.abspath(path), config_dir)

    # Factor the common root of the input bmodels out into "bmodel_dir".
    all_bmodels = [os.path.abspath(b) for group in groups for b in group]
    bmodel_root = os.path.commonpath(all_bmodels) if all_bmodels else config_dir
    bmodel_dir = os.path.relpath(bmodel_root, config_dir)

    # The tar takes the output bmodel's base name; append a timestamp
    # (same _YYYYMMDD_HHMMSS convention as llm_convert) when out_bmodel
    # does not already carry one, or when that tar already exists (e.g.
    # re-running llm_combine with the previous timestamped bmodel as -o),
    # so repeated runs do not overwrite the previous tar.
    if not re.search(r'_\d{8}_\d{6}$', out_base):
        out_base += datetime.datetime.now().strftime('_%Y%m%d_%H%M%S')
    if os.path.exists(f"{out_base}_pp.tar"):
        out_base += datetime.datetime.now().strftime('_%Y%m%d_%H%M%S')
    # Group outputs live in a folder named after the tar (unique per
    # model/timestamp) so that intermediates of models sharing one output
    # directory do not overwrite each other; the tar reuses the folder.
    tar_folder = f"{os.path.basename(out_base)}_pp"
    config = {
        "num_device": num_device,
        "tar": rel(f"{out_base}_pp.tar"),
    }
    if bmodel_dir != '.':
        # Common root of the input bmodels, relative to the config file's
        # directory; entries in "bmodels" are relative to it.
        config["bmodel_dir"] = bmodel_dir
    config["groups"] = [{
        "output":
        os.path.join(tar_folder, name),
        "bmodels": [os.path.relpath(os.path.abspath(b), bmodel_root) for b in group],
    } for name, group in zip(group_names, groups)]
    return config


def _resolve(path: str, config_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(config_dir, path))


def combine_by_config(config: dict, config_dir: str = '.') -> list:
    """Combine each group in the config, tar the generated bmodels, then
    remove them (the tar is the shippable artifact; group bmodels are kept
    only when "tar" is null).

    Returns the list of generated group bmodel paths.
    """
    generated_bmodels = []
    # Reject duplicate group outputs up front: a later group would silently
    # overwrite an earlier one's bmodel and the cleanup below would remove
    # the same path twice.
    outputs = [group["output"] for group in config["groups"]]
    duplicates = sorted({o for o in outputs if outputs.count(o) > 1})
    if duplicates:
        raise ValueError(f"duplicate group outputs in combine config: {duplicates}")
    # Input bmodels are relative to "bmodel_dir" (itself relative to the
    # config file's directory) when present, else to the config directory.
    bmodel_dir = config.get("bmodel_dir", "")
    for i, group in enumerate(config["groups"]):
        output = _resolve(group["output"], config_dir)
        bmodels = [_resolve(os.path.join(bmodel_dir, b), config_dir) for b in group["bmodels"]]
        if not bmodels:
            print(f"PP group {i} ({os.path.basename(output)}) is empty, skipping.")
            continue
        for bmodel in bmodels:
            if not os.path.exists(bmodel):
                raise FileNotFoundError(f"bmodel not found: {bmodel}")
        os.makedirs(os.path.dirname(output), exist_ok=True)
        combine_cmd = ['model_tool', '--combine', *bmodels, '-o', output]
        print(' '.join(combine_cmd))
        try:
            subprocess.run(combine_cmd, check=True)
        except subprocess.CalledProcessError as e:
            # Mirror LlmConverter.run_command: a clean error message and
            # exit with the failed command's return code, no traceback.
            print(f"Error: command failed with return code {e.returncode}")
            sys.exit(e.returncode)
        group_size = os.path.getsize(output)
        print(f"PP group {i} bmodel size: {group_size / (1024.0 ** 3):.4f} GB, "
              f"models: {len(bmodels)}, output: {output}")
        generated_bmodels.append(output)

    # Tar all PP bmodels into a single archive so the whole pipeline-parallel
    # bundle can be shipped / loaded as one file. The tar holds a folder
    # (named after the tar file) with short bmodel names, so `tar xvf
    # out_pp.tar` extracts to `out_pp/embed_vit.bmodel`, `out_pp/block_00.bmodel`, ...
    tar_path = config.get("tar")
    if tar_path and generated_bmodels:
        tar_path = _resolve(tar_path, config_dir)
        folder = os.path.splitext(os.path.basename(tar_path))[0]
        with tarfile.open(tar_path, "w") as tar:
            for bmodel in generated_bmodels:
                tar.add(bmodel, arcname=os.path.join(folder, os.path.basename(bmodel)))
        tar_size = os.path.getsize(tar_path)
        print(f"PP tar size: {tar_size / (1024.0 ** 3):.4f} GB, "
              f"bmodels: {len(generated_bmodels)}, output: {tar_path}")
        # The intermediate group bmodels are no longer needed once they are
        # in the tar.
        for bmodel in generated_bmodels:
            os.remove(bmodel)
        print(f"removed {len(generated_bmodels)} intermediate group bmodels")
        try:  # drop the now-empty output folder, if any
            os.rmdir(os.path.join(os.path.dirname(tar_path), folder))
        except OSError:
            pass
    return generated_bmodels


def save_combine_config(config: dict, config_path: str):
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
        f.write("\n")
    print(f"combine config saved to: {config_path}")


def load_combine_config(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config, os.path.dirname(os.path.abspath(config_path))


def default_config_path(out_bmodel: str) -> str:
    """Default combine config path for an output bmodel:
    {out_base}_combine.json next to it. Named after the output (instead of
    a fixed combine.json) so models sharing one output directory do not
    overwrite each other's config."""
    out_base = os.path.splitext(os.path.basename(out_bmodel))[0]
    return os.path.join(os.path.dirname(os.path.abspath(out_bmodel)), f"{out_base}_combine.json")


def combine_by_num_device(bmodel_list: list,
                          num_device: int,
                          out_bmodel: str,
                          config_path: str = None,
                          no_tar: bool = False,
                          gen_config_only: bool = False) -> list:
    """Group bmodels by num_device, write the combine config, then combine."""
    if config_path is None:
        config_path = default_config_path(out_bmodel)
    config_dir = os.path.dirname(os.path.abspath(config_path))
    config = gen_combine_config(bmodel_list, num_device, out_bmodel, config_dir)
    if no_tar:
        config["tar"] = None
    save_combine_config(config, config_path)
    if gen_config_only:
        return []
    return combine_by_config(config, config_dir)


def find_bmodels(input_dir: str) -> list:
    """Search input_dir recursively for LLM bmodels.

    llm_convert lays bmodels out as {name}/{name}.bmodel; files following
    this convention are preferred so that already-combined outputs (e.g.
    when the parent out_dir is passed by mistake) are excluded. If no
    bmodel follows the convention (e.g. a flat directory of bmodels), all
    *.bmodel files found are used.
    """
    all_bmodels = sorted(
        os.path.join(dirpath, f) for dirpath, _, filenames in os.walk(input_dir) for f in filenames
        if f.endswith(".bmodel"))
    conventional = [
        b for b in all_bmodels
        if os.path.splitext(os.path.basename(b))[0] == os.path.basename(os.path.dirname(b))
    ]
    if conventional and len(conventional) < len(all_bmodels):
        print(f"find_bmodels: using {len(conventional)} '{{name}}/{{name}}.bmodel' files, "
              f"ignoring {len(all_bmodels) - len(conventional)} other .bmodel files")
    # All conventional bmodels must belong to a single model directory;
    # otherwise (e.g. the parent out_dir of two models was passed) blocks
    # of different models would be merged into the same PP groups.
    model_dirs = {os.path.dirname(os.path.dirname(b)) for b in conventional}
    if len(model_dirs) > 1:
        raise ValueError(
            f"find_bmodels: bmodels from {len(model_dirs)} model directories found under "
            f"{input_dir}; pass a single model's bmodel directory")
    bmodels = conventional or all_bmodels
    if not bmodels:
        raise FileNotFoundError(f"no .bmodel found under: {input_dir}")
    return bmodels


def infer_num_device(dir_name: str) -> int:
    """Infer num_device from the "_Ndev" in a bmodel directory name."""
    m = re.search(r'_(\d+)dev', dir_name)
    return int(m.group(1)) if m else 0


def read_num_device_from_config(out_bmodel: str) -> int:
    """Read num_device from a combine config sitting next to out_bmodel
    (written by an earlier llm_convert / llm_combine run). Returns 0 when
    no config is found. This covers bmodel directories whose name carries
    no "_Ndev" (e.g. non-bm1684x chips use "_Ncore")."""
    config_dir = os.path.dirname(os.path.abspath(out_bmodel))
    out_base = os.path.splitext(os.path.basename(out_bmodel))[0]
    for name in (f"{out_base}_combine.json", "combine.json"):
        path = os.path.join(config_dir, name)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return int(json.load(f).get("num_device", 0))
            except (OSError, ValueError):
                continue
    return 0


def default_output(input_dir: str) -> str:
    """Default output bmodel for a bmodel directory: the existing combined
    bmodel next to it ({dir_name}_{timestamp}.bmodel) if present, otherwise
    {dir_name}.bmodel next to the directory."""
    input_dir = os.path.normpath(input_dir)
    parent = os.path.dirname(os.path.abspath(input_dir))
    dir_name = os.path.basename(input_dir)
    pattern = re.compile(re.escape(dir_name) + r'_\d{8}_\d{6}\.bmodel$')
    candidates = sorted(f for f in os.listdir(parent) if pattern.match(f))
    if candidates:
        return os.path.join(parent, candidates[-1])
    return os.path.join(parent, f"{dir_name}.bmodel")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine LLM bmodels into pipeline-parallel groups.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    # yapf: disable
    parser.add_argument("inputs", nargs="*", type=str,
                        help="bmodel directories (searched recursively) and/or bmodel files")
    parser.add_argument("--num_device", type=int, default=0,
                        help="number of devices to distribute the bmodels over "
                        "(default: inferred from the _Ndev in the directory name)")
    parser.add_argument("-o", "--output", type=str, default="",
                        help="output bmodel path used to derive group names and the combine "
                        "config location (default: the combined bmodel next to the input "
                        "directory)")
    parser.add_argument("--config", type=str, default="",
                        help="combine from an existing combine config instead of --num_device")
    parser.add_argument("--config_out", type=str, default="",
                        help="where to write the combine config (default: {out_base}_combine.json "
                        "next to --output)")
    parser.add_argument("--gen_config_only", action="store_true",
                        help="only generate combine.json, do not combine")
    parser.add_argument("--no_tar", action="store_true",
                        help="do not tar the generated group bmodels")
    # yapf: enable
    args = parser.parse_args()

    if args.config:
        config, config_dir = load_combine_config(args.config)
        combine_by_config(config, config_dir)
    else:
        if not args.inputs:
            parser.error("no input bmodels or directory given")
        bmodel_list = []
        for inp in args.inputs:
            if os.path.isdir(inp):
                bmodel_list.extend(find_bmodels(inp))
            elif os.path.isfile(inp) and inp.endswith(".bmodel"):
                bmodel_list.append(inp)
            else:
                parser.error(f"input is neither a directory nor a .bmodel file: {inp}")
        num_device = args.num_device
        output = args.output
        if len(args.inputs) == 1 and os.path.isdir(args.inputs[0]):
            if not output:
                output = default_output(args.inputs[0])
                print(f"output not given, using: {output}")
            if num_device == 0:
                dir_name = os.path.basename(os.path.normpath(args.inputs[0]))
                num_device = infer_num_device(dir_name)
                if num_device > 0:
                    print(f"num_device inferred from directory name: {num_device}")
            if num_device == 0:
                num_device = read_num_device_from_config(output)
                if num_device > 0:
                    print(f"num_device read from combine config next to: {output}")
        if num_device < 3:
            parser.error("--num_device must be at least 3 (and could not be inferred)")
        if not output:
            parser.error("--output is required when inputs are bmodel files")
        combine_by_num_device(bmodel_list,
                              num_device,
                              output,
                              config_path=args.config_out or None,
                              no_tar=args.no_tar,
                              gen_config_only=args.gen_config_only)
