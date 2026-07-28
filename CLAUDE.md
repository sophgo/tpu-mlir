# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TPU-MLIR is an MLIR-based compiler that converts ONNX / PyTorch / TFLite / Caffe / HuggingFace models into `bmodel` binaries for SOPHGO TPUs (`bm1684x`, `bm1688`, `bm1690`, `cv186ah`, `cv180x`, `cv181x`, `mars3`, `sgtpuv8`, …). All build/test commands assume you are inside the `sophgo/tpuc_dev:latest` Docker container (Ubuntu 22.04, Python ≥ 3.10).

## Environment & build

Always source the env first — `build.sh` aborts unless `INSTALL_PATH` is set and `ENVSETUP_LAST_UPDATED` matches the date pinned in `envsetup.sh`.

```bash
pip install -r requirements.txt
source ./envsetup.sh        # exports PROJECT_ROOT, BUILD_PATH, INSTALL_PATH, REGRESSION_PATH, PYTHONPATH, PATH; installs git hooks
./build.sh                  # RELEASE (default) — also runs release_doc.sh and strips binaries
./build.sh DEBUG            # debug build with -ggdb, no doc/strip — recommended for development
./build.sh DEBUG CUDA       # enable -DTPUMLIR_USE_CUDA=ON
```

`envsetup.sh` puts `python/tools`, `python/utils`, `python/test`, `python/samples` on `PATH`, so scripts like `model_transform.py`, `model_deploy.py`, `model_runner.py`, `run_calibration.py`, `llm_convert.py`, `test_onnx.py` are run by bare name.

## Tests

There are three layers of tests; there is no single "run all" command for routine work.

1. **Python op / model regression** via `regression/main_entry.py`:
   ```bash
   regression/run.sh op       # torch op set + custom layer rebuild
   regression/run.sh model    # onnx op set
   regression/run.sh script   # check-tpumlir + script + model basic set
   regression/main_entry.py --test_type basic --test_set onnx torch script model
   ```
2. **Single op / model case** — run the underlying `test_*.py` directly (they are on `PATH`):
   ```bash
   test_onnx.py    --case Conv2d     --chip bm1684x
   test_torch.py   --case LayerNorm  --chip bm1688
   test_tflite.py  --case <Case>     --chip bm1684x
   test_tpulang.py --case <Case>     --chip bm1684x
   python regression/run_model.py <model_name> --chip bm1684x --mode f16   # end-to-end, configured by regression/config/
   ```
   Logs land in `regression/regression_out/`. `--simple` skips heavy checks.

**Do all test/run commands in `./tmp`, not in the source tree** — this avoids polluting the source tree with generated files.

## Lint / format

Pre-commit hooks installed by `envsetup.sh` enforce these (run manually before pushing):

- C/C++ in `lib/`, `include/`, `tools/`: `clang-format -i` (config `.clang-format`, LLVM style).
- Python in `python/`, `regression/`: `yapf -i` (config `.style.yapf`, **100-column limit**, 4-space indent).
- Comments must be English (checked by `hooks/check_comment_language.py`).

## Architecture

The compiler is an MLIR pipeline with two principal dialects:

```
front-end importer (python/transform/) ──► Top dialect ──lowering──► Tpu dialect ──► codegen ──► bmodel
                                          (framework-          (chip-specific ops, layer-group
                                           neutral graph)       memory planning, quantization)
```

- **Dialects** live in `include/tpu_mlir/Dialect/{Top,Tpu}/{IR,Transforms}` with implementations in `lib/Dialect/{Top,Tpu}`. Ops are defined in TableGen (`*.td`); regenerated headers go to `$BUILD_PATH`.
- **Conversions** between dialects: `lib/Conversion/{TopToTpu,TopToTosa,TopToLinalg}`. `TopToTpu` is per-chip (subdirectories for BM1684X, BM1688, etc.).
- **Backends** (`lib/Backend`) wrap chip backend libraries (`BM168x`, `CV18xx`); `lib/PplBackend` is the PPL kernel backend, built separately by `lib/PplBackend/build.sh`.
- **Driver tools** (`tools/`): `tpuc-opt` (the MLIR opt tool with all TPU passes), `tpuc-tool`, `model_tool` (bmodel inspector), `chiprunner`.
- **Python front-end** (`python/transform/`) imports framework graphs and emits Top-dialect MLIR via the C-API in `capi/` / `bindings/`. `python/tools/model_transform.py` and `model_deploy.py` are the user-facing entry points; `llm_convert.py` is the one-shot LLM pipeline.
- **Calibration / quantization**: `python/calibration/` (PTQ, AutoTune, search), `python/tools/run_calibration.py`. INT8 deploy requires a calibration table.
- **Layer-group / memory planning** is a major Tpu-dialect transform — see `lib/Dialect/Tpu/Transforms/LayerGroup`.
- **Custom ops**: `third_party/customlayer` is sourced separately (`source $PROJECT_ROOT/third_party/customlayer/envsetup.sh`); `regression/run.sh op` rebuilds plugin/backend/firmware before testing.

## Conventions

- Always import via the `tpu_mlir` namespace in C++; new passes register in `lib/InitAll.cpp` and the corresponding `Passes.td`.
- Per-chip code paths key off the `processor`/`chip` argument. The canonical chip list and support matrix live in `regression/chip.py` — mirror it when adding chip switches.
- Quantize modes are spelled `F32 / BF16 / F16 / INT8` (uppercase) in user-facing flags but `f32/bf16/f16/int8` in regression configs — follow the surrounding file.
- `model_deploy.py` uses `--processor`, not `--chip`; `test_*.py` and most internal scripts use `--chip`. Don't conflate them.
- Tolerances in deploy/test are `<cos>,<euclid>` pairs (e.g. `0.99,0.90`); INT8 typically needs looser values like `0.85,0.45`.
- New ops require: a `.td` entry in the relevant dialect, a shape-inference + lowering pattern, a Python importer hook in `python/transform/`, and a regression case in `python/test/test_onnx.py` (or the matching framework file).
- Commit messages: short imperative summary; sign commits with a GitHub-registered email (CONTRIBUTING.md). One logical change per PR; CI must pass.
- Do **not** edit anything under `third_party/`, `install/`, `build/`, `dist/`, or `tmp/` casually — those are vendored/submoduled or build artefacts.

## Compiling LLMs

LLM compilation falls into two scenarios, controlled by `--use_history_kv` (using Qwen3.5 as an example):

**1. Without history KV** — for single-turn conversations with short context (e.g. within 4K):

```bash
llm_convert.py -m Qwen3.5-2B-int4-AutoRound -c bm1688 -s 2048 --max_input_length 1024 --out_dir qwen3_5_bm1688
```

- Compiles two instruction groups: `block_` (prefill) and `block_cache_` (decode).
- `-s` sets the max total sequence length; `--max_input_length` sets the max input length.

**2. With history KV** — for multi-turn conversations, long contexts (e.g. 8K), or when unsure. More flexible with good overall performance, so prefer this mode in those cases:

```bash
llm_convert.py -m Qwen3.5-2B-int4-AutoRound -c bm1688 -s 8192 --use_history_kv --chunk_length 1024 --out_dir qwen3_5_bm1688
```

- Compiles three instruction groups: `block_` (prefill), `block_kv_` (prefill with history), and `block_cache_` (decode).
- `--chunk_length` sets the segment length for chunked inference. For example, with 1K chunks, a 7K input runs prefill as `block_` + 7 × `block_kv_`; decode is also segmented by KV-cache length, so performance differs at 1K / 2K / 4K / 8K lengths.

**Other functional flags:**

- `--dynamic` — dynamic-shape compilation; recommended to always pass it. Static compilation is still the default for backward compatibility (Qwen3.5 forces dynamic); this flag may be removed once all models go dynamic.
- `--do_sample` — enable random sampling.
- `--max_pixels` — image size for VLMs; leave unset to use the internal defaults.
- `--embedding_disk` — store word embeddings in a bin file and run them on CPU.
- `--lora_max_rank` — max LoRA rank; setting it compiles a LoRA-enabled version (LoRA for Qwen3.5 is not yet tuned).

The thinking behind the compiler-based approach for LLMs is summarized in this paper: <https://arxiv.org/pdf/2607.15865>

## LLM demo usage

The LLM demo accepts slash commands (e.g. `/exit`, `/clear`) and uses `@` to attach files:

- Image: `what is the image about? @./test.jpg`
- Text file (`.txt` / `.md`): `what is it talking about? @./story.txt`

## Working style

- **English refinement:** Users are mostly non-native English speakers. When the user's input or a description contains awkward or incorrect English, render the corresponding output (reports, docs, commit messages) in clear, natural English rather than mirroring the broken phrasing. If the user's English is already correct, preserve it as-is.
- **No auto-commit:** When making code fixes, do not `git commit` them directly. Leave the changes in the working tree for the user to review and commit themselves.
- **Preserve file ownership:** Do not change file ownership. Edits made through the Edit/Write tools run as root and silently change the edited file's owner to `root` — after editing, copying, moving, or regenerating any file, restore its original owner (repo files are uid/gid 1018; verify against untouched neighbors with `ls -l`), e.g. `chown 1018:1018 <files>`.
- **Remember in CLAUDE.md:** When the user asks to remember something (a rule, preference, or lesson learned), always record it in this `CLAUDE.md` so it persists in the repo for every session — not in private/session-only memory.
