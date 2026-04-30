# Copilot Instructions for TPU-MLIR

TPU-MLIR is an MLIR-based compiler that turns ONNX/PyTorch/TFLite/Caffe/HuggingFace models into `bmodel` binaries for SOPHGO TPUs (bm1684x, bm1688, bm1690, cv186ah, etc.).

## Environment

All build/test commands assume you are inside the `sophgo/tpuc_dev:latest` Docker container (Ubuntu 22.04, Python ≥ 3.10). Always source the env first — `build.sh` aborts unless `INSTALL_PATH` is set and `ENVSETUP_LAST_UPDATED` matches the date pinned in `envsetup.sh`.

```bash
pip install -r requirements.txt
source ./envsetup.sh        # exports PROJECT_ROOT, BUILD_PATH, INSTALL_PATH, REGRESSION_PATH, PYTHONPATH, PATH; also installs git hooks
```

`envsetup.sh` puts `python/tools`, `python/utils`, `python/test`, `python/samples` on `PATH`, so scripts like `model_transform.py`, `model_deploy.py`, `model_runner.py`, `run_calibration.py`, `llm_convert.py`, `test_onnx.py` are run by name.

CMODEL vs. real chip: by default `USING_CMODEL=True` and `LD_LIBRARY_PATH` points at the cmodel libs. Switch with the shell helpers `use_cmodel`, `use_chip`, `use_chip_cmodel` defined in `envsetup.sh`.

## Build

```bash
./build.sh              # RELEASE (default), also runs ./release_doc.sh and strips binaries
./build.sh DEBUG        # debug build with -ggdb, no doc/strip
./build.sh RELEASE CUDA # enable -DTPUMLIR_USE_CUDA=ON
```

The build uses Ninja + clang + lld, installs into `$INSTALL_PATH` (default `./install`), then builds `passes_json_files`, `builder_python`, `install_passes_files` targets and `lib/PplBackend/build.sh`. To iterate quickly without re-running CMake:

```bash
cmake --build $BUILD_PATH --target install -j$(nproc)
cmake --build $BUILD_PATH --target check-tpumlir   # lit + unit tests
```

Set `ENABLE_COVERAGE=True` (or call `enable_coverage`) before `./build.sh` to build with coverage instrumentation.

## Tests

There are three layers of tests; run the relevant one — there is no single "run all" command for routine work.

1. **Lit / C++ unit tests** (fast, run after every compiler change):
   ```bash
   cmake --build $BUILD_PATH --target check-tpumlir
   # or directly:
   $BUILD_PATH/bin/llvm-lit -sv $BUILD_PATH/test
   ```
   Lit cases live under `test/` (Dialect, Transforms, Linalg, TDB, Unit). C++ gtests under `unittests/`.

2. **Python op / model regression** via `regression/main_entry.py`:
   ```bash
   # full sets used by CI:
   regression/run.sh op       # torch op set + custom layer rebuild
   regression/run.sh model    # onnx op set
   regression/run.sh script   # check-tpumlir + script + model basic set
   # direct invocation:
   regression/main_entry.py --test_type basic --test_set onnx torch script model
   ```

3. **Single op / model case** — run the underlying `test_*.py` directly (they are on `PATH`):
   ```bash
   test_onnx.py   --case Conv2d   --chip bm1684x
   test_torch.py  --case LayerNorm --chip bm1688
   test_tflite.py --case <Case>   --chip bm1684x
   test_tpulang.py --case <Case>  --chip bm1684x
   run_model.py <model_name> --chip bm1684x --mode f16   # end-to-end model regression, configured by regression/config/
   ```
   Logs land in `regression/regression_out/`. `--simple` skips heavy checks.

## Lint / format

Pre-commit hooks installed by `envsetup.sh` enforce these — run them manually before pushing:

- C/C++ in `lib/`, `include/`, `tools/`: `clang-format -i` (config `.clang-format`, LLVM style).
- Python in `python/`, `regression/`: `yapf -i` (config `.style.yapf`, **100-column limit**, 4-space indent).
- Comments must be English (checked by `hooks/check_comment_language.py`).

## Architecture

The compiler is structured as an MLIR pipeline with two principal dialects:

```
front-end importer (python/transform/) ──► Top dialect ──lowering──► Tpu dialect ──► codegen ──► bmodel
                                          (framework-          (chip-specific ops, layer-group
                                           neutral graph)       memory planning, quantization)
```

- **Dialects** live in `include/tpu_mlir/Dialect/{Top,Tpu}/{IR,Transforms}` with implementations in `lib/Dialect/{Top,Tpu}`. Ops are defined in TableGen (`*.td`); regenerated headers go to `$BUILD_PATH`.
- **Conversions** between dialects: `lib/Conversion/{TopToTpu,TopToTosa,TopToLinalg}`. `TopToTpu` is per-chip (subdirectories for BM1684X, BM1688, etc.).
- **Backends** (`lib/Backend`) wrap chip backend libraries; `lib/PplBackend` is the PPL kernel backend (built separately by `lib/PplBackend/build.sh`).
- **Driver tools** (`tools/`): `tpuc-opt` (the MLIR opt tool with all TPU passes), `tpuc-tool`, `model_tool` (bmodel inspector), `chiprunner`.
- **Python front-end** (`python/transform/`) imports framework graphs and emits Top-dialect MLIR via the C-API in `capi/` / `bindings/`. `python/tools/model_transform.py` and `model_deploy.py` are the user-facing entry points; `llm_convert.py` is the one-shot LLM pipeline.
- **Calibration / quantization**: `python/calibration/` (PTQ, AutoTune, search), `python/tools/run_calibration.py`. INT8 deploy requires a calibration table.
- **Layer-group / memory planning** is a major Tpu-dialect transform — see `lib/Dialect/Tpu/Transforms/LayerGroup`.
- **Custom ops**: `third_party/customlayer` is sourced separately (`source $PROJECT_ROOT/third_party/customlayer/envsetup.sh`); `regression/run.sh op` rebuilds plugin/backend/firmware before testing.

## Conventions

- Always import via the `tpu_mlir` namespace in C++; new passes register in `lib/InitAll.cpp` and the corresponding `Passes.td`.
- Per-chip code paths key off the `processor`/`chip` argument (`bm1684x`, `bm1684`, `bm1688`, `bm1690`, `cv186ah`, `cv183x`, `cv182x`, `mars3`, `sgtpuv8`). The canonical list is in `regression/chip.py` and `python/utils/`; mirror it when adding chip switches.
- Quantize modes are spelled `F32 / BF16 / F16 / INT8` (uppercase) in user-facing flags but `f32/bf16/f16/int8` in regression configs — follow the surrounding file.
- `model_deploy.py` uses `--processor`, not `--chip`; `test_*.py` and most internal scripts use `--chip`. Don't conflate them.
- Tolerances in deploy/test are `<cos>,<euclid>` pairs (e.g. `0.99,0.90`); INT8 typically needs looser values like `0.85,0.45`.
- New ops require: a `.td` entry in the relevant dialect, a shape-inference + lowering pattern, a Python importer hook in `python/transform/`, and a regression case in `python/test/test_onnx.py` (or the matching framework file).
- Commit messages: short imperative summary; sign commits with a GitHub-registered email (CONTRIBUTING.md). One logical change per PR; CI must pass.
- Do **not** edit anything under `third_party/` casually — those are vendored or submoduled. The same goes for `install/`, `build/`, `dist/`, `tmp/` (build artefacts).
