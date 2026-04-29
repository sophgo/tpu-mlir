# Contributing to TPU-MLIR

Thanks for taking the time to contribute! 🎉

## Reporting issues

Before opening an issue, please:

1. Search [existing issues](https://github.com/sophgo/tpu-mlir/issues) to avoid duplicates.
2. If you find a related but unresolved issue, comment there instead of opening a new one.
3. When opening a new issue, use the appropriate template and include:
   - TPU-MLIR version (`pip show tpu_mlir` or commit SHA)
   - Target chip (e.g. `bm1684x`)
   - Source framework + model link (if reproducible)
   - Exact command line and full error output
   - Minimal reproducer when possible

## Submitting pull requests

1. **Discuss first** — for non-trivial changes, open an issue describing the design before sending a PR.
2. **One topic per PR** — keep changes focused and reviewable.
3. **Match the existing style**:
   - C/C++ — formatted with `clang-format` (config: `.clang-format`).
   - Python — formatted with `yapf` (config: `.style.yapf`, 100-column limit).
   - The repo also ships an `.editorconfig` that most editors pick up automatically.
4. **Run the regression tests** under `regression/` for any change touching the compiler or runtime.
5. **Write good commit messages** — short imperative summary, optional body explaining *why*.
6. **Sign your commits** with an email registered on your GitHub account so the commit is properly attributed (see GitHub → Settings → Emails).

## Building from source

See [README.md → Installation](./README.md#-installation) and [README_cn.md → 安装](./README_cn.md#-安装).

Quick reference inside the official Docker image:

```shell
pip install -r requirements.txt
source ./envsetup.sh
./build.sh           # RELEASE build
./build.sh DEBUG     # debug build with symbols
```

## Code review

- A maintainer will review your PR and may request changes.
- CI must pass before merge.
- Once approved, a maintainer will squash-merge or rebase-merge the change.

## License

By contributing, you agree that your contributions will be licensed under the same terms as the project (see [LICENSE](./LICENSE)).
