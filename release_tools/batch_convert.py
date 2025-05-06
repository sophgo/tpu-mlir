#!/usr/bin/env python3
import os
import re
from pathlib import Path
import shutil

BASE_PACKAGE = "tpu_mlir.python"

def validate_module_path(module_name, project_root):
    """
    Verify the actual existence path of the module in the project
    Return the absolute path format that can be imported (or None indicates invalid)
    """
    project_module = project_root.name.replace('-', '_')
    if not module_name.startswith(f"{project_module}."):
        return None

    rel_module = module_name[len(project_module)+1:]
    module_rel = rel_module.replace('.', os.sep)

    candidates = [
        (project_root / module_rel, True),
        (project_root / f"{module_rel}.py", False)
    ]

    for path, is_package in candidates:
        if is_package:
            if path.is_dir() and (path / "__init__.py").exists():
                return module_name
        else:
            if path.exists():
                return module_name
    return None

def convert_imports(content, project_root, prefix):

    from_pattern = re.compile(
        r'^(\s*)from\s+((?!\.+)[\w.]+)\s+(import\s+[\w*., ]+)(\r?\n)',
        re.MULTILINE
    )
    import_pattern = re.compile(
        r'^(\s*)import\s+([\w., ]+)(\r?\n)',
        re.MULTILINE
    )

    def replace_from(match):
        indent, module, imports, line_end = match.groups()
        full_module = f"{prefix}.{module}" if prefix else module
        if validated := validate_module_path(full_module, project_root):
            return f"{indent}from {validated} {imports}{line_end}"
        return match.group(0)

    def replace_import(match):
        indent, module, line_end = match.groups()
        full_module = f"{prefix}.{module}" if prefix else module
        if validated := validate_module_path(full_module, project_root):
            return f"{indent}import {validated}{line_end}"
        return match.group(0)


    content = from_pattern.sub(replace_from, content)
    content = import_pattern.sub(replace_import, content)
    return content

def process_file(file_path, project_root, prefix):
    """
    Process single files and replace the imports
    """
    file_path = Path(file_path)

    '''
    Create a backup file(optional)
    '''
    # backup = file_path.with_name(f"{file_path.stem}_bak{file_path.suffix}")
    # shutil.copy2(file_path, backup)

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = convert_imports(content, project_root, prefix)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"Processed: {file_path}")

def main():
    project_root = Path("/workspace/tpu-mlir")
    prefix = "tpu_mlir.python"

    target_dirs = [
        project_root / "python/tools",
        project_root / "python/utils"
    ]
    target_files = []
    for target_dir in target_dirs:
        target_files.extend(target_dir.rglob("*.py"))

    for file_path in target_files:
        if not file_path.is_file():
            print(f"[Skip] no file: {file_path}")
            continue
        process_file(file_path, project_root, prefix)
        print(f"[Success]: {file_path.relative_to(project_root)}")

if __name__ == "__main__":
    main()
