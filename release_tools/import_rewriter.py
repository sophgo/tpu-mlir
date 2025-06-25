#!/usr/bin/env python3
import re
from pathlib import Path
import argparse

def convert_imports(content, module_type, base_package):
    """
    Convert imports of specified module_type (tools/utils) to prefixed format
    Only processes imports matching the pattern:
    - import <module_type>...
    - from <module_type>... import ...
    """
    # Pattern for 'from module... import ...'
    from_pattern = re.compile(
        rf'^(\s*)from\s+({module_type})(\.\S*)?\s+(import\s+[\w*., ]+)(\r?\n)',
        re.MULTILINE
    )

    # Pattern for 'import module...'
    import_pattern = re.compile(
        rf'^(\s*)import\s+({module_type})(\.\S*)?([\w., ]*)(\r?\n)',
        re.MULTILINE
    )

    def replace_from(match):
        indent, module, submodule, imports, line_end = match.groups()
        new_module = f"{base_package}.{module}{submodule if submodule else ''}"
        return f"{indent}from {new_module} {imports}{line_end}"

    def replace_import(match):
        indent, module, submodule, others, line_end = match.groups()
        new_module = f"{base_package}.{module}{submodule if submodule else ''}"
        return f"{indent}import {new_module}{others}{line_end}"

    # Process from...import statements
    content = from_pattern.sub(replace_from, content)
    # Process import statements
    content = import_pattern.sub(replace_import, content)

    return content

def process_file(file_path, module_type, base_package):
    """
    Process a single file to update imports for specified module type
    """
    file_path = Path(file_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = convert_imports(content, module_type, base_package)

    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False

def main():
    parser = argparse.ArgumentParser(
        description='Add package prefix to specific module imports (tools/utils) in all Python files'
    )
    parser.add_argument('--project-root', required=True,
                       help='Root directory of the project')
    parser.add_argument('--module-type', required=True, choices=['tools', 'utils'],
                       help='Type of module to process (tools or utils)')
    parser.add_argument('--base-package', default='tpu_mlir.python',
                       help='Base package prefix to add')

    args = parser.parse_args()

    # Scan all Python files under python directory
    python_dir = Path(args.project_root) / "python"

    if not python_dir.is_dir():
        print(f"Error: Python directory not found - {python_dir}")
        return

    processed_files = 0
    for file_path in python_dir.rglob('*.py'):
        if not file_path.is_file():
            continue

        if process_file(file_path, args.module_type, args.base_package):
            print(f"Updated imports in: {file_path.relative_to(args.project_root)}")
            processed_files += 1

    print(f"\nProcessing complete. Modified {processed_files} files containing {args.module_type} imports.")

if __name__ == "__main__":
    main()
