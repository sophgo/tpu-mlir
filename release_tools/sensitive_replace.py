import os
import re
import argparse
from pathlib import Path

def replace_keywords_in_file(file_path, search_pattern, replace_with, dry_run=False):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        new_content = re.sub(search_pattern, replace_with, content, flags=re.IGNORECASE)

        if new_content == original_content:
            return 0

        if not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            print(f"✓ modified: {file_path}")
        else:
            print(f"⚠ going to modify: {file_path}")

        return 1

    except UnicodeDecodeError:
        return 0
    except Exception as e:
        print(f"✗ ERROR: {file_path}: {e}")
        return 0

def find_files(root_dir, extensions):
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env', 'node_modules']]

        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                yield os.path.join(root, file)

def main():
    parser = argparse.ArgumentParser(description='replace src_words with dst_words in source code files.')
    parser.add_argument('--path', '-r', default='.',
                       help='')
    parser.add_argument('--dry-run', '-d', action='store_true',
                       help='Simulated operation, without actually modifying the files.')
    parser.add_argument('--search', '-s', default='(sophon|sophgo)',
                       help='search pattern, split with | : (sophon|sophgo)')
    parser.add_argument('--replace', '-w', default='OEM',
                       help='the word used to replace the search pattern.')
    args = parser.parse_args()

    extensions = ['.h', '.hpp', '.hxx', '.hh', '.py', '.cuh', '.cu']

    total_files = 0
    modified_files = 0

    for file_path in find_files(args.path, extensions):
        total_files += 1

        modified_count = replace_keywords_in_file(
            file_path,
            args.search,
            args.replace,
            args.dry_run
        )
        modified_files += modified_count

    print("-" * 50)
    print(f"total file num: {total_files}")
    print(f"modified file num: {modified_files}")

    if args.dry_run:
        print("dry_run only , no files were actually modified.")

if __name__ == "__main__":
    main()
