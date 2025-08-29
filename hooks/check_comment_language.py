#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check if comments in source code files contain Chinese characters.
Supports C/C++, Python, Shell, and HTML files.
"""

import re
import sys
import argparse
from pathlib import Path

def has_chinese(text):
    """Check if text contains Chinese characters."""
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    return bool(chinese_pattern.search(text))

def extract_comments_cpp(content):
    """Extract comments from C/C++ files."""
    comments = []
    lines = content.split('\n')

    for line_num, line in enumerate(lines, 1):
        # Single line comments
        if '//' in line:
            comment_start = line.find('//')
            comment = line[comment_start + 2:].strip()
            if comment:
                comments.append((line_num, comment, line.strip()))

        # Multi-line comments (simple detection)
        if '/*' in line and '*/' in line:
            start = line.find('/*')
            end = line.find('*/', start)
            comment = line[start + 2:end].strip()
            if comment:
                comments.append((line_num, comment, line.strip()))

    # Handle multi-line comments spanning multiple lines
    in_multiline = False
    multiline_start = 0
    for line_num, line in enumerate(lines, 1):
        if '/*' in line and not in_multiline:
            in_multiline = True
            multiline_start = line_num
            start = line.find('/*')
            if '*/' not in line[start:]:
                comment = line[start + 2:].strip()
                if comment:
                    comments.append((line_num, comment, line.strip()))
        elif in_multiline:
            if '*/' in line:
                in_multiline = False
                end = line.find('*/')
                comment = line[:end].strip()
                if comment.startswith('*'):
                    comment = comment[1:].strip()
                if comment:
                    comments.append((line_num, comment, line.strip()))
            else:
                comment = line.strip()
                if comment.startswith('*'):
                    comment = comment[1:].strip()
                if comment:
                    comments.append((line_num, comment, line.strip()))

    return comments

def extract_comments_python(content):
    """Extract comments from Python files."""
    comments = []
    lines = content.split('\n')

    for line_num, line in enumerate(lines, 1):
        # Single line comments
        if '#' in line:
            # Check if # is inside a string
            in_string = False
            quote_char = None
            escaped = False

            for i, char in enumerate(line):
                if escaped:
                    escaped = False
                    continue

                if char == '\\':
                    escaped = True
                    continue

                if char in ['"', "'"]:
                    if not in_string:
                        in_string = True
                        quote_char = char
                    elif char == quote_char:
                        in_string = False
                        quote_char = None
                elif char == '#' and not in_string:
                    comment = line[i + 1:].strip()
                    if comment:
                        comments.append((line_num, comment, line.strip()))
                    break

    return comments

def extract_comments_shell(content):
    """Extract comments from Shell files."""
    comments = []
    lines = content.split('\n')

    for line_num, line in enumerate(lines, 1):
        # Skip shebang line
        if line.startswith('#!'):
            continue

        # Single line comments
        if '#' in line:
            # Simple check for # not in quotes
            comment_start = line.find('#')
            # Basic check to avoid # in strings (not perfect but good enough)
            before_hash = line[:comment_start]
            single_quotes = before_hash.count("'")
            double_quotes = before_hash.count('"')

            # If even number of quotes before #, it's likely a comment
            if single_quotes % 2 == 0 and double_quotes % 2 == 0:
                comment = line[comment_start + 1:].strip()
                if comment:
                    comments.append((line_num, comment, line.strip()))

    return comments

def extract_comments_html(content):
    """Extract comments from HTML files."""
    comments = []
    lines = content.split('\n')

    # Find HTML comments <!-- -->
    comment_pattern = re.compile(r'<!--(.*?)-->', re.DOTALL)

    for line_num, line in enumerate(lines, 1):
        # Single line HTML comments
        matches = comment_pattern.finditer(line)
        for match in matches:
            comment = match.group(1).strip()
            if comment:
                comments.append((line_num, comment, line.strip()))

    return comments

def check_file_comments(file_path):
    """Check comments in a single file for Chinese characters."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                content = f.read()
        except UnicodeDecodeError:
            print(f"Warning: Cannot decode file {file_path}")
            return []

    file_extension = Path(file_path).suffix.lower()

    if file_extension in ['.cpp', '.c', '.h', '.hpp', '.cc', '.cxx']:
        comments = extract_comments_cpp(content)
    elif file_extension in ['.py']:
        comments = extract_comments_python(content)
    elif file_extension in ['.sh', '.bash']:
        comments = extract_comments_shell(content)
    elif file_extension in ['.html', '.htm']:
        comments = extract_comments_html(content)
    else:
        return []

    chinese_comments = []
    for line_num, comment, full_line in comments:
        if has_chinese(comment):
            chinese_comments.append((line_num, comment, full_line))

    return chinese_comments

def main():
    parser = argparse.ArgumentParser(description='Check for Chinese comments in source code files')
    parser.add_argument('files', nargs='+', help='Files to check')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    total_issues = 0

    for file_path in args.files:
        if not Path(file_path).exists():
            continue

        chinese_comments = check_file_comments(file_path)

        if chinese_comments:
            print(f"\033[31mERROR\033[0m: Chinese comments found in {file_path}")
            for line_num, comment, full_line in chinese_comments:
                print(f"  \033[33mLine {line_num}\033[0m: {comment}")
                if args.verbose:
                    print(f"    Full line: {full_line}")
            total_issues += len(chinese_comments)
            print()

    if total_issues > 0:
        print(f"\033[31mFound {total_issues} Chinese comment(s). Please use English comments only.\033[0m")
        return 1
    else:
        print("\033[32mSuccess: All comments are in English\033[0m")
        return 0

if __name__ == '__main__':
    sys.exit(main())
