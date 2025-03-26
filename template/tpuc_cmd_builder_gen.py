#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from hashlib import md5
import os
import json
import re
from jinja2 import Environment, FileSystemLoader
from typing import List, Dict, Any
import sys

try:
    from loguru import logger
    logger.remove()
    logger.add(sys.stdout, level="ERROR")
except ImportError:
    from logging import root as logger

def camel_to_snake(name: str) -> str:
    """Convert camel case naming to snake case naming"""
    # Handle special cases like "QDQConvert" -> "qdq_convert"
    if name.isupper():
        return name.lower()
    pattern = re.compile(r"(?<!^)(?=[A-Z])")
    return pattern.sub("_", name).lower()


def extract_pass_methods(
    json_files: List[str], passes_dir: str
) -> List[Dict[str, Any]]:
    """Extract pass method information from JSON files, including detailed options"""

    # First load all option data
    pass_methods = []
    options_ret = []

    for json_file in json_files:
        passes_data = {}
        options_data = {}
        file_path = os.path.join(passes_dir, json_file)
        if not os.path.exists(file_path):
            logger.info(f"Warning: Unable to find file {file_path}")
            continue

        with open(file_path, "r") as f:
            try:
                content = f.read()
                md5_content = md5(content.encode()).hexdigest()[:4]
                data = json.loads(content)
                # Load all option definitions
                for key, value in data.items():
                    if (
                        isinstance(value, dict)
                        and "!superclasses" in value
                        and isinstance(value["!superclasses"], list)
                        and "Option" in value["!superclasses"]
                    ):
                        cur = {
                            "name": f"{md5_content}_{key}",
                            "argument": value.get("argument", ""),
                            "type": value.get("type", "string"),
                            "description": value.get("description", ""),
                            "default_value": value.get("defaultValue", ""),
                        }
                        options_data[f"{md5_content}_{key}"] = cur
                        options_ret.append(cur)

                # Load all pass definitions
                for key, value in data.items():
                    if (
                        isinstance(value, dict)
                        and "!superclasses" in value
                        and isinstance(value["!superclasses"], list)
                        and "Pass" in value["!superclasses"]
                    ):
                        passes_data[key] = value
            except json.JSONDecodeError:
                logger.error(f"Error: File {file_path} is not a valid JSON file")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")

        # Then create method information for each pass
        for pass_name, pass_data in passes_data.items():
            argument = pass_data.get("argument", "")
            if not argument:
                continue  # Skip passes without argument

            # Create method name based on pass name
            method_name = argument.replace("-", "_")
            if method_name.endswith("_pass"):
                method_name = method_name[:-5]  # Remove trailing "_pass"

            # Extract options for this pass
            pass_options = []
            for opt in pass_data.get("options", []):
                if isinstance(opt, dict) and "def" in opt:
                    opt_def_name = f"{md5_content}_{opt['def']}"
                    if opt_def_name in options_data:
                        pass_options.append(options_data[opt_def_name])
            logger.debug(
                f"pass_name: {pass_name}, argument: {argument}, method_name: {method_name}"
            )

            pass_methods.append(
                {
                    "method_name": method_name,
                    "argument_name": argument,
                    "pass_name": pass_name,
                    "pass_argument": argument,
                    "has_options": bool(pass_options),
                    "options": pass_options,
                    "summary": pass_data.get("summary", ""),
                    "description": pass_data.get("description", ""),
                }
            )

    return pass_methods


def extract_json_files_content(
    json_files: List[str], passes_dir: str
) -> Dict[str, Dict[str, Any]]:
    """Extract all JSON file contents for direct embedding into generated files"""
    all_data = {}

    for json_file in json_files:
        file_path = os.path.join(passes_dir, json_file)
        if not os.path.exists(file_path):
            logger.error(f"Warning: Unable to find file {file_path}")
            continue

        with open(file_path, "r") as f:
            try:
                data = json.load(f)

                # Filter out pass and option type data
                filtered_data = {}
                for key, value in data.items():
                    if (
                        isinstance(value, dict)
                        and "!superclasses" in value
                        and isinstance(value["!superclasses"], list)
                        and (
                            "Pass" in value["!superclasses"]
                            or "Option" in value["!superclasses"]
                        )
                    ):
                        filtered_data[key] = value

                all_data[json_file] = filtered_data
            except json.JSONDecodeError:
                logger.error(f"Error: File {file_path} is not a valid JSON file")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")

    return all_data


def generate_builder(template_file: str, output_file: str, passes_dir: str):
    """Generate builder.py file using a template"""
    # Get JSON file list
    json_files = [f for f in os.listdir(passes_dir) if f.endswith(".json")]

    try:
        # Extract pass method information
        pass_methods = extract_pass_methods(json_files, passes_dir)

        # Extract JSON file content
        json_data = extract_json_files_content(json_files, passes_dir)

        # Sort by method name
        pass_methods.sort(key=lambda x: x["method_name"])

        # Create Jinja2 environment
        template_dir = os.path.abspath(os.path.dirname(__file__))
        logger.info(template_dir)
        env = Environment(loader=FileSystemLoader(template_dir))
        # template = env.get_template(os.path.join(os.path.dirname(__file__), os.path.basename(template_file)))
        template = env.get_template(os.path.basename(template_file))

        # Render template
        rendered_content = template.render(
            json_files=json_files, pass_methods=pass_methods, json_data=json_data
        )

        # Write to output file
        with open(output_file, "w") as f:
            f.write(rendered_content)

        logger.info(f"Successfully generated file: {output_file}")
        logger.info(
            f"Processed {len(json_files)} JSON files, generated {len(pass_methods)} pass methods"
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate TPU-MLIR Command Builder")
    parser.add_argument(
        "--template", default="builder_template.j2", help="Jinja2 template file"
    )
    parser.add_argument(
        "--output", default="builder_generated.py", help="Output builder.py file path"
    )
    parser.add_argument(
        "--passes_dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="passes_json directory path",
    )

    args = parser.parse_args()

    generate_builder(args.template, args.output, args.passes_dir)
