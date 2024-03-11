import argparse
import os

entrygen_functions = []
function_names = []


def entrygen_single(execute_file_path):
    entrygen_functions.append(f'# f"{{package_path}}/{execute_file_path}\n')
    entrygen_count = 0

    file_abspath = os.path.join(os.path.dirname(__file__), execute_file_path)
    assert os.path.isdir(file_abspath) == False
    file = os.path.basename(file_abspath)
    file_name = os.path.splitext(file)[0]
    ext_name = os.path.splitext(file)[-1]

    if ext_name == ".py":
        codegen = f"""def {file_name.replace("-","_")}():\n\t\
file_name = f"{{os.getenv('TPUC_ROOT')}}/{execute_file_path}"\n\t\
run_subprocess_py(file_name)\n\n"""
    else:
        codegen = f"""def {file_name.replace("-","_")}():\n\t\
file_name = f"{{os.getenv('TPUC_ROOT')}}/{execute_file_path}"\n\t\
run_subprocess_c(file_name)\n\n"""

    function_names.append(file_name)
    entrygen_functions.append(codegen)
    entrygen_count += 1

    if entrygen_count == 0:
        entrygen_functions.pop()
    else:
        entrygen_functions.append(
            f'### total {entrygen_count} entry generated for f"{{package_path}}/{execute_file_path}\n\n'
        )


def entrygen(execute_path):
    abspath = os.path.join(os.path.dirname(__file__), execute_path)
    files = os.listdir(abspath)
    entrygen_functions.append(f'# f"{{package_path}}/{execute_path}\n')
    entrygen_count = 0
    exclude_file_type = [".prototxt", ".caffemodel", ".pt", ".onnx", ".tflite"]

    for file in files:
        file_abspath = os.path.join(os.path.dirname(execute_path), file)
        if os.path.isdir(file_abspath):
            continue

        file_name = os.path.splitext(file)[0]
        ext_name = os.path.splitext(file)[-1]

        if ext_name == ".py":
            codegen = f"""def {file_name.replace("-","_")}():\n\t\
file_name = f"{{os.getenv('TPUC_ROOT')}}/{os.path.join(execute_path,file)}"\n\t\
run_subprocess_py(file_name)\n\n"""
        elif ext_name in exclude_file_type:
            continue
        else:
            codegen = f"""def {file_name.replace("-","_")}():\n\t\
file_name = f"{{os.getenv('TPUC_ROOT')}}/{os.path.join(execute_path,file)}"\n\t\
run_subprocess_c(file_name)\n\n"""

        function_names.append(file_name)
        entrygen_functions.append(codegen)
        entrygen_count += 1

    if entrygen_count == 0:
        entrygen_functions.pop()
    else:
        entrygen_functions.append(
            f'### total {entrygen_count} entry generated for f"{{package_path}}/{execute_path}\n\n'
        )


def entryset(project_path):
    with open(os.path.join(project_path, "setup.py"), "r") as f:
        lines = f.readlines()

    marker_index = None
    for i, line in enumerate(lines):
        flag = "### Command Entries Will Be Set From Here. Do Not Delete This Line! ###"
        if flag in line:
            marker_index = i
            break

    if marker_index is not None:
        marker_indent = lines[marker_index].split(flag)[0]
        insert_code_list = []
        for function_name in function_names:
            insert_code_list.append(
                f'"{function_name}=tpu_mlir.entry:{function_name.replace("-","_")}",'
            )
            insert_code_list.append(
                f'"{function_name}.py=tpu_mlir.entry:{function_name.replace("-","_")}",'
            )

        insert_code = "\n".join([marker_indent + line for line in insert_code_list])
        insert_code += "\n"
        lines.insert(marker_index + 1, insert_code)

        with open(os.path.join(project_path, "setup.py"), "w") as g:
            g.writelines(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute_path",
        nargs="*",
        help="execute_path of files needed to codegen relative to the **tpu_mlir (copied) folder**",
    )
    parser.add_argument(
        "--execute_file",
        nargs="*",
        help="execute_path of specific files needed to codegen relative to the **tpu_mlir (copied) folder**",
    )
    args = parser.parse_args()

    # entry gen
    if args.execute_path:
        for r_path in args.execute_path:
            entrygen(r_path)
    if args.execute_file:
        for r_file in args.execute_file:
            entrygen_single(r_file)

    dirname, filename = os.path.split(os.path.abspath(__file__))
    with open(os.path.join(dirname, "entry.py"), "w+") as f:
        f.write(
            "import tpu_mlir,os\nfrom tpu_mlir import run_subprocess_c, run_subprocess_py\n\n"
        )
        f.writelines(entrygen_functions)

    # entry set
    entryset(os.getenv("PROJECT_ROOT"))
