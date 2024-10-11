#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Error: No file name provided."
    exit 1
fi

script_name=$(basename "$0")
ppl_package="$1"


if [[ ${ppl_package} != ppl_*.tar.gz ]]; then
  echo "Error: Invalid ppl file"
  exit 1
elif [[ ! -f ${ppl_package} ]]; then
  echo "Error: file[${ppl_package}] is not exist"
  exit 1
else
  echo "Update ppl package...."
fi

version="${ppl_package#*_}"
version="${version%.tar.gz}"

keep_files=("README.md" "${ppl_package}" "${script_name}")

# remove current files and folders
for file in *; do
    if [[ ! " ${keep_files[@]} " =~ " ${file} " ]]; then
        if [ -d "$file" ]; then
            echo "Deleting directory: $file"
            rm -rf "$file"
        else
            echo "Deleting file: $file"
            rm -f "$file"
        fi
    fi
done

echo $version > version

# clean ppl
tar xvf ${ppl_package}

mv ppl_${version}/* .
rm -rf ppl_${version}
rm -rf ${ppl_package}
rm -rf samples examples docker python runtime/bm1690/tpuv7-runtime* requirements.txt envsetup.sh
find .  -name "*.so*" -exec rm -f {} +
find .  -name "*.a" -exec rm -f {} +
chmod +x bin/*
