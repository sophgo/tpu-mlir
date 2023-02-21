#!/bin/bash
# set -e
# set -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [ -z $SET_CHIP_NAME ]; then
  echo "please set SET_CHIP_NAME"
  exit 1
fi
export WORKING_PATH=${WORKING_PATH:-$SCRIPT_DIR/regression_out}
export WORKSPACE_PATH=${WORKING_PATH}/${SET_CHIP_NAME}
export CVIMODEL_REL_PATH=$WORKSPACE_PATH/cvimodel_regression
export MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-8}
export OMP_NUM_THREADS=4
echo "WORKING_PATH: ${WORKING_PATH}"
echo "WORKSPACE_PATH: ${WORKSPACE_PATH}"
echo "CVIMODEL_REL_PATH: ${CVIMODEL_REL_PATH}"
echo "MAX_PARALLEL_JOBS: ${MAX_PARALLEL_JOBS}"

run_generic()
{
  local net=$1
  echo "generic regression $net"
  ./generic_model.sh -n $net -r ${CVIMODEL_REL_PATH} > $WORKSPACE_PATH/$1.log 2>&1 | true
  if [ "${PIPESTATUS[0]}" -ne "0" ]; then
    cat $WORKSPACE_PATH/$1.log >> verdict.log
    echo "$net generic regression FAILED" >> verdict.log
    return 1
  else
    echo "$net generic regression PASSED" >> verdict.log
    return 0
  fi
}
export -f run_generic

run_generic_all_parallel()
{
  local run_extra=$1

  rm -f regression.txt
  # bach_size 1
  for ((i=0; i < ${#net_list_generic[@]}; i++))
  do
    echo "run_generic ${net_list_generic[$i]}" >> regression.txt
  done
  if [ $run_extra -eq 1 ]; then
    for ((i=0; i < ${#net_list_generic_extra[@]}; i++))
    do
      echo "run_generic ${net_list_generic_extra[$i]}" >> regression.txt
    done
  fi
  # bach_size 4
  # for ((i=0; i < ${#net_list_batch[@]}; i++))
  # do
  #   echo "run_generic ${net_list_batch[$i]}" >> regression.txt
  # done
  # if [ $run_extra -eq 1 ]; then
  #   for ((i=0; i < ${#net_list_batch_extra[@]}; i++))
  #   do
  #     echo "run_generic ${net_list_batch_extra[$i]}" >> regression.txt
  #   done
  # fi
  cat regression.txt
  parallel -j${MAX_PARALLEL_JOBS} --delay 5  --joblog job_regression.log < regression.txt
  return $?
}

usage()
{
   echo ""
   echo "Usage: $0 [-b batch_size] [-n net_name] [-e] [-a count]"
   echo -e "\t-b Description of batch size for test"
   echo -e "\t-n Description of net name for test"
   echo -e "\t-e Enable extra net list"
   echo -e "\t-a Enable run accuracy, with given image count"
   echo -e "\t-f Model list filename"
   exit 1
}

run_extra=0
bs=1
run_accuracy=10
while getopts "n:a:f:e" opt
do
  case "$opt" in
    n ) network="$OPTARG" ;;
    e ) run_extra=1 ;;
    a ) run_accuracy="$OPTARG" ;;
    f ) model_list_file="$OPTARG" ;;
    h ) usage ;;
  esac
done

# default run in parallel
if [ -z "$RUN_IN_PARALLEL" ]; then
  export RUN_IN_PARALLEL=1
fi

# run regression for all
mkdir -p $WORKSPACE_PATH
mkdir -p $CVIMODEL_REL_PATH

net_list_generic=()
net_list_batch=()
net_list_generic_extra=()
net_list_batch_extra=()

if [ -z $model_list_file ]; then
  if [ ${SET_CHIP_NAME} = "cv183x" ]; then
    model_list_file=$SCRIPT_DIR/config/model_list_cv183x.txt
  elif [ ${SET_CHIP_NAME} = "cv181x" ]; then
    model_list_file=$SCRIPT_DIR/config/model_list_cv181x.txt
  elif [ ${SET_CHIP_NAME} = "cv180x" ]; then
    model_list_file=$SCRIPT_DIR/config/model_list_cv180x.txt
  else
    model_list_file=$SCRIPT_DIR/config/model_list_cv182x.txt
  fi
fi

while read net bs1 bs4 acc bs1_ext bs4_ext acc_ext
do
  [[ $net =~ ^#.* ]] && continue
  # echo "net='$net' bs1='$bs1' bs4='$bs4' acc='$acc' bs1_ext='$bs1_ext' bs4_ext='$bs4_ext' acc_ext='$acc_ext'"

  if [ "$bs1" = "Y" ]; then
    net_list_generic+=("$net")
  fi
  if [ "$bs1_ext" = "Y" ]; then
    net_list_generic_extra+=("$net")
  fi
done < ${model_list_file}

# printf 'net_list_generic: %s\n' "${net_list_generic[@]}"
# printf 'net_list_batch: %s\n' "${net_list_batch[@]}"
# printf 'net_list_generic_accuracy: %s\n' "${net_list_generic_accuracy[@]}"
# printf 'net_list_generic_extra: %s\n' "${net_list_generic_extra[@]}"
# printf 'net_list_batch_extra: %s\n' "${net_list_batch_extra[@]}"
# printf 'net_list_generic_accuracy_extra: %s\n' "${net_list_generic_accuracy_extra[@]}"

echo "" > verdict.log
# run specified model and exit
if [ ! -z "$network" ]; then
  run_generic $network
  ERR=$?
  if [ $ERR -eq 0 ]; then
    echo $network TEST PASSED
  else
    echo $network FAILED
  fi
  popd
  exit $ERR
fi

ERR=0
# run all models in model_lists.txt
run_generic_all_parallel $run_extra
if [ "$?" -ne 0 ]; then
  ERR=1
fi

cat verdict.log


# VERDICT
if [ $ERR -eq 0 ]; then
  echo $0 ALL TEST PASSED
else
  echo $0 FAILED
fi

exit $ERR
