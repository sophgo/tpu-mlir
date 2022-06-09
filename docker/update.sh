#!/bin/bash

#docker pull ubuntu:18.04

SUDO=sudo
if [ "$(id -u)" == "0" ]; then
  SUDO=
fi
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

CMD="
$SUDO docker build \
    -t sophgo/sophgo_dev:1.2-ubuntu-18.04 \
    -f $DIR/Dockerfile_update \
    .
"

echo $CMD
eval $CMD
