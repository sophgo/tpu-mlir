#!/bin/bash

#docker pull ubuntu:18.04

SUDO=sudo
if [ "$(id -u)" == "0" ]; then
  SUDO=
fi
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $DIR/docker.env

CMD="
$SUDO docker build \
    -t $REPO/$IMAGE:$TAG_BASE \
    -f $DIR/Dockerfile_ubuntu-${BASE_IMAGE_VERSION} \
    .
"

echo $CMD
eval $CMD
