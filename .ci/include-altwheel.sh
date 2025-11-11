#!/bin/bash

function replace_hash() {
  local yaml="$1"
  local host_wheel_dir="$2"
  local container_wheel_dir="$3"
  shift 3
  for althash in "$@"; do
    local shorthash=$(echo ${althash}|head -c 8)
    local bname=$(basename ${host_wheel_dir}/triton-*+git${shorthash}*.whl)
    local wheel="${container_wheel_dir}/${bname}"
    sed -i "s|${althash}|${wheel}|" "${yaml}"
  done
}
