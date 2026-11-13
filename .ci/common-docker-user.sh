#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Who the CI containers run as. Source this; do not execute it.
#
# The images default to the non-root uid 65534 (nobody, base.Dockerfile), but
# every `docker run` here passes DOCKER_USER_ARGS instead, to run as the user
# invoking the script. That keeps what the container writes into bind mounts
# (wheels, tarballs, caches, release output) owned by the host user who has
# to manage it. That uid has no passwd entry in the image, so HOME would
# default to /, which is not writable; /tmp is.
DOCKER_USER_ARGS=(--user "$(id -u):$(id -g)" -e HOME=/tmp)

# Fail unless everything under each <dir> is writable by the host user.
#
# For the host directories bind-mounted into the containers. Runs from before
# the containers stopped running as root left root-owned files in them, and a
# container would otherwise fail on one only after hours of building.
function require_writable() {
  local dir
  local blocked
  for dir in "$@"; do
    blocked=$(find "${dir}" ! -writable -print -quit)
    if [[ -n "${blocked}" ]]; then
      echo "Error: ${blocked} is not writable by $(id -un), and the CI containers run as that user." >&2
      echo "Files left by earlier root runs can be reclaimed once with:" >&2
      echo "  sudo chown -R $(id -u):$(id -g) ${dir}" >&2
      return 1
    fi
  done
}

# Make <volume> writable by the host user, running <image> to do it.
#
# A new docker volume is root-owned, and so is everything a container wrote
# into one before the containers stopped running as root. This is the only
# step that runs as root, and only when the volume's root directory is not
# already owned by the host user.
function claim_volume() {
  local volume="$1"
  local image="$2"
  local owner
  owner=$(docker run --rm "${DOCKER_USER_ARGS[@]}" -v "${volume}:/volume" \
    "${image}" stat -c '%u:%g' /volume) || return 1
  if [[ "${owner}" != "$(id -u):$(id -g)" ]]; then
    docker run --rm --user 0:0 -v "${volume}:/volume" \
      "${image}" chown -R "$(id -u):$(id -g)" /volume
  fi
}
