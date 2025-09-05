#!/bin/bash

set -ex
PATCH_FILE="$1"
triton_patch_url='https://github.com/ROCm/triton.git'
PATCH_REMOTE='__ci_patching'

. "${PATCH_FILE}"
git merge-base --is-ancestor ${expect_triton_commit} HEAD
if [ $? -ne 0 ]; then
  echo "HEAD does not contain ${expect_triton_commit}" >&2
  exit 1
fi

git remote add ${PATCH_REMOTE} ${triton_patch_url} || git remote set-url ${PATCH_REMOTE} ${triton_patch_url}
git fetch ${PATCH_REMOTE} ${triton_patch_branch}

for commit in "${triton_cherry_picks[@]}"; do
  git cherry-pick ${commit}
done
