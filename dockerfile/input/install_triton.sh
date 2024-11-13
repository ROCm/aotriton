#!/bin/bash

TRITON_LLVM_HASH="$1"
fn="llvm-${TRITON_LLVM_HASH}-almalinux-x64.tar.gz"

if [ -f "/input/$fn" ]; then
  mkdir -p "$HOME/.triton/llvm"
  cd "$HOME/.triton/llvm"
  echo "Unpacking $fn" && tar xf "/input/$fn"
  echo -n "https://oaitriton.blob.core.windows.net/public/llvm-builds/$fn" > "llvm-${TRITON_LLVM_HASH}-almalinux-x64/version.txt"
fi
