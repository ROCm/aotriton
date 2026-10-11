#!/bin/sh
# Source this file to use the ROCm SDK installed with the active Python stack.
# Login shells may reset PATH; retain the image's selected Python environment.
if [ -n "${VIRTUAL_ENV:-}" ]; then
    case ":$PATH:" in
        *":$VIRTUAL_ENV/bin:"*) ;;
        *) PATH="$VIRTUAL_ENV/bin:$PATH" ;;
    esac
fi
ROCM_PATH="$(rocm-sdk path --root)" || return 1
rocm_sdk_bin="$(rocm-sdk path --bin)" || return 1
export ROCM_PATH

case ":$PATH:" in
    *":$rocm_sdk_bin:"*) ;;
    *) PATH="$rocm_sdk_bin:$PATH" ;;
esac
# Keep the LLVM drivers beside their companion executables in lib/llvm/bin.
case ":$PATH:" in
    *":$ROCM_PATH/lib/llvm/bin:"*) ;;
    *) PATH="$ROCM_PATH/lib/llvm/bin:$PATH" ;;
esac
export PATH
unset rocm_sdk_bin
