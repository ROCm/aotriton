# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

message("CMAKE_SOURCE_DIR ${CMAKE_SOURCE_DIR}")
message("CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}")
cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH CMAKE_CURRENT_SOURCE_PARENT_DIR)
message("CMAKE_CURRENT_SOURCE_PARENT_DIR ${CMAKE_CURRENT_SOURCE_PARENT_DIR}")
message("CMAKE_CURRENT_LIST_DIR ${CMAKE_CURRENT_LIST_DIR}")
message("CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}")
set(AOTRITON_V2_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(AOTRITON_KERNEL_STORAGE_V2_DIR "${AOTRITON_V2_BUILD_DIR}/aotriton.images")
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${AOTRITON_V2_BUILD_DIR}")
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${AOTRITON_KERNEL_STORAGE_V2_DIR}")

execute_process(COMMAND ${CMAKE_COMMAND}
  -E env VIRTUAL_ENV=${VENV_DIR}
  "${VENV_BIN_PYTHON}" -m aotriton.gpu_targets
  --target_arch ${AOTRITON_TARGET_ARCH}
  --target_gpus ${AOTRITON_OVERRIDE_TARGET_GPUS}
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.."
  OUTPUT_VARIABLE EFFECTIVE_TARGET_GPUS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
message(STATUS "AOTRITON_TARGET_ARCH ${AOTRITON_TARGET_ARCH}")
message(STATUS "AOTRITON_OVERRIDE_TARGET_GPUS ${AOTRITON_OVERRIDE_TARGET_GPUS}")
message(STATUS "EFFECTIVE_TARGET_GPUS ${EFFECTIVE_TARGET_GPUS}")

if(AOTRITON_TUNING_DATABASE_ROOT)
  # External override: same per-family shape as modules/*/database
  # (<family>/database/amd/<arch>/...), just rooted elsewhere.
  file(GLOB DBXZ_BASES "${AOTRITON_TUNING_DATABASE_ROOT}/*/database")
  message(STATUS "Using external tuning database root: ${AOTRITON_TUNING_DATABASE_ROOT}")
else()
  # Per-family databases: each modules/<family>/database is its own base. A new
  # family drops in modules/<fam>/database, no edit here.
  file(GLOB DBXZ_BASES "${PROJECT_SOURCE_DIR}/modules/*/database")
endif()
foreach(DBXZ_BASE IN LISTS DBXZ_BASES)
  # Compose independently into build_dir/<family>/database/ - matching the
  # per-family Factory lookup (DatabaseFactories.create_factory(build_dir /
  # family)). No cross-family merge: each family's tuning_database/op_database
  # stays self-contained.
  cmake_path(GET DBXZ_BASE PARENT_PATH DBXZ_FAMILY_DIR)
  cmake_path(GET DBXZ_FAMILY_DIR FILENAME DBXZ_FAMILY)
  set(FAMILY_DB_DIR "${AOTRITON_V2_BUILD_DIR}/${DBXZ_FAMILY}/database")
  file(GLOB_RECURSE DB_TARS CONFIGURE_DEPENDS "${DBXZ_BASE}/*.sqlite3.tar.xz")
  foreach(DBXZ IN LISTS DB_TARS)
    cmake_path(RELATIVE_PATH DBXZ BASE_DIRECTORY "${DBXZ_BASE}" OUTPUT_VARIABLE REL_DBXZ)
    cmake_path(REMOVE_FILENAME REL_DBXZ)
    file(ARCHIVE_EXTRACT INPUT "${DBXZ}" DESTINATION "${FAMILY_DB_DIR}/${REL_DBXZ}")
    message(STATUS "Extracting ${DBXZ} to ${FAMILY_DB_DIR}/${REL_DBXZ}")
  endforeach(DBXZ)
  message(STATUS "Assembling ${DBXZ_FAMILY} Database...")
  execute_process(
      COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR}
      "${VENV_BIN_PYTHON}" -m aotriton.database_compose
      --database_file "${FAMILY_DB_DIR}/tuning_database.sqlite3"
      --decomposed "${FAMILY_DB_DIR}"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_PARENT_DIR}"
      COMMAND_ERROR_IS_FATAL ANY)
endforeach(DBXZ_BASE)

get_filename_component(AOTRITON_COMPILER "${CMAKE_CURRENT_LIST_DIR}/../python/compile.py" ABSOLUTE)
message("AOTRITON_COMPILER ${AOTRITON_COMPILER}")

set(HSACO_COMPILE_OPTION "")
if(AOTRITON_TERMINATE_WHEN_GPU_BUILD_TIMEOUT)
  list(APPEND HSACO_COMPILE_OPTION "--error_when_timeout")
endif()

if(AOTRITON_BUILD_FOR_TUNING)
  set(GENERATE_OPTION "--build_for_tuning")
else(AOTRITON_BUILD_FOR_TUNING)
  set(GENERATE_OPTION "")
endif()

if(AOTRITON_BUILD_FOR_TUNING_BUT_SKIP_KERNEL)
  if (NOT AOTRITON_BUILD_FOR_TUNING)
    message(FATAL_ERROR "AOTRITON_BUILD_FOR_TUNING_BUT_SKIP_KERNEL is only relevant with AOTRITON_BUILD_FOR_TUNING=ON")
  endif()
  list(APPEND GENERATE_OPTION "--build_for_tuning_but_skip_kernel")
  list(APPEND GENERATE_OPTION ${AOTRITON_BUILD_FOR_TUNING_BUT_SKIP_KERNEL})
endif()

if(AOTRITON_NOIMAGE_MODE)
  list(APPEND GENERATE_OPTION "--noimage_mode")
endif()

if(AOTRITON_ALT_TRITON_WHEEL_CONFIG_FILE)
  list(APPEND GENERATE_OPTION "--alt_venv_config")
  list(APPEND GENERATE_OPTION "${AOTRITON_ALT_TRITON_WHEEL_CONFIG_FILE}")
endif()

if(AOTRITON_ENABLE_FP32_INPUTS)
  set(AOTRITON_ENABLE_FP32 1)
else()
  set(AOTRITON_ENABLE_FP32 0)
endif()

# Normalise ON/OFF to 1/0 here to make python `bool(int(os.getenv(...)))`
# idiom work.
if(AOTRITON_DEBUG_SKIP_TRITON_KERNELS)
  set(AOTRITON_DEBUG_SKIP_TRITON_KERNELS_ENV 1)
  message(WARNING
    "[AOTriton] AOTRITON_DEBUG_SKIP_TRITON_KERNELS=ON: the generator will emit no Triton "
    "image rules, so aotriton.images is INCOMPLETE and the resulting library cannot serve "
    "any Triton-backed operator at runtime. Debugging aid for iterating on other backends; "
    "never ship the result.")
else()
  set(AOTRITON_DEBUG_SKIP_TRITON_KERNELS_ENV 0)
endif()

if(NOT AOTRITON_NOIMAGE_MODE)
  find_package(Git REQUIRED)
  ## Handle aiter.txt
  # CAVEAT: MUST USE GIT TAG OTHERWISE GIT_SHALLOW WILL NOT WORK
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E cat "${CMAKE_CURRENT_SOURCE_PARENT_DIR}/third_party/aiter.txt"
    OUTPUT_VARIABLE AITER_GIT_TAG
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  # CAVEAT: FetchContent_Declare is buggy and unsuitable for our needs
  #         See https://stackoverflow.com/questions/78237952
  set(AITER_DIR "${AOTRITON_V2_BUILD_DIR}/aiter")
  if(NOT EXISTS "${AITER_DIR}/.git")
    execute_process(
      COMMAND ${GIT_EXECUTABLE} clone --depth 1 --branch ${AITER_GIT_TAG}
      https://github.com/ROCm/aiter.git
      WORKING_DIRECTORY "${AOTRITON_V2_BUILD_DIR}"
      COMMAND_ERROR_IS_FATAL ANY
    )
  else()
    # Check current HEAD tag to avoid unnecessary network fetches on every configure
    execute_process(
      COMMAND ${GIT_EXECUTABLE} describe --tags --exact-match HEAD
      WORKING_DIRECTORY "${AITER_DIR}"
      RESULT_VARIABLE AITER_DESCRIBE_RESULT
      OUTPUT_VARIABLE AITER_CURRENT_TAG
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(NOT AITER_DESCRIBE_RESULT EQUAL 0 OR NOT AITER_CURRENT_TAG STREQUAL "${AITER_GIT_TAG}")
      execute_process(
        COMMAND ${GIT_EXECUTABLE} fetch origin --depth 1 "tag" "${AITER_GIT_TAG}"
        WORKING_DIRECTORY "${AITER_DIR}"
        COMMAND_ERROR_IS_FATAL ANY
      )
      # cmake doc: "For sequential execution of multiple commands use multiple
      # execute_process calls each with a single COMMAND argument."
      execute_process(
        COMMAND ${GIT_EXECUTABLE} checkout ${AITER_GIT_TAG}
        WORKING_DIRECTORY "${AITER_DIR}"
        COMMAND_ERROR_IS_FATAL ANY
      )
    endif()
  endif()
endif()

## Generate Bare.* files and shim files
# AOTRITON_DEBUG_SKIP_TRITON_KERNELS is passed to the generator rather than acted on
# here: it makes the generator omit Triton image rules from Bare.compile/Bare.cluster,
# so the rule files this build reads below are simply empty and every loop over them
# does nothing. No build-system special-casing, and the flatzip stage keeps working for
# whatever non-Triton backends did emit rules.
execute_process(
COMMAND ${CMAKE_COMMAND}
-E env VIRTUAL_ENV=${VENV_DIR}
AOTRITON_ENABLE_FP32=${AOTRITON_ENABLE_FP32}
AOTRITON_DEBUG_SKIP_TRITON_KERNELS=${AOTRITON_DEBUG_SKIP_TRITON_KERNELS_ENV}
"${VENV_BIN_PYTHON}" -X utf8 -m aotriton.generate
--target_gpus ${EFFECTIVE_TARGET_GPUS}
--build_dir "${AOTRITON_V2_BUILD_DIR}"
--root_dir "${CMAKE_CURRENT_SOURCE_PARENT_DIR}"
${GENERATE_OPTION}
COMMAND_ECHO STDOUT
WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_PARENT_DIR}"
COMMAND_ERROR_IS_FATAL ANY
)

# Compile HSACO Kernels
if(NOT AOTRITON_NOIMAGE_MODE)
  set(SIGNATURE_FILE "${AOTRITON_V2_BUILD_DIR}/__signature__.json")
  add_custom_command(OUTPUT "${SIGNATURE_FILE}"
    COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR}
    "${VENV_BIN_PYTHON}"
    -m aotriton.write_image_signature
    --build_dir "${AOTRITON_V2_BUILD_DIR}"
    --root_dir "${CMAKE_CURRENT_SOURCE_PARENT_DIR}"
    --git_sha1 ${AOTRITON_GIT_SHA1}
    --git_treesha1 ${AOTRITON_GIT_TREESHA1}
    --target_arch ${AOTRITON_TARGET_ARCH}
    --output_file "${SIGNATURE_FILE}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_PARENT_DIR}"
  )

  message("Bare.compile: ${AOTRITON_V2_BUILD_DIR}/Bare.compile")
  message("Bare.cluster: ${AOTRITON_V2_BUILD_DIR}/Bare.cluster")
  set(AOTRITON_HSACO_RECORD "${AOTRITON_V2_BUILD_DIR}/Bare.targets")
  file(STRINGS "${AOTRITON_V2_BUILD_DIR}/Bare.compile" HSACO_RULES ENCODING UTF-8)
  list(LENGTH HSACO_RULES NUM_HSACO_RULES)
  message("Bare.compile ${NUM_HSACO_RULES} rules read")

  ## Use Bare.compile file to generate custom rules
  file(WRITE "${AOTRITON_HSACO_RECORD}" "")
  set(I "0")  # Progress
  foreach(RULE IN LISTS HSACO_RULES)
    # message("${RULE}")
    list(POP_FRONT RULE ALTNAME)
    list(POP_FRONT RULE COMPILER_PYTHON)
    list(POP_FRONT RULE HSACO)
    list(POP_FRONT RULE SRC)
    list(POP_FRONT RULE KNAME)
    list(POP_FRONT RULE NWARPS)
    list(POP_FRONT RULE NSTAGES)
    list(POP_FRONT RULE WAVESPEREU)
    list(POP_FRONT RULE TGTGPU)
    list(POP_FRONT RULE SIG)
    add_custom_command(OUTPUT "${HSACO}"
      COMMAND ${CMAKE_COMMAND} -E env
      TRITON_CACHE_DIR=${CMAKE_BINARY_DIR}/triton-cache.${ALTNAME}
      TRITON_F32_DEFAULT="ieee"
      TRITON_STORE_BINARY_ONLY=1
      "${COMPILER_PYTHON}"
      "${AOTRITON_COMPILER}"
      "${SRC}"
      "--kernel_name" "${KNAME}"
      "-o" "${HSACO}"
      "-g" "1,1,1"
      "--num_warps" "${NWARPS}"
      "--num_stages" "${NSTAGES}"
      "--waves_per_eu" "${WAVESPEREU}"
      "--target" "${TGTGPU}"
      "--signature" "${SIG}"
      "--timeout" "${AOTRITON_GPU_BUILD_TIMEOUT}"
      ${HSACO_COMPILE_OPTION}
      DEPENDS aotriton_venv_triton
    )
    # DO NOT USE list(APPEND). It is quadratic growth
    file(APPEND "${AOTRITON_HSACO_RECORD}" "${HSACO}" "\n")
    message(CONFIGURE_LOG "hsaco progress ${I}/${NUM_HSACO_RULES}")
  endforeach(RULE)

  # flyc: kernels compiled during the build from a FlyDSL description, via a separate
  # driver (`aotriton.flyc_compile`) and a disjoint rule file (`Fly.compile`, not
  # `Bare.compile`) with its own row shape -- see codegen/root.py:write_flyc_hsaco.
  # Deliberately NOT gated by AOTRITON_DEBUG_SKIP_TRITON_KERNELS: that flag only
  # silences the Triton image pipeline above, and flyc rows are how a skipping build
  # exercises the rest of the packaging pipeline cheaply.
  #
  # No TRITON_* env vars here -- flyc_bootstrap.setup() (invoked by flyc_compile
  # itself) resolves ROCM_PATH and sets ARCH/COMPILE_ONLY/FLYDSL_RUNTIME_ENABLE_CACHE.
  # AOTRITON_FLYDSL_KERNEL_ROOT is passed explicitly because, unlike ROCM_PATH, it has
  # NO fallback (flyc_bootstrap._find_flydsl_checkout_root raises if unset) and a ninja
  # build subprocess is not guaranteed to inherit the configure-time environment.
  #
  # Each flyc HSACO is also appended to the SAME AOTRITON_HSACO_RECORD (Bare.targets)
  # the Triton loop wrote above, so the existing aotriton_v2_compile target (DEPENDS
  # ${ALL_HSACOS}, defined below) covers both backends' hsacos -- required because
  # ADD_FROM_CLUSTER_RULES gives every per-functional .aks2 rule a blanket
  # `DEPENDS aotriton_v2_compile`, including flyc's own .aks2 rows, which are folded
  # into the EXISTING Bare.cluster/Bare.flatzip files rather than into rule files of
  # their own.
  set(ALL_FLYC_HSACOS "")
  if(EXISTS "${AOTRITON_V2_BUILD_DIR}/Fly.compile")
    file(STRINGS "${AOTRITON_V2_BUILD_DIR}/Fly.compile" FLYC_RULES ENCODING UTF-8)
    list(LENGTH FLYC_RULES NUM_FLYC_RULES)
    message("Fly.compile ${NUM_FLYC_RULES} rules read")
    foreach(RULE IN LISTS FLYC_RULES)
      list(POP_FRONT RULE VENVPYTHON)
      list(POP_FRONT RULE HSACO)
      list(POP_FRONT RULE DESC)
      list(POP_FRONT RULE KERNEL_NAME)
      list(POP_FRONT RULE TGTGPU)
      list(POP_FRONT RULE SIGNATURE)
      list(POP_FRONT RULE HINTS)
      add_custom_command(OUTPUT "${HSACO}"
        COMMAND ${CMAKE_COMMAND} -E env
        VIRTUAL_ENV=${VENV_DIR}
        AOTRITON_FLYDSL_KERNEL_ROOT=${AOTRITON_FLYDSL_KERNEL_ROOT}
        "${VENVPYTHON}" -m aotriton.flyc_compile
        "${DESC}" --kernel_name "${KERNEL_NAME}" --target "${TGTGPU}"
        --signature "${SIGNATURE}" --hints "${HINTS}"
        --out_path "${HSACO}" --timeout "${AOTRITON_GPU_BUILD_TIMEOUT}"
        # Same switch the Triton loop passes, and for the same reason: without
        # it the driver answers a timeout or an exception by writing a
        # zero-length .hsaco and exiting 0, aks2 packs the empty blob, and the
        # failure surfaces only as hipModuleLoadData on hardware.
        ${HSACO_COMPILE_OPTION}
        DEPENDS aotriton_venv_flydsl
        VERBATIM)
      list(APPEND ALL_FLYC_HSACOS "${HSACO}")
      # DO NOT USE list(APPEND) on the on-disk record. Same quadratic-growth
      # caveat as the Triton loop above.
      file(APPEND "${AOTRITON_HSACO_RECORD}" "${HSACO}" "\n")
    endforeach()
    add_custom_target(aotriton_v3_flyc_compile ALL DEPENDS ${ALL_FLYC_HSACOS})
  endif()

  ## Add a virtual target
  ### FIXME: AOTRITON_HSACO_RECORD file is duplicated from Bare.compile
  file(STRINGS "${AOTRITON_HSACO_RECORD}" ALL_HSACOS ENCODING UTF-8)
  add_custom_target(aotriton_v2_compile ALL DEPENDS ${ALL_HSACOS})

  # AKS2 intermediate output directory (not installed; used only during build).
  set(AOTRITON_AKS2_DIR "${AOTRITON_V2_BUILD_DIR}/aks2")
  execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${AOTRITON_AKS2_DIR}")

  # Kernel Storage V2: build per-functional AKS2 files under aks2/ intermediate dir.
  # Bare.cluster format: <vendor-arch> <family> <kernel> <sha256> <abs_hsaco>...
  set(ALL_AKS2 "")
  set(ALL_ARCH_DIRS "")
  macro(ADD_FROM_CLUSTER_RULES)
    foreach(RULE IN LISTS CLUSTER_RULES)
      list(POP_FRONT RULE DIR_ARCH)
      list(POP_FRONT RULE DIR_FAMILY)
      list(POP_FRONT RULE DIR_KERNEL)
      list(POP_FRONT RULE SHA256)
      string(JOIN "/" AKS2
        ${AOTRITON_AKS2_DIR}
        ${DIR_ARCH}
        ${DIR_FAMILY}
        ${DIR_KERNEL}
        "${SHA256}.aks2")
      string(JOIN "/" HSACO_MANIFEST
        ${AOTRITON_AKS2_DIR}
        ${DIR_ARCH}
        ${DIR_FAMILY}
        ${DIR_KERNEL}
        "${SHA256}.nsv")
      # message(STATUS "Add AKS2 ${AKS2}")
      add_custom_command(OUTPUT "${AKS2}"
        COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR}
        "${VENV_BIN_PYTHON}"
        -m aotriton.aks2
        -o "${AKS2}"
        ${AKS2_OPTIONS}
        --hsaco_manifest "${HSACO_MANIFEST}"
        DEPENDS aotriton_v2_compile "${HSACO_MANIFEST}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_PARENT_DIR}"
        COMMAND_EXPAND_LISTS)
      list(APPEND ALL_AKS2 "${AKS2}")
    endforeach(RULE)
  endmacro()
  file(STRINGS "${AOTRITON_V2_BUILD_DIR}/Bare.cluster" CLUSTER_RULES ENCODING UTF-8)
  set(AKS2_OPTIONS "")
  ADD_FROM_CLUSTER_RULES()
  # Affine kernels: emit AKS2 blobs to the intermediate aks2/ tree (same as Bare),
  # then pack them into the per-family affine_kernels.zip below.
  if(EXISTS "${AOTRITON_V2_BUILD_DIR}/Affine.cluster")
    file(STRINGS "${AOTRITON_V2_BUILD_DIR}/Affine.cluster" AFFINE_CLUSTER_RULES ENCODING UTF-8)
    foreach(RULE IN LISTS AFFINE_CLUSTER_RULES)
      list(POP_FRONT RULE DIR_ARCH)
      list(POP_FRONT RULE DIR_FAMILY)
      list(POP_FRONT RULE DIR_KERNEL)
      list(POP_FRONT RULE FONLY)
      string(JOIN "/" AKS2
        ${AOTRITON_AKS2_DIR}
        ${DIR_ARCH}
        ${DIR_FAMILY}
        ${DIR_KERNEL}
        "${FONLY}.aks2")
      string(JOIN "/" HSACO_MANIFEST
        ${AOTRITON_AKS2_DIR}
        ${DIR_ARCH}
        ${DIR_FAMILY}
        ${DIR_KERNEL}
        "${FONLY}.nsv")
      add_custom_command(OUTPUT "${AKS2}"
        COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR}
        "${VENV_BIN_PYTHON}"
        -m aotriton.aks2
        -o "${AKS2}"
        --ignore_json
        --hsaco_manifest "${HSACO_MANIFEST}"
        DEPENDS aotriton_v2_compile "${HSACO_MANIFEST}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_PARENT_DIR}"
        COMMAND_EXPAND_LISTS)
      list(APPEND ALL_AKS2 "${AKS2}")
    endforeach(RULE)
  endif()
  add_custom_target(aotriton_v3_aks2 ALL DEPENDS ${ALL_AKS2})

  # Pack AKS2 files into per-kernel uncompressed ZIP archives under aotriton.images/.
  # Bare.flatzip format: <vendor-arch> <family> <kernel> <abs_aks2>...
  # ZIP is STORED (no compression) so entries are readable via lseek+read at fixed offsets.
  set(ALL_ZIPS "")
  if(EXISTS "${AOTRITON_V2_BUILD_DIR}/Bare.flatzip")
    file(STRINGS "${AOTRITON_V2_BUILD_DIR}/Bare.flatzip" FLATZIP_RULES ENCODING UTF-8)
    foreach(RULE IN LISTS FLATZIP_RULES)
      list(POP_FRONT RULE DIR_ARCH)
      list(POP_FRONT RULE DIR_FAMILY)
      list(POP_FRONT RULE DIR_KERNEL)
      # Remaining RULE entries are the absolute paths of the .aks2 files that
      # feed this ZIP (see python/codegen/root.py write_cluster()). Depending
      # on the concrete inputs ensures the ZIP is rebuilt when any contributing
      # .aks2 changes, even if the .nsv manifest text is unchanged.
      set(ZIP_AKS2_DEPS ${RULE})
      set(ZIP_OUT "${AOTRITON_KERNEL_STORAGE_V2_DIR}/${DIR_ARCH}/${DIR_FAMILY}/${DIR_KERNEL}.zip")
      set(ZIP_MANIFEST "${AOTRITON_KERNEL_STORAGE_V2_DIR}/${DIR_ARCH}/${DIR_FAMILY}/${DIR_KERNEL}.nsv")
      # ZIP_MANIFEST is written by write_cluster() at generate time alongside Bare.flatzip.
      add_custom_command(OUTPUT "${ZIP_OUT}"
        COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR}
        "${VENV_BIN_PYTHON}" -m aotriton.flatzip
        --manifest "${ZIP_MANIFEST}"
        -o "${ZIP_OUT}"
        DEPENDS aotriton_v3_aks2 "${ZIP_MANIFEST}" ${ZIP_AKS2_DEPS}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_PARENT_DIR}"
        COMMAND_EXPAND_LISTS)
      list(APPEND ALL_ZIPS "${ZIP_OUT}")
      # Place __signature__ alongside the first ZIP in each arch dir.
      if(NOT "${DIR_ARCH}" IN_LIST ALL_ARCH_DIRS)
        set(IMAGE_ARCH_SIGNATURE "${AOTRITON_KERNEL_STORAGE_V2_DIR}/${DIR_ARCH}/__signature__")
        add_custom_command(OUTPUT "${IMAGE_ARCH_SIGNATURE}"
          COMMAND ${CMAKE_COMMAND} -E copy "${SIGNATURE_FILE}" "${IMAGE_ARCH_SIGNATURE}"
          DEPENDS "${ZIP_OUT}" "${SIGNATURE_FILE}"
          VERBATIM)
        list(APPEND ALL_ZIPS "${IMAGE_ARCH_SIGNATURE}")
        list(APPEND ALL_ARCH_DIRS "${DIR_ARCH}")
      endif()
    endforeach(RULE)
  endif()
  add_custom_target(aotriton_kernel_storage_v3 ALL DEPENDS ${ALL_ZIPS})
endif(NOT AOTRITON_NOIMAGE_MODE)

# The build logic is slightly different when AOTRITON_BUILD_FOR_TUNING=ON
# Some GPU kernels may fail to compile and autotune files should be re-generated accordingly
if(AOTRITON_BUILD_FOR_TUNING)
  # Regenerate with --ignore_missing_kernels
  add_custom_target(aotriton_v2_regen_shim
    COMMAND ${CMAKE_COMMAND}
    -E env VIRTUAL_ENV=${VENV_DIR}
    AOTRITON_ENABLE_FP32=${AOTRITON_ENABLE_FP32}
    AOTRITON_DEBUG_SKIP_TRITON_KERNELS=${AOTRITON_DEBUG_SKIP_TRITON_KERNELS_ENV}
    "${VENV_BIN_PYTHON}" -m aotriton.generate
    --target_gpus ${EFFECTIVE_TARGET_GPUS}
    --build_dir "${AOTRITON_V2_BUILD_DIR}"
    --root_dir "${CMAKE_CURRENT_SOURCE_PARENT_DIR}"
    ${GENERATE_OPTION}
    --build_for_tuning_second_pass
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_PARENT_DIR}")
  if(NOT AOTRITON_NOIMAGE_MODE)
    add_dependencies(aotriton_v2_regen_shim aotriton_v2_compile)
  endif(NOT AOTRITON_NOIMAGE_MODE)
endif(AOTRITON_BUILD_FOR_TUNING)

set(AOTRITON_SHIM_FLAGS "")
if(AOTRITON_NAME_SUFFIX)
  list(APPEND AOTRITON_SHIM_FLAGS "--library_suffix" "${AOTRITON_NAME_SUFFIX}")
endif()
message(STATUS "AOTRITON_SHIM_FLAGS ${AOTRITON_SHIM_FLAGS}")

add_subdirectory(common)
if(NOT AOTRITON_HIPFREE_MODE)
  add_subdirectory(hip-dependent)
endif()
