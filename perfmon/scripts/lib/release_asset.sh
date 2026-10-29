# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Shared mechanics for pulling a release asset and installing the kernel
# images out of it. Source this from a build-<tag>.sh.
#
# WHAT IS SHARED AND WHAT IS NOT
#
# Downloading a URL and untarring it does not drift between releases, so it
# lives here. WHICH asset to ask for drifts a great deal, so it does not --
# each build-<tag>.sh names its own asset. Verified against the actual
# release pages:
#
#   0.9.2b, 0.10b   No SEPARATE images package. There is one jumbo tarball,
#                   aotriton-<tag>-manylinux_2_28_x86_64-rocm<ver>-shared.tar.gz,
#                   and the GPU images are inside it along with the runtime.
#   0.11b, 0.11.2b  aotriton-<tag>-images-amd-<group>.tar.gz, with gfx11xx as
#                   ONE group.
#   0.12.1b, 0.13b  same shape, but gfx11xx was split into gfx110x and gfx115x,
#                   and gfx1250 appeared. +asan variants also exist and must
#                   never be selected.
#   0.14.2b         same shape and groups (named after the pack directory of
#                   python/gpu_targets.py's AOTRITON_ARCH_TO_PACK).
#
# EVERY released tarball has the same internal structure: aotriton/lib/*. The
# split at 0.11b did not change that -- it moved the aotriton/lib/aotriton.images
# hierarchy into its own package while aotriton/lib/libaotriton*.so stayed in the
# runtime one. So 0.9.2b/0.10b's -shared tarball carries both, and the newer
# images tarballs carry only the images, at the same path either way. That is
# why one extraction (strip the aotriton/lib/ prefix) serves both shapes, and
# why asking for the -shared tarball is the right move for the two releases
# that predate the split rather than a workaround.
#
# Note there is no git sha in any of these names -- an earlier version of this
# code matched on one, which would never have found anything.

# Where this library lives, resolved at source time: the digest manifest sits
# beside it, and a build-<tag>.sh runs from wherever the caller happens to be.
_RELEASE_ASSET_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# _asset_expected_sha256 <asset_name> -- prints the recorded digest, or nothing
#
# Matched on the asset NAME alone, which is unique across releases because
# every one of these names carries its own tag.
_asset_expected_sha256() {
  local asset="$1"
  local manifest="${PERFMON_ASSET_SHA256:-${_RELEASE_ASSET_LIB_DIR}/asset_sha256.txt}"
  [ -f "${manifest}" ] || return 0
  awk -v want="${asset}" '$1 !~ /^#/ && $2 == want { print $1; exit }' "${manifest}"
}

# _verify_asset <file> <asset_name>
#
# An asset with no recorded digest WARNS rather than fails: a new tag has to be
# buildable before anyone has recorded its digests, and refusing would make
# adding a release a two-step dance. A recorded digest that does not match is a
# hard failure -- and takes the file with it, so a retry re-downloads instead of
# finding the bad copy sitting in the cache.
_verify_asset() {
  local file="$1" asset="$2" want got
  want="$(_asset_expected_sha256 "${asset}")"

  if [ -z "${want}" ]; then
    echo "[release_asset] WARNING: no recorded sha256 for ${asset}; not verified." >&2
    echo "[release_asset]          Record it with refresh_asset_sha256.sh to make" >&2
    echo "[release_asset]          this checked." >&2
    return 0
  fi

  got="$(sha256sum "${file}" | cut -d\  -f1)"
  if [ "${got}" != "${want}" ]; then
    echo "[release_asset] ERROR: sha256 mismatch for ${asset}" >&2
    echo "[release_asset]        expected ${want}" >&2
    echo "[release_asset]        got      ${got}" >&2
    echo "[release_asset]        Removing it. A re-uploaded release needs" >&2
    echo "[release_asset]        asset_sha256.txt refreshed and reviewed; anything" >&2
    echo "[release_asset]        else means the bytes are not what was expected." >&2
    rm -f "${file}"
    return 1
  fi
  echo "[release_asset] sha256 ok: ${asset}" >&2
}

# fetch_release_asset <tag> <asset_name> <fallback_dir> -- prints the file's PATH
#
# PERFMON_CACHE_DIR, when set, is where downloads are kept and looked for. It
# is shared across tags and arches (asset names are unique, and the images are
# arch-group- rather than arch-specific), so a tarball is fetched once for the
# whole fleet rather than once per subject -- these run to several GB for the
# pre-0.11b releases, which carry the runtime and the images in one file.
# <fallback_dir> is used when it is unset, which keeps this script usable
# standalone with no cache configured.
#
# PERFMON_IMAGES_TARBALL overrides the download entirely, for a tarball
# obtained by other means. PERFMON_RELEASE_REPO overrides the repository, for
# a mirror.
fetch_release_asset() {
  local tag="$1" asset="$2" fallback="$3"
  local repo="${PERFMON_RELEASE_REPO:-ROCm/aotriton}"

  if [ -n "${PERFMON_IMAGES_TARBALL:-}" ]; then
    # Deliberately NOT verified: this exists for a tarball obtained by other
    # means, including one built locally, which has no published digest. The
    # caller has said which file to use; second-guessing that would leave no
    # way to use the override at all.
    echo "[release_asset] using PERFMON_IMAGES_TARBALL: ${PERFMON_IMAGES_TARBALL}" >&2
    printf '%s' "${PERFMON_IMAGES_TARBALL}"
    return 0
  fi

  local dest="${PERFMON_CACHE_DIR:-${fallback}}"
  mkdir -p "${dest}"
  local target="${dest}/${asset}"

  if [ -f "${target}" ]; then
    # Verified on the way out of the cache as well as into it: the check is
    # cheap next to a multi-GB download, and it is what makes a cache entry
    # trustworthy after anything else has had a chance to touch it.
    if _verify_asset "${target}" "${asset}"; then
      echo "[release_asset] cache hit: ${target}" >&2
      printf '%s' "${target}"
      return 0
    fi
    echo "[release_asset] cached copy rejected; re-downloading" >&2
  fi

  # curl, not the gh CLI: one HTTP GET does not justify that dependency, and
  # it is not in the perfmon image. The URL is deterministic, so no API query
  # is needed either. GITHUB_TOKEN is honoured only for the rate limit.
  local url="https://github.com/${repo}/releases/download/${tag}/${asset}"
  local auth=()
  [ -n "${GITHUB_TOKEN:-}" ] && auth=(-H "Authorization: Bearer ${GITHUB_TOKEN}")

  # Download beside the target and rename on success. An interrupted curl
  # otherwise leaves a short file at the cache path, which the next run finds
  # and -- but for the verification above -- would happily untar.
  echo "[release_asset] curl ${url}" >&2
  if ! curl -fL "${auth[@]}" -o "${target}.part" "${url}"; then
    rm -f "${target}.part"
    echo "[release_asset] ERROR: could not download ${url}" >&2
    echo "[release_asset]        If that asset name is wrong for this release," >&2
    echo "[release_asset]        check https://github.com/${repo}/releases/tag/${tag}" >&2
    echo "[release_asset]        and fix this tag's build script. To bypass the" >&2
    echo "[release_asset]        download, set PERFMON_IMAGES_TARBALL=/path/to.tar.gz" >&2
    return 1
  fi
  mv "${target}.part" "${target}"

  _verify_asset "${target}" "${asset}" || return 1
  printf '%s' "${target}"
}

# install_images_from_tarball <tarball> <install_dir>
#
# Extracts just lib/aotriton.images/ and lands it at
# <install_dir>/lib/aotriton.images/. Every one of these tarballs is rooted at
# "aotriton/", so that prefix is stripped.
install_images_from_tarball() {
  local tarball="$1" install_dir="$2"

  mkdir -p "${install_dir}/lib"

  # Extract first and judge the result, rather than pre-flighting with
  # `tar tzf | grep -q`. That pre-flight looks obvious and is wrong twice
  # over: grep -q exits at its first match, tar then dies of SIGPIPE, and the
  # `set -o pipefail` every build-<tag>.sh runs under turns that into a
  # failure on a perfectly good tarball. It also decompresses the whole
  # archive an extra time -- for the pre-0.11b jumbo tarballs that is several
  # GB of pointless work before the real extraction even starts.
  local rc=0
  tar xzf "${tarball}" -C "${install_dir}/lib" \
      --strip-components=2 'aotriton/lib/aotriton.images' || rc=$?

  # Fail on a tarball that does not contain what this whole step exists for,
  # rather than "succeeding" and leaving a subject that cannot measure
  # anything. The runner loads images by dladdr, relative to the library, so a
  # missing directory would surface much later as a kernel-not-found at
  # measurement time.
  if [ "${rc}" -ne 0 ] || [ ! -d "${install_dir}/lib/aotriton.images" ]; then
    echo "[release_asset] ERROR: ${tarball} yielded no lib/aotriton.images/." >&2
    echo "[release_asset]        Contents begin:" >&2
    { tar tzf "${tarball}" | head -10 | sed 's/^/[release_asset]          /'; } >&2 || true
    return 1
  fi
  echo "[release_asset] installed ${install_dir}/lib/aotriton.images" >&2
}
