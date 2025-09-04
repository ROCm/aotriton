# Overview

This directory contains scripts to patch upstream Triton for the compiling of
AOTriton.

# Specification

## Naming

Script files are named as `patch-YYYYMMDD.sh` (GNU coreutils `date +%Y%m%d`) for chronological order.

### Timezone

Timezone is not considered here because

* New script file is only created when old `git cherry-pick` does not work.
* The date is only used to iterate script files in a sensible order.

## Content Scripts

Script files should assign:

* a variable `expect_triton_commit`
* an optional variable `triton_patch_url`
  - If unassigned, it defaults to `https://github.com/ROCm/triton.git`
* a variable `triton_patch_branch`
* assign an array `triton_cherry_picks`

The patching logic is if `expect_triton_commit` is in upstream Triton history,
`triton_patch_url` is added as a remote, `triton_patch_branch` is fetched, and
then `triton_cherry_picks` will be cherry-picked into the branch.

CAVEAT: ALL PATCHING SCRIPTS ARE SOURCED
