# Script inventory and layout of `.def/`, `.tune/`, `.perfmon/`

Written before the move (perfmon-after-def-rev0.md section 6.4). "Old path" is
the path before the reorganisation; "New path" is where the script lives after
it. Every directory is at depth 2 (`.<app>/<role>/<file>`), so a script's
`$SCRIPT_DIR/..` is always its application root, and the relative
`$SCRIPT_DIR/../../` hops keep working.

## Layout

Top level, flat: `.def/` is everything shared by DEF applications, `.tune/`
is tune-only, `.perfmon/` is perfmon-only.

Inside each directory, the directory says where a script runs and who invokes
it:

| Directory | Runs on | Invoked by |
|---|---|---|
| `bin/` | control host | the operator (and the webui) |
| `libexec/` | control host | other programs only: `bin/` commands and the webui; helpers (shell or Python) |
| `host/` | a worker or build host, outside containers | ssh, `tsp`, SLURM, or an operator logged into that host |
| `container/` | inside a worker container | `docker exec` / `docker run` command lines, `.ci/` |
| `lib/` | sourced into another shell | shell scripts; sourced libraries only |
| `image.scripts/` | inside the container, at image build | the generated Dockerfile (data of `.def/`, name kept: `prepwkdir` copies it to `<workdir>/image.scripts`) |

Deviations from the suggested names: none in the directory names. Placement
notes:

* The per-host primitives of the old `single/` (`start_worker.sh`,
  `stop_worker.sh`, ...) run on the control host and reach the worker over ssh.
  They are `libexec/`, not `host/`: `wkctl`, `srun`, `imgbld`, `detect-gpus`
  and the webui call them; the operator does not. `host/` is only for what
  executes on the other host.
* `worker_service.sh` is in `container/` although the SLURM job
  (`host/slurm_worker_job.sh`) and `host/run_worker.sh` run it straight in a
  venv with no container. Its main use is inside the container
  (`docker exec`), and the container is the stricter environment.
* `libbld`, `testbld`, `build_arch.sh` and `build_triton_wheel.sh` run inside
  the container (`remotebld` starts them there), so they are `.tune/container/`
  rather than `.tune/bin/`.
* The webui entry point `dashboard.py` is an operator command: `.def/bin/`.

## Shared by every DEF application: `.def/`

| Old path | New path | Runs on | Called by | Apps |
|---|---|---|---|---|
| `.tune/bin/deploy` | `.def/bin/deploy` | control host | operator; webui (`DeployWorkdirCommand`) | tune, perfmon |
| `.tune/bin/wkctl` | `.def/bin/wkctl` | control host | operator; webui scheduler, `tasks.py` | tune, perfmon |
| `.tune/bin/fetchbuild` | `.def/bin/fetchbuild` | control host | operator; webui | tune, perfmon |
| `.tune/bin/initdb` | `.def/bin/initdb` | control host | operator; webui | all |
| `.tune/bin/initproj` | `.def/bin/initproj` | control host | operator | all |
| `.tune/bin/srvctl` | `.def/bin/srvctl` | control host | operator; webui | all |
| `.tune/bin/psql` | `.def/bin/psql` | control host | operator | all |
| `.tune/bin/pg_dump` | `.def/bin/pg_dump` | control host | operator | all |
| `.tune/bin/detect-gpus` | `.def/bin/detect-gpus` | control host | operator | all |
| `.tune/bin/probe-status` | `.def/bin/probe-status` | control host | operator; webui (Workers page) | all |
| `.tune/bin/srun` | `.def/bin/srun` | control host | operator | all |
| `.tune/bin/imgbld` | `.def/bin/imgbld` | control host | operator; webui | all |
| `.tune/bin/prepwkdir` | `.def/bin/prepwkdir` | control host | operator; webui | all |
| `.tune/dashboard.py` | `.def/bin/dashboard.py` | control host | operator | all |
| `.tune/single/start_worker.sh` | `.def/libexec/start_worker.sh` | control host, ssh to worker host | `wkctl`; `stopstart_worker.sh`; webui | all |
| `.tune/single/stop_worker.sh` | `.def/libexec/stop_worker.sh` | control host, ssh | `wkctl`; `stopstart_worker.sh`; webui | all |
| `.tune/single/restart_worker.sh` | `.def/libexec/restart_worker.sh` | control host, ssh | `wkctl`; webui | all |
| `.tune/single/stopstart_worker.sh` | `.def/libexec/stopstart_worker.sh` | control host | webui | all |
| `.tune/single/stopwait_worker.sh` | `.def/libexec/stopwait_worker.sh` | control host, ssh | `wkctl` | all |
| `.tune/single/detect_gpu.sh` | `.def/libexec/detect_gpu.sh` | control host, ssh | `detect-gpus`; webui | all |
| `.tune/single/get_worker_status.sh` | `.def/libexec/get_worker_status.sh` | control host, ssh | `probe-status`; webui | all |
| `.tune/single/submit_slurm_job.sh` | `.def/libexec/submit_slurm_job.sh` | control host, ssh to SLURM login node | `srun` | all |
| `.tune/single/build_image.sh` | `.def/libexec/build_image.sh` | control host, ssh + `tsp` | `imgbld`; webui | all |
| `.tune/single/sync_workdir.sh` | `.def/libexec/sync_workdir.sh` | control host (rsync to the host) | `deploy`, `remotebld`; webui | all |
| `.tune/single/start_services.sh` | `.def/libexec/start_services.sh` | control host (docker) | `srvctl` | all |
| `.tune/single/stop_services.sh` | `.def/libexec/stop_services.sh` | control host (docker) | `srvctl` | all |
| `.tune/lib/create_dockerfile.sh` | `.def/libexec/create_dockerfile.sh` | control host | `prepwkdir` (it generates a file; it is not sourced) | all |
| `.tune/remote/run_worker.sh` | `.def/host/run_worker.sh` | worker host | operator on that host | all |
| `.tune/remote/slurm_worker_job.sh` | `.def/host/slurm_worker_job.sh` | SLURM compute node | `sbatch` from `submit_slurm_job.sh` | all |
| `.tune/remote/get_status.sh` | `.def/host/get_status.sh` | worker host | `ssh bash -s` from `get_worker_status.sh` | all |
| `.tune/remote/worker_service.sh` | `.def/container/worker_service.sh` | container (or SLURM venv) | `docker exec` in `start/stop/restart/stopwait_worker.sh`; `slurm_worker_job.sh`; `run_worker.sh` | all |
| `.tune/remote/install_aotriton_pkg.sh` | `.def/container/install_aotriton_pkg.sh` | container | `start_worker.sh` (docker run payload); `bash_direct`, `testrun_direct` | all |
| `.tune/lib/category.sh` | `.def/lib/category.sh` | sourced | `deploy`, `wkctl`, `fetchbuild`, `sync_workdir.sh`, `worker_service.sh` | all |
| `.tune/lib/manifest.sh` | `.def/lib/manifest.sh` | sourced | same as category.sh; `python/test/pgdef/test_manifest.py` | all |
| `.tune/lib/paths.sh` | `.def/lib/paths.sh` | sourced | nearly every script; `.perfmon/bin/*`, `perfmon/launch_runner.sh`; `test_paths.py` | all |
| `.tune/lib/config_load.sh` | `.def/lib/config_load.sh` | sourced | every control-host script | all |
| `.tune/lib/db_query.sh` | `.def/lib/db_query.sh` | sourced | control-host scripts reading `workers.db` | all |
| `.tune/lib/sqlite3_compat.sh` | `.def/lib/sqlite3_compat.sh` | sourced | `config_load.sh`, `db_query.sh`, `detect_gpu.sh`, `testbld` | all |
| `.tune/lib/ssh_helpers.sh` | `.def/lib/ssh_helpers.sh` | sourced | (no caller found) | all |
| `.tune/lib/host_filter.sh` | `.def/lib/host_filter.sh` | sourced | `wkctl`, `detect-gpus`, `imgbld`, `runtest` | all |
| `.tune/lib/pkgroot/aotriton` (symlink to `python/`) | `.def/lib/pkgroot/aotriton` | sourced scripts | `manifest.sh` finds `aotriton.pgdef` without an install | all |
| `.tune/image.scripts/*.sh` (4 files) | `.def/image.scripts/*.sh` | container, image build | the generated Dockerfile | all |
| `.tune/webui/**` | `.def/webui/**` | control host | `dashboard.py`; `python/tune/pq/export_visperf.py` reads `static/js/perf.js` | all |
| `.tune/requirements-tuning_ui.txt` | `.def/requirements-tuning_ui.txt` | control host | pip | all |

## Tune only: `.tune/`

| Old path | New path | Runs on | Called by |
|---|---|---|---|
| `.tune/bin/bake_lut` | `.tune/bin/bake_lut` | control host | operator; webui |
| `.tune/bin/compute_best_results` | unchanged | control host | operator; `bake_lut`; webui |
| `.tune/bin/export_best_results` | unchanged | control host | operator; `bake_lut`; webui |
| `.tune/bin/recreate_materialized_view` | unchanged | control host | operator; `bake_lut`; webui |
| `.tune/bin/update_materialized_view` | unchanged | control host | operator; `bake_lut`; webui |
| `.tune/bin/query_broken_entries` | unchanged | control host | operator; `update_materialized_view` |
| `.tune/bin/sancheck` | unchanged | control host | operator; `bake_lut`; webui |
| `.tune/bin/decomposedb` | unchanged | control host | operator; `bake_lut`; webui |
| `.tune/bin/dbsum` | unchanged | control host | operator |
| `.tune/bin/manual_pick_missing_entry` | unchanged | control host | operator |
| `.tune/bin/reset_task` | unchanged | control host | operator |
| `.tune/bin/retry_broken_entries` | unchanged | control host | operator |
| `.tune/bin/retry_missing_entries` | unchanged | control host | operator |
| `.tune/bin/dispatch` | unchanged | control host | operator |
| `.tune/bin/runtest` | unchanged | control host | operator |
| `.tune/bin/remotebld` | unchanged | control host (ssh + `tsp` to the build host) | operator; webui |
| `.tune/bin/amend_sel_to_adiffs.sh` | unchanged | control host | operator |
| `.tune/bin/libbld` | `.tune/container/libbld` | container | `remotebld` (`docker run`) |
| `.tune/bin/testbld` | `.tune/container/testbld` | container | `remotebld` (`docker run`) |
| `.tune/bin/append_oom_to_adiffs.sh` | `.tune/container/append_oom_to_adiffs.sh` | container | `.ci/run-test.sh` |
| `.tune/single/build_arch.sh` | `.tune/container/build_arch.sh` | container | `libbld` |
| `.tune/single/build_triton_wheel.sh` | `.tune/container/build_triton_wheel.sh` | container | `libbld`, `testbld` |
| `.tune/single/prebuild_wheel.sh` | `.tune/host/prebuild_wheel.sh` | build host (`tsp`) | `remotebld` |
| `.tune/remote/bash_direct` | `.tune/host/bash_direct` | worker host, starts a container | operator on that host |
| `.tune/remote/testrun_direct` | `.tune/host/testrun_direct` | worker host, starts a container | operator on that host |
| `.tune/single/run-test.sh` | `.tune/libexec/run-test.sh` | control host (ssh + `tsp`) | `runtest`; webui |
| `.tune/libexec/broken_entries_to_db` | unchanged | control host | `retry_broken_entries` |
| `.tune/libexec/reset_broken_to_pending` | unchanged | control host | `retry_broken_entries` |
| `.tune/libexec/pq_helpers.py` | unchanged | control host | `reset_task`, `retry_missing_entries` |
| `.tune/lib/aotriton_version.sh` | unchanged | sourced | `testbld`, `build_arch.sh`, `prebuild_wheel.sh` |
| `.tune/lib/find_flydsl.sh` | unchanged | sourced | `testbld`, `build_arch.sh`, `prebuild_wheel.sh` |
| `.tune/lib/require_aotriton_pkg.sh` | unchanged | sourced | the Python-wrapping `bin/` commands |

## Perfmon only: `.perfmon/`

| Path | Runs on | Called by |
|---|---|---|
| `.perfmon/bin/build_subjects` | control host | operator; webui (phase 4) |
| `.perfmon/bin/dispatch` | control host | operator |
| `.perfmon/bin/export` | control host | operator |

These source `.def/lib/paths.sh` after the move. `perfmon/launch_runner.sh`
(not an operator tool; runs inside the perfmon image) does too.

## Compatibility (removed next release)

For one release, these are symlinks, so muscle memory and old docs keep
working. **They go away next release**; use the new paths.

* `.tune/bin/<name>` -> `.def/bin/<name>` for `deploy`, `wkctl`, `fetchbuild`,
  `initdb`, `initproj`, `srvctl`, `psql`, `pg_dump`, `detect-gpus`,
  `probe-status`, `srun`, `imgbld`, `prepwkdir`.
* `.tune/bin/libbld`, `.tune/bin/testbld` -> `.tune/container/<name>`
  (the tuning docs told operators to run them from `bin/`).
* `.tune/dashboard.py` -> `.def/bin/dashboard.py`.

The `.def/bin` scripts resolve their own path with `readlink -f`, so a link
finds `.def/` and not `.tune/`.

The per-host scripts of `single/` and the `remote/` scripts get no link: only
the `bin/` commands were operator muscle memory, and webui and scripts are
updated.
