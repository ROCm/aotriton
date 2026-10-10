# Tuner v3.5 Notes

See `DESIGN.md` for the architecture documentation. This file collects
operational notes and known issues.

# Known Issues

## Workers showing on all GPUs from `amd-smi process`

On a multi-GPU host, every worker process appears in the `amd-smi process`
output of **every** GPU, not just the one it tunes on. With 8 workers on an
8-GPU host, `.tune/remote/get_status.sh` reports `8/8/8/8/8/8/8/8` instead of
`1/1/1/1/1/1/1/1`.

Only the GPU actually in use reports a nonzero `vram_mem`; the other rows are
all `0 B`.

### Root cause

This is expected upstream KFD behaviour, not a misconfiguration. A single
`open("/dev/kfd", O_RDWR)` makes the driver create a `kfd_process_device`
(pdd) for *every* GPU in the topology. The kernel path is
`kfd_open()` -> `kfd_create_process()` -> `create_process()` ->
`kfd_init_apertures()`, whose device loop is (`kfd_flat_memory.c`):

```c
	/*Iterating over all devices*/
	while (kfd_topology_enum_kfd_devices(id, &dev) == 0) {
		if (!dev || kfd_devcgroup_check_permission(dev)) {
			/* Skip non GPU devices and devices to which the
			 * current process have no access to. Access can be
			 * limited by placing the process in a specific
			 * cgroup hierarchy
			 */
			id++; continue;
		}
		pdd = kfd_create_process_device_data(dev, process);
```

The only filters are "is a CPU node" and **the device cgroup**. Actual GPU use
is never consulted. Each pdd gets its own `vram_<gpu_id>`, `stats_<gpu_id>`
and `sdma_<gpu_id>` entries under `/sys/class/kfd/kfd/proc/<host_pid>/`, and
`amdsmi_get_gpu_process_list_by_pid()` reads exactly those, so the process
surfaces on all GPUs.

Verified by a process that only calls `os.open("/dev/kfd")` — no ioctl, no
HIP, no render node, no torch. It gets 8 pdd entries at the `open()` itself
and shows up on all 8 GPUs. The behaviour is identical in mainline Linux and
in ROCm's `ROCK-Kernel-Driver`.

A secondary (non-causal) contributor: ROCr opens **all** `/dev/dri/renderD*`
nodes at init regardless of any device mask.

### Environment variables do not fix this

None of the usual masks help. All were measured on 8x gfx942 (ROCm 7.14,
torch 2.12.0+rocm7.14.0, kernel 6.8) with the process allocating on one GPU:

| Setting | `torch.cuda.device_count()` | pdd entries | GPUs with nonzero `vram_mem` |
|---|---|---|---|
| *(none)* | 8 | 8 | 1 |
| `ROCR_VISIBLE_DEVICES=3` | 1 | 8 | 1 |
| `HIP_VISIBLE_DEVICES=3` | 1 | 8 | 1 |
| `GPU_DEVICE_ORDINAL=3` | 8 (ignored) | 8 | 1 |
| `CUDA_VISIBLE_DEVICES=3` | 1 | 8 | 1 |

The masks are applied by ROCr in userspace when enumerating HSA agents, well
after `hsaKmtOpenKFD()` has already opened `/dev/kfd` and registered the
process against every GPU. AMD's GPU isolation documentation makes the same
point: these variables "shouldn't be used for isolating untrusted
applications".

### TODO

* [tune] The one mechanism that *does* work is the **device cgroup**, since
  `kfd_devcgroup_check_permission()` keys on the render node minor. Giving a
  worker container only its own GPU:

  ```
  docker run --device=/dev/kfd --device=/dev/dri/renderD<N> ...
  ```

  produces exactly one pdd and one row in `amd-smi process`. This works on
  cgroup v2 hosts as well — `devcgroup_check_permission()` runs the
  `BPF_CGROUP_RUN_PROG_DEVICE_CGROUP()` filter that runc uses to implement
  `--device`. It is defeated by `--privileged` or by mapping all of
  `/dev/dri`, which is what the current worker containers do. Investigate
  whether per-GPU device mapping is workable for the worker containers.
