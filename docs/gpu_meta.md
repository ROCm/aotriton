# GPU Metadata Detection Plan

## Goal
Automatically detect GPU information from each worker host and store in workers.db for WebUI display.

## Data to Extract

Using `amd-smi static --json`:
- **Architecture**: GPU architecture string (gfx942, gfx90a, gfx1100, etc.)
- **PCIe IDs**: vendor:device ID pairs (e.g., `1002:74a1`)
- **GPU Count**: Number of GPUs detected on the host

## Storage Format

Store in `workers.db` config table with keys using double colon `::` as separator:
```
<hostname>::gpu::arch    = "gfx942"
<hostname>::gpu::pciid   = "1002:74a1"
<hostname>::gpu::number  = "8"
```

**Key format rationale:**
- Double colon `::` is extremely unlikely in hostnames (not RFC-valid)
- Clearly separates namespace components: `<hostname>`, `gpu`, `<field>`
- Easy to query: `SELECT * FROM config WHERE key LIKE '<hostname>::gpu::%'`
- Easy to split in code: `key.split('::')`

## Components

### 1. `.tune/single/detect_gpu.sh <workdir> <hostname>`

Detection script that:
1. Loads config.rc to get `CELERY_WORKER_IMAGE`
2. SSHs to hostname
3. Runs Docker container with the worker image:
   ```bash
   docker run --rm \
     --device=/dev/kfd \
     --device=/dev/dri \
     --group-add video \
     "$CELERY_WORKER_IMAGE" \
     amd-smi static --json
   ```
4. Parses JSON output (using Python on dev node)
5. Updates `workers.db` config table with `<hostname>::gpu::*` keys

**JSON parsing:**
```python
data = json.loads(output)
gpu_count = len(data['gpu'])
first_gpu = data['gpu'][0]
arch = first_gpu['asic']['target_graphics_version']
vendor_id = first_gpu['asic']['vendor_id']
device_id = first_gpu['asic']['device_id']
```

**Assumptions:**
- All GPUs on a host are the same model (use first GPU for arch/pciid)
- Worker image has amd-smi available
- Worker image doesn't need to be "started" - just run ephemeral container

### 2. `.tune/bin/detect-gpus <workdir> [--host <hostname>...]`

Batch wrapper that:
- Queries `workers.db` for all registered workers
- Calls `detect_gpu.sh` for each hostname (or specified hosts only)
- Displays summary table

**Output format:**
```
Detecting GPU info for 3 workers...

gpu-01.example.com:
  Architecture: gfx942
  PCIe ID: 1002:74a1
  GPU Count: 8

gpu-02.example.com:
  Architecture: gfx90a
  PCIe ID: 1002:740c
  GPU Count: 4

gpu-03.example.com:
  Error: Failed to detect GPUs

Summary:
  Success: 2/3
  Failed: 1/3
```

### 3. WebUI Integration (Future)

**UI Features:**
- Trigger detection for all/selected workers (button: "Detect GPUs")
- Display GPU metadata in worker list table
- Auto-populate arch field when adding workers
- Warning if detected arch doesn't match registered arch

**API Endpoints:**
- `POST /api/workers/detect-gpu` - Trigger detection (accepts hostnames)
- `GET /api/workers/<hostname>/gpu-info` - Retrieve stored GPU metadata

## Implementation Steps

1. ✅ Write plan to `docs/gpu_meta.md`
2. **Review plan with user**
3. Create `.tune/single/detect_gpu.sh`
4. Create `.tune/bin/detect-gpus`
5. Test on real hardware
6. Update `docs/How To Generate Tuning Database.md` with GPU detection step
7. WebUI integration (separate task)

## Error Handling

The detection script should handle and report:
- Worker unreachable via SSH
- Docker image not built on worker
- amd-smi not available in image
- Docker can't access GPU devices (/dev/kfd, /dev/dri)
- amd-smi returns invalid/unparseable JSON

Since detection is triggered manually (not automated), retry logic is not needed. Errors should be displayed to the user for manual investigation.

## Testing Plan

1. Test on single worker with known GPU (gfx942)
2. Verify JSON parsing extracts correct fields
3. Verify database storage
4. Test batch detection across multiple workers
5. Test error cases:
   - Worker unreachable via SSH
   - Docker image not built
   - amd-smi fails
6. Test with different GPU architectures (gfx90a, gfx1100)

## Future Enhancements

- Detect other metadata: VRAM size, temperature limits, power caps
- Detect NUMA topology for multi-GPU systems
- Validate detected arch matches tuning results in database
- Auto-update arch in workers table if mismatch detected
