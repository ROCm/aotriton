# Comparison of .tune vs .celery scripts

## Scripts Ported:
1. ✅ initproj (create-project-directory.sh) - Commands match
2. ✅ wkctl (wkctl.sh) - Docker run commands match
3. ✅ srvctl (srvctl.sh) - Docker run commands match
4. ✅ deploy (deploy-workdir.sh) - Rsync commands match (with --mkpath added)
5. ✅ imgbld (build-worker-image.sh) - Docker build commands match
6. ✅ libbld (build-for-tuning.sh) - Triton wheel + cmake commands match
7. ✅ prepwkdir (prepare-workdir.sh) - Git clone + rsync commands match
8. ✅ srun (srun.sh) - sbatch commands match (SLURM)

## Issues Found:

### CRITICAL: worker-service.sh references
All .tune scripts reference `.celery/worker-service.sh` but should reference `.tune/remote/worker_service.sh`:
- .tune/single/start_worker.sh
- .tune/single/stop_worker.sh
- .tune/single/restart_worker.sh
- .tune/single/stopwait_worker.sh
- .tune/remote/slurm_worker_job.sh

## Scripts NOT Yet Ported:
- manage-workers.py (database management)
- install-hostos-packages.sh
- build-slurm-venv.sh
- dispatch-tasks.sh
- ssh-all.sh
- redeploy.sh
- psql-debug.sh

## Command Verification Details:

### Worker Control (wkctl)
Docker run command matches exactly between .celery/wkctl.sh and .tune/single/start_worker.sh:
- Same device mappings (--device=/dev/kfd, --device=/dev/dri)
- Same security options (--cap-add=SYS_PTRACE, --security-opt seccomp=unconfined)
- Same PYTHONPATH setting
- Same bash -c command chain

### Server Control (srvctl)
Docker run commands for RabbitMQ and PostgreSQL match exactly:
- RabbitMQ: Uses rabbitmq:4-management
- PostgreSQL: Uses configured POSTGRES_DOCKER_IMAGE with max_connections=500, shared_buffers=2GB

### Image Build (imgbld)
Docker build command matches:
- Uses tsp for task queueing
- Uses --network=host flag
- Same Dockerfile path and context

### Library Build (libbld)
Triton wheel building and cmake options match exactly:
- Triton wheel cached by git hash + Python version
- Same cmake options: DAOTRITON_TARGET_ARCH, DAOTRITON_BUILD_FOR_TUNING, etc.
- Uses ninja install/strip

### Deploy (deploy)
Rsync commands match with one addition:
- Added --mkpath flag in .tune version for safer directory creation
- Same exclusions: /build/, /installed/, /run/, /scratch/
- Same architecture-specific sync logic
