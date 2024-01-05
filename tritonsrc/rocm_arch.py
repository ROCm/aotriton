import subprocess

def rocm_get_gpuarch():
    return subprocess.check_output(['/opt/rocm/bin/offload-arch']).decode('utf8', errors='ignore').strip()
