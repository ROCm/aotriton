import subprocess

def rocm_get_gpuarch():
    # return subprocess.check_output(['/opt/rocm/bin/offload-arch']).decode('utf8', errors='ignore').strip()
    out = subprocess.check_output(['rocm_agent_enumerator -name'], shell=True).decode('utf8', errors='ignore').strip()
    lines = out.splitlines()
    assert lines
    # Example: gfx942:sramecc+:xnack-
    return lines[0].split(':')[0]
