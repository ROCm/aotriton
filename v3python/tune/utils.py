import sys
from dataclasses import fields
import json
import dacite
import subprocess
import select
import errno

def safeload(s):
    return json.loads(s) if s else None

def parse_python(line: str) -> dict:
    args = line.split(';')
    d = {}
    for assignment in args:
        # print(f'{assignment=}', file=sys.stderr)
        k, v = assignment.split('=', maxsplit=1)
        d[k] = eval(v)
    return d

def asdict_shallow(obj) -> dict:
    return {field.name: getattr(obj, field.name) for field in fields(obj)}

dacite_tuple = dacite.Config(cast=[tuple])

def safe_readline(
    process: subprocess.Popen,
    timeout: float = 5.0
) -> tuple[str|None, int, str|None]:
    """
    Defensively read one line from subprocess stdout with timeout and crash protection.

    Args:
        process: subprocess.Popen instance with stdout=subprocess.PIPE
        timeout: Maximum time to wait for a line (seconds)

    Returns:
        Tuple of (line, exitcode, stderr) where:
        - line: The output line if successfully read, None otherwise
        - exitcode: Process exitcode if crashed, errno.ETIMEDOUT if timedout, 0 otherwise
        - error_msg: stderr if any problem occurred, None if successful

    CAVEAT: This function still blocks with partial lines.
            Although it is good enough for our use.

    Note: do not support Win32
    """
    assert sys.platform != 'win32'
    def check_crash():
        return_code = process.poll()
        if return_code is not None and return_code != 0:
            try:
                error_msg = process.stderr.read() if process.stderr else ""
            except Exception as e:
                error_msg = "Could not read stderr: {e}"
            return (None, return_code, error_msg)
        return None
    # First check if process has already crashed
    ret = check_crash()
    if ret is not None:
        return ret

    # Platform-specific reading
    try:
        line = _read_line_unix(process, timeout)
        # Check if process crashed during read
        ret = check_crash()
        if ret is not None:
            return ret
        if line is None:
            return (None, errno.ETIMEDOUT, f"Timeout after {timeout} seconds waiting for output")
        return (line.rstrip('\n\r'), 0, None)
    except Exception as e:
        return (None, -1,  f"Unexpected error reading from subprocess: {e}")


def _read_line_unix(process: subprocess.Popen, timeout: float) -> str|None:
    """
    Read one line with timeout on Unix systems using select.

    CAVEAT: This function still blocks with partial lines.
            Although it is good enough for our use.
    """
    if not process.stdout:
        raise ValueError("Process stdout is None")

    ready, _, _ = select.select([process.stdout], [], [], timeout)

    if ready:
        line = process.stdout.readline()
        return line if line else None

    return None  # Timeout


