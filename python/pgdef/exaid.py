# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Crash-isolated subprocess runners ("exaid").

An application's GPU work runs in a child process, so a crash or a hang kills
the child and not the worker. The two halves of the line protocol live here:

    ExaidProxy         the client side. Owns a pipe to a subprocess and speaks
                       the line protocol to it. It is application-neutral: it
                       never knows what it launched, because the argv comes
                       from a `spawn_argv` callable that the application's
                       runner class supplies.

    CommandProcessor   the server side skeleton, for runners written in Python
                       (`aotriton.tune.testrun`). `command_<name>` methods are
                       the commands. `serve()` is the stdin/stdout loop.

Wire protocol: the client writes one command per line. The server answers with
one line, `OK` or `OK <json>` on success; `OVERHEATING:` lines (any text after
them) are keep-alives that carry no answer, and anything else is an error.

Runner classes themselves (e.g. `ExaidTunerWorker`) belong to
the applications and are handed to `pgdef.localq.node.run_node_role`; there is
no registry here.
"""

import errno
import json
import logging
import subprocess
import sys
from pathlib import Path

from aotriton.utils.line_reader import safe_readline

logger = logging.getLogger(__name__)

DEFAULT_KEEPALIVE_PREFIX = 'OVERHEATING:'

AOTRITON_ROOT = Path(__file__).resolve().parent.parent.parent


def first(line, sep=" "):
    seps = line.split(sep, maxsplit=1)
    if len(seps) > 1:
        return seps
    return seps[0], None


class ExaidSubprocessNotOK(RuntimeError):
    def __init__(self, stdout: str | None, stderr: str | None):
        super().__init__(stdout)
        self.stdout = stdout
        self.stderr = stderr


class ExaidProxy(object):
    """Pipe and line protocol. Knows nothing about what it launched.

    `spawn_argv` is a callable returning the argv to exec. It is called on
    every (re)launch rather than once, so a runner that changes what it wants
    to run needs no cooperation from this class beyond `shutdown()`.

    `keepalive_prefix` is the first token of a line the child may print while
    it works (thermal throttling, for example). Such lines are logged and
    skipped, and they do not count against the call's timeout budget more than
    any other waiting does.
    """

    def __init__(self, spawn_argv, base_dir: str | None = None, *,
                 keepalive_prefix: str = DEFAULT_KEEPALIVE_PREFIX):
        self._spawn_argv = spawn_argv
        self._base_dir = base_dir
        self._keepalive_prefix = keepalive_prefix
        self._process = None
        self._last_error = None

    def get_base_dir(self):
        return self._base_dir if self._base_dir is not None else AOTRITON_ROOT.as_posix()

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _reap(self):
        """Forget a process that has already exited, so the next use relaunches."""
        if self._process is None:
            return
        pid = self._process.pid
        try:
            self._process.wait(timeout=0)
        except Exception:
            pass
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            try:
                stream.close()
            except Exception:
                pass
        self._process = None
        logger.warning(f"Exaid worker process {pid} is gone; will relaunch on next use")

    @property
    def process(self):
        # Detect a process that died since we last spoke to it. Without this,
        # only readinfo() noticed a crash (it clears _process on failure) while
        # write() did not -- so after a runner was killed, every subsequent
        # write hit EPIPE against the dead handle forever and the proxy never
        # relaunched. Crash relaunch is this class's job, so the check belongs
        # here, on the one path both write() and readinfo() go through.
        if self._process is not None and self._process.poll() is not None:
            self._reap()
        if self._process is None:
            args = self._spawn_argv()
            logger.info(f"Starting exaid worker process: {' '.join(str(a) for a in args)}")
            self._process = subprocess.Popen(args,
                                             stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             cwd=self.get_base_dir())
            logger.info(f"Exaid worker process started: pid={self._process.pid}")
        return self._process

    def write(self, *objects, sep=' '):
        cmd = sep.join(str(o) for o in objects)
        proc = self.process
        logger.info(f"Sending command to worker (pid={proc.pid}): {cmd}")
        line = cmd + '\n'
        try:
            proc.stdin.write(line.encode('utf-8'))
            proc.stdin.flush()
        except (BrokenPipeError, ValueError) as e:
            # It died between the liveness check above and this write. Clear
            # the handle so the caller's next attempt starts a fresh process
            # rather than retrying against a corpse.
            self._reap()
            raise OSError(errno.EPIPE,
                          f"exaid worker died while sending {cmd!r}: {e}") from e

    def readinfo(self, *, timeout: int | float = 10):
        logger.info(f"Waiting for response from worker (pid={self.process.pid}, timeout={timeout}s)")
        while True:
            (line, eno, error_msg) = safe_readline(self.process, timeout=timeout)
            if eno != 0 or line is None:
                if eno == errno.ETIMEDOUT:
                    logger.error(f"Worker timeout after {timeout}s, killing process (pid={self._process.pid})")
                    self._process.kill()
                elif line is None:
                    logger.error(f"Worker closed stdout unexpectedly (pid={self._process.pid})")
                else:
                    logger.error(f"Worker error (pid={self._process.pid}, errno={eno}): {error_msg}")
                self._process.wait()
                stdout = self._process.stdout.read().decode('utf-8', errors='replace')
                stderr = self._process.stderr.read().decode('utf-8', errors='replace')
                logger.error(f"Worker stdout: {stdout}")
                logger.error(f"Worker stderr: {stderr}")
                del self._process
                self._process = None
                error_desc = error_msg if error_msg else "stdout closed unexpectedly"
                raise OSError(eno if eno != 0 else errno.EPIPE,
                              error_desc + "\nSTDOUT:\n" + stdout + "\nSTDERR:\n" + stderr)
            ret, info = first(line)
            if ret == self._keepalive_prefix:
                logger.warning(f"Worker keep-alive: {line}")
                continue
            if ret != "OK":
                logger.error(f"Worker returned non-OK status: {line}")
                raise ExaidSubprocessNotOK(line, error_msg)
            logger.info(f"Received response from worker (pid={self.process.pid}): {ret} {info}")
            break
        return info

    def join(self):
        if self._process is None:
            return
        pid = self._process.pid
        try:
            self._process.wait(0.2)
            logger.info(f"Worker process exited cleanly (pid={pid})")
        except subprocess.TimeoutExpired:
            logger.warning(f"Worker process did not exit in 0.2s, killing (pid={pid})")
            self._process.kill()
            self._process.wait()
            logger.info(f"Worker process killed (pid={pid})")
            del self._process
            self._process = None

    def shutdown(self):
        """Terminate the current process, if any, and forget it.

        The next `process` access relaunches, calling `spawn_argv` afresh --
        which is how a runner swaps what it is running underneath itself.

        `exit` is tried first; EOF on stdin is the fallback for a process
        already too wedged to parse a command.
        """
        if self._process is None:
            return
        pid = self._process.pid
        logger.info(f"Shutting down exaid worker process (pid={pid})")
        try:
            self.write('exit')
        except (OSError, ValueError, BrokenPipeError):
            try:
                self._process.stdin.close()   # EOF
            except Exception:
                pass
        self.join()
        if self._process is not None:
            self._process.kill()
            self._process.wait()
        self._process = None


class MaxIndentEncoder(json.JSONEncoder):
    """Pretty printer for the interactive (tty) side of `serve()`: indents the
    first `MAX_INDENT_LEVEL` levels and keeps the rest on one line."""
    MAX_INDENT_LEVEL = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._indent_unit = self.indent if isinstance(self.indent, str) else ' ' * self.indent if self.indent is not None else ''

    def encode(self, obj):
        return self._recursive_encode(obj, 0)

    def _recursive_encode(self, obj, current_indent_level, max_indent_delta=0):
        indent_str = self._indent_unit * current_indent_level

        if current_indent_level >= self.MAX_INDENT_LEVEL + max_indent_delta:
            # If max indent is reached, use compact representation.
            return json.dumps(obj, separators=(',', ':'))

        if isinstance(obj, dict):
            if not obj:
                return '{}'
            items = []
            for k, v in obj.items():
                key_str = self.default(k) if not isinstance(k, str) else json.dumps(k)
                # Hack
                max_indent_delta = 1 if k == 'adiffs' else 0
                value_str = self._recursive_encode(v, current_indent_level + 1, max_indent_delta)
                items.append(f"{indent_str}{self._indent_unit}{key_str}: {value_str}")
            return "{\n" + ",\n".join(items) + f"\n{indent_str}}}"
        elif isinstance(obj, (list, tuple)):
            if not obj:
                return '[]'
            items = []
            for item in obj:
                items.append(f"{indent_str}{self._indent_unit}{self._recursive_encode(item, current_indent_level + 1)}")
            return "[\n" + ",\n".join(items) + f"\n{indent_str}]"
        else:
            # Fall back to the default encoder for other types (primitives).
            return super().encode(obj)


class CommandProcessor(object):
    """Server side skeleton for a Python runner.

    Subclasses add `command_<name>(self, line)` methods. `line` is the rest of
    the command line after the command name, or None if there was none. A
    command returns None (answered `OK`), a str (an error message) or any
    JSON-serialisable object (answered `OK <json>`).
    """

    def __init__(self):
        self._all_commands = None

    def _gen_all_commands(self):
        PFX = 'command_'
        for attr in dir(self):
            if attr.startswith(PFX):
                yield attr.removeprefix(PFX)

    @property
    def all_commands(self):
        if self._all_commands is None:
            self._all_commands = list(self._gen_all_commands())
        return self._all_commands

    def command_help(self, line):
        print("Available Commands:")
        for command in self._gen_all_commands():
            print("\t", command)

    def process_input(self, line):
        command, tail = first(line)
        if command.lower() == 'exit':
            return 'exit'
        attr = 'command_' + command
        if not hasattr(self, attr):
            return f"Unknown Command {command}"
        return getattr(self, attr)(tail)


def serve(processor: CommandProcessor, *, stdin=None):
    """Run `processor` over stdin/stdout until `exit` or EOF.

    Interactive (tty) use gets a prompt and pretty-printed JSON. Piped use
    speaks the exaid wire protocol that `ExaidProxy` expects.
    """
    stdin = stdin if stdin is not None else sys.stdin
    if stdin.isatty():
        import readline  # noqa: F401  -- line editing for input()

        def gen_line():
            while True:
                try:
                    raw = input("> ")
                    line = raw.strip()
                    if line.lower() == "exit":
                        break
                    yield line
                except EOFError:  # Ctrl+D
                    break

        def report_ok(ret):
            if ret is None:
                return
            print(json.dumps(ret, indent=2, cls=MaxIndentEncoder))

        def report_error(error):
            print(error)
    else:
        def gen_line():
            for raw in stdin:
                line = raw.strip()
                if line.lower() == "exit":
                    break
                yield line

        def report_ok(ret):
            if ret is None:
                print('OK', flush=True)
            else:
                print('OK', json.dumps(ret, separators=(',', ':')), flush=True)

        def report_error(error):
            print("Error", flush=True)
            print(error, file=sys.stderr)
    for line in gen_line():
        ret = processor.process_input(line)
        if isinstance(ret, str):
            report_error(ret)
        else:
            report_ok(ret)
