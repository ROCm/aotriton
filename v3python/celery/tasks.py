from .celery import app
from celery.signals import celeryd_after_setup, celeryd_init
from celery import current_task, chain, group, chord
import os
import time
import logging
import sys
import socket
from pathlib import Path

LOCALQ = socket.gethostname() + '_localqueue'

def _stub_probe_nhsaco(kname):
    # if kname == 'attn_fwd':
    #     return 32
    # if kname == 'bwd_kernel_dk_dv':
    #     return 12
    # if kname == 'bwd_kernel_dq':
    #     return 20
    return 3

probe_nhsaco = _stub_probe_nhsaco

@app.task
def add(x, y):
    return x + y

@app.task
def mul(x, y):
    return x * y

@app.task
def xsum(numbers):
    return sum(numbers)

@app.task
def preprocess(task_config):
    godel_number = task_config["godel_number"]
    p = Path('/dev/shm/aotriton-tuner') / str(godel_number)
    p.mkdir(parents=True, exist_ok=False)  # exist_ok=False for debugging
    return task_config

@app.task
def postprocess(task_config):
    godel_number = task_config["godel_number"]
    p = Path('/dev/shm/aotriton-tuner') / str(godel_number)
    # Note: Real tuning uses rm -rf not rmdir
    #       Using rmdir is to confirm all sub tasks complete (otherwise error)
    p.rmdir()

@app.task
def tune_hsaco(task_config, kname, hsaco_id):
    godel_number = task_config["godel_number"]
    fn = f'/dev/shm/aotriton-tuner/{godel_number}-{kname}-{hsaco_id}.run'
    Path(fn).touch()
    time.sleep(0.1)
    Path(fn).unlink()
    d = {
        'node'      : socket.gethostname(),
        'worker'    : current_task.request.hostname,
        'runfile'   : fn,
    }
    return d

@app.task
def tune_subkernel(task_config, kname):
    print(f"tune_subkernel {task_config=}")
    nhsaco = probe_nhsaco(kname)
    res = group([tune_hsaco.s(task_config, kname, hsaco_index).set(queue=LOCALQ) for hsaco_index in range(nhsaco)])
    res()

@app.task
def do_tune_kernel(task_config):
    sub_kernels = task_config['sub_kernels']
    res = group([tune_subkernel.s(task_config, kname).set(queue=LOCALQ) for kname in sub_kernels])
    res()
    return task_config

@app.task
def tune_kernel(task_config):
    # worker_hostname = current_task.request.hostname
    res = chain(preprocess.s(task_config).set(queue=LOCALQ),
                do_tune_kernel.s().set(queue=LOCALQ),
                postprocess.s().set(queue=LOCALQ))
    res()

@app.task
def get_worker_name_task():
    worker_hostname = current_task.request.hostname
    return worker_hostname

def route_task(name, args, kwargs, options, task=None, **kw):
    arch = kwargs['arch']
    d = {
        'queue': arch,
        'routing_key': f'{arch}.{name}',
    }
    return d
