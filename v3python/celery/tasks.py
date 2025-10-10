from .celery import app
from celery.signals import (
    celeryd_after_setup,
    celeryd_init,
    worker_shutting_down,
)
from celery import current_task, chain, group, chord
import os
import time
import logging
import sys
import socket
from pathlib import Path
from v3python.tune.exaid import (
    exaid_create,
    exaid_exitall,
    ExaidSubprocessNotOK,
)
import shutil

# NOTE: gethostname() may return FQDN
SHORT_HOSTNAME = socket.gethostname().split('.')[0]
GPUQ = SHORT_HOSTNAME + '_gpuqueue'
CPUQ = SHORT_HOSTNAME + '_cpuqueue'

def _stub_probe_nhsaco(kname):
    # if kname == 'attn_fwd':
    #     return 32
    # if kname == 'bwd_kernel_dk_dv':
    #     return 12
    # if kname == 'bwd_kernel_dq':
    #     return 20
    return 3

def get_exaid(task_config, hostname):
    module = task_config["module"]
    gpu_id_str = hostname.split('@')[0].split('_')[1]
    return exaid_create(module, int(gpu_id_str))

def get_exaid_with_tmpdir(task_config, hostname):
    exaid = get_exaid(task_config, hostname)
    if 'tmpdir' in task_config:
        tmpdir = Path(task_config['tmpdir'])
    else:
        tmpdir = exaid.get_tmpfs_for(task_config["entry"])
    return exaid, tmpdir

@worker_shutting_down.connect
def on_worker_shutting_down(sender, sig, how, exitcode, **kwargs):
    exaid_exitall()

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
    print(f'Enter preprocess')
    print(f'preprocess {type(task_config)=}')
    print(f'{type(current_task)=}', flush=True)
    print(f'{type(current_task.request)=}', flush=True)
    print(f'{type(current_task.request.hostname)=}', flush=True)
    worker_hostname = current_task.request.hostname
    exaid, p = get_exaid_with_tmpdir(task_config, worker_hostname)
    try:
        exaid.prepare_data(task_config["entry"], p)
        task_config['tmpdir'] = p.as_posix()
        return task_config
    except OSError as e:
        print('[exaid][prepare_data] subprocess exited with errno:',
              e.errno,
              'stderr:',
              e.strerror)
    except ExaidSubprocessNotOK as e:
        print('[exaid][prepare_data] subprocess does not report OK.',
              'stdout:', e.stdout,
              'stderr:', e.stderr,
              sep='\n')
    # TODO: Celery error handling
    return task_config

@app.task
def postprocess(task_config):
    print(f'postprocess {task_config=}')
    # # Note: Real tuning uses rm -rf not rmdir
    # #       Using rmdir is to confirm all sub tasks complete (otherwise error)
    # p.rmdir()
    worker_hostname = current_task.request.hostname
    exaid, p = get_exaid_with_tmpdir(task_config, worker_hostname)
    shutil.rmtree(p)

@app.task
def tune_hsaco(task_config, kname, hsaco_id):
    worker_hostname = current_task.request.hostname
    exaid, p = get_exaid_with_tmpdir(task_config, worker_hostname)
    try:
        result_data = exaid.benchmark(p, kname, hsaco_id)
        task_config['result'] = "OK"
        task_config['result_data'] = result_data
    except OSError as e:
        print(f'[exaid][benchmark] {task_config} {kname}={hsaco_id} subprocess exited with errno:',
              e.errno,
              'stderr:',
              e.strerror)
        task_config['result'] = "crash"
        task_config['result'] = {
            "errno" : e.errno,
            "stderr": e.strerror
        }
    except ExaidSubprocessNotOK as e:
        task_config['result'] = "NotOK"
        task_config['result'] = {
            "stdout": e.stdout,
            "stderr": e.strerror
        }
    return task_config

# @app.task
# def tune_subkernel(task_config, kname):
#     print(f"tune_subkernel {task_config=}")
#     nhsaco = probe_nhsaco(kname)
#     res = group([tune_hsaco.s(task_config, kname, hsaco_index).set(queue=GPUQ) for hsaco_index in range(nhsaco)])
#     res()

@app.task
def do_tune_kernel(task_config):
    worker_hostname = current_task.request.hostname
    exaid, p = get_exaid_with_tmpdir(task_config, worker_hostname)
    max_hsaco_dict = task_config.get("max_hsaco", {})
    max_hsaco_global = max_hsaco_dict.get("*", None)
    try:
        kernel_dict = exaid.probe(p)
        def gen():
            for kname, hsaco_list in kernel_dict.items():
                max_hsaco = max_hsaco_dict.get(kname, max_hsaco_global)
                for hsaco_index in range(len(hsaco_list[:max_hsaco])):
                    yield tune_hsaco.s(task_config, kname, hsaco_index).set(queue=GPUQ)
        res = group([sig for sig in gen()])
        res()
    except OSError as e:
        print('[exaid][probe] subprocess exited with errno:',
              e.errno,
              'stderr:',
              e.strerror)
    except ExaidSubprocessNotOK as e:
        print('[exaid][probe] subprocess does not report OK.',
              'stdout:', e.stdout,
              'stderr:', e.stderr,
              sep='\n')
    return task_config
    # TODO: Celery error handling

'''
This is a dummpy task to block
'''
@app.task
def tune_kernel_done(task_config):
    pass

@app.task
def tune_kernel(task_config):
    arch = task_config['arch']
    # worker_hostname = current_task.request.hostname
    print(f'tune_kernel {task_config=} {GPUQ=}')
    res = chain(preprocess.s(task_config).set(queue=GPUQ),
                do_tune_kernel.s().set(queue=GPUQ),
                postprocess.s().set(queue=GPUQ))
    res().get()  # TODO: add rule to only route tune_kernel to "gfx*" queues

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
