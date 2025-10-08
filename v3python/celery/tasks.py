from .celery import app
from celery.signals import celeryd_after_setup, celeryd_init
from celery import current_task, chain, group, chord
import os
import time
import logging
import sys
import socket
from pathlib import Path
from v3python.tune.exaid import exaid_create, ExaidSubprocessNotOK
import shutil

LOCALQ = socket.gethostname() + '_localqueue'

def _stub_probe_nhsaco(kname):
    # if kname == 'attn_fwd':
    #     return 32
    # if kname == 'bwd_kernel_dk_dv':
    #     return 12
    # if kname == 'bwd_kernel_dq':
    #     return 20
    return 3

def get_exaid(task_config):
    module = task_config["module"]
    worker_hostname = current_task.request.hostname
    _, gpu_id_str = worker_hostname.split('_')
    return exaid_create(module, int(gpu_id_str))

def get_exaid_with_tmpdir(task_config):
    exaid = get_exaid(task_config)
    if 'tmpdir' in task_config:
        tmpdir = Path(task_config['tmpdir'])
    else:
        tmpdir = exaid.get_tmpfs_for(task_config["entry"])
    return exaid, tmpdir

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
    exaid, p = get_exaid_with_tmpdir(task_config)
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

@app.task
def postprocess(task_config):
    # # Note: Real tuning uses rm -rf not rmdir
    # #       Using rmdir is to confirm all sub tasks complete (otherwise error)
    # p.rmdir()
    exaid, p = get_exaid_with_tmpdir(task_config)
    shutil.rmtree(p)

@app.task
def tune_hsaco(task_config, kname, hsaco_id):
    exaid, p = get_exaid_with_tmpdir(task_config)
    try:
        result_data = exaid.benchmark(p, kname, hsaco_id)
        task_config['result'] = "OK"
        task_config['result_data'] = result_data
    except OSError as e:
        print(f'[exaid][benchmark] {config} {kname}={hsaco_id} subprocess exited with errno:',
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
            "stdout" : e.errno,
            "stderr": e.strerror
        }
    return task_config

# @app.task
# def tune_subkernel(task_config, kname):
#     print(f"tune_subkernel {task_config=}")
#     nhsaco = probe_nhsaco(kname)
#     res = group([tune_hsaco.s(task_config, kname, hsaco_index).set(queue=LOCALQ) for hsaco_index in range(nhsaco)])
#     res()

@app.task
def do_tune_kernel(task_config):
    exaid, p = get_exaid_with_tmpdir(task_config)
    try:
        kernel_dict = exaid.probe(p)
        def gen():
            for kname, hsaco_list in kernel_dict.items():
                for hsaco_index in range(len(hsaco_list)):
                    yield tune_hsaco.s(task_config, kname, hsaco_index).set(queue=LOCALQ)
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
    # TODO: Celery error handling

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
