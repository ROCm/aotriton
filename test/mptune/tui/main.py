from .tunerapp import TunerApp
from ..core import MonadAction, TuningResult, KernelIndexProress
import time

def main(info_queue : 'Queue',
         src : 'Monad',
         workers : 'list[Monad]',
         dbaccessor : 'Monad'):
    app = TunerApp(info_queue=info_queue, src=src, workers=workers, dbaccessor=dbaccessor)
    app.run()

class PseudoMessage:
    def __init__(self, task_id, source, action, payload):
        self.task_id = task_id
        self.source = source
        self.action = action
        self.payload = payload

class PseudoMonad:
    def __init__(self, identifier):
        self.identifier = identifier

def pseudo_state_tracker(q):
    for i in range(10):
        payload = TuningResult(tup=(i*3+1, i*3+2, i*3+3))
        q.put(PseudoMessage(task_id=i, source='src', action=MonadAction.Pass, payload=payload))
        time.sleep(0.1)
    worker_index = 0
    SRC = ['worker_0_on_gpu_0',
           'worker_1_on_gpu_7',
          ]
    for i in range(10):
        payload = TuningResult(tup=(i*3+1, i*3+2, i*3+3))
        payload.kig_dict = {}
        for kn in ['fwd', 'bwd_dkdv', 'bwd_dq']:
            payload.kig_dict[kn] = KernelIndexProress(kernel_index=-1, total_number_of_kernels=15)
        for kn in ['fwd', 'bwd_dkdv', 'bwd_dq']:
            payload.profiled_kernel_name = kn
            for ki in range(payload.kig_dict[kn].total_number_of_kernels):
                payload.kig_dict[kn].kernel_index = ki
                q.put(PseudoMessage(task_id=i, source=SRC[worker_index], action=MonadAction.Pass, payload=payload))
                # print(payload)
                time.sleep(0.01)
        worker_index = (worker_index + 1) % 2
    q.put(PseudoMessage(task_id=None, source=SRC[0], action=MonadAction.Exit, payload=None))
    q.put(PseudoMessage(task_id=None, source=SRC[1], action=MonadAction.Exit, payload=None))
    q.put(None)

def pseudo_main():
    from multiprocessing import Process, Queue

    q = Queue()
    p = Process(target=pseudo_state_tracker, args=(q,))
    p.start()
    src = PseudoMonad('src')
    workers = [ PseudoMonad('worker_0_on_gpu_0'),
                PseudoMonad('worker_1_on_gpu_7'),
              ]
    dbaccessor = PseudoMonad('dbaccessor')
    main(q, src, workers, dbaccessor)

if __name__ == '__main__':
    pseudo_main()
