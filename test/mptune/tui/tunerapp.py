from textual.app import App, ComposeResult
from textual.message import Message
from textual.widgets import Header, Footer, Label, ProgressBar
# from .monad import MonadWidget
from ..core import MonadAction
from threading import Thread

class TunerApp(App):
    CSS_PATH = "tunerapp.tcss"
    TITLE = "Tuner System Viewer"

    class StateChange(Message):
        def __init__(self,
                     monad_status=None,
                     text=None,
                     progress=None, # tuple[int, int]
                     ):
            super().__init__()
            self.monad_status = monad_status
            self.text = text
            self.progress = progress

    def __init__(self,
                 args,
                 watchdog,
                 info_queue: 'Queue',
                 src : 'Monad',
                 workers : 'list[Monad]',
                 dbaccessor : 'Monad'):
        super().__init__()
        self._args = args
        self._info_queue = info_queue
        self._src = src
        self._gpu_workers = workers
        self._dbaccessor = dbaccessor
        self._all_monads = [self._src] + self._gpu_workers + [self._dbaccessor]
        self._identifier_to_monad = { monad.identifier : monad for monad in self._all_monads }
        # Not canonicial in terms of finding widgets but works for any
        # Monad.identifier (textual's id has limitations on character set)
        self._thread = Thread(target=self.pull_updates)
        self._thread.start()
        self._watchdog = watchdog

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(f"Monad ID", classes='gridheader')
        yield Label(f"Status", classes='gridheader')
        yield Label(f"Current Task/Progress", classes='gridheader')
        for monad in self._all_monads:
            identifier = monad.identifier
            yield Label(f"{monad.identifier}", classes='box')
            yield Label(f"Unknown", id=f'{identifier}_status', classes='box')
            yield Label(f"N/A", id=f'{identifier}_current', classes='bar')
            yield ProgressBar(total=1, id=f'{identifier}_pbar', classes='bar')
        yield Footer()

    def on_mount(self) -> None:
        self._watchdog_timer = self.set_interval(0.2, self._watchdog)

    def on_unmount(self) -> None:
        # self._state_tracker.get_ui_update_queue().put(None)
        self._info_queue.put(None)

    def on_tuner_app_state_change(self, msg : StateChange):
        source = msg.source
        if msg.monad_status is not None:
            self.query_one(f'#{source}_status').update(msg.monad_status)
        if msg.text is not None:
            self.query_one(f'#{source}_current').update(msg.text)
        if msg.progress is not None:
            done, total = msg.progress
            self.query_one(f'#{source}_pbar').update(progress=done, total=total)

    def pull_updates(self):
        # print('Thread pull_updates started')
        info_queue = self._info_queue
        total_recv = 0
        while True:
            try:
                info = info_queue.get(timeout=0.1)
                total_recv += 1
            except:
                self.print(f'App info total_recv timedout')
            self.print(f'App info total_recv {total_recv}')
            self.print(f'App info queue get {info}')
            if info is None:
                self.print(f'App pull_updates exits')
                return
            action = info.action
            payload = info.payload
            # monad = self._identifier_to_monad[info.source]
            msg = None
            if action == MonadAction.Exit:
                msg = self.StateChange(monad_status='EXIT')
            if action == MonadAction.OOB_Init:
                msg = self.StateChange(monad_status='Running')
            if action == MonadAction.OOB_Died:
                msg = self.StateChange(monad_status=f'Died of {payload.exitcode}')
            if action == MonadAction.OOB_Restart:
                msg = self.StateChange(monad_status=f'Restarting')
            if action == MonadAction.OOB_AckRecv:
                if info.task_id is not None and payload.tup is not None:
                    msg = self.StateChange(text=f'{info.task_id:06d} {payload.tup}')
                else:
                    msg = self.StateChange(text=f'info={info}')
            if action == MonadAction.Pass:
                text = f'{info.task_id:06d} {payload.tup} {payload.profiled_kernel_name}'
                kig_dict = payload.kig_dict
                if kig_dict and payload.profiled_kernel_name in kig_dict:
                    kig = kig_dict[payload.profiled_kernel_name]
                    progress = (kig.kernel_index + 1, kig.total_number_of_kernels)
                    text += f' {progress[0]:4d}/{progress[1]:4d}'
                    text += f' Pass/Fail/VSpill/NoImage/Uncertain = {kig.passed_kernels}/{kig.failed_kernels}/{kig.vspill_kernels}/{kig.noimage_kernels}/{kig.uncertain_errors}'
                    text += f' Last adiff {kig.last_adiff}'
                else:
                    progress = None
                msg = self.StateChange(text=text, progress=progress)
            if action == MonadAction.Exception:
                self.print(info.exception)
            if msg:
                msg.source = info.source
                self.post_message(msg)

    @property
    def verbose(self):
        return self._args.verbose

    def print(self, text):
        if self.verbose:
            print(text, flush=True)
