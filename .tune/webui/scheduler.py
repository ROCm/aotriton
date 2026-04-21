# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
WorkerScheduler: Manages office hours timers for scheduled workers.
"""

import threading
import logging
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
import subprocess

logger = logging.getLogger(__name__)


class ScheduledTimer(threading.Timer):
    """Timer subclass that tracks metadata for display"""

    def __init__(self, interval, function, args=None, kwargs=None, hostname=None, action=None):
        super().__init__(interval, function, args or [], kwargs or {})
        self.hostname = hostname
        self.action = action  # 'START' or 'STOP'
        self.fire_time = datetime.now() + timedelta(seconds=interval)

    def get_remaining_seconds(self):
        """Get seconds remaining until timer fires"""
        if self.finished.is_set():
            return 0
        return max(0, (self.fire_time - datetime.now()).total_seconds())


class WorkerScheduler:
    """Manages office hours timers for all scheduled workers"""

    def __init__(self, workdir: str):
        self.workdir = workdir
        self.timers = {}  # hostname -> ScheduledTimer
        self.worker_states = {}  # hostname -> {'state': 'ALLOWED'|'BLOCKED', 'next_action': 'START'|'STOP'}

    def arm_all_scheduled_workers(self):
        """Scan database and arm timers for all workers with enabled schedules"""
        from .tasks import get_scheduled_workers

        workers = get_scheduled_workers(self.workdir)
        logger.info(f"Arming timers for {len(workers)} scheduled workers")

        for hostname, schedule in workers.items():
            self._arm_timer_for_worker(hostname, schedule)

        return len(workers)

    def _arm_timer_for_worker(self, hostname: str, schedule: dict):
        """Calculate next transition and arm timer"""
        now = datetime.now()

        # Determine current state (ALLOWED or BLOCKED)
        current_state = self._is_allowed_now(now, schedule)

        # Calculate next transition time
        next_transition = self._calculate_next_transition(now, schedule, current_state)

        # Determine action (STOP if currently allowed, START if currently blocked)
        next_action = 'STOP' if current_state else 'START'

        # Cancel existing timer if any
        if hostname in self.timers:
            self.timers[hostname].cancel()

        # Arm new timer
        delay_seconds = (next_transition - now).total_seconds()
        timer = ScheduledTimer(
            delay_seconds,
            self._execute_transition,
            args=(hostname, next_action),
            hostname=hostname,
            action=next_action
        )
        timer.start()

        self.timers[hostname] = timer
        self.worker_states[hostname] = {
            'state': 'ALLOWED' if current_state else 'BLOCKED',
            'next_action': next_action,
            'next_transition': next_transition
        }

        logger.info(f"Armed timer for {hostname}: {next_action} at {next_transition} ({delay_seconds:.0f}s)")

    def _execute_transition(self, hostname: str, action: str):
        """Execute stop/start action when timer fires"""
        logger.info(f"Timer fired for {hostname}: {action}")

        if action == 'STOP':
            self._graceful_stop_worker(hostname)
        elif action == 'START':
            self._start_worker(hostname)

        # Re-arm timer for next transition
        from .tasks import get_worker_schedule, get_default_schedule
        schedule = get_worker_schedule(self.workdir, hostname)
        if not schedule:
            logger.warning(f"No schedule found for {hostname}, not re-arming")
            return

        # Merge with defaults
        defaults = get_default_schedule(self.workdir) or {}
        for key in ['weekday_start', 'weekday_end', 'weekend_allowed']:
            if key not in schedule or not schedule[key]:
                schedule[key] = defaults.get(key)

        self._arm_timer_for_worker(hostname, schedule)

    def _graceful_stop_worker(self, hostname: str):
        """Gracefully stop worker with --graceful flag"""
        logger.info(f"Gracefully stopping {hostname} for office hours...")

        # Get aotriton root (parent of .tune)
        aotriton_root = Path(self.workdir).resolve().parent.parent
        script_path = aotriton_root / '.tune/single/stop_worker.sh'

        result = subprocess.run(
            [script_path.as_posix(), self.workdir, hostname, '--graceful'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            logger.info(f"Successfully stopped {hostname}")
        else:
            logger.error(f"Failed to stop {hostname}: {result.stderr}")

    def _start_worker(self, hostname: str):
        """Start worker when entering allowed hours"""
        logger.info(f"Starting {hostname} for allowed hours...")

        # Get aotriton root (parent of .tune)
        aotriton_root = Path(self.workdir).resolve().parent.parent
        script_path = aotriton_root / '.tune/single/start_worker.sh'

        result = subprocess.run(
            [script_path.as_posix(), self.workdir, hostname],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            logger.info(f"Successfully started {hostname}")
        else:
            logger.error(f"Failed to start {hostname}: {result.stderr}")

    def _is_allowed_now(self, now: datetime, schedule: dict) -> bool:
        """Check if current time is in allowed window"""
        is_weekday = now.weekday() < 5  # 0=Mon, 6=Sun

        if is_weekday:
            start = schedule.get('weekday_start')  # e.g., "18:00"
            end = schedule.get('weekday_end')      # e.g., "08:00"

            if not start or not end:
                return True  # No restrictions if times not set

            current_time = now.time()
            start_time = datetime.strptime(start, '%H:%M').time()
            end_time = datetime.strptime(end, '%H:%M').time()

            if start_time <= end_time:
                # Normal range (e.g., 08:00-18:00)
                return start_time <= current_time <= end_time
            else:
                # Wrap-around range (e.g., 18:00-08:00)
                return current_time >= start_time or current_time <= end_time
        else:
            # Weekend
            weekend_allowed = schedule.get('weekend_allowed', True)
            if isinstance(weekend_allowed, str):
                weekend_allowed = weekend_allowed.lower() == 'true'
            return weekend_allowed

    def _calculate_next_transition(self, now: datetime, schedule: dict, current_state: bool) -> datetime:
        """Calculate next time state will change (ALLOWED <-> BLOCKED)"""
        is_weekday = now.weekday() < 5

        weekday_start = schedule.get('weekday_start', '18:00')
        weekday_end = schedule.get('weekday_end', '08:00')
        weekend_allowed = schedule.get('weekend_allowed', True)
        if isinstance(weekend_allowed, str):
            weekend_allowed = weekend_allowed.lower() == 'true'

        start_time = datetime.strptime(weekday_start, '%H:%M').time()
        end_time = datetime.strptime(weekday_end, '%H:%M').time()

        # Find next boundary (start or end time)
        candidates = []

        # Check today
        for boundary_time in [start_time, end_time]:
            candidate = datetime.combine(now.date(), boundary_time)
            if candidate > now:
                candidates.append(candidate)

        # Check next 7 days
        for days_ahead in range(1, 8):
            future_date = now.date() + timedelta(days=days_ahead)
            future_weekday = future_date.weekday() < 5

            for boundary_time in [start_time, end_time]:
                candidate = datetime.combine(future_date, boundary_time)

                # Check if this transition changes state
                # (e.g., don't schedule weekend start if weekends already allowed)
                candidates.append(candidate)

        # Sort by time and return first one that changes state
        candidates.sort()

        for candidate in candidates:
            # Check if state at candidate time differs from current state
            # Simulate time just after candidate
            future_state = self._is_allowed_now(candidate + timedelta(seconds=1), schedule)
            if future_state != current_state:
                return candidate

        # Fallback: re-check in 1 day
        return now + timedelta(days=1)

    def get_scheduler_status(self) -> dict:
        """Get status of all armed timers for display"""
        status = {}
        for hostname, timer in self.timers.items():
            state_info = self.worker_states.get(hostname, {})
            status[hostname] = {
                'state': state_info.get('state', 'UNKNOWN'),
                'next_action': timer.action,
                'next_transition': timer.fire_time.isoformat(),
                'remaining_seconds': timer.get_remaining_seconds(),
                'armed': not timer.finished.is_set()
            }
        return status

    def cancel_all_timers(self):
        """Cancel all armed timers (for cleanup)"""
        for timer in self.timers.values():
            timer.cancel()
        self.timers.clear()
        self.worker_states.clear()
        logger.info("Cancelled all scheduled timers")
