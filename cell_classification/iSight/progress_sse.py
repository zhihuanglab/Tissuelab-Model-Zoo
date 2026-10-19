"""
Per-TaskNode /progress SSE helpers (copy into each TaskNode folder).

Lifecycle:
  - /execute calls begin_execution() (resets to 0, marks active)
  - workers update state.value during the run
  - on success: mark_terminal_and_wait(100) before HTTP return
  - /execute finally: end_execution()
  - /progress clears stale terminal state only when idle
  - stream end does NOT reset (reconnects must not wipe in-flight progress)
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import AsyncIterator


class ProgressSSEState:
    """Thread-safe progress state shared by /execute and /progress."""

    def __init__(self) -> None:
        self.value = 0
        self.complete = False
        self.cancelled = False
        self.execution_active = False
        self._flush = threading.Event()
        self._clients = 0
        self._lock = threading.Lock()

    def begin_execution(self) -> None:
        with self._lock:
            self.value = 0
            self.complete = False
            self.cancelled = False
            self.execution_active = True
            self._flush.clear()

    def end_execution(self) -> None:
        self.execution_active = False

    def set_value(self, value: int) -> None:
        self.value = max(0, min(100, int(value)))

    def mark_cancelled(self) -> None:
        self.cancelled = True
        self.complete = False
        self.value = 0
        self._flush.set()

    def mark_terminal_and_wait(self, value: int = 100, timeout: float = 2.0) -> bool:
        """
        Publish terminal progress and wait until SSE has emitted it.

        Returns True if flushed (or no subscribers). False on timeout.
        AI scheduler should still publish 100% on HTTP return as a backstop.
        """
        with self._lock:
            # Clear BEFORE publishing terminal state. Clearing after complete=True
            # races with SSE notify_flushed() and can drop the handshake forever.
            self._flush.clear()
            self.value = max(0, min(100, int(value)))
            self.complete = True
            clients = self._clients
        if clients <= 0:
            return True
        return self._flush.wait(timeout=timeout)

    def notify_flushed(self) -> None:
        self._flush.set()

    def clear_stale_if_idle(self) -> bool:
        """Clear terminal leftovers only when idle. Returns True if cleared."""
        if self.execution_active:
            return False
        if self.complete or self.cancelled or self.value >= 100:
            self.value = 0
            self.complete = False
            self.cancelled = False
            return True
        return False

    def _add_client(self) -> None:
        with self._lock:
            self._clients += 1

    def _remove_client(self) -> None:
        with self._lock:
            self._clients = max(0, self._clients - 1)


async def iter_progress_events(
    state: ProgressSSEState,
    *,
    keepalive_sec: float = 2.0,
    log_prefix: str = "[SSE]",
    quiet: bool = False,
) -> AsyncIterator[dict]:
    """Async generator body for EventSourceResponse(/progress).

    quiet=True: skip per-tick Progress prints (CellCast keeps tqdm on one line).
    """
    state._add_client()
    try:
        if state.clear_stale_if_idle():
            if not quiet:
                print(f"{log_prefix} Clearing stale terminal progress before next execution.")

        last_value = -1
        last_emit = time.monotonic()

        while True:
            now = time.monotonic()
            value = state.value
            complete = state.complete
            cancelled = state.cancelled
            active = state.execution_active

            changed = (
                value != last_value
                or (value == 100 and complete)
                or (value == 0 and last_value > 0)
                or cancelled
            )
            keepalive = active and not changed and (now - last_emit) >= keepalive_sec

            if changed or keepalive:
                if changed and (last_value > value or cancelled):
                    if not quiet:
                        if cancelled:
                            print(f"{log_prefix} Task cancelled, sending reset signal")
                        else:
                            print(f"{log_prefix} Progress reset: {last_value}% -> {value}%")
                    yield {"data": str(-1)}
                if changed and not quiet:
                    print(f"{log_prefix} Progress: {value}%")
                yield {"data": str(value)}
                last_value = value
                last_emit = now

                if cancelled:
                    state.notify_flushed()
                    if not quiet:
                        print(f"{log_prefix} Task cancelled, closing connection.")
                    await asyncio.sleep(0.5)
                    break

                if value >= 100 and complete:
                    state.notify_flushed()

            # Keep waiters unblocked once 100 has been emitted at least once.
            if value >= 100 and complete and last_value == 100:
                state.notify_flushed()

            # Close after execute finishes so final 100% is not raced against HTTP return.
            if value >= 100 and complete and not active:
                if last_value != 100:
                    if not quiet:
                        print(f"{log_prefix} Progress: 100%")
                    yield {"data": "100"}
                    last_value = 100
                state.notify_flushed()
                if not quiet:
                    print(f"{log_prefix} Progress complete, closing connection.")
                await asyncio.sleep(0.5)
                break

            await asyncio.sleep(0.1)

        await asyncio.sleep(1)
        if not quiet:
            print(f"{log_prefix} Progress stream closed.")
    finally:
        state._remove_client()
