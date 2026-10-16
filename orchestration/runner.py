"""
Pipeline controller; owns a long-lived worker subprocess.
All callbacks fire on the listener thread — wrap with root.after(0, ...) for Tk safety.
"""

import multiprocessing as mp
import queue as _queue
import threading
from typing import Callable, Optional

from orchestration.worker import worker_main


class PipelineController:

    def __init__(self) -> None:
        self.on_ready:    Optional[Callable] = None
        self.on_stage:    Optional[Callable] = None
        self.on_preview:  Optional[Callable] = None
        self.on_complete: Optional[Callable] = None
        self.on_reject:   Optional[Callable] = None
        self.on_error:    Optional[Callable] = None
        self.on_cancel:   Optional[Callable] = None

        # spawn is required for PyTorch + MPS; fork breaks CUDA/MPS context reuse.
        self._ctx = mp.get_context("spawn")

        self._proc: Optional[mp.Process] = None
        self._req_q = None
        self._resp_q = None
        self._listener: Optional[threading.Thread] = None
        self._listener_stop = threading.Event()

        self._running = False  # True between run() and the terminal event
        self._ready   = False  # True once worker emits worker_ready=ready

        self._spawn_worker()

    def run(self, image_path: str, mode: str = "Clean Scan") -> None:
        if self._running:
            return
        self._running = True
        self._req_q.put({"path": str(image_path), "mode": mode})

    def repolish(self, mode: str) -> None:
        if self._running or not self._ready:
            return
        self._running = True
        self._req_q.put({"action": "repolish", "mode": mode})

    def cancel(self) -> None:
        if not self._running:
            return
        self._running = False
        self._ready = False
        self._tear_down_worker()
        if self.on_cancel:
            self.on_cancel()
        self._spawn_worker()

    def shutdown(self) -> None:
        self._running = False
        self._ready = False
        self._tear_down_worker(send_sentinel=True)

    def is_ready(self) -> bool:
        return self._ready

    def is_running(self) -> bool:
        return self._running

    def _spawn_worker(self) -> None:
        self._req_q  = self._ctx.Queue()
        self._resp_q = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=worker_main,
            args=(self._req_q, self._resp_q),
            daemon=True,
        )
        self._proc.start()

        self._listener_stop = threading.Event()
        self._listener = threading.Thread(
            target=self._listen, name="pipeline-listener", daemon=True,
        )
        self._listener.start()

    def _tear_down_worker(self, send_sentinel: bool = False) -> None:
        # Stop the listener first so it doesn't race on the queue being discarded.
        self._listener_stop.set()

        if send_sentinel and self._req_q is not None:
            try:
                self._req_q.put_nowait(None)
            except Exception:
                pass

        if self._proc is not None:
            try:
                if send_sentinel:
                    self._proc.join(timeout=2.0)
                if self._proc.is_alive():
                    self._proc.terminate()
                    self._proc.join(timeout=1.0)
            except Exception:
                pass
            self._proc = None

        self._req_q = None
        self._resp_q = None

    def _listen(self) -> None:
        # Capture locals — _spawn_worker replaces instance queues after cancel(),
        # so each listener thread must hold its own queue + stop event reference.
        resp_q = self._resp_q
        stop   = self._listener_stop
        if resp_q is None:
            return
        while not stop.is_set():
            try:
                ev = resp_q.get(timeout=0.1)
            except _queue.Empty:
                continue
            except (OSError, EOFError, ValueError):
                return

            self._dispatch(ev)

    def _dispatch(self, ev: dict) -> None:
        kind = ev.get("type")

        if kind == "worker_ready":
            if ev.get("state") == "ready":
                self._ready = True
                if self.on_ready:
                    self.on_ready()

        elif kind == "stage":
            if self.on_stage:
                self.on_stage(
                    ev["stage"], ev["state"], ev.get("message", ""),
                    ev.get("elapsed_ms"),
                )

        elif kind == "preview":
            if self.on_preview:
                self.on_preview(ev["stage"], ev["image"], ev.get("meta", {}))

        elif kind == "reject":
            self._running = False
            if self.on_reject:
                self.on_reject(ev["message"])

        elif kind == "complete":
            self._running = False
            if self.on_complete:
                self.on_complete(
                    ev["image"],
                    ev.get("text", ""),
                    ev.get("total_seconds", 0.0),
                    ev.get("cached", False),
                    ev.get("diagnostics", {}),
                )

        elif kind == "error":
            self._running = False
            if self.on_error:
                self.on_error(ev.get("message", "Unknown error"))
