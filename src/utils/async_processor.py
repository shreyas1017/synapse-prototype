"""
Runs heavy tasks (OCR, captioning) on background threads
so the main loop never freezes.
"""

import threading
from src.utils.logger import logger


class AsyncProcessor:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self._last_result = None
        self._is_running = False
        logger.info("[ASYNC] Background processor initialized")

    def run(self, fn, *args, callback=None):
        """
        Run fn(*args) on a background thread.
        Optionally call callback(result) when done.
        Ignores call if a task is already running.
        """
        if self._is_running:
            logger.debug("[ASYNC] Task already running, skipping")
            return

        def _worker():
            self._is_running = True
            try:
                result = fn(*args)
                with self._lock:
                    self._last_result = result
                if callback:
                    callback(result)
            except Exception as e:
                logger.error(f"[ASYNC] Error in background task: {e}")
            finally:
                self._is_running = False

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def get_last_result(self):
        """Return last completed result (non-blocking)."""
        with self._lock:
            return self._last_result

    @property
    def is_running(self):
        return self._is_running
