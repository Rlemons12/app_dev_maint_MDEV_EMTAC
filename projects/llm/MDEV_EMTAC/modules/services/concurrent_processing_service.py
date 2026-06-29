from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Any, List, Optional, Dict, Tuple

from modules.configuration.log_config import (
    debug_id,
    info_id,
    warning_id,
    error_id,
    with_request_id,
    get_request_id,
)


class ConcurrentProcessingService:
    """
    ThreadPool wrapper.

    IMPORTANT:
    - Your worker MUST create its own DB session/transaction.
      Do NOT share SQLAlchemy sessions across threads.

    This service just runs callables and collects results.
    """

    @with_request_id
    def run(
        self,
        *,
        items: List[Any],
        worker: Callable[[Any], Any],
        max_workers: int = 4,
        request_id: Optional[str] = None,
    ) -> List[Tuple[bool, Any]]:
        rid = request_id or get_request_id()

        if not items:
            return []

        max_workers = max(1, min(max_workers, len(items)))
        info_id(f"[CONCURRENT] running items={len(items)} max_workers={max_workers}", rid)

        results: List[Tuple[bool, Any]] = [(False, None)] * len(items)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(worker, item): idx
                for idx, item in enumerate(items)
            }

            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    value = future.result()
                    results[idx] = (True, value)
                except Exception as e:
                    error_id(f"[CONCURRENT] worker failed idx={idx} err={e}", rid)
                    results[idx] = (False, e)

        ok = sum(1 for r in results if r[0])
        warning_id(f"[CONCURRENT] completed ok={ok}/{len(items)}", rid) if ok != len(items) else debug_id(
            f"[CONCURRENT] completed ok={ok}/{len(items)}", rid
        )

        return results