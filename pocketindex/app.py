import asyncio
import inspect
import time
from typing import Any, Callable, List
from pocketindex.stats import UpdateStats
from . import context
from .runtime import _any_watchable, _sources_signature


class App:
    def __init__(self, name: str, main_func: Callable, **kwargs):
        self.name = name
        self.main_func = main_func
        self.kwargs = kwargs
        # Most recent run statistics, for monitoring/log cross-checking.
        self.last_stats: UpdateStats = None

    def update_blocking(
        self,
        live: bool = False,
        report_to_stdout: bool = True,
        live_interval: float = 2.0,
        full_reprocess: bool = False,
    ) -> UpdateStats:
        return asyncio.run(
            self.run_async(
                live=live,
                report_to_stdout=report_to_stdout,
                live_interval=live_interval,
                full_reprocess=full_reprocess,
            )
        )

    async def run_async(
        self,
        live: bool = False,
        report_to_stdout: bool = True,
        live_interval: float = 2.0,
        full_reprocess: bool = False,
    ) -> UpdateStats:
        # 1. Run lifespan to set up context
        builder = context.EnvironmentBuilder()
        active_managers = []
        gen = None

        if context._lifespan_func:
            # _lifespan_func is an async generator
            gen = context._lifespan_func(builder)
            try:
                await gen.__anext__()
            except StopAsyncIteration:
                pass

            # For any provided values that are async context managers, enter them
            for key_name, val in list(context._CONTEXT.items()):
                if hasattr(val, "__aenter__"):
                    entered_val = await val.__aenter__()
                    context._CONTEXT[key_name] = entered_val
                    active_managers.append((val, entered_val))

        # Sources scanned by the most recent run; live mode watches these for
        # change events (W2 push). Refreshed by every _run_once invocation.
        watched_sources: List[Any] = []

        async def _run_once(force_reprocess: bool = False) -> UpdateStats:
            stats = UpdateStats()
            context._ACTIVE_STATS.set(stats)
            context._FULL_REPROCESS.set(force_reprocess)
            scan_token = context._SCANNED_SOURCES.set([])
            started = time.monotonic()
            try:
                if inspect.iscoroutinefunction(self.main_func):
                    await self.main_func(**self.kwargs)
                else:
                    self.main_func(**self.kwargs)
            finally:
                context._ACTIVE_STATS.set(None)
                context._FULL_REPROCESS.set(False)
                watched_sources[:] = context._SCANNED_SOURCES.get() or []
                context._SCANNED_SOURCES.reset(scan_token)
            self.last_stats = stats
            if report_to_stdout:
                elapsed = time.monotonic() - started
                print(
                    f"[pocketindex] run complete in {elapsed:.2f}s\n{stats}",
                    flush=True,
                )
            return stats

        try:
            # 2. Run main function (once for catch-up, repeatedly for live mode)
            # full_reprocess applies to the catch-up pass; subsequent live polls
            # revert to incremental so we don't re-run everything every interval.
            stats = await _run_once(force_reprocess=full_reprocess)
            if live:
                # Push-style live mode: when the scanned sources expose change
                # signatures, the loop only re-runs the pipeline after an actual
                # add/edit/delete — idle periods cost just a cheap stat scan
                # instead of a full re-embedding pass. Sources without a
                # signature() fall back to the original interval polling so no
                # change is ever silently missed.
                watchable = _any_watchable(watched_sources)
                last_sig = _sources_signature(watched_sources)
                if report_to_stdout:
                    mode = "watching for changes" if watchable else "polling"
                    print(
                        f"[pocketindex] entering live mode "
                        f"({mode} every {live_interval:.1f}s; Ctrl+C to stop)",
                        flush=True,
                    )
                try:
                    while True:
                        await asyncio.sleep(live_interval)
                        if watchable:
                            current_sig = _sources_signature(watched_sources)
                            if current_sig == last_sig:
                                continue  # no change event — stay idle
                        stats = await _run_once()
                        watchable = _any_watchable(watched_sources)
                        last_sig = _sources_signature(watched_sources)
                except (KeyboardInterrupt, asyncio.CancelledError):
                    if report_to_stdout:
                        print("[pocketindex] live mode stopped.", flush=True)
            return stats
        finally:
            # 3. Clean up context managers
            for mgr, entered_val in reversed(active_managers):
                try:
                    await mgr.__aexit__(None, None, None)
                except Exception as e:
                    print(f"Error exiting context manager: {e}")

            if gen is not None:
                try:
                    await gen.__anext__()
                except StopAsyncIteration:
                    pass
