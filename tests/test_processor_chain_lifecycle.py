# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

"""
Regression tests for the global processor-chain alias lifecycle:

- observable chain generations and context-local chain views,
- reentrant, nested and concurrently isolated capture_logs,
- fresh built-in processor instances after reset_defaults(),
- stable ConsoleRenderer layout snapshots under hot reconfiguration,
- ProcessorFormatter foreign-record isolation,
- cache_logger_on_first_use clear-context semantics of new().
"""

from __future__ import annotations

import asyncio
import logging
import threading

from copy import copy

import pytest

import structlog

from structlog import (
    ProcessorChain,
    get_processor_chain,
    get_processor_generation,
)
from structlog.dev import Column, ConsoleRenderer
from structlog.stdlib import ProcessorFormatter
from structlog.testing import LogCapture, capture_logs


@pytest.fixture(autouse=True)
def _reset_structlog():
    structlog.reset_defaults()
    yield
    structlog.reset_defaults()


def _passthrough(logger, name, event_dict):
    return event_dict


# ---------------------------------------------------------------------------
# Generations and aliases
# ---------------------------------------------------------------------------


class TestChainGenerations:
    def test_configure_installs_generation_aware_alias(self):
        """
        get_processor_chain() returns the same list instance that is handed
        to loggers bound without explicit processors.
        """
        structlog.configure(processors=[_passthrough])

        chain = get_processor_chain()
        assert isinstance(chain, ProcessorChain)
        assert chain is structlog.get_config()["processors"]

        bound = structlog.get_logger()
        assert bound.bind()._processors is chain

    def test_configure_advances_generation(self):
        g0 = get_processor_generation()

        structlog.configure(processors=[_passthrough])

        assert get_processor_generation() != g0
        assert get_processor_generation() == get_processor_chain().generation

    def test_reset_defaults_advances_generation(self):
        structlog.configure(processors=[_passthrough])
        g_configured = get_processor_generation()

        structlog.reset_defaults()

        g_reset = get_processor_generation()
        assert g_reset != g_configured
        assert isinstance(get_processor_chain(), ProcessorChain)

    def test_chain_is_still_a_plain_list(self):
        chain = get_processor_chain()

        assert isinstance(chain, list)
        assert len(chain) == 6
        assert chain[-1] in chain
        assert list(reversed(chain)) == list(chain)[::-1]


# ---------------------------------------------------------------------------
# capture_logs: generations, nesting, concurrency
# ---------------------------------------------------------------------------


class TestCaptureLifecycle:
    def test_already_bound_logger_is_still_captured(self):
        """
        #408 contract: a logger bound *before* capturing and holding the
        configured chain reference must still be captured.
        """
        log = structlog.get_logger().bind(pre="yes")

        with capture_logs() as entries:
            log.info("hi", x=1)

        assert entries == [
            {"event": "hi", "pre": "yes", "x": 1, "log_level": "info"}
        ]

    def test_capture_advances_and_restores_generation(self):
        g_before = get_processor_generation()

        with capture_logs() as entries:
            g_inside = get_processor_generation()
            assert g_inside != g_before
            structlog.get_logger().info("a")

        assert get_processor_generation() == g_before
        assert [e["event"] for e in entries] == ["a"]

    def test_nested_captures_have_distinct_generations_and_entries(self):
        g_outer_before = get_processor_generation()
        log = structlog.get_logger()

        with capture_logs() as outer:
            g_outer = get_processor_generation()
            log.info("outer-1")

            with capture_logs() as inner:
                g_inner = get_processor_generation()
                assert g_inner != g_outer
                log.info("inner-only")
                assert [e["event"] for e in inner] == ["inner-only"]

            # Inner generation popped, outer generation visible again.
            assert get_processor_generation() == g_outer
            log.info("outer-2")
            assert [e["event"] for e in outer] == ["outer-1", "outer-2"]

        assert get_processor_generation() == g_outer_before

    def test_physical_chain_is_not_mutated(self):
        """
        Capturing installs a view; the configured processors survive entry
        and exit unchanged and keep their list identity.
        """
        structlog.configure(
            processors=[_passthrough, structlog.testing.ReturnLogger()]
        )
        chain = get_processor_chain()
        physical = list(chain)

        with capture_logs() as entries:
            # The *view* resolves to the capture processors only.
            assert len(chain) == 1
            assert isinstance(chain[-1], LogCapture)
            structlog.get_logger().bind().msg("captured")

        assert len(entries) == 1
        assert list(get_processor_chain()) == physical
        assert get_processor_chain() is chain

    def test_concurrent_thread_captures_are_isolated(self):
        """
        Two threads capturing at the same time must each only see their own
        events; their temporary chains must not cross.
        """
        results = {}
        barrier = threading.Barrier(2)

        def worker(name):
            with capture_logs() as entries:
                log = structlog.get_logger()
                barrier.wait()
                for i in range(50):
                    event = f"{name}-{i}"
                    log.info(event)
                results[name] = entries

        t1 = threading.Thread(target=worker, args=("t1",))
        t2 = threading.Thread(target=worker, args=("t2",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(results["t1"]) == 50
        assert len(results["t2"]) == 50
        assert all(e["event"].startswith("t1-") for e in results["t1"])
        assert all(e["event"].startswith("t2-") for e in results["t2"])
        # The production chain never received a LogCapture physically.
        assert not any(
            isinstance(p, LogCapture) for p in get_processor_chain()
        )

    def test_thread_without_capture_uses_production_chain(self):
        """
        A capture in one thread must not redirect loggers running in another
        thread that never captured.
        """
        physical_calls = []

        def physical_processor(logger, name, event_dict):
            physical_calls.append(event_dict["event"])
            return "ok"

        structlog.configure(
            processors=[physical_processor],
            wrapper_class=structlog.BoundLogger,
            logger_factory=structlog.PrintLoggerFactory(),
        )
        shared = structlog.get_logger()
        seen = {}

        def uncaptured():
            shared.msg("from-other-thread")
            seen["done"] = True

        with capture_logs() as entries:
            t = threading.Thread(target=uncaptured)
            t.start()
            t.join()
            shared.msg("captured-here")

        assert seen == {"done": True}
        assert physical_calls == ["from-other-thread"]
        assert [e["event"] for e in entries] == ["captured-here"]

    def test_concurrent_asyncio_task_captures_are_isolated(self):
        physical_calls = []

        def physical_processor(logger, name, event_dict):
            physical_calls.append(event_dict["event"])
            return "ok"

        structlog.configure(
            processors=[physical_processor],
            wrapper_class=structlog.BoundLogger,
            logger_factory=structlog.PrintLoggerFactory(),
        )

        async def task(name):
            with capture_logs() as entries:
                log = structlog.get_logger()
                for i in range(10):
                    event = f"{name}-{i}"
                    log.msg(event)
                    await asyncio.sleep(0)
                return entries

        async def uncaptured_task():
            log = structlog.get_logger()
            for i in range(3):
                event = f"plain-{i}"
                log.msg(event)
                await asyncio.sleep(0)

        async def main():
            r1, r2, _ = await asyncio.gather(
                task("a"), task("b"), uncaptured_task()
            )
            return r1, r2

        a, b = asyncio.run(main())

        assert [e["event"] for e in a] == [f"a-{i}" for i in range(10)]
        assert [e["event"] for e in b] == [f"b-{i}" for i in range(10)]
        assert sorted(physical_calls) == [f"plain-{i}" for i in range(3)]

    def test_explicit_processors_are_not_redirected(self):
        """
        Views only overlay the configured chain; loggers with their own
        processor list keep it (parity with the pre-view behavior).
        """
        own = LogCapture()
        log = structlog.wrap_logger(
            structlog.PrintLoggerFactory(),
            processors=[own],
            wrapper_class=structlog.BoundLogger,
        )

        with capture_logs() as entries:
            log.msg("own-chain")

        assert [e["event"] for e in own.entries] == ["own-chain"]
        assert entries == []


# ---------------------------------------------------------------------------
# Fresh built-in instances after reset_defaults()
# ---------------------------------------------------------------------------


class TestFreshBuiltins:
    def test_reset_replaces_polluted_active_renderer(self):
        """
        State hot-changed on the active ConsoleRenderer must not survive
        reset_defaults().
        """
        renderer = ConsoleRenderer.get_active()
        renderer.pad_event_to = 99
        renderer.columns = [
            Column("", lambda k, v: f"d={k}={v}"),
            Column("custom", lambda k, v: "CUSTOM"),
        ]

        structlog.reset_defaults()

        fresh = ConsoleRenderer.get_active()
        assert fresh is not renderer
        assert fresh.pad_event_to != 99

        fresh_default = ConsoleRenderer(colors=renderer.colors)
        assert [c.key for c in fresh.columns] == [
            c.key for c in fresh_default.columns
        ]

    def test_reset_installs_fresh_stateful_processors(self):
        before = list(get_processor_chain())

        structlog.reset_defaults()

        after = list(get_processor_chain())
        assert [type(p) for p in before] == [type(p) for p in after]
        # Stateful built-ins are new instances; module-level functions stay.
        for old, new in zip(before, after, strict=True):
            if isinstance(old, ConsoleRenderer) or old.__class__.__name__ in (
                "TimeStamper",
                "StackInfoRenderer",
            ):
                assert new is not old

    def test_production_path_observes_get_active_instance(self):
        """
        The renderer used by the production info path is the same object
        get_active() returns; hot changes are visible to both.
        """
        structlog.configure(
            processors=[ConsoleRenderer()],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.testing.ReturnLoggerFactory(),
        )
        renderer = ConsoleRenderer.get_active()
        assert renderer is get_processor_chain()[-1]
        log = structlog.get_logger()

        renderer.colors = True
        assert "\x1b[" in log.info("hello")

        renderer.colors = False
        assert "\x1b[" not in log.info("hello")

    def test_renderer_layout_snapshot_is_stable_within_call(self):
        """
        A column formatter that swaps the whole layout mid-__call__ cannot
        make that same call render a half-old/half-new layout: the layout is
        snapshotted once at entry.
        """
        renderer = ConsoleRenderer(colors=False)

        def old_default(key, value):
            return f"OLD<{key}={value}>"

        def new_default(key, value):
            return f"NEW<{key}={value}>"

        new_layout = [
            Column("", new_default),
            Column("b", lambda k, v: "NEW-B"),
        ]

        class SwappingColumn:
            def __init__(self):
                self.swapped = False

            def __call__(self, key, value):
                if not self.swapped:
                    self.swapped = True
                    renderer.columns = new_layout
                return f"SWAP<{key}={value}>"

        swap = SwappingColumn()
        renderer.columns = [
            Column("", old_default),
            Column("a", swap),
        ]

        out = renderer(None, "info", {"event": "e", "a": 1, "b": 2})

        assert "SWAP<a=1>" in out
        # The leftover kv pair must still be rendered by the snapshot's
        # default column, never by the newly installed one.
        assert "OLD<b=2>" in out
        assert "NEW<" not in out
        assert "NEW-B" not in out

    def test_concurrent_reconfigure_and_render_never_breaks(self):
        """
        Stress: hot layout changes in one thread can't make a render in
        another thread raise or return a non-string.
        """
        renderer = ConsoleRenderer(colors=False)
        stop = threading.Event()
        errors = []

        layouts = [
            [Column("", lambda k, v: f"{k}={v}")],
            [
                Column("", lambda k, v: f"<{k}:{v}>"),
                Column("event", lambda k, v: f"*{v}*"),
            ],
        ]

        def mutator():
            i = 0
            while not stop.is_set():
                renderer.columns = layouts[i % 2]
                renderer.colors = bool(i % 2)
                i += 1

        def emitter():
            try:
                while not stop.is_set():
                    out = renderer(
                        None, "info", {"event": "e", "n": 1, "x": "y"}
                    )
                    assert isinstance(out, str)
            except Exception as e:  # pragma: no cover - concurrency guard
                errors.append(e)

        threads = [
            threading.Thread(target=mutator),
            *[threading.Thread(target=emitter) for _ in range(3)],
        ]
        for t in threads:
            t.start()
        stop.wait(0.5)
        stop.set()
        for t in threads:
            t.join()

        assert errors == []


# ---------------------------------------------------------------------------
# ProcessorFormatter foreign-record isolation
# ---------------------------------------------------------------------------


class _StatefulProcessor:
    def __init__(self):
        self.calls = 0
        self.seen = []

    def __call__(self, logger, name, event_dict):
        self.calls += 1
        marker = event_dict.get("_from_structlog", False)
        self.seen.append(marker)
        if "mutated" not in event_dict:
            event_dict["mutated"] = []
        event_dict["mutated"].append(marker)
        return event_dict


def _render(logger, name, event_dict):
    event_dict.pop("mutated", None)
    return event_dict.get("event", "")


class TestProcessorFormatterIsolation:
    def _formatter(self, shared, **kw):
        return ProcessorFormatter(
            processors=[ProcessorFormatter.remove_processors_meta, shared, _render],
            **kw,
        )

    def test_foreign_record_does_not_mutate_shared_instance(self):
        shared = _StatefulProcessor()
        formatter = self._formatter(shared, isolate_foreign_processors=True)

        record = logging.LogRecord(
            "name", logging.INFO, "path", 1, "foreign", None, None
        )
        assert formatter.format(record) == "foreign"
        assert shared.calls == 0

    def test_structlog_record_still_uses_shared_instance(self):
        shared = _StatefulProcessor()
        formatter = self._formatter(shared, isolate_foreign_processors=True)

        record = logging.LogRecord(
            "name",
            logging.INFO,
            "path",
            1,
            {"event": "native"},
            None,
            None,
        )
        record._logger = logging.getLogger("x")  # type: ignore[attr-defined]
        record._name = "info"  # type: ignore[attr-defined]

        assert formatter.format(record) == "native"
        assert shared.calls == 1

    def test_foreign_pre_chain_is_isolated_too(self):
        shared_pre = _StatefulProcessor()
        shared_renderer = _StatefulProcessor()
        formatter = ProcessorFormatter(
            processors=[
                ProcessorFormatter.remove_processors_meta,
                shared_renderer,
                _render,
            ],
            foreign_pre_chain=[shared_pre],
            isolate_foreign_processors=True,
        )

        record = logging.LogRecord(
            "name", logging.INFO, "path", 1, "foreign", None, None
        )
        formatter.format(record)

        assert shared_pre.calls == 0
        assert shared_renderer.calls == 0

    def test_default_keeps_sharing_behavior(self):
        """
        Without the opt-in flag instances continue to be shared as before.
        """
        shared = _StatefulProcessor()
        formatter = self._formatter(shared)

        record = logging.LogRecord(
            "name", logging.INFO, "path", 1, "foreign", None, None
        )
        formatter.format(record)

        assert shared.calls == 1

    def test_stateless_callables_remain_shared_under_isolation(self):
        assert copy(structlog.processors.add_log_level) is (
            structlog.processors.add_log_level
        )


# ---------------------------------------------------------------------------
# cache_logger_on_first_use: clear-context semantics of new()
# ---------------------------------------------------------------------------


class TestCachedLoggerNew:
    def test_materialized_proxy_new_clears_initial_values(self):
        """
        Regression: after the lazy proxy materialized its cached logger,
        new() must clear instead of silently adding on top of the cached
        context.
        """
        structlog.configure(cache_logger_on_first_use=True)
        proxy = structlog.get_logger(a=1)

        proxy.bind()  # first use -> materializes cached assembled logger

        assert proxy.new(b=2)._context == {"b": 2}
        assert proxy.new(c=3)._context == {"c": 3}
        assert proxy.bind(d=4)._context == {"d": 4}

    def test_materialized_logger_new_matches_bound_logger_base(self):
        """
        Loggers derived from the cached logger keep BoundLoggerBase.new
        semantics: clear context, then bind.
        """
        structlog.configure(cache_logger_on_first_use=True)
        child = structlog.get_logger().bind(a=1)

        assert child._context == {"a": 1}
        replaced = child.new(b=2)
        assert replaced._context == {"b": 2}

    def test_new_semantics_without_caching_unchanged(self):
        """
        Before materialization the lazy proxy keeps its long-standing
        contract: initial_values survive new() (upstream-locked).
        """
        structlog.configure(cache_logger_on_first_use=False)
        proxy = structlog.get_logger(a=1)

        assert proxy.new(b=2)._context == {"a": 1, "b": 2}
        assert proxy.bind(c=3)._context == {"a": 1, "c": 3}
