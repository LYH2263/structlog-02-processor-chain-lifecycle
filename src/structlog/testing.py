# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

"""
Helpers to test your application's logging behavior.

.. versionadded:: 20.1.0

See :doc:`testing`.
"""

from __future__ import annotations

from collections.abc import Generator, Iterable
from contextlib import contextmanager
from typing import Any, NamedTuple, NoReturn

from . import _config
from ._log_levels import map_method_name
from .exceptions import DropEvent
from .typing import EventDict, Processor, WrappedLogger


__all__ = [
    "CapturedCall",
    "CapturingLogger",
    "CapturingLoggerFactory",
    "LogCapture",
    "ReturnLogger",
    "ReturnLoggerFactory",
    "capture_logs",
]


class LogCapture:
    """
    Class for capturing log messages in its entries list.
    Generally you should use `structlog.testing.capture_logs`,
    but you can use this class if you want to capture logs with other patterns.

    :ivar List[structlog.typing.EventDict] entries: The captured log entries.

    .. versionadded:: 20.1.0

    .. versionchanged:: 24.3.0
       Added mapping from "exception" to "error"
       Added mapping from "warn" to "warning"
    """

    entries: list[EventDict]

    def __init__(self) -> None:
        self.entries = []

    def __call__(
        self, _: WrappedLogger, method_name: str, event_dict: EventDict
    ) -> NoReturn:
        event_dict["log_level"] = map_method_name(method_name)
        self.entries.append(event_dict)

        raise DropEvent


@contextmanager
def capture_logs(
    processors: Iterable[Processor] = (),
) -> Generator[list[EventDict], None, None]:
    """
    Context manager that appends all logging statements to its yielded list
    while it is active. Disables all configured processors for the duration
    of the context manager.

    The capture is installed as a temporary generation on top of the
    currently configured processor chain:

    - Loggers that were already bound and still hold the configured chain's
      list reference are captured too -- the list itself is never replaced
      (this used to cause already-bound loggers to be missed, see issue
      #408).
    - Captures can be nested.  Events emitted while the inner capture is
      active only appear in the inner list; after it is left, the outer
      capture receives events again.
    - Concurrent captures in other threads or asyncio tasks are isolated:
      each context only sees its own temporary chain and its own entries.
      Logging from contexts without an active capture keeps using the
      physical production chain and is not redirected.

    Configuring *structlog* itself (for example with `configure`) remains
    global and is not thread-safe, just like the rest of the global
    configuration.

    Args:
        processors: Processors to apply before the logs are captured.

    .. versionadded:: 20.1.0
    .. versionadded:: 25.5.0 *processors* parameter
    .. versionchanged:: 26.2.0
       The capture no longer mutates the configured processor list in place.
       It is now reentrant, can be nested, and is isolated between concurrent
       threads and asyncio tasks.
    """
    cap = LogCapture()
    # Install a context-local view on the *current* configured chain alias.
    # The chain's physical contents stay untouched, which is what makes the
    # overlay reentrant and isolated while still reaching already-bound
    # loggers (they iterate the same alias and thus resolve the view).
    chain = _config._CONFIG.default_processors
    temporary_processors: tuple[Processor, ...] = (*processors, cap)
    token, _view = _config._push_chain_view(
        chain, temporary_processors, label="capture_logs"
    )
    try:
        yield cap.entries
    finally:
        _config._pop_chain_view(token)


class ReturnLogger:
    """
    Return the arguments that it's called with.

    >>> from structlog import ReturnLogger
    >>> ReturnLogger().info("hello")
    'hello'
    >>> ReturnLogger().info("hello", when="again")
    (('hello',), {'when': 'again'})

    .. versionchanged:: 0.3.0
        Allow for arbitrary arguments and keyword arguments to be passed in.
    """

    def msg(self, *args: Any, **kw: Any) -> Any:
        """
        Return tuple of ``args, kw`` or just ``args[0]`` if only one arg passed
        """
        # Slightly convoluted for backwards compatibility.
        if len(args) == 1 and not kw:
            return args[0]

        return args, kw

    log = debug = info = warn = warning = msg
    fatal = failure = err = error = critical = exception = msg


class ReturnLoggerFactory:
    r"""
    Produce and cache `ReturnLogger`\ s.

    To be used with `structlog.configure`\ 's *logger_factory*.

    Positional arguments are silently ignored.

    .. versionadded:: 0.4.0
    """

    def __init__(self) -> None:
        self._logger = ReturnLogger()

    def __call__(self, *args: Any) -> ReturnLogger:
        return self._logger


class CapturedCall(NamedTuple):
    """
    A call as captured by `CapturingLogger`.

    Can also be unpacked like a tuple.

    Args:
        method_name: The method name that got called.

        args: A tuple of the positional arguments.

        kwargs: A dict of the keyword arguments.

    .. versionadded:: 20.2.0
    """

    method_name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class CapturingLogger:
    """
    Store the method calls that it's been called with.

    This is nicer than `ReturnLogger` for unit tests because the bound logger
    doesn't have to cooperate.

    **Any** method name is supported.

    .. versionadded:: 20.2.0
    """

    calls: list[CapturedCall]

    def __init__(self) -> None:
        self.calls = []

    def __repr__(self) -> str:
        return f"<CapturingLogger with {len(self.calls)} call(s)>"

    def __getattr__(self, name: str) -> Any:
        """
        Capture call to `calls`
        """

        def log(*args: Any, **kw: Any) -> None:
            self.calls.append(CapturedCall(name, args, kw))

        return log


class CapturingLoggerFactory:
    r"""
    Produce and cache `CapturingLogger`\ s.

    Each factory produces and reuses only **one** logger.

    You can access it via the ``logger`` attribute.

    To be used with `structlog.configure`\ 's *logger_factory*.

    Positional arguments are silently ignored.

    .. versionadded:: 20.2.0
    """

    logger: CapturingLogger

    def __init__(self) -> None:
        self.logger = CapturingLogger()

    def __call__(self, *args: Any) -> CapturingLogger:
        return self.logger
