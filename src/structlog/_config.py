# SPDX-License-Identifier: MIT OR Apache-2.0
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the MIT License.  See the LICENSE file in the root of this
# repository for complete details.

"""
Global state department.  Don't reload this module or everything breaks.
"""

from __future__ import annotations

import contextvars
import itertools
import os
import sys
import warnings

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, SupportsIndex, cast, overload


if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from ._native import make_filtering_bound_logger
from ._output import PrintLoggerFactory
from .contextvars import merge_contextvars
from .dev import ConsoleRenderer, _has_colors, set_exc_info
from .processors import StackInfoRenderer, TimeStamper, add_log_level
from .typing import BindableLogger, Context, Processor, WrappedLogger


"""
Any changes to these defaults must be reflected in:

- `getting-started`.
- structlog.stdlib.recreate_defaults()'s docstring.
"""

_no_colors = os.environ.get("NO_COLOR", "") != ""
_force_colors = os.environ.get("FORCE_COLOR", "") != ""


# ---------------------------------------------------------------------------
# Processor chain generations
# ---------------------------------------------------------------------------

_chain_generation_counter: Iterator[int] = itertools.count(1)


def _next_chain_generation() -> int:
    return next(_chain_generation_counter)


@dataclass(frozen=True)
class _ChainView:
    """
    A temporary, context-local overlay installed over a `ProcessorChain`.

    A view does *not* change the physical contents of its target chain.  It
    only changes what the chain resolves to when it is iterated, indexed, or
    measured from the same `contextvars` context (in other words, the current
    thread or asyncio task).  This is what makes `~structlog.testing.capture_logs`
    reentrant and isolated between concurrent contexts without replacing the
    list instance held by already-bound loggers (c.f. issue #408).
    """

    target: ProcessorChain
    processors: tuple[Processor, ...]
    generation: int
    label: str | None = None


_active_chain_views: contextvars.ContextVar[tuple[_ChainView, ...] | None] = (
    contextvars.ContextVar("structlog_active_chain_views", default=None)
)


def _push_chain_view(
    chain: ProcessorChain,
    processors: Iterable[Processor],
    *,
    label: str | None = None,
) -> tuple[contextvars.Token[tuple[_ChainView, ...] | None], _ChainView]:
    """
    Install a context-local *processors* overlay on *chain*.

    Returns the ContextVar token needed to remove the overlay again and the
    view that was installed.
    """
    view = _ChainView(
        target=chain,
        processors=tuple(processors),
        generation=_next_chain_generation(),
        label=label,
    )
    current = _active_chain_views.get()
    token = _active_chain_views.set(
        (*current, view) if current is not None else (view,)
    )

    return token, view


def _pop_chain_view(
    token: contextvars.Token[tuple[_ChainView, ...] | None],
) -> None:
    """
    Remove the overlay installed with `_push_chain_view`.
    """
    _active_chain_views.reset(token)


class ProcessorChain(list[Processor]):
    """
    The processor chain as a first-class, generation-aware configuration
    object.

    It behaves exactly like the plain processor `list` it replaces (bound
    loggers iterate it on every emission), but adds two capabilities:

    - An observable, monotonically increasing :attr:`generation`.  A new
      generation starts whenever a *new* chain alias is installed through
      `configure` / `reset_defaults`, and whenever a temporary context-local
      view (like `~structlog.testing.capture_logs`) is entered or left.
    - Context-local *views*.  A view replaces what the chain resolves to only
      within the current thread or asyncio task, so loggers that were bound
      before the view was installed and still hold this list reference are
      transparently redirected, while loggers in other threads or tasks keep
      seeing the physical chain.

    The physical contents of the chain are only changed through regular list
    operations (and through `configure`, which installs a new alias
    altogether); views never mutate them.

    .. versionadded:: 26.2.0
    """

    def __init__(self, processors: Iterable[Processor] = ()) -> None:
        super().__init__(processors)
        self._generation = _next_chain_generation()

    # -- observation ------------------------------------------------------

    @property
    def generation(self) -> int:
        """
        Generation of the chain visible from the current context.

        Either the chain's structural generation, or -- if a context-local
        view targets this chain -- the view's generation.
        """
        view = self._active_view()

        return view.generation if view is not None else self._generation

    def _active_view(self) -> _ChainView | None:
        views = _active_chain_views.get()
        if views:
            # The innermost view targeting this alias wins, which makes
            # nested overlays work.
            for view in reversed(views):
                if view.target is self:
                    return view

        return None

    # -- context-aware sequence interface --------------------------------

    def __iter__(self) -> Iterator[Processor]:
        view = self._active_view()
        if view is not None:
            return iter(view.processors)

        return list.__iter__(self)

    def __len__(self) -> int:
        view = self._active_view()
        if view is not None:
            return len(view.processors)

        return list.__len__(self)

    @overload
    def __getitem__(self, index: SupportsIndex, /) -> Processor: ...

    @overload
    def __getitem__(self, index: slice, /) -> list[Processor]: ...

    def __getitem__(
        self, index: SupportsIndex | slice
    ) -> Processor | list[Processor]:
        view = self._active_view()
        if view is None:
            return list.__getitem__(self, index)

        if isinstance(index, slice):
            return list(view.processors[index])

        return view.processors[index]

    def __contains__(self, item: object) -> bool:
        view = self._active_view()
        if view is not None:
            return item in view.processors

        return list.__contains__(self, item)

    def __reversed__(self) -> Iterator[Processor]:
        view = self._active_view()
        if view is not None:
            return reversed(view.processors)

        return list.__reversed__(self)

    def __repr__(self) -> str:
        view = self._active_view()
        items = list(view.processors) if view is not None else list.copy(self)
        view_note = f" view={view.label!r}" if view is not None and view.label else ""

        return (
            f"<ProcessorChain generation={self.generation}{view_note} "
            f"processors={items!r}>"
        )

    def copy(self) -> Self:
        """
        Shallow copy of the processors visible from the current context.

        The copy is a new, independent chain alias (and therefore starts at
        a new generation).
        """
        view = self._active_view()
        source = view.processors if view is not None else tuple(self)

        return type(self)(source)


def _as_processor_chain(processors: Iterable[Processor]) -> ProcessorChain:
    """
    Return *processors* as a `ProcessorChain`, installing a fresh generation
    if it isn't one already.
    """
    if isinstance(processors, ProcessorChain):
        return processors

    return ProcessorChain(processors)


def _make_default_console_renderer() -> ConsoleRenderer:
    return ConsoleRenderer(
        colors=not _no_colors
        and (
            _force_colors
            or (
                _has_colors
                and sys.stdout is not None
                and hasattr(sys.stdout, "isatty")
                and sys.stdout.isatty()
            )
        ),
        force_colors=_force_colors,
    )


def _make_default_processors() -> list[Processor]:
    """
    Build a fresh set of built-in default processors with fresh mutable
    state.
    """
    return [
        merge_contextvars,
        add_log_level,
        StackInfoRenderer(),
        set_exc_info,
        TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        _make_default_console_renderer(),
    ]


_BUILTIN_DEFAULT_PROCESSORS: Sequence[Processor] = _make_default_processors()
_BUILTIN_DEFAULT_CONTEXT_CLASS = cast(type[Context], dict)
_BUILTIN_DEFAULT_WRAPPER_CLASS = make_filtering_bound_logger(0)
_BUILTIN_DEFAULT_LOGGER_FACTORY = PrintLoggerFactory()
_BUILTIN_CACHE_LOGGER_ON_FIRST_USE = False


class _Configuration:
    """
    Global defaults.
    """

    is_configured: bool = False
    default_processors: ProcessorChain = ProcessorChain(
        _BUILTIN_DEFAULT_PROCESSORS
    )
    default_context_class: type[Context] = _BUILTIN_DEFAULT_CONTEXT_CLASS
    default_wrapper_class: Any = _BUILTIN_DEFAULT_WRAPPER_CLASS
    logger_factory: Callable[..., WrappedLogger] = (
        _BUILTIN_DEFAULT_LOGGER_FACTORY
    )
    cache_logger_on_first_use: bool = _BUILTIN_CACHE_LOGGER_ON_FIRST_USE


_CONFIG = _Configuration()
"""
Global defaults used when arguments to `wrap_logger` are omitted.
"""


def is_configured() -> bool:
    """
    Return whether *structlog* has been configured.

    If `False`, *structlog* is running with builtin defaults.

    .. versionadded: 18.1.0
    """
    return _CONFIG.is_configured


def get_config() -> dict[str, Any]:
    """
    Get a dictionary with the current configuration.

    .. note::

       Changes to the returned dictionary do *not* affect *structlog*.

    .. versionadded: 18.1.0
    """
    return {
        "processors": _CONFIG.default_processors,
        "context_class": _CONFIG.default_context_class,
        "wrapper_class": _CONFIG.default_wrapper_class,
        "logger_factory": _CONFIG.logger_factory,
        "cache_logger_on_first_use": _CONFIG.cache_logger_on_first_use,
    }


def get_processor_chain() -> ProcessorChain:
    """
    Return the processor chain alias currently installed by `configure`.

    The returned `ProcessorChain` is the same list instance that is handed to
    loggers bound without explicit *processors*.  It is generation-aware and
    reflects context-local overlays (such as
    `structlog.testing.capture_logs`) when iterated from the same context.

    .. versionadded:: 26.2.0
    """
    return _CONFIG.default_processors


def get_processor_generation() -> int:
    """
    Return the generation of the processor chain visible in this context.

    The value advances whenever `configure` installs a new chain, whenever
    `reset_defaults` restores the built-in defaults, and for the duration of
    a context-local chain overlay like `structlog.testing.capture_logs`.
    After the overlay is left, the previous generation becomes visible
    again.

    .. versionadded:: 26.2.0
    """
    return _CONFIG.default_processors.generation


def get_logger(*args: Any, **initial_values: Any) -> Any:
    """
    Convenience function that returns a logger according to configuration.

    >>> from structlog import get_logger
    >>> log = get_logger(y=23)
    >>> log.info("hello", x=42)
    y=23 x=42 event='hello'

    Args:
        args:
            *Optional* positional arguments that are passed unmodified to the
            logger factory.  Therefore it depends on the factory what they
            mean.

        initial_values: Values that are used to pre-populate your contexts.

    Returns:
        A proxy that creates a correctly configured bound logger when
        necessary. The type of that bound logger depends on your configuration
        and is `structlog.BoundLogger` by default.

    See `configuration` for details.

    If you prefer CamelCase, there's an alias for your reading pleasure:
    `structlog.getLogger`.

    .. versionadded:: 0.4.0 *args*
    """
    return wrap_logger(None, logger_factory_args=args, **initial_values)


getLogger = get_logger  # noqa: N816
"""
CamelCase alias for `structlog.get_logger`.

This function is supposed to be in every source file -- we don't want it to
stick out like a sore thumb in frameworks like Twisted or Zope.
"""


def wrap_logger(
    logger: WrappedLogger | None,
    processors: Iterable[Processor] | None = None,
    wrapper_class: type[BindableLogger] | None = None,
    context_class: type[Context] | None = None,
    cache_logger_on_first_use: bool | None = None,
    logger_factory_args: Iterable[Any] | None = None,
    **initial_values: Any,
) -> Any:
    """
    Create a new bound logger for an arbitrary *logger*.

    Default values for *processors*, *wrapper_class*, and *context_class* can
    be set using `configure`.

    If you set an attribute here, `configure` calls have *no* effect for the
    *respective* attribute.

    In other words: selective overwriting of the defaults while keeping some
    *is* possible.

    Args:
        initial_values: Values that are used to pre-populate your contexts.

        logger_factory_args:
            Values that are passed unmodified as ``*logger_factory_args`` to
            the logger factory if not `None`.

    Returns:
        A proxy that creates a correctly configured bound logger when
        necessary.

    See `configure` for the meaning of the rest of the arguments.

    .. versionadded:: 0.4.0 *logger_factory_args*
    """
    return BoundLoggerLazyProxy(
        logger,
        wrapper_class=wrapper_class,
        processors=processors,
        context_class=context_class,
        cache_logger_on_first_use=cache_logger_on_first_use,
        initial_values=initial_values,
        logger_factory_args=logger_factory_args,
    )


def configure(
    processors: Iterable[Processor] | None = None,
    wrapper_class: type[BindableLogger] | None = None,
    context_class: type[Context] | None = None,
    logger_factory: Callable[..., WrappedLogger] | None = None,
    cache_logger_on_first_use: bool | None = None,
) -> None:
    """
    Configures the **global** defaults.

    They are used if `wrap_logger` or `get_logger` are called without
    arguments.

    Can be called several times, keeping an argument at `None` leaves it
    unchanged from the current setting.

    After calling for the first time, `is_configured` starts returning `True`.

    Use `reset_defaults` to undo your changes.

    Args:
        processors: The processor chain. See :doc:`processors` for details.

        wrapper_class:
            Class to use for wrapping loggers instead of
            `structlog.BoundLogger`.  See `standard-library`, :doc:`twisted`,
            and `custom-wrappers`.

        context_class:
            Class to be used for internal context keeping. The default is a
            `dict` and since dictionaries are ordered as of Python 3.6, there's
            few reasons to change this option.

        logger_factory:
            Factory to be called to create a new logger that shall be wrapped.

        cache_logger_on_first_use:
            `wrap_logger` doesn't return an actual wrapped logger but a proxy
            that assembles one when it's first used. If this option is set to
            `True`, this assembled logger is cached. See `performance`.

    .. versionadded:: 0.3.0 *cache_logger_on_first_use*
    """
    _CONFIG.is_configured = True

    if processors is not None:
        _CONFIG.default_processors = _as_processor_chain(processors)
    if wrapper_class is not None:
        _CONFIG.default_wrapper_class = wrapper_class
    if context_class is not None:
        _CONFIG.default_context_class = context_class
    if logger_factory is not None:
        _CONFIG.logger_factory = logger_factory
    if cache_logger_on_first_use is not None:
        _CONFIG.cache_logger_on_first_use = cache_logger_on_first_use


def configure_once(
    processors: Iterable[Processor] | None = None,
    wrapper_class: type[BindableLogger] | None = None,
    context_class: type[Context] | None = None,
    logger_factory: Callable[..., WrappedLogger] | None = None,
    cache_logger_on_first_use: bool | None = None,
) -> None:
    """
    Configures if structlog isn't configured yet.

    It does *not* matter whether it was configured using `configure` or
    `configure_once` before.

    Raises:
        RuntimeWarning: if repeated configuration is attempted.
    """
    if not _CONFIG.is_configured:
        configure(
            processors=processors,
            wrapper_class=wrapper_class,
            context_class=context_class,
            logger_factory=logger_factory,
            cache_logger_on_first_use=cache_logger_on_first_use,
        )
    else:
        warnings.warn(
            "Repeated configuration attempted.", RuntimeWarning, stacklevel=2
        )


def reset_defaults() -> None:
    """
    Resets global default values to builtin defaults.

    A fresh processor `ProcessorChain` generation with *fresh* built-in
    processor instances is installed, so state that was changed through the
    active renderer (for example via
    ``structlog.dev.ConsoleRenderer.get_active()``) can't leak across the
    reset.

    `is_configured` starts returning `False` afterwards.
    """
    _CONFIG.is_configured = False
    # Fresh alias (new generation) *and* fresh stateful processor instances,
    # so resetting really restores pristine defaults.
    _CONFIG.default_processors = ProcessorChain(_make_default_processors())
    _CONFIG.default_wrapper_class = _BUILTIN_DEFAULT_WRAPPER_CLASS
    _CONFIG.default_context_class = _BUILTIN_DEFAULT_CONTEXT_CLASS
    _CONFIG.logger_factory = _BUILTIN_DEFAULT_LOGGER_FACTORY
    _CONFIG.cache_logger_on_first_use = _BUILTIN_CACHE_LOGGER_ON_FIRST_USE


class BoundLoggerLazyProxy:
    """
    Instantiates a bound logger on first usage.

    Takes both configuration and instantiation parameters into account.

    The only points where a bound logger changes state are ``bind()``,
    ``unbind()``, and ``new()`` and that return the actual ``BoundLogger``.

    If and only if configuration says so, that actual bound logger is cached on
    first usage.

    .. versionchanged:: 0.4.0 Added support for *logger_factory_args*.
    """

    # fulfill BindableLogger protocol without carrying accidental state
    @property
    def _context(self) -> dict[str, str]:
        return self._initial_values

    def __init__(
        self,
        logger: WrappedLogger | None,
        wrapper_class: type[BindableLogger] | None = None,
        processors: Iterable[Processor] | None = None,
        context_class: type[Context] | None = None,
        cache_logger_on_first_use: bool | None = None,
        initial_values: dict[str, Any] | None = None,
        logger_factory_args: Any = None,
    ) -> None:
        self._logger = logger
        self._wrapper_class = wrapper_class
        self._processors = processors
        self._context_class = context_class
        self._cache_logger_on_first_use = cache_logger_on_first_use
        self._initial_values = initial_values or {}
        self._logger_factory_args = logger_factory_args or ()

    def __repr__(self) -> str:
        return (
            f"<BoundLoggerLazyProxy(logger={self._logger!r}, wrapper_class="
            f"{self._wrapper_class!r}, processors={self._processors!r}, "
            f"context_class={self._context_class!r}, "
            f"initial_values={self._initial_values!r}, "
            f"logger_factory_args={self._logger_factory_args!r})>"
        )

    def bind(self, **new_values: Any) -> BindableLogger:
        """
        Assemble a new BoundLogger from arguments and configuration.
        """
        if self._context_class:
            ctx = self._context_class(self._initial_values)
        else:
            ctx = _CONFIG.default_context_class(self._initial_values)

        _logger = self._logger
        if not _logger:
            _logger = _CONFIG.logger_factory(*self._logger_factory_args)

        procs: Iterable[Processor]
        if self._processors is None:
            procs = _CONFIG.default_processors
        else:
            procs = self._processors

        cls = self._wrapper_class or _CONFIG.default_wrapper_class
        # Looks like Protocols ignore definitions of __init__ so we have to
        # silence Mypy here.
        logger = cls(
            _logger,
            processors=procs,
            context=ctx,  # type: ignore[call-arg]
        )

        def finalized_bind(**new_values: Any) -> BindableLogger:
            """
            Use cached assembled logger to bind potentially new values.
            """
            if new_values:
                return logger.bind(**new_values)

            return logger

        def finalized_new(**new_values: Any) -> BindableLogger:
            """
            Clear the cached assembled logger's context, then bind.

            Routing this through ``logger.new`` (instead of silently adding to
            the cached context via bind) keeps the clear-context semantics of
            `new` intact once the proxy has materialized its logger.
            """
            return logger.new(**new_values)

        if self._cache_logger_on_first_use is True or (
            self._cache_logger_on_first_use is None
            and _CONFIG.cache_logger_on_first_use is True
        ):
            self.bind = finalized_bind  # type: ignore[method-assign]
            self.new = finalized_new  # type: ignore[method-assign]

        return finalized_bind(**new_values)

    def unbind(self, *keys: str) -> BindableLogger:
        """
        Same as bind, except unbind *keys* first.

        In our case that could be only initial values.
        """
        return self.bind().unbind(*keys)

    def try_unbind(self, *keys: str) -> BindableLogger:
        return self.bind().try_unbind(*keys)

    def new(self, **new_values: Any) -> BindableLogger:
        """
        Clear context, then bind.
        """
        if self._context_class:
            self._context_class().clear()
        else:
            _CONFIG.default_context_class().clear()

        return self.bind(**new_values)

    def __getattr__(self, name: str) -> Any:
        """
        If a logging method if called on a lazy proxy, we have to create an
        ephemeral BoundLogger first.
        """
        if name == "__isabstractmethod__":
            raise AttributeError

        bl = self.bind()

        return getattr(bl, name)

    def __getstate__(self) -> dict[str, Any]:
        """
        Our __getattr__ magic makes this necessary.
        """
        return self.__dict__

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Our __getattr__ magic makes this necessary.
        """
        for k, v in state.items():
            setattr(self, k, v)
