"""Prevent repeated execution of the same tool call within one question."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Any, Hashable


NormalizedArgs = Hashable
ArgsNormalizer = Callable[[Any], NormalizedArgs]


def normalize_args(args: Any) -> NormalizedArgs:
    """Convert tool arguments to a stable, hashable structural value.

    Mapping key order and list/tuple representation do not change identity.
    Strings (including mapping keys) are preserved exactly. The team has not
    specified tool-specific text equivalence, so Unicode, whitespace, case,
    and punctuation must not be changed here. In particular, NFKC would merge
    distinct calculator expressions such as "2³" and "23". A team-approved
    tool-specific normalizer can be supplied to LoopBreaker at integration.

    Args:
        args: JSON-like tool arguments.

    Returns:
        Stable hashable value suitable for a ``(tool_name, normalized_args)`` key.

    Raises:
        TypeError: If an unsupported argument type is supplied.
    """

    if args is None or isinstance(args, (bool, int, float)):
        return (type(args).__name__, args)
    if isinstance(args, str):
        return ("str", args)
    if isinstance(args, Mapping):
        normalized_items = [
            (normalize_args(key), normalize_args(value)) for key, value in args.items()
        ]
        normalized_items.sort(key=lambda item: repr(item[0]))
        return ("mapping", tuple(normalized_items))
    if isinstance(args, Sequence) and not isinstance(args, (str, bytes, bytearray)):
        return ("sequence", tuple(normalize_args(value) for value in args))
    if isinstance(args, Set) and not isinstance(args, (str, bytes, bytearray)):
        values = [normalize_args(value) for value in args]
        values.sort(key=repr)
        return ("set", tuple(values))
    raise TypeError(f"Unsupported tool argument type: {type(args).__name__}")


@dataclass(frozen=True)
class ToolCallOutcome:
    """Outcome returned to the graph after checking a requested tool call."""

    result: Any
    blocked: bool
    answer_directly: bool
    normalized_args: NormalizedArgs

    @property
    def executed(self) -> bool:
        """Whether the tool was actually executed for this request."""

        return not self.blocked


class LoopBreaker:
    """Cache tool results and stop duplicate calls within one question.

    Create one instance per question, or call :meth:`reset` before the next
    question. ``run`` accepts a zero-argument execution closure so this module
    does not need to know whether a LangChain tool uses ``invoke``, keyword
    arguments, or another calling convention.

    Args:
        normalizer: Tool-argument normalizer. Defaults to :func:`normalize_args`.
    """

    def __init__(self, normalizer: ArgsNormalizer = normalize_args) -> None:
        self._normalizer = normalizer
        self._results: dict[tuple[str, NormalizedArgs], Any] = {}
        self.loop_breaks = 0

    def run(
        self,
        tool_name: str,
        args: Any,
        execute: Callable[[], Any],
    ) -> ToolCallOutcome:
        """Execute a new tool call or return its prior result when duplicated.

        Args:
            tool_name: Stable registered tool name.
            args: Tool arguments used to detect a duplicate.
            execute: Zero-argument closure that performs the actual tool call.

        Returns:
            Tool result and routing flags. On a duplicate, ``blocked`` and
            ``answer_directly`` are both true and ``execute`` is not called.
        """

        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("tool_name must be a non-empty string")
        if not callable(execute):
            raise TypeError("execute must be callable")

        normalized = self._normalizer(args)
        key = (tool_name, normalized)
        if key in self._results:
            self.loop_breaks += 1
            return ToolCallOutcome(
                result=self._results[key],
                blocked=True,
                answer_directly=True,
                normalized_args=normalized,
            )

        result = execute()
        self._results[key] = result
        return ToolCallOutcome(
            result=result,
            blocked=False,
            answer_directly=False,
            normalized_args=normalized,
        )

    def reset(self) -> None:
        """Clear remembered calls and the per-question loop-break count."""

        self._results.clear()
        self.loop_breaks = 0
