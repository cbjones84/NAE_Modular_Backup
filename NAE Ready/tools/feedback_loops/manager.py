"""Utility for orchestrating multiple feedback loops."""

from __future__ import annotations

from typing import Dict, Optional

from .base import FeedbackLoop, FeedbackResult


class FeedbackManager:
    """Register and run feedback loops by name."""

    def __init__(self) -> None:
        self._loops: Dict[str, FeedbackLoop] = {}

    def register(self, loop: FeedbackLoop) -> None:
        self._loops[loop.name] = loop

    def unregister(self, name: str) -> None:
        self._loops.pop(name, None)

    def get(self, name: str) -> Optional[FeedbackLoop]:
        return self._loops.get(name)

    def run(self, name: str, context: Dict) -> Optional[FeedbackResult]:
        loop = self._loops.get(name)
        if not loop:
            return None
        return loop.run(context)

    def run_all(self, context: Dict) -> Dict[str, Optional[FeedbackResult]]:
        return {name: loop.run(context) for name, loop in self._loops.items()}



