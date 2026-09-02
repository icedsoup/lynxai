from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

Observation = TypeVar("Observation")
Action = TypeVar("Action")


class Environment(ABC, Generic[Observation, Action]):
    """abstract contract for all training and simulation environments.
       The agent only depends on this interface.
    """

    @abstractmethod
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Observation:
        """reset the environment and return initial observation"""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Apply an action and return (observation, reward, done, info)."""
        raise NotImplementedError
