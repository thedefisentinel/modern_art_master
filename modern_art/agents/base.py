"""Agent 基类。"""

from __future__ import annotations

import abc
import random

from ..engine import game
from ..engine.state import GameState
from ..engine.actions import Action


class Agent(abc.ABC):
    """决策智能体。act 必须返回可直接交给 game.apply 的具体动作。"""

    def __init__(self, name: str = "agent", seed: int | None = None):
        self.name = name
        self.rng = random.Random(seed)

    @abc.abstractmethod
    def act(self, state: GameState) -> Action:
        """给出当前应行动玩家(game.current_player(state))的动作。"""

    # 便捷属性
    @staticmethod
    def me(state: GameState) -> int:
        return game.current_player(state)
