"""薄自对弈环境:把引擎包成"每次一个玩家行动"的多智能体环境。

引擎本身就是轮流制(AEC),这里只做编解码 + 奖励,不引入 pettingzoo。

用法(训练循环里):
    env = SelfPlayEnv()
    env.reset(num_players=4, seed=...)
    while not env.done:
        me = env.current_player()
        obs, mask = env.observe()
        discrete, amount01 = policy(obs, mask)      # 由训练器/策略给出
        env.step(discrete, amount01)
    rewards = env.rewards                            # 每个玩家的相对分(终局)
"""

from __future__ import annotations

import numpy as np

from ..engine import game
from ..engine.state import GameState
from . import encoding


def terminal_rewards(state: GameState, scale: float = 100.0) -> list[float]:
    """相对分奖励:每人终局现金 − 全场平均,再除以 scale。零和,均值为 0。"""
    scores = game.final_scores(state)
    mean = sum(scores) / len(scores)
    return [(s - mean) / scale for s in scores]


class SelfPlayEnv:
    def __init__(self, reward_scale: float = 100.0):
        self.state: GameState | None = None
        self.reward_scale = reward_scale
        self.rewards: list[float] | None = None

    def reset(self, num_players: int, seed: int | None = None) -> None:
        self.state = game.new_game(num_players, seed=seed)
        self.rewards = None

    @property
    def done(self) -> bool:
        return self.state is None or game.is_over(self.state)

    def current_player(self) -> int:
        return game.current_player(self.state)

    def observe(self, me: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        if me is None:
            me = self.current_player()
        return encoding.encode_obs(self.state, me), encoding.legal_mask(self.state, me)

    def step(self, discrete: int, amount01: float = 0.0) -> None:
        """当前玩家执行 (离散动作, 连续金额)。到终局时填充 self.rewards。"""
        me = self.current_player()
        action = encoding.decode_action(self.state, me, discrete, amount01)
        self.step_action(action)

    def step_action(self, action) -> None:
        """当前玩家执行一个原始引擎动作(供基线对手复用同一循环)。"""
        self.state = game.apply(self.state, action)
        if self.done:
            self.rewards = terminal_rewards(self.state, self.reward_scale)


def random_rollout(num_players: int, seed: int, rng: np.random.Generator) -> list[float]:
    """用随机合法动作跑完一整局(测试用):随机离散(掩码内)+ 随机金额。"""
    env = SelfPlayEnv()
    env.reset(num_players, seed=seed)
    steps = 0
    while not env.done:
        _obs, mask = env.observe()
        legal = np.flatnonzero(mask)
        discrete = int(rng.choice(legal))
        amount01 = float(rng.random())
        env.step(discrete, amount01)
        steps += 1
        if steps > 5000:
            raise RuntimeError("rollout 步数异常")
    return env.rewards
