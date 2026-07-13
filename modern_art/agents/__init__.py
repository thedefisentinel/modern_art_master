"""AI 智能体。当前提供随机与简单启发式基线,后续接入强化学习(rl/)。

统一接口:Agent.act(state) -> 具体可执行的 Action(已填好金额)。
所有 Agent 只应依赖 game.observation(state, me) 中的公开信息 + 自己的私有信息,
不得窥探对手手牌/现金/暗标(基线实现遵守此约定)。
"""

import os

from .base import Agent
from .random_agent import RandomAgent
from .heuristic_agent import HeuristicAgent
from .standard_agent import StandardAgent
from .archetype_agents import AggressiveAgent, TightAgent

AGENT_REGISTRY: dict[str, type[Agent]] = {
    "standard": StandardAgent,     # 一般难度(推荐试玩对手)
    "heuristic": HeuristicAgent,   # 简单启发式基线(便宜狙击手)
    "aggressive": AggressiveAgent,  # 莽夫:高价抢画、出钱大方
    "tight": TightAgent,           # 铁公鸡:能不买就不买
    "random": RandomAgent,         # 随机(送分)
}

#: 训练好的 RL 策略默认检查点(接入 CLI/网页时用)。
#: v5(修正公开拍卖规则下训练)> v4 > v3;2v2 对轰 v5:v4 = 57:43。
DEFAULT_RL_CHECKPOINT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "rl", "checkpoints", "policy_v5.pt"))


def make_agent(name: str, seed=None):
    """按名字构造对手。"rl" 特殊:懒加载训练好的策略(需 torch + 检查点)。"""
    if name == "rl":
        if not os.path.exists(DEFAULT_RL_CHECKPOINT):
            raise FileNotFoundError(
                f"RL 检查点不存在:{DEFAULT_RL_CHECKPOINT}(请先运行 rl.train 训练)")
        from .rl_agent import RLAgent   # 懒导入,避免非 RL 场景引入 torch
        return RLAgent(DEFAULT_RL_CHECKPOINT, name="rl", seed=seed)
    return AGENT_REGISTRY[name](seed=seed)


def available_opponents() -> list[str]:
    """可选对手名(RL 检查点存在时才含 'rl')。"""
    names = list(AGENT_REGISTRY)
    if os.path.exists(DEFAULT_RL_CHECKPOINT):
        names = ["rl"] + names
    return names


__all__ = [
    "Agent", "RandomAgent", "HeuristicAgent", "StandardAgent",
    "AggressiveAgent", "TightAgent", "AGENT_REGISTRY",
    "make_agent", "available_opponents", "DEFAULT_RL_CHECKPOINT",
]
