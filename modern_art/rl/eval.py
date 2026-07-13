"""评估一个训练好的检查点:对各基线的胜率与相对分(3/4/5 人)。

    python -m modern_art.rl.eval --ckpt modern_art/rl/checkpoints/policy.pt --games 400

胜率:该策略坐 seat0,其余座位为某基线;记 seat0 夺冠的比例(并列按 1/并列数 计)。
相对分:seat0 的(终局现金 − 全场平均)。>0 表示强于平均对手。
"""

from __future__ import annotations

import argparse

import numpy as np

from ..engine import game
from ..agents import AGENT_REGISTRY
from ..agents.rl_agent import RLAgent
from .env import SelfPlayEnv


def eval_vs(agent, baseline_name: str, num_players: int, games: int, seed: int = 999):
    rng = np.random.default_rng(seed)
    wins = 0.0
    rel = 0.0
    for _ in range(games):
        env = SelfPlayEnv()
        env.reset(num_players, seed=int(rng.integers(1 << 30)))
        opps = {i: AGENT_REGISTRY[baseline_name](seed=int(rng.integers(1 << 30)))
                for i in range(1, num_players)}
        while not env.done:
            me = env.current_player()
            env.step_action((agent if me == 0 else opps[me]).act(env.state))
        w = game.winner(env.state)
        wins += (1.0 / len(w)) if 0 in w else 0.0
        rel += env.rewards[0]
    return wins / games, rel / games


def main(argv=None):
    ap = argparse.ArgumentParser(description="评估 RL 检查点")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--counts", default="3,4,5")
    args = ap.parse_args(argv)

    agent = RLAgent(args.ckpt, device=args.device, greedy=True)
    counts = [int(x) for x in args.counts.split(",")]
    print(f"检查点 {args.ckpt}")
    print(f"{'对手':<10}{'人数':<6}{'胜率':<10}{'相对分':<10}  (随机基准胜率=1/人数)")
    for name in AGENT_REGISTRY:
        for n in counts:
            wr, rel = eval_vs(agent, name, n, args.games)
            base = 1.0 / n
            flag = "✓强于均势" if wr > base else "✗弱于均势"
            print(f"{name:<10}{n:<6}{wr:>7.1%}   {rel:>+7.2f}    (基准{base:.0%}) {flag}")


if __name__ == "__main__":
    main()
