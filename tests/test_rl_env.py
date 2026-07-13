"""RL 编解码 / 环境的单元测试 —— 零依赖(numpy),`python tests/test_rl_env.py` 直接跑。

核心保证:
  * 观测维度恒定(与人数、阶段无关);
  * 掩码内的每个离散动作解码出的引擎动作**一定合法**(apply 不抛错);
  * 每个决策点至少有一个合法动作;
  * 金额解码落在合法区间;
  * 随机自对弈能整局跑通、相对分奖励均值为 0。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from modern_art.engine import game
from modern_art.rl import encoding, env
from modern_art.rl.encoding import (
    encode_obs, legal_mask, decode_action, amount_range, OBS_DIM, NUM_DISCRETE,
)


def test_obs_dim_constant():
    dims = set()
    rng = np.random.default_rng(0)
    for n in (3, 4, 5):
        for seed in range(20):
            s = game.new_game(n, seed=seed)
            steps = 0
            while not game.is_over(s):
                me = game.current_player(s)
                obs = encode_obs(s, me)
                dims.add(obs.shape[0])
                assert obs.dtype == np.float32
                assert np.all(np.isfinite(obs))
                mask = legal_mask(s, me)
                legal = np.flatnonzero(mask)
                d = int(rng.choice(legal))
                s = game.apply(s, decode_action(s, me, d, float(rng.random())))
                steps += 1
                assert steps < 5000
    assert dims == {OBS_DIM}, dims


def test_every_masked_action_is_legal():
    # 在大量真实状态下,穷举掩码内每个离散动作 × 若干金额,decode+apply 都不应抛错
    rng = np.random.default_rng(1)
    checked = 0
    for n in (3, 4, 5):
        for seed in range(40):
            s = game.new_game(n, seed=seed)
            steps = 0
            while not game.is_over(s):
                me = game.current_player(s)
                mask = legal_mask(s, me)
                legal = np.flatnonzero(mask)
                assert legal.size >= 1, f"无合法动作 phase={s.phase}"
                for d in legal:
                    d = int(d)
                    rgn = amount_range(s, me, d)
                    for amt01 in (0.0, 0.37, 1.0):
                        act = decode_action(s, me, d, amt01)
                        # 金额在区间内
                        if rgn is not None:
                            lo, hi = rgn
                            val = getattr(act, "amount", getattr(act, "price", None))
                            assert lo <= val <= hi, (d, val, rgn)
                        # 合法:apply 不抛错(用克隆,避免推进当前对局)
                        game.apply(s, act)
                        checked += 1
                # 真正推进一步(随机)
                d = int(rng.choice(legal))
                s = game.apply(s, decode_action(s, me, d, float(rng.random())))
                steps += 1
                assert steps < 5000
    assert checked > 1000


def test_random_rollout_rewards_zero_sum():
    rng = np.random.default_rng(2)
    for n in (3, 4, 5):
        for seed in range(30):
            rewards = env.random_rollout(n, seed, rng)
            assert len(rewards) == n
            assert abs(sum(rewards)) < 1e-6, sum(rewards)   # 相对分零和


def test_num_discrete_layout():
    assert NUM_DISCRETE == 36
    # 选牌 25 + 补牌 5 + 6 个单动作
    assert encoding.PASS_BUY == 35 and encoding.CHOOSE_BASE == 0


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"\nRL 环境测试全部通过:{len(tests)} 项  (OBS_DIM={OBS_DIM}, NUM_DISCRETE={NUM_DISCRETE})")


if __name__ == "__main__":
    _run_all()
