"""自对弈 PPO 训练(参数共享 + 参数化动作 + 辅助信念头 + 基线对手池)。

用法:
    python -m modern_art.rl.train --iters 50 --games 128        # 正式训练
    python -m modern_art.rl.train --smoke                        # 快速冒烟(验证能学起来)

目标:相对分(现金−平均)最大化;评估用对基线胜率 + 相对分。
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from ..engine import game
from ..engine.rules import ARTISTS, AUCTION_TYPES
from ..agents import AGENT_REGISTRY
from . import encoding
from .encoding import MAX_PLAYERS, NA, NT, OBS_DIM, NUM_DISCRETE
from .env import SelfPlayEnv
from .model import ActorCritic, AUX_OPP_DIM, AUX_RANK_DIM

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


class FrozenNet:
    """把某个(冻结的)网络包成能 .act(state) 的对手,供 league 自对弈用。"""

    def __init__(self, net, device):
        self.net = net
        self.device = device

    @torch.no_grad()
    def act(self, state):
        me = game.current_player(state)
        obs = torch.from_numpy(encoding.encode_obs(state, me)).to(self.device)
        mask = torch.from_numpy(encoding.legal_mask(state, me)).to(self.device)
        out = self.net.act(obs, mask,
                           needs_amount_fn=lambda d, st=state, m=me: encoding.needs_amount(st, m, d))
        return encoding.decode_action(state, me, out["discrete"], out["amount01"])


# ── 辅助标签 ──────────────────────────────────────────────────────────────

def opp_hand_label(state, me):
    """对手手牌构成标签(座位相对,slot 1..n-1),+ 有效掩码。"""
    n = state.num_players
    rel = [(me + k) % n for k in range(n)]
    label = np.zeros(AUX_OPP_DIM, dtype=np.float32)
    valid = np.zeros(AUX_OPP_DIM, dtype=np.float32)
    for slot in range(1, MAX_PLAYERS):
        if slot < n:
            q = rel[slot]
            ct = Counter((c.artist, c.auction) for c in state.players[q].hand)
            base = (slot - 1) * NA * NT
            for ai, a in enumerate(ARTISTS):
                for ti, t in enumerate(AUCTION_TYPES):
                    label[base + ai * NT + ti] = ct.get((a, t), 0) / 4.0
                    valid[base + ai * NT + ti] = 1.0
    return label, valid


# ── 采样一批自对弈数据 ────────────────────────────────────────────────────

#: league 对手风格池(含莽夫/铁公鸡等原型,扩大 general 覆盖;random 占比低)
_OPP_POOL = ["heuristic", "standard", "standard", "aggressive", "aggressive", "tight", "random"]


def default_opp_provider(n, rng):
    """league 对手:多风格混合。可整桌同风格,也可混搭不同风格(如理性人 vs 莽夫群)。"""
    if rng.random() < 0.5:
        name = rng.choice(_OPP_POOL)                    # 整桌同一风格
        return {i: AGENT_REGISTRY[name](seed=int(rng.integers(1 << 30))) for i in range(1, n)}
    # 混搭:每个对手座位独立抽风格
    return {i: AGENT_REGISTRY[rng.choice(_OPP_POOL)](seed=int(rng.integers(1 << 30)))
            for i in range(1, n)}


def collect(net, device, num_games, rng, gamma=0.997, lam=0.95,
            league_frac=0.5, player_counts=(3, 4, 5), opp_provider=None):
    steps = []  # 每个学习者决策点一条记录
    opp_provider = opp_provider or default_opp_provider

    for _ in range(num_games):
        n = int(rng.choice(player_counts))
        league_game = rng.random() < league_frac
        # 座位分配:league 局仅 seat0 为学习者,其余为对手池;否则全自对弈
        opp = None
        if league_game:
            opp = opp_provider(n, rng)   # {seat: 具备 .act(state) 的对手}

        env = SelfPlayEnv()
        env.reset(n, seed=int(rng.integers(1 << 30)))
        game_records = []  # (player, dict) 学习者步

        guard = 0
        while not env.done:
            me = env.current_player()
            if opp is not None and me in opp:
                env.step_action(opp[me].act(env.state))
            else:
                obs_np, mask_np = env.observe(me)
                obs = torch.from_numpy(obs_np).to(device)
                mask = torch.from_numpy(mask_np).to(device)
                out = net.act(obs, mask,
                              needs_amount_fn=lambda d, st=env.state, m=me: encoding.needs_amount(st, m, d))
                opp_l, opp_v = opp_hand_label(env.state, me)
                rec = {
                    "obs": obs_np, "mask": mask_np,
                    "discrete": out["discrete"], "amount01": out["amount01"],
                    "needs": 1.0 if out["needs"] else 0.0,
                    "logp": out["logp"], "value": out["value"],
                    "opp_l": opp_l, "opp_v": opp_v,
                    "round": env.state.round,
                }
                game_records.append((me, rec))
                env.step(out["discrete"], out["amount01"])
            guard += 1
            assert guard < 8000

        rewards = env.rewards
        final_markers = env.state.value_markers
        # 每个学习者按其决策序列做 GAE;补 rank 标签
        by_player = {}
        for pl, rec in game_records:
            by_player.setdefault(pl, []).append(rec)
        for pl, recs in by_player.items():
            R = rewards[pl]
            k = len(recs)
            adv = 0.0
            for t in reversed(range(k)):
                v_t = recs[t]["value"]
                next_v = recs[t + 1]["value"] if t + 1 < k else 0.0
                r_t = R if t == k - 1 else 0.0
                delta = r_t + gamma * next_v - v_t
                adv = delta + gamma * lam * adv
                recs[t]["adv"] = adv
                recs[t]["ret"] = adv + v_t
                # rank 标签:该步所在回合最终放的标码 /30
                rr = recs[t]["round"]
                recs[t]["rank_l"] = np.array(
                    [final_markers[a][rr - 1] / 30.0 for a in ARTISTS], dtype=np.float32)
            steps.extend(recs)

    # 打包成张量
    def stack(key, dtype=torch.float32):
        return torch.as_tensor(np.stack([s[key] for s in steps]), dtype=dtype, device=device)

    batch = {
        "obs": stack("obs"),
        "mask": torch.as_tensor(np.stack([s["mask"] for s in steps]), dtype=torch.bool, device=device),
        "discrete": torch.as_tensor([s["discrete"] for s in steps], dtype=torch.long, device=device),
        "amount01": torch.as_tensor([s["amount01"] for s in steps], dtype=torch.float32, device=device),
        "needs": torch.as_tensor([s["needs"] for s in steps], dtype=torch.float32, device=device),
        "logp": torch.as_tensor([s["logp"] for s in steps], dtype=torch.float32, device=device),
        "adv": torch.as_tensor([s["adv"] for s in steps], dtype=torch.float32, device=device),
        "ret": torch.as_tensor([s["ret"] for s in steps], dtype=torch.float32, device=device),
        "opp_l": stack("opp_l"),
        "opp_v": stack("opp_v"),
        "rank_l": stack("rank_l"),
    }
    return batch, len(steps)


# ── PPO 更新 ──────────────────────────────────────────────────────────────

def ppo_update(net, opt, batch, epochs=4, minibatch=1024, clip=0.2,
               vcoef=0.5, entcoef=0.03, aux_opp_coef=0.5, aux_rank_coef=0.5, max_grad=0.5):
    N = batch["obs"].shape[0]
    adv = batch["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    idx_all = np.arange(N)
    stats = {}
    for _ in range(epochs):
        np.random.shuffle(idx_all)
        for start in range(0, N, minibatch):
            mb = torch.as_tensor(idx_all[start:start + minibatch], device=batch["obs"].device)
            logp, ent, value, aux_opp, aux_rank = net.evaluate_actions(
                batch["obs"][mb], batch["mask"][mb], batch["discrete"][mb],
                batch["amount01"][mb], batch["needs"][mb])
            ratio = torch.exp(logp - batch["logp"][mb])
            a = adv[mb]
            pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
            vloss = 0.5 * ((value - batch["ret"][mb]) ** 2).mean()
            entropy = ent.mean()
            ov = batch["opp_v"][mb]
            aux_opp_loss = (((aux_opp - batch["opp_l"][mb]) ** 2) * ov).sum() / ov.sum().clamp(min=1)
            aux_rank_loss = ((aux_rank - batch["rank_l"][mb]) ** 2).mean()
            loss = pg + vcoef * vloss - entcoef * entropy + aux_opp_coef * aux_opp_loss + aux_rank_coef * aux_rank_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad)
            opt.step()
            stats = {"pg": pg.item(), "v": vloss.item(), "ent": entropy.item(),
                     "aux_opp": aux_opp_loss.item(), "aux_rank": aux_rank_loss.item()}
    return stats


# ── 评估:对基线胜率 + 相对分 ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(net, device, baseline_name, num_players, games, seed=12345):
    rng = np.random.default_rng(seed)
    wins = 0.0
    rel_sum = 0.0
    for g in range(games):
        env = SelfPlayEnv()
        env.reset(num_players, seed=int(rng.integers(1 << 30)))
        opps = {i: AGENT_REGISTRY[baseline_name](seed=int(rng.integers(1 << 30)))
                for i in range(1, num_players)}
        while not env.done:
            me = env.current_player()
            if me == 0:
                obs = torch.from_numpy(env.observe(0)[0]).to(device)
                mask = torch.from_numpy(env.observe(0)[1]).to(device)
                d, x = net.act_greedy(obs, mask)
                env.step(d, x)
            else:
                env.step_action(opps[me].act(env.state))
        w = game.winner(env.state)
        wins += (1.0 / len(w)) if 0 in w else 0.0
        rel_sum += env.rewards[0]
    return wins / games, rel_sum / games


# ── 主循环 ────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description="现代艺术 自对弈 PPO 训练")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--games", type=int, default=128, help="每轮采样局数")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.03, help="熵系数(防过早收敛)")
    ap.add_argument("--league-frac", type=float, default=0.5)
    ap.add_argument("--snapshot-every", type=int, default=10, help="每多少轮存一个自对弈快照")
    ap.add_argument("--pool-cap", type=int, default=8, help="快照对手池上限")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-games", type=int, default=200)
    ap.add_argument("--save", default=os.path.join(CKPT_DIR, "policy.pt"))
    ap.add_argument("--smoke", action="store_true", help="快速冒烟:小规模跑通并看是否学起来")
    args = ap.parse_args(argv)

    if args.smoke:
        args.iters, args.games, args.eval_games = 8, 64, 100

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    net = ActorCritic().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"设备 {device} | OBS_DIM {OBS_DIM} | 动作 {NUM_DISCRETE} | 参数 "
          f"{sum(p.numel() for p in net.parameters())/1e3:.0f}k | ent {args.ent_coef} league {args.league_frac}")
    wr0, rel0 = evaluate(net, device, "random", 4, args.eval_games)
    print(f"[初始] vs random(4人): 胜率 {wr0:.2%}  相对分 {rel0:+.2f}")

    # 自对弈快照对手池:league 局的对手 = 结构化基线 或 冻结的历史自己
    snapshots: list[FrozenNet] = []

    def opp_provider(n, rng_):
        if snapshots and rng_.random() < 0.5:
            fn = snapshots[int(rng_.integers(len(snapshots)))]
            return {i: fn for i in range(1, n)}
        return default_opp_provider(n, rng_)

    for it in range(1, args.iters + 1):
        t0 = time.time()
        batch, nsteps = collect(net, device, args.games, rng,
                                league_frac=args.league_frac, opp_provider=opp_provider)
        stats = ppo_update(net, opt, batch, epochs=args.epochs, minibatch=args.minibatch,
                           entcoef=args.ent_coef)
        # 定期把当前策略冻结进对手池
        if it % args.snapshot_every == 0:
            frozen = ActorCritic().to(device)
            frozen.load_state_dict({k: v.detach().clone() for k, v in net.state_dict().items()})
            frozen.eval()
            snapshots.append(FrozenNet(frozen, device))
            if len(snapshots) > args.pool_cap:
                snapshots.pop(0)
        dt = time.time() - t0
        if it % max(1, args.iters // 10) == 0 or it == args.iters or args.smoke:
            wr_r, _ = evaluate(net, device, "random", 4, args.eval_games // 2)
            wr_h, _ = evaluate(net, device, "heuristic", 4, args.eval_games // 2)
            wr_s, rel_s = evaluate(net, device, "standard", 4, args.eval_games // 2)
            wr_a, _ = evaluate(net, device, "aggressive", 4, args.eval_games // 2)
            print(f"[{it:>3}/{args.iters}] {nsteps}步 {dt:.1f}s | "
                  f"pg {stats['pg']:+.3f} v {stats['v']:.3f} ent {stats['ent']:.2f} "
                  f"aux_opp {stats['aux_opp']:.3f} aux_rank {stats['aux_rank']:.3f} | "
                  f"vs 随机 {wr_r:.0%} 启发 {wr_h:.0%} 标准 {wr_s:.0%} 莽夫 {wr_a:.0%} (标准相对分 {rel_s:+.1f})")
        torch.save({"model": net.state_dict(), "obs_dim": OBS_DIM}, args.save)

    print(f"完成。检查点已存:{args.save}")


if __name__ == "__main__":
    main()
