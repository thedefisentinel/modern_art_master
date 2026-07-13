"""导出对拍数据:随机玩多局,每步记录 (前态, 观测, 掩码, 合法动作, 所选动作, 后态)。
供 Node 用 engine.js 复现比对。输出 tools/parity.json。"""
import json, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from modern_art.engine import game
from modern_art.engine.actions import ChooseCard, AddSecond, Bid, SealedBid
from modern_art.engine.rules import AuctionType as T
from modern_art.rl import encoding

def concretize(s, act, rng):
    me = game.current_player(s)
    if isinstance(act, ChooseCard):
        c = s.players[me].hand[act.card_index]
        return ChooseCard(act.card_index, price=rng.randint(0, s.players[me].money)) if c.auction==T.FIXED_PRICE else ChooseCard(act.card_index)
    if isinstance(act, AddSecond):
        c = s.players[me].hand[act.card_index]
        return AddSecond(act.card_index, price=rng.randint(0, s.players[me].money)) if c.auction==T.FIXED_PRICE else AddSecond(act.card_index)
    if isinstance(act, Bid):
        lo, hi = game.bid_bounds(s); return Bid(rng.randint(lo, hi))
    if isinstance(act, SealedBid):
        lo, hi = game.bid_bounds(s); return SealedBid(rng.randint(lo, hi))
    return act

records = []
for n in (3, 4, 5):
    for seed in range(12):
        rng = random.Random(9000 + n*100 + seed)
        s = game.new_game(n, seed=seed)
        steps = 0
        while not game.is_over(s):
            me = game.current_player(s)
            before = s.to_dict()
            obs = encoding.encode_obs(s, me).tolist()
            mask = encoding.legal_mask(s, me).tolist()
            legal = [a.to_dict() for a in game.legal_actions(s)]
            act = concretize(s, rng.choice(game.legal_actions(s)), rng)
            s2 = game.apply(s, act)
            records.append({"me": me, "before": before, "obs": obs, "mask": [bool(x) for x in mask],
                            "legal": legal, "action": act.to_dict(), "after": s2.to_dict()})
            s = s2
            steps += 1
            assert steps < 5000

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parity.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(records, f)
print(f"导出 {len(records)} 条对拍记录 -> {out}")
