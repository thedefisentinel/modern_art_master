"""观测 / 动作 的编解码 + 合法掩码。纯函数,不依赖 torch,便于单测。

观测:定长 float 向量(与人数无关,用 MAX_PLAYERS 补零 + 有效位)。
动作:参数化 ——
  * 离散头:36 个动作(选牌 25 + 补牌 5 + 放弃补牌/出价/放弃出价/暗标/买/不买 6),带掩码;
  * 连续头:1 个标量 amount∈[0,1],仅金额型动作使用,按上下文映射到 [lo,hi]。
解码保证:只要传入的离散动作在掩码内,decode 出的引擎动作一定合法(有测试保证)。
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..engine import game
from ..engine.state import GameState, Phase
from ..engine.rules import (
    ARTISTS, AUCTION_TYPES, ARTIST_PRIORITY, RANK_VALUES, AuctionType,
)
from ..engine.actions import (
    Action, ChooseCard, AddSecond, DeclineAdd, Bid, PassBid, SealedBid, Buy, PassBuy,
)

MAX_PLAYERS = 5
NA = len(ARTISTS)        # 5 画家
NT = len(AUCTION_TYPES)  # 5 拍卖类型
_AIDX = {a: i for i, a in enumerate(ARTISTS)}          # 画家 -> 0..4(=优先级顺序)
_TIDX = {t: i for i, t in enumerate(AUCTION_TYPES)}    # 类型 -> 0..4

# ── 离散动作索引布局 ──────────────────────────────────────────────────────
CHOOSE_BASE = 0                 # 0..24 : 上架 (画家 a, 类型 t) = a*NT + t
ADD_BASE = NA * NT              # 25..29: 双张补牌,补类型 t(画家固定=拍卖画家)
DECLINE_ADD = ADD_BASE + NT     # 30
BID = DECLINE_ADD + 1           # 31
PASS_BID = BID + 1              # 32
SEALED_BID = PASS_BID + 1       # 33
BUY = SEALED_BID + 1            # 34
PASS_BUY = BUY + 1              # 35
NUM_DISCRETE = PASS_BUY + 1     # 36


# ══════════════════════════════════════════════════════════════════════════
# 估值(与标准 AI 同一套:按本回合张数预计名次)
# ══════════════════════════════════════════════════════════════════════════

def projected_value(state: GameState, artist) -> int:
    counts = state.round_counts
    ranked = sorted(ARTISTS, key=lambda a: (-counts[a], ARTIST_PRIORITY[a]))
    pos = ranked.index(artist)
    bonus = RANK_VALUES[pos] if pos < 3 else 5
    return state.value_board[artist] + bonus


# ══════════════════════════════════════════════════════════════════════════
# 观测编码
# ══════════════════════════════════════════════════════════════════════════

_PHASES = ["choose", "double_offer", "bid_open", "bid_once", "bid_sealed", "buy_fixed", "game_over"]


def encode_obs(state: GameState, me: int) -> np.ndarray:
    """从 me 视角编码定长观测。对手手牌/金钱不泄露;收藏、面板、当前拍卖为公开。"""
    s = state
    n = s.num_players
    rel = [(me + k) % n for k in range(n)]     # 座位相对顺序:自己在 slot 0
    f: list[float] = []

    # 1) 自己手牌:每 (画家,类型) 张数(25)+ 每画家合计(5)
    hand_ct = Counter((c.artist, c.auction) for c in s.players[me].hand)
    for a in ARTISTS:
        for t in AUCTION_TYPES:
            f.append(hand_ct.get((a, t), 0) / 4.0)
    for a in ARTISTS:
        f.append(sum(hand_ct.get((a, t), 0) for t in AUCTION_TYPES) / 8.0)

    # 2) 自己现金(1)
    f.append(s.players[me].money / 200.0)

    # 3) 回合 one-hot(4)、阶段 one-hot(7)、人数 one-hot(3)
    for r in range(1, 5):
        f.append(1.0 if s.round == r else 0.0)
    for ph in _PHASES:
        f.append(1.0 if s.phase.value == ph else 0.0)
    for pc in (3, 4, 5):
        f.append(1.0 if n == pc else 0.0)

    # 4) 价值板:每回合标码(5×4=20)、累计价值(5)、本回合张数(5)
    for a in ARTISTS:
        for r in range(4):
            f.append(s.value_markers[a][r] / 30.0)
    for a in ARTISTS:
        f.append(s.value_board[a] / 90.0)
    for a in ARTISTS:
        f.append(s.round_counts[a] / 5.0)

    # 5) 各座位公开信息(MAX_PLAYERS 槽,座位相对,含自己;金钱不含)
    #    每槽:有效位、手牌数、收藏每画家(5)、收藏总数、累计成交付款 = 9
    #    累计付款 = 对手"花钱/激进程度"的公开信号(暗标看不到当前出价时靠它推断)
    for slot in range(MAX_PLAYERS):
        if slot < n:
            q = rel[slot]
            f.append(1.0)
            f.append(len(s.players[q].hand) / 16.0)
            col = Counter(c.artist for c in s.players[q].purchases)
            for a in ARTISTS:
                f.append(col.get(a, 0) / 5.0)
            f.append(len(s.players[q].purchases) / 8.0)
            f.append(s.players[q].paid_total / 200.0)
        else:
            f.extend([0.0] * (1 + 1 + NA + 1 + 1))

    # 6) 当前拍卖上下文
    a = s.auction
    if a is not None:
        f.append(1.0)
        for art in ARTISTS:
            f.append(1.0 if a.artist == art else 0.0)
        for t in AUCTION_TYPES:
            f.append(1.0 if a.auction_type == t else 0.0)
        f.append(1.0 if len(a.cards) == 2 else 0.0)
        f.append((len(a.cards) or 1) / 2.0)
        srel = rel.index(a.seller) if a.seller in rel else -1
        for slot in range(MAX_PLAYERS):
            f.append(1.0 if slot == srel else 0.0)
        f.append(a.high / 200.0)
        hbrel = rel.index(a.high_bidder) if (a.high_bidder is not None and a.high_bidder in rel) else -1
        for slot in range(MAX_PLAYERS):
            f.append(1.0 if slot == hbrel else 0.0)
        f.append(1.0 if a.high_bidder is None else 0.0)
        for t in AUCTION_TYPES:
            f.append(1.0 if (a.double_card is not None and a.double_card.auction == t) else 0.0)
        f.append(projected_value(s, a.artist) / 90.0)
        # 各座位在本次拍卖里的当前出价 + 是否仍在公开竞价(让非公开拍卖也能"看别人出价")
        # 公开/轮流:player_bids 是各家已喊的价(公开);暗标:为 0(规则上看不到)
        for slot in range(MAX_PLAYERS):
            if slot < n:
                q = rel[slot]
                f.append(a.player_bids.get(q, 0) / 200.0)
                f.append(1.0 if q in a.active else 0.0)
            else:
                f.extend([0.0, 0.0])
    else:
        zlen = (1 + NA + NT + 1 + 1 + MAX_PLAYERS + 1 + MAX_PLAYERS + 1 + NT + 1
                + MAX_PLAYERS * 2)
        f.extend([0.0] * zlen)

    return np.asarray(f, dtype=np.float32)


# 观测维度(编码一次空局求得,供网络使用)
OBS_DIM = int(encode_obs(game.new_game(3, seed=0), 0).shape[0])


# ══════════════════════════════════════════════════════════════════════════
# 合法掩码
# ══════════════════════════════════════════════════════════════════════════

def legal_mask(state: GameState, me: int) -> np.ndarray:
    """返回 (NUM_DISCRETE,) 的 bool 掩码:True 表示该离散动作在当前阶段合法。"""
    mask = np.zeros(NUM_DISCRETE, dtype=bool)
    s = state
    phase = s.phase
    hand = s.players[me].hand

    if phase == Phase.CHOOSE:
        present = {(c.artist, c.auction) for c in hand}
        for (art, t) in present:
            mask[CHOOSE_BASE + _AIDX[art] * NT + _TIDX[t]] = True

    elif phase == Phase.DOUBLE_OFFER:
        mask[DECLINE_ADD] = True
        art = s.auction.artist
        for c in hand:
            if c.artist == art and c.auction != AuctionType.DOUBLE:
                mask[ADD_BASE + _TIDX[c.auction]] = True

    elif phase in (Phase.BID_OPEN, Phase.BID_ONCE):
        mask[PASS_BID] = True
        lo, hi = game.bid_bounds(s)
        if hi >= lo:  # 有能力加价
            mask[BID] = True

    elif phase == Phase.BID_SEALED:
        mask[SEALED_BID] = True

    elif phase == Phase.BUY_FIXED:
        mask[PASS_BUY] = True
        if s.players[me].money >= s.auction.high:
            mask[BUY] = True

    return mask


# ══════════════════════════════════════════════════════════════════════════
# 金额区间 + 解码
# ══════════════════════════════════════════════════════════════════════════

def max_round_value(state: GameState, artist) -> int:
    """一张该画家的画在本回合能兑现的**最大价值**:累计价值 + 本轮升到第1名的 +30。
    (本回合买的画只在本回合末结算一次,之后进美术馆,故理性出价不超过此上限。)"""
    return state.value_board[artist] + RANK_VALUES[0]


def _value_cap(state: GameState, me: int, artist, ncards: int) -> int:
    """出价上限:min(现金, 张数 × 单张本回合最大价值)。"""
    money = state.players[me].money
    return max(0, min(money, ncards * max_round_value(state, artist)))


def amount_range(state: GameState, me: int, discrete: int) -> tuple[int, int] | None:
    """该离散动作若需金额,返回 (lo, hi);否则 None。
    金额上限 = min(现金, 张数×(累计价值+30)) —— 本回合该(组)画的最大可能价值。"""
    s = state
    if CHOOSE_BASE <= discrete < ADD_BASE:
        t = AUCTION_TYPES[(discrete - CHOOSE_BASE) % NT]
        if t != AuctionType.FIXED_PRICE:
            return None
        art = ARTISTS[(discrete - CHOOSE_BASE) // NT]
        return (0, _value_cap(s, me, art, 1))          # 直接一口价:单张
    if ADD_BASE <= discrete < DECLINE_ADD:
        t = AUCTION_TYPES[discrete - ADD_BASE]
        if t != AuctionType.FIXED_PRICE:
            return None
        return (0, _value_cap(s, me, s.auction.artist, 2))  # 双张补牌:两张一起
    if discrete == BID:
        a = s.auction
        lo = a.high + 1
        return (lo, max(lo, _value_cap(s, me, a.artist, max(1, len(a.cards)))))
    if discrete == SEALED_BID:
        a = s.auction
        return (0, _value_cap(s, me, a.artist, max(1, len(a.cards))))
    return None


def _amount_from01(rng: tuple[int, int], amount01: float) -> int:
    lo, hi = rng
    if hi < lo:
        return lo
    x = min(1.0, max(0.0, float(amount01)))
    return int(round(lo + x * (hi - lo)))


def _find_hand_index(hand, artist, atype) -> int:
    for i, c in enumerate(hand):
        if c.artist == artist and c.auction == atype:
            return i
    raise ValueError(f"手牌中找不到 {artist}/{atype}(掩码应已排除)")


def decode_action(state: GameState, me: int, discrete: int, amount01: float = 0.0) -> Action:
    """把 (离散动作, 连续金额∈[0,1]) 解成引擎动作。前提:discrete 在 legal_mask 内。"""
    s = state
    hand = s.players[me].hand
    rng = amount_range(s, me, discrete)
    amt = _amount_from01(rng, amount01) if rng is not None else None

    if CHOOSE_BASE <= discrete < ADD_BASE:
        art = ARTISTS[(discrete - CHOOSE_BASE) // NT]
        t = AUCTION_TYPES[(discrete - CHOOSE_BASE) % NT]
        idx = _find_hand_index(hand, art, t)
        return ChooseCard(idx, price=amt) if t == AuctionType.FIXED_PRICE else ChooseCard(idx)

    if ADD_BASE <= discrete < DECLINE_ADD:
        t = AUCTION_TYPES[discrete - ADD_BASE]
        art = s.auction.artist
        idx = _find_hand_index(hand, art, t)
        return AddSecond(idx, price=amt) if t == AuctionType.FIXED_PRICE else AddSecond(idx)

    if discrete == DECLINE_ADD:
        return DeclineAdd()
    if discrete == BID:
        return Bid(amt)
    if discrete == PASS_BID:
        return PassBid()
    if discrete == SEALED_BID:
        return SealedBid(amt)
    if discrete == BUY:
        return Buy()
    if discrete == PASS_BUY:
        return PassBuy()
    raise ValueError(f"未知离散动作 {discrete}")


def needs_amount(state: GameState, me: int, discrete: int) -> bool:
    """该动作是否用到连续金额头(用于训练时决定是否计入连续项 log-prob)。"""
    return amount_range(state, me, discrete) is not None
