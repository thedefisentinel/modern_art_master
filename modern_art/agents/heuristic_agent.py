"""简单启发式基线(占位,待强化学习取代)。

核心思想:一张画的"心理价位"≈ 该艺术家累计价值 + 本回合热度带来的名次增益期望。
  * 买入:出价不超过心理价位就有利可图(回合结束按累计价值变现),故尽量低价拿下,
         心理价位不足则放弃。
  * 上架:优先拍卖"当前最有价值且自己手上还有存货"的艺术家,以抬升其名次。
  * 双张:若手上有同艺术家第二张,愿意补牌当卖家收钱。
这套策略不追求最优,只作为可解释的对照与陪练。
"""

from __future__ import annotations

from ..engine import game
from ..engine.state import GameState
from ..engine.rules import AuctionType, RANK_VALUES
from ..engine.actions import (
    Action, ChooseCard, AddSecond, DeclineAdd, Bid, PassBid, SealedBid, Buy, PassBuy,
)
from .base import Agent

_TYPE_PREFERENCE = [
    AuctionType.ONCE_AROUND,
    AuctionType.OPEN,
    AuctionType.SEALED,
    AuctionType.FIXED_PRICE,
    AuctionType.DOUBLE,
]


class HeuristicAgent(Agent):
    # ── 估值 ──────────────────────────────────────────────────────────────
    @staticmethod
    def _estimate(state: GameState, artist) -> int:
        vb = state.value_board[artist]
        cnt = state.round_counts[artist]
        if cnt >= 4:
            momentum = RANK_VALUES[0]
        elif cnt >= 2:
            momentum = RANK_VALUES[1]
        elif cnt >= 1:
            momentum = RANK_VALUES[2]
        else:
            momentum = 0
        return vb + momentum

    # ── 决策入口 ──────────────────────────────────────────────────────────
    def act(self, state: GameState) -> Action:
        phase = state.phase.value
        me = game.current_player(state)
        if phase == "choose":
            return self._choose(state, me)
        if phase == "double_offer":
            return self._double_offer(state, me)
        if phase in ("bid_open", "bid_once"):
            return self._bid(state, me)
        if phase == "bid_sealed":
            return self._sealed(state, me)
        if phase == "buy_fixed":
            return self._buy_fixed(state, me)
        # 兜底
        return game.legal_actions(state)[0]

    # ── 上架选牌 ──────────────────────────────────────────────────────────
    def _choose(self, state: GameState, me: int) -> Action:
        hand = state.players[me].hand
        # 按 (估值, 手上同艺术家存货数) 排序,优先拍卖高价值且存货多的艺术家
        from collections import Counter
        counts = Counter(c.artist for c in hand)
        best_idx = 0
        best_key = (-1, -1, 99)
        for i, card in enumerate(hand):
            pref = _TYPE_PREFERENCE.index(card.auction)
            key = (self._estimate(state, card.artist), counts[card.artist], -pref)
            if key > best_key:
                best_key = key
                best_idx = i
        card = hand[best_idx]
        if card.auction == AuctionType.FIXED_PRICE:
            est = self._estimate(state, card.artist)
            price = max(0, min(state.players[me].money, int(est * 0.6)))
            return ChooseCard(best_idx, price=price)
        return ChooseCard(best_idx)

    # ── 双张补牌 ──────────────────────────────────────────────────────────
    def _double_offer(self, state: GameState, me: int) -> Action:
        acts = game.legal_actions(state)
        adds = [a for a in acts if isinstance(a, AddSecond)]
        if not adds:
            return DeclineAdd()
        # 有同艺术家牌可补 -> 补一张当卖家收钱(优先非一口价)
        add = min(adds, key=lambda a: _TYPE_PREFERENCE.index(
            state.players[me].hand[a.card_index].auction))
        card = state.players[me].hand[add.card_index]
        if card.auction == AuctionType.FIXED_PRICE:
            est = self._estimate(state, card.artist)
            price = max(0, min(state.players[me].money, int(est * 0.6)))
            return AddSecond(add.card_index, price=price)
        return AddSecond(add.card_index)

    # ── 公开 / 轮流出价 ──────────────────────────────────────────────────
    def _bid(self, state: GameState, me: int) -> Action:
        a = state.auction
        lo, hi = game.bid_bounds(state)
        willing = self._estimate(state, a.artist) * len(a.cards)
        # 卖家自己一般不抬价
        if me == a.seller:
            return PassBid()
        if lo <= willing and lo <= hi:
            return Bid(lo)  # 以最低必要价拿下
        return PassBid()

    # ── 暗标 ──────────────────────────────────────────────────────────────
    def _sealed(self, state: GameState, me: int) -> Action:
        a = state.auction
        lo, hi = game.bid_bounds(state)
        willing = self._estimate(state, a.artist) * len(a.cards)
        bid = max(0, min(hi, int(willing * 0.6)))
        return SealedBid(bid)

    # ── 一口价 ────────────────────────────────────────────────────────────
    def _buy_fixed(self, state: GameState, me: int) -> Action:
        a = state.auction
        price = a.high
        willing = self._estimate(state, a.artist) * len(a.cards)
        if price <= willing and price <= state.players[me].money:
            return Buy()
        return PassBuy()
