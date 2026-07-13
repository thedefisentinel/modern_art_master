"""标准 AI("一般"难度):可解释的中等强度对手,用于试玩与验证游戏逻辑。

核心估价:一张画的心理价位 ≈ 该画家「若按当前本回合张数所处名次结算」时的每张价值
         = 累计价值(value_board) + 当前名次对应的 30/20/10(名次外给一点点期望值)。

行为:
  * 买入:出价一路加到心理价位为止(不再只出最低价),便宜就买,贵了就放弃;
  * 卖家亦参与竞价:若场上价低于自己心理价位,宁可自购(付银行仍有赚);
  * 暗标:按心理价位打折出价,规避"赢家诅咒";
  * 上架:优先拍卖当前预计价值最高、且自己有存货的画家,尽量把现金拿到手;
  * 双张:手上有同画家备牌且该画家有价值时补牌(对分规则下仍有收益)。
比启发式基线更接近真实博弈,足以让人类试玩时感到有对抗、便于检验逻辑是否自洽。
"""

from __future__ import annotations

from collections import Counter

from ..engine import game
from ..engine.state import GameState
from ..engine.rules import ARTISTS, ARTIST_PRIORITY, RANK_VALUES, AuctionType
from ..engine.actions import (
    Action, ChooseCard, AddSecond, DeclineAdd, Bid, PassBid, SealedBid, Buy, PassBuy,
)
from .base import Agent

# 上架/补牌时,同等价值下的拍卖类型偏好(市场化的类型更能拍出价)
_TYPE_PREF = [
    AuctionType.ONCE_AROUND,
    AuctionType.OPEN,
    AuctionType.SEALED,
    AuctionType.FIXED_PRICE,
    AuctionType.DOUBLE,
]


class StandardAgent(Agent):
    # ── 估价 ──────────────────────────────────────────────────────────────
    @staticmethod
    def _projected_value(state: GameState, artist) -> int:
        """按当前本回合张数推断该画家的名次,估算其每张画的结算价值。"""
        counts = state.round_counts
        ranked = sorted(ARTISTS, key=lambda a: (-counts[a], ARTIST_PRIORITY[a]))
        pos = ranked.index(artist)                 # 0-based 预计名次
        bonus = RANK_VALUES[pos] if pos < 3 else 5  # 名次外仍给一点爬升期望
        return state.value_board[artist] + bonus

    def _willing(self, state: GameState, artist, ncards: int) -> int:
        return self._projected_value(state, artist) * ncards

    # ── 决策入口 ──────────────────────────────────────────────────────────
    def act(self, state: GameState) -> Action:
        ph = state.phase.value
        me = game.current_player(state)
        if ph == "choose":
            return self._choose(state, me)
        if ph == "double_offer":
            return self._double_offer(state, me)
        if ph in ("bid_open", "bid_once"):
            return self._bid(state, me)
        if ph == "bid_sealed":
            return self._sealed(state, me)
        if ph == "buy_fixed":
            return self._buy_fixed(state, me)
        return game.legal_actions(state)[0]

    # ── 上架选牌 ──────────────────────────────────────────────────────────
    def _choose(self, state: GameState, me: int) -> Action:
        hand = state.players[me].hand
        counts = Counter(c.artist for c in hand)
        best_i, best_key = 0, None
        for i, card in enumerate(hand):
            v = self._projected_value(state, card.artist)
            # 尽量不主动打出第 5 张(会立刻结束回合);略微降低其优先级
            if state.round_counts[card.artist] == 4:
                v -= 15
            # 同价值下:自己存货多的画家更值得拍(抬升名次),市场化类型优先
            key = (v, counts[card.artist], -_TYPE_PREF.index(card.auction))
            if best_key is None or key > best_key:
                best_key, best_i = key, i
        card = hand[best_i]
        if card.auction == AuctionType.FIXED_PRICE:
            v = self._projected_value(state, card.artist)
            price = min(state.players[me].money, max(1, int(v * 0.8)))
            return ChooseCard(best_i, price=price)
        return ChooseCard(best_i)

    # ── 双张补牌 ──────────────────────────────────────────────────────────
    def _double_offer(self, state: GameState, me: int) -> Action:
        acts = game.legal_actions(state)
        adds = [a for a in acts if isinstance(a, AddSecond)]
        if not adds:
            return DeclineAdd()
        artist = state.auction.artist
        # 该画家没什么价值就不浪费备牌
        if self._projected_value(state, artist) < 12:
            return DeclineAdd()
        add = min(adds, key=lambda a: _TYPE_PREF.index(state.players[me].hand[a.card_index].auction))
        card = state.players[me].hand[add.card_index]
        if card.auction == AuctionType.FIXED_PRICE:
            v = self._projected_value(state, artist)
            price = min(state.players[me].money, max(1, int(v * 0.8)))
            return AddSecond(add.card_index, price=price)
        return AddSecond(add.card_index)

    # ── 公开 / 轮流出价 ──────────────────────────────────────────────────
    def _bid(self, state: GameState, me: int) -> Action:
        a = state.auction
        lo, hi = game.bid_bounds(state)
        willing = self._willing(state, a.artist, len(a.cards))
        # 一路加到心理价位为止;卖家同样参与(价低则自购,付银行仍有赚)
        if lo <= willing and lo <= hi:
            return Bid(lo)
        return PassBid()

    # ── 暗标 ──────────────────────────────────────────────────────────────
    def _sealed(self, state: GameState, me: int) -> Action:
        a = state.auction
        lo, hi = game.bid_bounds(state)
        willing = self._willing(state, a.artist, len(a.cards))
        bid = max(0, min(hi, int(willing * 0.75)))  # 打折规避赢家诅咒
        return SealedBid(bid)

    # ── 一口价 ────────────────────────────────────────────────────────────
    def _buy_fixed(self, state: GameState, me: int) -> Action:
        a = state.auction
        price = a.high
        willing = self._willing(state, a.artist, len(a.cards))
        if price <= willing and price <= state.players[me].money:
            return Buy()
        return PassBuy()
