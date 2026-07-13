"""性格原型对手,用于扩大训练/评估的对手多样性(群体训练,而非只自对弈)。

真实对局里对手风格各异。若只跟"理性的自己"对练,学出来的策略在遇到非理性对手时
可能吃亏——例如一个理性人 + 三个莽夫:莽夫疯狂高价抢画,理性人若也去抢就亏,
真正该做的是把画高价卖给莽夫。要学会这类 general 打法,训练里必须有这些原型。

  AggressiveAgent(莽夫):买画很狠、出价大方(按预计价值的高倍数出价),常常超额付款。
  TightAgent(铁公鸡):几乎只出最低价、能不买就不买。
"""

from __future__ import annotations

from ..engine import game
from ..engine.rules import ARTISTS, ARTIST_PRIORITY, RANK_VALUES, AuctionType
from ..engine.actions import (
    Action, ChooseCard, AddSecond, DeclineAdd, Bid, PassBid, SealedBid, Buy, PassBuy,
)
from .base import Agent


def _proj(state, artist) -> int:
    """按本回合张数预计名次的估值(理性人视角:更保守)。"""
    counts = state.round_counts
    ranked = sorted(ARTISTS, key=lambda a: (-counts[a], ARTIST_PRIORITY[a]))
    pos = ranked.index(artist)
    return state.value_board[artist] + (RANK_VALUES[pos] if pos < 3 else 5)


def _max_value(state, artist) -> int:
    """这张画本回合的**最大可能价值**:累计价值 + 升到第 1 名的 30。
    任何不傻的玩家(含莽夫)都不会出到超过它。"""
    return state.value_board[artist] + RANK_VALUES[0]


class AggressiveAgent(Agent):
    """莽夫:乐观又大方,愿意按"最大可能价值"出满,但**绝不超过**这个上限。

    与理性人的区别:理性人按预计名次估值(留利润空间、且并非每画家都进前三),
    出价明显低于上限;莽夫假设每张画都能升到第 1 名,几乎顶格出价,常常在期望上亏,
    因此理性人可以"把画高价卖给莽夫"获利——但莽夫不会送钱到上限之外。
    """

    def __init__(self, aggr: float = 0.85, name: str = "aggressive", seed=None):
        super().__init__(name=name, seed=seed)
        self.aggr = aggr    # 出价占"最大可能价值"的比例(<1,留点余量;仍明显高于理性人)

    def _want(self, state, me, artist, ncards):
        cap = int(self.aggr * ncards * _max_value(state, artist))
        return max(0, min(state.players[me].money, cap))

    def act(self, state) -> Action:
        me = game.current_player(state)
        ph = state.phase.value
        if ph == "choose":
            hand = state.players[me].hand
            i = self.rng.randrange(len(hand))       # 莽夫随手拍一张
            c = hand[i]
            if c.auction == AuctionType.FIXED_PRICE:
                # 一口价定得偏低,想快点脱手(或引诱别人接)
                price = min(state.players[me].money, max(1, _proj(state, c.artist) // 2))
                return ChooseCard(i, price=price)
            return ChooseCard(i)
        if ph == "double_offer":
            adds = [a for a in game.legal_actions(state) if isinstance(a, AddSecond)]
            if not adds:
                return DeclineAdd()
            a = adds[0]
            c = state.players[me].hand[a.card_index]
            if c.auction == AuctionType.FIXED_PRICE:
                return AddSecond(a.card_index,
                                 price=min(state.players[me].money, max(1, _proj(state, c.artist) // 2)))
            return AddSecond(a.card_index)
        if ph in ("bid_open", "bid_once"):
            a = state.auction
            lo, hi = game.bid_bounds(state)
            want = self._want(state, me, a.artist, len(a.cards))
            if lo <= min(want, hi):
                return Bid(min(hi, want))            # 一路加到高价
            return PassBid()
        if ph == "bid_sealed":
            a = state.auction
            lo, hi = game.bid_bounds(state)
            want = self._want(state, me, a.artist, len(a.cards))
            return SealedBid(max(0, min(hi, want)))
        if ph == "buy_fixed":
            a = state.auction
            return Buy() if state.players[me].money >= a.high else PassBuy()
        return game.legal_actions(state)[0]


class TightAgent(Agent):
    """铁公鸡:只肯出最低价,能不买就不买;当卖家时定高价想套现。"""

    def act(self, state) -> Action:
        me = game.current_player(state)
        ph = state.phase.value
        if ph == "choose":
            hand = state.players[me].hand
            i = self.rng.randrange(len(hand))
            c = hand[i]
            if c.auction == AuctionType.FIXED_PRICE:
                price = min(state.players[me].money, max(1, _proj(state, c.artist)))
                return ChooseCard(i, price=price)
            return ChooseCard(i)
        if ph == "double_offer":
            return DeclineAdd()
        if ph in ("bid_open", "bid_once"):
            a = state.auction
            lo, hi = game.bid_bounds(state)
            # 只在极便宜(低于半价)时才最低价跟一手,否则放弃
            if lo <= hi and lo <= _proj(state, a.artist) * len(a.cards) // 2:
                return Bid(lo)
            return PassBid()
        if ph == "bid_sealed":
            lo, hi = game.bid_bounds(state)
            return SealedBid(lo)                      # 最低(通常 0)
        if ph == "buy_fixed":
            a = state.auction
            if state.players[me].money >= a.high and a.high <= _proj(state, a.artist):
                return Buy()
            return PassBuy()
        return game.legal_actions(state)[0]
