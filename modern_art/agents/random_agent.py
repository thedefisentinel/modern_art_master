"""随机基线:在合法动作中随机选择,金额在合法范围内随机(适度克制,避免总是梭哈)。"""

from __future__ import annotations

from ..engine import game
from ..engine.state import GameState
from ..engine.rules import AuctionType
from ..engine.actions import (
    Action, ChooseCard, AddSecond, Bid, SealedBid,
)
from .base import Agent


class RandomAgent(Agent):
    def act(self, state: GameState) -> Action:
        me = game.current_player(state)
        acts = game.legal_actions(state)
        act = self.rng.choice(acts)
        return self._concretize(state, me, act)

    def _concretize(self, state: GameState, me: int, act: Action) -> Action:
        money = state.players[me].money
        if isinstance(act, ChooseCard):
            card = state.players[me].hand[act.card_index]
            if card.auction == AuctionType.FIXED_PRICE:
                return ChooseCard(act.card_index, price=self.rng.randint(0, money))
            return ChooseCard(act.card_index)
        if isinstance(act, AddSecond):
            card = state.players[me].hand[act.card_index]
            if card.auction == AuctionType.FIXED_PRICE:
                return AddSecond(act.card_index, price=self.rng.randint(0, money))
            return AddSecond(act.card_index)
        if isinstance(act, Bid):
            lo, hi = game.bid_bounds(state)
            hi = min(hi, lo + 40)  # 克制加价幅度
            return Bid(self.rng.randint(lo, hi))
        if isinstance(act, SealedBid):
            lo, hi = game.bid_bounds(state)
            hi = min(hi, 40)
            return SealedBid(self.rng.randint(lo, hi))
        return act
