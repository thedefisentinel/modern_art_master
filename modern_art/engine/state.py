"""游戏状态的数据结构。

状态是纯数据(可深拷贝、可序列化为 JSON)。所有状态转移逻辑在 game.py,
这里只定义"形状"。设计成对 RL 友好:状态可克隆用于搜索,可序列化为观测。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .rules import ArtistId, AuctionType


class Phase(str, Enum):
    """状态机的阶段。决定当前"该谁、做什么"。"""

    CHOOSE = "choose"            # 拍卖师从手牌选一张上架
    DOUBLE_OFFER = "double_offer"  # 双张:依次询问谁来补第二张
    BID_OPEN = "bid_open"        # 公开拍卖竞价中
    BID_ONCE = "bid_once"        # 轮流出价中
    BID_SEALED = "bid_sealed"    # 暗标收集出价中
    BUY_FIXED = "buy_fixed"      # 一口价:依次询问是否购买
    GAME_OVER = "game_over"


@dataclass(frozen=True)
class Card:
    """一张画:由艺术家与其上印刷的拍卖类型唯一描述(同类牌互相等价)。"""

    artist: ArtistId
    auction: AuctionType
    # 画作编号(每位画家内 0..张数-1),仅用于显示不同画面;compare=False 使其
    # 不参与相等/哈希 —— 同(画家,类型)的牌在游戏逻辑上仍然完全等价。
    art_id: int = field(default=0, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {"artist": self.artist.value, "auction": self.auction.value, "art_id": self.art_id}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Card":
        return Card(ArtistId(d["artist"]), AuctionType(d["auction"]), art_id=int(d.get("art_id", 0)))


@dataclass
class PlayerState:
    hand: list[Card] = field(default_factory=list)
    money: int = 0
    # 本回合已购入的画(回合结束时统一按当回合价值变现,然后清空)
    purchases: list[Card] = field(default_factory=list)
    # 全局累计:该玩家赢得拍卖时付出的总金额(成交价公开,可作"激进/花钱程度"信号)
    paid_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hand": [c.to_dict() for c in self.hand],
            "money": self.money,
            "purchases": [c.to_dict() for c in self.purchases],
            "paid_total": self.paid_total,
        }


@dataclass
class Auction:
    """一次拍卖的临时上下文。仅在拍卖/补牌阶段有效,空闲时为 None。"""

    artist: ArtistId
    auction_type: AuctionType          # 已解析的类型(双张时为第二张的类型)
    cards: list[Card]                  # 本次一起售出的画(1 张,双张时 2 张)
    turn_holder: int                   # 本轮"该谁当拍卖师"的玩家(回合推进的基准)
    seller: int                        # 收款方(通常=turn_holder;双张补牌者可不同)

    # 竞价过程状态(按拍卖类型使用其中一部分)
    order: list[int] = field(default_factory=list)     # 行动顺序
    to_act: int | None = None                          # 当前该行动的玩家
    idx: int = 0                                        # order 中的指针(轮流/暗标/一口价/补牌)
    high: int = 0                                       # 当前最高价 / 一口价的定价
    high_bidder: int | None = None                     # 当前最高出价者
    active: list[int] = field(default_factory=list)    # 公开拍卖:可参与竞价的玩家(不因过牌淘汰)
    pass_streak: int = 0                               # 公开拍卖:自上次加价以来连续过牌数
    sealed_bids: dict[int, int] = field(default_factory=dict)  # 暗标:已收集的出价
    player_bids: dict[int, int] = field(default_factory=dict)  # 公开/轮流:各玩家当前站着的出价(公开)

    # 双张专用
    double_card: Card | None = None                    # 原始双张牌
    offer_order: list[int] = field(default_factory=list)  # 补牌询问顺序

    def to_dict(self) -> dict[str, Any]:
        return {
            "artist": self.artist.value,
            "auction_type": self.auction_type.value,
            "cards": [c.to_dict() for c in self.cards],
            "turn_holder": self.turn_holder,
            "seller": self.seller,
            "order": list(self.order),
            "to_act": self.to_act,
            "idx": self.idx,
            "high": self.high,
            "high_bidder": self.high_bidder,
            "active": list(self.active),
            "pass_streak": self.pass_streak,
            "sealed_bids": dict(self.sealed_bids),
            "player_bids": dict(self.player_bids),
            "double_card": self.double_card.to_dict() if self.double_card else None,
            "offer_order": list(self.offer_order),
        }


@dataclass
class GameState:
    num_players: int
    round: int                                  # 当前回合 1..4
    phase: Phase
    players: list[PlayerState]
    deck: list[Card]                            # 尚未发出的牌(供后续回合发牌)
    value_board: dict[ArtistId, int]            # 各艺术家累计价值(= value_markers 每行之和)
    value_markers: dict[ArtistId, list[int]]    # 每回合放置的价值标码 [r1,r2,r3,r4],未进前三为 0
    round_counts: dict[ArtistId, int]           # 本回合各艺术家已上架张数
    active_player: int                          # 当前回合流程中的拍卖师(turn holder)
    start_player: int                           # 本回合的起始玩家
    auction: Auction | None = None
    # 事件日志(供 CLI/UI/回放展示;不影响逻辑)
    log: list[str] = field(default_factory=list)

    def clone(self) -> "GameState":
        """快速克隆(用于每步 apply / 搜索)。

        Card 是 frozen(不可变),可安全共享引用;只需新建可变容器(列表/字典)与
        PlayerState/Auction 实例。比 copy.deepcopy 快很多,且行为等价。
        """
        a = self.auction
        return GameState(
            num_players=self.num_players,
            round=self.round,
            phase=self.phase,
            players=[
                PlayerState(hand=p.hand.copy(), money=p.money,
                            purchases=p.purchases.copy(), paid_total=p.paid_total)
                for p in self.players
            ],
            deck=self.deck.copy(),
            value_board=self.value_board.copy(),
            value_markers={k: v.copy() for k, v in self.value_markers.items()},
            round_counts=self.round_counts.copy(),
            active_player=self.active_player,
            start_player=self.start_player,
            auction=None if a is None else Auction(
                artist=a.artist,
                auction_type=a.auction_type,
                cards=a.cards.copy(),
                turn_holder=a.turn_holder,
                seller=a.seller,
                order=a.order.copy(),
                to_act=a.to_act,
                idx=a.idx,
                high=a.high,
                high_bidder=a.high_bidder,
                active=a.active.copy(),
                pass_streak=a.pass_streak,
                sealed_bids=a.sealed_bids.copy(),
                player_bids=a.player_bids.copy(),
                double_card=a.double_card,
                offer_order=a.offer_order.copy(),
            ),
            log=self.log.copy(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_players": self.num_players,
            "round": self.round,
            "phase": self.phase.value,
            "players": [p.to_dict() for p in self.players],
            "deck": [c.to_dict() for c in self.deck],
            "value_board": {a.value: v for a, v in self.value_board.items()},
            "value_markers": {a.value: list(v) for a, v in self.value_markers.items()},
            "round_counts": {a.value: v for a, v in self.round_counts.items()},
            "active_player": self.active_player,
            "start_player": self.start_player,
            "auction": self.auction.to_dict() if self.auction else None,
            "log": list(self.log),
        }
