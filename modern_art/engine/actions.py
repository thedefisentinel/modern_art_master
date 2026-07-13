"""玩家动作定义。

每个阶段合法的动作类型不同(见 game.legal_actions)。动作都是不可变的小对象,
可无损序列化为 dict(便于网页传输、RL 编码、日志回放)。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class Action:
    """所有动作的基类。子类通过 kind 区分。"""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = type(self).__name__
        return d


@dataclass(frozen=True)
class ChooseCard(Action):
    """CHOOSE 阶段:拍卖师从手牌选一张上架拍卖。

    card_index: 手牌中的索引。
    price:      仅当所选牌为『一口价』类型时必填(拍卖师定的售价, >= 0)。
    """

    card_index: int
    price: int | None = None


@dataclass(frozen=True)
class AddSecond(Action):
    """DOUBLE_OFFER 阶段:为双张补上同艺术家的第二张牌。

    补牌者成为本次拍卖的卖家;拍卖类型由第二张牌决定;赢家获得两张画。
    price: 仅当第二张牌为『一口价』类型时必填。
    """

    card_index: int
    price: int | None = None


@dataclass(frozen=True)
class DeclineAdd(Action):
    """DOUBLE_OFFER 阶段:放弃为双张补牌。"""


@dataclass(frozen=True)
class Bid(Action):
    """公开拍卖 / 轮流出价:出价。金额必须高于当前最高价。

    amount 为 None 时是 legal_actions 返回的"需填金额"模板;apply 时必须为具体整数。
    """

    amount: int | None = None


@dataclass(frozen=True)
class PassBid(Action):
    """公开拍卖 / 轮流出价:放弃出价(退出本次竞价)。"""


@dataclass(frozen=True)
class SealedBid(Action):
    """暗标:秘密出价(>= 0)。所有人出价后统一比较。

    amount 为 None 时是 legal_actions 返回的模板;apply 时必须为具体整数。
    """

    amount: int | None = None


@dataclass(frozen=True)
class Buy(Action):
    """一口价:以拍卖师定的价格买下。"""


@dataclass(frozen=True)
class PassBuy(Action):
    """一口价:不买,轮到下一位。"""


# 反序列化:kind 字符串 -> 类
_ACTION_CLASSES: dict[str, type[Action]] = {
    cls.__name__: cls
    for cls in (ChooseCard, AddSecond, DeclineAdd, Bid, PassBid, SealedBid, Buy, PassBuy)
}


def from_dict(d: dict[str, Any]) -> Action:
    """从 to_dict() 的结果重建动作对象。"""
    d = dict(d)
    kind = d.pop("kind")
    cls = _ACTION_CLASSES[kind]
    return cls(**d)
