"""游戏引擎:纯逻辑,无 IO。唯一的规则真相来源。

对外主要接口(见 game.py):
    new_game(num_players, seed) -> GameState
    legal_actions(state) -> list[Action]
    apply(state, action) -> GameState        # 返回新状态,不修改入参
    current_player(state) -> int
    is_over(state) -> bool
    final_scores(state) -> list[int]
    observation(state, player) -> dict        # 隐藏对手私有信息的视图
"""

from .rules import (
    ArtistId,
    AuctionType,
    ARTISTS,
    AUCTION_TYPES,
    ARTIST_NAME_ZH,
    AUCTION_NAME_ZH,
)
from .state import GameState, PlayerState, Phase, Card
from . import actions
from .game import (
    new_game,
    legal_actions,
    apply,
    current_player,
    is_over,
    final_scores,
    observation,
)

__all__ = [
    "ArtistId",
    "AuctionType",
    "ARTISTS",
    "AUCTION_TYPES",
    "ARTIST_NAME_ZH",
    "AUCTION_NAME_ZH",
    "GameState",
    "PlayerState",
    "Phase",
    "Card",
    "actions",
    "new_game",
    "legal_actions",
    "apply",
    "current_player",
    "is_over",
    "final_scores",
    "observation",
]
