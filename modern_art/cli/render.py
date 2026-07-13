"""中文终端渲染。所有输出集中于此,便于后续替换为网页 UI 时对照。"""

from __future__ import annotations

from ..engine.state import GameState, Card
from ..engine import game
from ..engine.rules import (
    ARTISTS, ARTIST_NAME_ZH, ARTIST_COLOR_ZH, AUCTION_NAME_ZH,
)

BAR = "─" * 60


def card_str(card: Card) -> str:
    return f"{ARTIST_COLOR_ZH[card.artist]}·{ARTIST_NAME_ZH[card.artist]}/{AUCTION_NAME_ZH[card.auction]}"


def board_str(state: GameState, viewer: int | None = None) -> str:
    """价值板 + 本回合上架统计 + 各玩家概览。viewer 之外的现金保密。"""
    lines = [BAR, f"第 {state.round}/4 回合   阶段:{_phase_zh(state)}"]
    lines.append("价值板(累计) / 本回合上架:")
    for a in ARTISTS:
        vb = state.value_board[a]
        cnt = state.round_counts[a]
        mark = "  ← 已达4张,再来1张即结束回合" if cnt == 4 else ""
        lines.append(
            f"  {ARTIST_COLOR_ZH[a]}·{ARTIST_NAME_ZH[a]:<8} 累计 {vb:>3}   本回合 {cnt} 张{mark}"
        )
    lines.append("玩家:")
    for p in range(state.num_players):
        ps = state.players[p]
        money = str(ps.money) if viewer is None or p == viewer else "***"
        tag = "(你)" if p == viewer else ""
        lines.append(
            f"  P{p}{tag:<4} 现金 {money:>5}   手牌 {len(ps.hand)} 张   本回合已购 {len(ps.purchases)} 张"
        )
    lines.append(BAR)
    return "\n".join(lines)


def hand_str(state: GameState, player: int) -> str:
    hand = state.players[player].hand
    if not hand:
        return "  (手牌为空)"
    return "\n".join(f"  [{i}] {card_str(c)}" for i, c in enumerate(hand))


def auction_str(state: GameState) -> str:
    a = state.auction
    if a is None:
        return ""
    kind = AUCTION_NAME_ZH[a.auction_type] if a.auction_type.value != "double" else "双张(待补牌)"
    head = (
        f"【拍卖中】卖家 P{a.seller} · {ARTIST_NAME_ZH[a.artist]} × {len(a.cards) or '?'} · {kind}"
    )
    if state.phase.value in ("bid_open", "bid_once"):
        hb = f"P{a.high_bidder}" if a.high_bidder is not None else "无"
        head += f"   当前最高 {a.high}(出价者 {hb})"
    if state.phase.value == "buy_fixed":
        head += f"   定价 {a.high}"
    return head


def _phase_zh(state: GameState) -> str:
    return {
        "choose": "选牌上架",
        "double_offer": "双张补牌",
        "bid_open": "公开拍卖竞价",
        "bid_once": "轮流出价",
        "bid_sealed": "暗标出价",
        "buy_fixed": "一口价认购",
        "game_over": "游戏结束",
    }[state.phase.value]


def recent_log(state: GameState, since: int) -> list[str]:
    return state.log[since:]


def final_str(state: GameState) -> str:
    scores = game.final_scores(state)
    win = game.winner(state)
    lines = [BAR, "游戏结束!最终现金:"]
    ranking = sorted(range(state.num_players), key=lambda p: -scores[p])
    for rank, p in enumerate(ranking, 1):
        crown = " 👑" if p in win else ""
        lines.append(f"  第{rank}名  P{p}  现金 {scores[p]}{crown}")
    lines.append(BAR)
    return "\n".join(lines)
