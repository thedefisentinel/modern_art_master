"""交互式命令行对局:人机混战 / AI 自对弈。

用法示例:
    python -m modern_art.cli.play
    python -m modern_art.cli.play --seats human,heuristic,heuristic --seed 7
    python -m modern_art.cli.play --auto --seats heuristic,heuristic,random --seed 1
"""

from __future__ import annotations

import argparse
import sys

from ..engine import game
from ..engine.state import GameState
from ..engine.rules import AuctionType, MIN_PLAYERS, MAX_PLAYERS
from ..engine.actions import (
    ChooseCard, AddSecond, DeclineAdd, Bid, PassBid, SealedBid, Buy, PassBuy,
)
from ..agents import AGENT_REGISTRY, Agent, make_agent, available_opponents
from . import render


# ── 人类输入 ──────────────────────────────────────────────────────────────

def _prompt(msg: str) -> str:
    try:
        return input(msg).strip()
    except EOFError:
        print("\n(检测到输入结束,退出)")
        sys.exit(0)


def _ask_int(msg: str, lo: int, hi: int) -> int:
    while True:
        raw = _prompt(f"{msg} [{lo}-{hi}]: ")
        try:
            v = int(raw)
        except ValueError:
            print("  请输入整数。")
            continue
        if lo <= v <= hi:
            return v
        print(f"  超出范围 {lo}-{hi}。")


def human_action(state: GameState):
    me = game.current_player(state)
    print()
    print(render.board_str(state, viewer=me))
    aline = render.auction_str(state)
    if aline:
        print(aline)
    phase = state.phase.value

    if phase == "choose":
        print(f"轮到你(P{me})上架一张画:")
        print(render.hand_str(state, me))
        idx = _ask_int("选择要拍卖的画编号", 0, len(state.players[me].hand) - 1)
        card = state.players[me].hand[idx]
        if card.auction == AuctionType.FIXED_PRICE:
            price = _ask_int("『一口价』请定价", 0, state.players[me].money)
            return ChooseCard(idx, price=price)
        return ChooseCard(idx)

    if phase == "double_offer":
        acts = game.legal_actions(state)
        addable = {a.card_index for a in acts if isinstance(a, AddSecond)}
        print(f"P{me}:是否为双张【{render.card_str(state.auction.double_card)}】补第二张(同艺术家)?")
        print(render.hand_str(state, me))
        if addable:
            print(f"  可补的手牌编号:{sorted(addable)};输入 -1 放弃补牌。")
            while True:
                v = _ask_int("补牌编号(-1=放弃)", -1, len(state.players[me].hand) - 1)
                if v == -1:
                    return DeclineAdd()
                if v in addable:
                    card = state.players[me].hand[v]
                    if card.auction == AuctionType.FIXED_PRICE:
                        price = _ask_int("『一口价』请定价", 0, state.players[me].money)
                        return AddSecond(v, price=price)
                    return AddSecond(v)
                print("  该牌不能作为补牌(需同艺术家且非双张)。")
        print("  你没有可补的同艺术家牌,只能放弃。")
        return DeclineAdd()

    if phase in ("bid_open", "bid_once"):
        lo, hi = game.bid_bounds(state)
        if hi < lo:
            print("  现金不足以加价,只能放弃。")
            return PassBid()
        while True:
            raw = _prompt(f"出价 [{lo}-{hi}],或输入 p 放弃: ")
            if raw.lower() in ("p", "pass", "放弃"):
                return PassBid()
            try:
                v = int(raw)
            except ValueError:
                print("  输入数字出价或 p。")
                continue
            if lo <= v <= hi:
                return Bid(v)
            print(f"  出价须在 {lo}-{hi}。")

    if phase == "bid_sealed":
        lo, hi = game.bid_bounds(state)
        print(f"P{me}:暗标出价(其他人看不到)。")
        v = _ask_int("你的暗标出价", lo, hi)
        return SealedBid(v)

    if phase == "buy_fixed":
        price = state.auction.high
        can = state.players[me].money >= price
        while True:
            raw = _prompt(f"是否以定价 {price} 买下?(y=买 / n=不买){'' if can else ' [现金不足,只能 n]'}: ")
            if raw.lower() in ("n", "no", "不"):
                return PassBuy()
            if raw.lower() in ("y", "yes", "买") and can:
                return Buy()
            print("  请输入 y 或 n。")

    raise RuntimeError(f"未处理的阶段 {phase}")


# ── 对局主循环 ────────────────────────────────────────────────────────────

def run_game(seats: list, seed: int | None, auto: bool, verbose: bool) -> GameState:
    """seats[i] 为 None(人类)或 Agent 实例。"""
    n = len(seats)
    state = game.new_game(n, seed=seed)
    printed = 0

    def flush_log():
        nonlocal printed
        for line in state.log[printed:]:
            print(line)
        printed = len(state.log)

    print(render.board_str(state))
    flush_log()

    while not game.is_over(state):
        me = game.current_player(state)
        agent = seats[me]
        if agent is None:
            action = human_action(state)
        else:
            action = agent.act(state)
            if verbose or not auto:
                # 让人类看到 AI 的动作(动作细节会体现在随后 flush 的日志里)
                pass
        state = game.apply(state, action)
        if agent is not None and (verbose or not auto):
            flush_log()
        elif agent is None:
            flush_log()
        elif auto and verbose:
            flush_log()

    flush_log()
    print(render.final_str(state))
    return state


# ── 座位配置 ──────────────────────────────────────────────────────────────

def parse_seats(spec: str, seed: int | None) -> list:
    seats: list = []
    for i, tok in enumerate(spec.split(",")):
        tok = tok.strip().lower()
        if tok in ("human", "h", "人", "人类"):
            seats.append(None)
        elif tok in AGENT_REGISTRY or tok == "rl":
            seats.append(make_agent(tok, seed=None if seed is None else seed + 100 + i))
        else:
            raise SystemExit(
                f"未知座位类型:{tok}(可选:human / {' / '.join(available_opponents())})")
    if not (MIN_PLAYERS <= len(seats) <= MAX_PLAYERS):
        raise SystemExit(f"人数须为 {MIN_PLAYERS}-{MAX_PLAYERS}")
    return seats


def interactive_setup(seed: int | None) -> list:
    print("=== 现代艺术 · 对局设置 ===")
    n = _ask_int("玩家人数", MIN_PLAYERS, MAX_PLAYERS)
    seats = []
    options = "human / " + " / ".join(available_opponents())
    for i in range(n):
        while True:
            tok = _prompt(f"P{i} 座位类型({options})[默认 human]: ").strip().lower() or "human"
            if tok in ("human", "h"):
                seats.append(None)
                break
            if tok in AGENT_REGISTRY or tok == "rl":
                seats.append(make_agent(tok, seed=None if seed is None else seed + 100 + i))
                break
            print(f"  未知类型,可选:{options}")
    return seats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="现代艺术(Modern Art)命令行版")
    ap.add_argument("--seats", help="逗号分隔的座位,如 human,heuristic,heuristic")
    ap.add_argument("--seed", type=int, default=None, help="随机种子(复现对局)")
    ap.add_argument("--auto", action="store_true", help="全 AI 自对弈,不等待输入")
    ap.add_argument("--verbose", action="store_true", help="自对弈时逐步打印过程")
    args = ap.parse_args(argv)

    if args.seats:
        seats = parse_seats(args.seats, args.seed)
    elif args.auto:
        seats = parse_seats("heuristic,heuristic,heuristic", args.seed)
    else:
        seats = interactive_setup(args.seed)

    # 自对弈但座位里其实有人类 -> 关闭 auto
    auto = args.auto and all(s is not None for s in seats)
    run_game(seats, seed=args.seed, auto=auto, verbose=args.verbose or not auto)


if __name__ == "__main__":
    main()
