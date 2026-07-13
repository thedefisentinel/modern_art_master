"""引擎逻辑单元测试 —— 零依赖,直接 `python tests/test_engine.py` 运行,也兼容 pytest。

用手工构造的确定性局面精确验证每条规则:牌库、五种拍卖的资金流、双张、
第 5 张触发回合结束、暗标平局优先级、累计计分与"仅前三变现"。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modern_art.engine import game, rules
from modern_art.engine.rules import ArtistId as A, AuctionType as T, ARTISTS, CARD_DISTRIBUTION
from modern_art.engine.state import GameState, PlayerState, Phase, Card
from modern_art.engine.actions import (
    ChooseCard, AddSecond, DeclineAdd, Bid, PassBid, SealedBid, Buy, PassBuy,
)


# ── 构造辅助 ──────────────────────────────────────────────────────────────

def make_state(hands: list[list[Card]], money: int = 100, rnd: int = 1) -> GameState:
    n = len(hands)
    return GameState(
        num_players=n,
        round=rnd,
        phase=Phase.CHOOSE,
        players=[PlayerState(hand=list(h), money=money) for h in hands],
        deck=[],
        value_board={a: 0 for a in ARTISTS},
        value_markers={a: [0, 0, 0, 0] for a in ARTISTS},
        round_counts={a: 0 for a in ARTISTS},
        active_player=0,
        start_player=0,
    )


# 常用牌简写
def C(artist: A, t: T) -> Card:
    return Card(artist, t)


FILLER = lambda: C(A.KRYPTO, T.OPEN)  # 填充牌:让回合在被测拍卖后不立即结束


# ══════════════════════════════════════════════════════════════════════════

def test_deck_integrity():
    deck = game._build_deck()
    assert len(deck) == 70
    from collections import Counter
    per_artist = Counter(c.artist for c in deck)
    assert per_artist[A.LITE_METAL] == 12
    assert per_artist[A.YOKO] == 13
    assert per_artist[A.CHRISTIN_P] == 14
    assert per_artist[A.KARL_GITTER] == 15
    assert per_artist[A.KRYPTO] == 16
    # 每(艺术家,类型)张数与分布表一致
    per_pair = Counter((c.artist, c.auction) for c in deck)
    for artist in ARTISTS:
        for t, cnt in CARD_DISTRIBUTION[artist].items():
            assert per_pair[(artist, t)] == cnt


def test_new_game_deal_counts():
    for n, r1 in ((3, 10), (4, 9), (5, 8)):
        s = game.new_game(n, seed=1)
        assert s.num_players == n
        for p in range(n):
            assert len(s.players[p].hand) == r1, (n, p, len(s.players[p].hand))
            assert s.players[p].money == rules.STARTING_MONEY
        assert s.round == 1 and s.phase == Phase.CHOOSE


def test_open_auction_buyer_pays_seller():
    # P0 拍卖一张公开牌;P1 出价 10 获胜,付给 P0
    s = make_state([
        [C(A.YOKO, T.OPEN), FILLER()],
        [FILLER()],
        [FILLER()],
    ])
    s = game.apply(s, ChooseCard(0))                 # P0 上架 -> BID_OPEN, to_act=P1
    assert s.phase == Phase.BID_OPEN
    assert game.current_player(s) == 1
    s = game.apply(s, Bid(10))                        # P1 出价 10
    s = game.apply(s, PassBid())                      # P2 放弃
    s = game.apply(s, PassBid())                      # P0(卖家)放弃 -> 结算
    assert s.players[1].money == 90                   # 买家 -10
    assert s.players[0].money == 110                  # 卖家 +10
    assert s.players[1].purchases == [C(A.YOKO, T.OPEN)]
    assert s.round_counts[A.YOKO] == 1
    assert s.phase == Phase.CHOOSE                     # 回合继续(有填充牌)


def test_auctioneer_wins_own_pays_bank():
    # 公开拍卖:别人都过,卖家出价;别人再过一圈后成交,卖家自购 -> 付银行
    s = make_state([
        [C(A.YOKO, T.OPEN), FILLER()],
        [FILLER()],
        [FILLER()],
    ])
    s = game.apply(s, ChooseCard(0))
    s = game.apply(s, PassBid())                      # P1 过
    s = game.apply(s, PassBid())                      # P2 过 -> to_act=P0
    assert game.current_player(s) == 0
    s = game.apply(s, Bid(7))                          # P0 出价 7(公开拍卖此时不立即成交)
    s = game.apply(s, PassBid())                      # P1 过
    s = game.apply(s, PassBid())                      # P2 过 -> 一圈无人加价,成交
    assert s.players[0].money == 93                    # 付给银行,净 -7
    assert s.players[0].purchases == [C(A.YOKO, T.OPEN)]


def test_open_auction_reentry_after_pass():
    # 公开拍卖核心:过牌后只要还有人加价,之后仍能再喊回来
    s = make_state([
        [C(A.YOKO, T.OPEN), FILLER()],
        [FILLER()],
        [FILLER()],
    ])
    s = game.apply(s, ChooseCard(0))                  # P0 卖家,order 从 P1 起
    s = game.apply(s, Bid(5))                          # P1 出 5
    s = game.apply(s, PassBid())                      # P2 过(此时觉得贵)
    s = game.apply(s, Bid(8))                          # P0 出 8
    s = game.apply(s, PassBid())                      # P1 过
    s = game.apply(s, Bid(12))                         # P2 又喊回来,出 12  ← 关键:过后再出价
    s = game.apply(s, PassBid())                      # P0 过
    s = game.apply(s, PassBid())                      # P1 过 -> 成交,P2 得
    assert s.players[2].purchases == [C(A.YOKO, T.OPEN)]
    assert s.players[2].money == 88                    # 买家 P2 -12
    assert s.players[0].money == 112                   # 卖家 P0 +12


def test_once_around_seller_last():
    # 轮流:P1、P2 出价,卖家 P0 最后;P0 放弃 -> 最高者 P2 获胜付 P0
    s = make_state([
        [C(A.YOKO, T.ONCE_AROUND), FILLER()],
        [FILLER()],
        [FILLER()],
    ])
    s = game.apply(s, ChooseCard(0))
    assert s.phase == Phase.BID_ONCE and game.current_player(s) == 1
    s = game.apply(s, Bid(10))                         # P1
    s = game.apply(s, Bid(20))                         # P2
    assert game.current_player(s) == 0                 # 卖家最后
    s = game.apply(s, PassBid())                       # P0 放弃 -> 结算
    assert s.players[2].money == 80                    # 买家 P2 -20
    assert s.players[0].money == 120                   # 卖家 +20
    assert s.players[2].purchases == [C(A.YOKO, T.ONCE_AROUND)]


def test_sealed_tie_breaks_toward_seller():
    # 暗标平局:同为最高价时,按"卖家优先"的顺序取靠前者
    # 卖家 P0=5, P1=10, P2=10 -> P1、P2 并列最高,order=[0,1,2],P1 靠前胜
    s = make_state([
        [C(A.YOKO, T.SEALED), FILLER()],
        [FILLER()],
        [FILLER()],
    ])
    s = game.apply(s, ChooseCard(0))
    assert s.phase == Phase.BID_SEALED and game.current_player(s) == 0
    s = game.apply(s, SealedBid(5))                    # P0
    s = game.apply(s, SealedBid(10))                   # P1
    s = game.apply(s, SealedBid(10))                   # P2 -> 开标
    assert s.players[1].purchases == [C(A.YOKO, T.SEALED)]
    assert s.players[1].money == 90
    assert s.players[0].money == 110


def test_sealed_seller_wins_own_on_tie():
    # 卖家自己也并列最高时,卖家优先 -> 自购付银行
    s = make_state([
        [C(A.YOKO, T.SEALED), FILLER()],
        [FILLER()],
        [FILLER()],
    ])
    s = game.apply(s, ChooseCard(0))
    s = game.apply(s, SealedBid(10))                   # P0
    s = game.apply(s, SealedBid(10))                   # P1
    s = game.apply(s, SealedBid(3))                    # P2
    assert s.players[0].purchases == [C(A.YOKO, T.SEALED)]
    assert s.players[0].money == 90                    # 付银行,净 -10


def test_fixed_price_first_buyer():
    s = make_state([
        [C(A.YOKO, T.FIXED_PRICE), FILLER()],
        [FILLER()],
        [FILLER()],
    ])
    s = game.apply(s, ChooseCard(0, price=15))
    assert s.phase == Phase.BUY_FIXED and game.current_player(s) == 1
    s = game.apply(s, Buy())                           # P1 买
    assert s.players[1].money == 85
    assert s.players[0].money == 115
    assert s.players[1].purchases == [C(A.YOKO, T.FIXED_PRICE)]


def test_fixed_price_all_pass_seller_self_buy():
    s = make_state([
        [C(A.YOKO, T.FIXED_PRICE), FILLER()],
        [FILLER()],
        [FILLER()],
    ])
    s = game.apply(s, ChooseCard(0, price=15))
    s = game.apply(s, PassBuy())                       # P1
    s = game.apply(s, PassBuy())                       # P2 -> 卖家自购
    assert s.players[0].money == 85                    # 付银行,净 -15
    assert s.players[0].purchases == [C(A.YOKO, T.FIXED_PRICE)]


def test_fixed_price_over_money_rejected():
    s = make_state([[C(A.YOKO, T.FIXED_PRICE)], [FILLER()], [FILLER()]], money=20)
    try:
        game.apply(s, ChooseCard(0, price=25))         # 定价 > 卖家现金
        assert False, "应拒绝超过卖家现金的定价"
    except ValueError:
        pass


def test_double_self_add_open():
    # P0 打双张(YOKO),自己补第二张(YOKO/公开),两张一起拍,卖家=P0
    s = make_state([
        [C(A.YOKO, T.DOUBLE), C(A.YOKO, T.OPEN), FILLER()],
        [FILLER()],
        [FILLER()],
    ])
    s = game.apply(s, ChooseCard(0))                   # 打双张 -> DOUBLE_OFFER, to_act=P0
    assert s.phase == Phase.DOUBLE_OFFER and game.current_player(s) == 0
    s = game.apply(s, AddSecond(0))                    # P0 补第二张(手牌现索引0=YOKO/OPEN)
    assert s.phase == Phase.BID_OPEN
    assert s.auction.seller == 0 and len(s.auction.cards) == 2
    s = game.apply(s, Bid(30))                         # P1
    s = game.apply(s, PassBid())                       # P2
    s = game.apply(s, PassBid())                       # P0 卖家放弃 -> 结算
    assert len(s.players[1].purchases) == 2            # 赢家得两张
    assert s.players[1].money == 70 and s.players[0].money == 130
    assert s.round_counts[A.YOKO] == 2                 # 两张都计名次


def test_double_other_player_adds_becomes_seller():
    # P0 打双张但无法/不补;P1 补牌成为卖家并收款
    s = make_state([
        [C(A.YOKO, T.DOUBLE), FILLER()],               # P0 手上没有第二张 YOKO
        [C(A.YOKO, T.OPEN), FILLER()],                 # P1 有
        [FILLER()],
    ])
    s = game.apply(s, ChooseCard(0))                   # DOUBLE_OFFER, to_act=P0
    s = game.apply(s, DeclineAdd())                    # P0 放弃 -> to_act=P1
    assert game.current_player(s) == 1
    s = game.apply(s, AddSecond(0))                    # P1 补 YOKO/OPEN -> 卖家=P1
    assert s.auction.seller == 1 and s.auction.turn_holder == 0
    s = game.apply(s, Bid(40))                         # P2 出价(order 从 seller+1=2 起)
    # 依 open 顺序继续,直到结算;简单起见让其余放弃
    while s.phase == Phase.BID_OPEN:
        s = game.apply(s, PassBid())
    assert len(s.players[2].purchases) == 2
    # 两人对分:成交价 40 → 原主持人 P0 得 20,补牌者 P1 得 20(奇数才归补牌者)
    assert s.players[2].money == 60          # 第三方 P2 付 40
    assert s.players[0].money == 120         # 原主持人分得 20
    assert s.players[1].money == 120         # 补牌者分得 20
    # 双张成交后,下一位拍卖师从补牌者(卖家 P1)左侧起 = P2;原主持人 P0 被跳过
    assert s.active_player == 2


def test_double_split_odd_remainder_to_adder():
    # 奇数成交价 → 原主持人得小半,补牌者得大半(含多出的 1)
    s = make_state([
        [C(A.YOKO, T.DOUBLE), FILLER()],
        [C(A.YOKO, T.OPEN), FILLER()],
        [FILLER()],
    ])
    s = game.apply(s, ChooseCard(0))         # P0 双张
    s = game.apply(s, DeclineAdd())          # P0 不补
    s = game.apply(s, AddSecond(0))          # P1 补牌成为卖家
    s = game.apply(s, Bid(41))               # P2 出价 41(奇数)
    while s.phase == Phase.BID_OPEN:
        s = game.apply(s, PassBid())
    # 41 拆分:P0=20(小半), P1=21(大半,含奇数)
    assert s.players[2].money == 59
    assert s.players[0].money == 120
    assert s.players[1].money == 121


def test_double_split_adder_self_buy_pays_bank():
    # 补牌者自购:原主持人分得小半,补牌者自己那份(含奇数)进银行
    s = make_state([
        [C(A.YOKO, T.DOUBLE), FILLER()],
        [C(A.YOKO, T.SEALED), FILLER()],
        [FILLER()],
    ])
    s = game.apply(s, ChooseCard(0))         # P0 双张
    s = game.apply(s, DeclineAdd())          # P0 不补
    s = game.apply(s, AddSecond(0))          # P1 补 YOKO/暗标 -> 卖家=P1, 暗标 order=[1,2,0]
    s = game.apply(s, SealedBid(30))         # P1
    s = game.apply(s, SealedBid(10))         # P2
    s = game.apply(s, SealedBid(0))          # P0 -> 开标,P1 以 30 自购
    assert s.players[1].purchases and len(s.players[1].purchases) == 2
    # 30 拆分:P0(原主持人)得 15;P1 自己那份 15 进银行
    assert s.players[0].money == 115
    assert s.players[1].money == 70          # 100 - 30(付款),自己那份未回自己
    assert s.players[2].money == 100


def test_double_all_decline_gives_free_card():
    # 无人能补第二张 -> 原主持人免费获得该双张画(计名次、结算时照价变现),无资金变动
    s = make_state([
        [C(A.YOKO, T.DOUBLE), FILLER()],
        [FILLER()],
        [FILLER()],
    ])
    s = game.apply(s, ChooseCard(0))
    s = game.apply(s, DeclineAdd())                    # P0(原主持人)
    s = game.apply(s, DeclineAdd())                    # P1
    s = game.apply(s, DeclineAdd())                    # P2 -> 无人补牌
    assert s.round_counts[A.YOKO] == 1                 # 双张计入名次
    assert s.players[0].purchases == [C(A.YOKO, T.DOUBLE)]  # 免费归原主持人
    assert all(len(s.players[q].purchases) == 0 for q in (1, 2))
    assert all(p.money == 100 for p in s.players)      # 免费,无资金变动
    assert s.phase == Phase.CHOOSE


def test_fifth_card_ends_round_not_sold():
    # 预置 YOKO 已上 4 张;P0 再打一张 YOKO -> 第 5 张,回合立即结束,该张不拍卖
    s = make_state([
        [C(A.YOKO, T.OPEN), FILLER()],
        [FILLER()],
        [FILLER()],
    ], rnd=1)
    s.round_counts[A.YOKO] = 4
    s = game.apply(s, ChooseCard(0))
    assert s.round_counts[A.YOKO] == 0                 # 已进入第2回合(计数已重置)
    assert s.round == 2
    assert all(len(p.purchases) == 0 for p in s.players)   # 第5张不成交
    # YOKO 本回合 5 张,应得第1名 +30
    assert s.value_board[A.YOKO] == 30
    # 打出第5张的是 P0 → 下一回合起始玩家为其左侧 P1
    assert s.start_player == 1


def test_scoring_top3_only_and_cumulative():
    # 直接测结算:A/B/C 进前三,D 不进;D 即使有历史价值也不变现
    s = make_state([[FILLER()], [FILLER()], [FILLER()]], rnd=4)
    s.round_counts = {a: 0 for a in ARTISTS}
    s.round_counts[A.LITE_METAL] = 3   # 第1
    s.round_counts[A.YOKO] = 2         # 第2
    s.round_counts[A.CHRISTIN_P] = 1   # 第3
    s.round_counts[A.KARL_GITTER] = 0  # 不进前三
    # 累计:LITE_METAL 上一回合已有 30
    s.value_board[A.LITE_METAL] = 30
    s.value_board[A.KARL_GITTER] = 50  # 历史价值,但本回合不在前三 -> 变现为 0
    # P0 购入:LITE_METAL(应得 30+30=60) + KARL_GITTER(应得 0)
    s.players[0].purchases = [C(A.LITE_METAL, T.OPEN), C(A.KARL_GITTER, T.OPEN)]
    # P1 购入:YOKO(应得 20) + CHRISTIN_P(应得 10)
    s.players[1].purchases = [C(A.YOKO, T.OPEN), C(A.CHRISTIN_P, T.OPEN)]
    game._end_round(s, last_player=0)  # 直接触发结算(round=4 -> 结算后终局)
    assert s.value_board[A.LITE_METAL] == 60           # 30 + 30
    assert s.value_board[A.YOKO] == 20
    assert s.value_board[A.CHRISTIN_P] == 10
    assert s.value_board[A.KARL_GITTER] == 50           # 未变,未进前三
    assert s.players[0].money == 100 + 60 + 0
    assert s.players[1].money == 100 + 20 + 10
    assert game.is_over(s)


def test_scoring_tie_uses_artist_priority():
    # 张数并列时,靠前的艺术家(枚举顺序更前)拿更高名次
    s = make_state([[FILLER()], [FILLER()], [FILLER()]], rnd=4)
    s.round_counts[A.LITE_METAL] = 2   # 与 YOKO 并列,但优先级更高 -> 第1
    s.round_counts[A.YOKO] = 2         # -> 第2
    s.round_counts[A.CHRISTIN_P] = 2   # -> 第3
    game._end_round(s, last_player=0)
    assert s.value_board[A.LITE_METAL] == 30
    assert s.value_board[A.YOKO] == 20
    assert s.value_board[A.CHRISTIN_P] == 10


def test_full_random_playthrough_no_crash():
    import random
    def concretize(s, act, rng):
        if isinstance(act, ChooseCard):
            card = s.players[game.current_player(s)].hand[act.card_index]
            if card.auction == T.FIXED_PRICE:
                return ChooseCard(act.card_index, price=rng.randint(0, s.players[game.current_player(s)].money))
            return ChooseCard(act.card_index)
        if isinstance(act, AddSecond):
            card = s.players[game.current_player(s)].hand[act.card_index]
            if card.auction == T.FIXED_PRICE:
                return AddSecond(act.card_index, price=rng.randint(0, s.players[game.current_player(s)].money))
            return AddSecond(act.card_index)
        if isinstance(act, Bid):
            lo, hi = game.bid_bounds(s)
            return Bid(rng.randint(lo, hi))
        if isinstance(act, SealedBid):
            lo, hi = game.bid_bounds(s)
            return SealedBid(rng.randint(lo, hi))
        return act

    for n in (3, 4, 5):
        for seed in range(60):
            rng = random.Random(1000 * n + seed)
            s = game.new_game(n, seed=seed)
            steps = 0
            while not game.is_over(s):
                acts = game.legal_actions(s)
                assert acts, s.phase
                s = game.apply(s, concretize(s, rng.choice(acts), rng))
                steps += 1
                assert steps < 5000
            assert all(m >= 0 for m in game.final_scores(s))
            assert len(game.winner(s)) >= 1


# ── 运行器 ────────────────────────────────────────────────────────────────

def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
        passed += 1
    print(f"\n全部通过:{passed}/{len(tests)}")


if __name__ == "__main__":
    _run_all()
