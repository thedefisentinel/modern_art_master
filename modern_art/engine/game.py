"""核心引擎:状态机 + 五种拍卖 + 回合结算。唯一的规则真相来源。

对外接口:
    new_game / legal_actions / apply / current_player / is_over / final_scores / observation
辅助:
    bid_bounds(state)          当前竞价决策的 (最低, 最高) 金额,便于 CLI/AI 取值
    winner(state)              终局赢家(可能多个,平局)

不变量:
  * apply(state, action) 不修改入参,返回新状态(内部先 clone)。
  * 金钱永不为负:出价/购买/一口价自购前均校验可负担。
  * 一张牌上桌即计入 round_counts(拍卖前计数);某艺术家第 5 张上桌 -> 回合立即结束,
    该张不拍卖但计入名次。
"""

from __future__ import annotations

import random

from . import rules
from .rules import (
    ArtistId,
    AuctionType,
    ARTISTS,
    ARTIST_PRIORITY,
    ARTIST_NAME_ZH,
    ARTIST_COLOR_ZH,
    AUCTION_NAME_ZH,
    CARD_DISTRIBUTION,
    CARDS_DEALT,
    STARTING_MONEY,
    RANK_VALUES,
    ROUND_END_TRIGGER_COUNT,
    NUM_ROUNDS,
    MIN_PLAYERS,
    MAX_PLAYERS,
)
from .state import GameState, PlayerState, Auction, Card, Phase
from . import actions
from .actions import (
    Action,
    ChooseCard,
    AddSecond,
    DeclineAdd,
    Bid,
    PassBid,
    SealedBid,
    Buy,
    PassBuy,
)


# ══════════════════════════════════════════════════════════════════════════
# 建局 / 发牌
# ══════════════════════════════════════════════════════════════════════════

def _build_deck() -> list[Card]:
    deck: list[Card] = []
    for artist in ARTISTS:
        aid = 0  # 每位画家内的画作编号 0..张数-1(仅用于显示不同画面)
        for auction, count in CARD_DISTRIBUTION[artist].items():
            for _ in range(count):
                deck.append(Card(artist, auction, art_id=aid))
                aid += 1
    assert len(deck) == rules.TOTAL_CARDS
    return deck


def _deal(state: GameState, rnd: int) -> None:
    """给每位玩家发本回合的新增手牌(从牌堆末尾取,牌堆已在建局时洗好)。"""
    n = state.num_players
    count = CARDS_DEALT[n][rnd]
    for _ in range(count):
        for p in range(n):
            if state.deck:
                state.players[p].hand.append(state.deck.pop())


def new_game(num_players: int, seed: int | None = None) -> GameState:
    if not (MIN_PLAYERS <= num_players <= MAX_PLAYERS):
        raise ValueError(f"人数须为 {MIN_PLAYERS}-{MAX_PLAYERS},收到 {num_players}")
    deck = _build_deck()
    random.Random(seed).shuffle(deck)
    state = GameState(
        num_players=num_players,
        round=1,
        phase=Phase.CHOOSE,
        players=[PlayerState(money=STARTING_MONEY) for _ in range(num_players)],
        deck=deck,
        value_board={a: 0 for a in ARTISTS},
        value_markers={a: [0] * NUM_ROUNDS for a in ARTISTS},
        round_counts={a: 0 for a in ARTISTS},
        active_player=0,
        start_player=0,
    )
    _deal(state, 1)
    _open_round(state)
    state.log.append(
        f"开局:{num_players} 人,每人初始资金 {STARTING_MONEY}。第 1 回合开始。"
    )
    return state


# ══════════════════════════════════════════════════════════════════════════
# 公共查询接口
# ══════════════════════════════════════════════════════════════════════════

def is_over(state: GameState) -> bool:
    return state.phase == Phase.GAME_OVER


def current_player(state: GameState) -> int:
    """当前应行动的玩家索引。终局返回 -1。"""
    if state.phase == Phase.CHOOSE:
        return state.active_player
    if state.phase == Phase.GAME_OVER:
        return -1
    assert state.auction is not None and state.auction.to_act is not None
    return state.auction.to_act


def final_scores(state: GameState) -> list[int]:
    """各玩家最终得分(现金)。"""
    return [p.money for p in state.players]


def winner(state: GameState) -> list[int]:
    """终局赢家(现金最多者;可能并列)。未结束返回 []。"""
    if not is_over(state):
        return []
    scores = final_scores(state)
    best = max(scores)
    return [i for i, s in enumerate(scores) if s == best]


def bid_bounds(state: GameState) -> tuple[int, int] | None:
    """当前竞价决策允许的 (最低, 最高) 出价;非竞价阶段返回 None。

    公开/轮流:最低 = 当前最高价+1,最高 = 该玩家现金。
    暗标:      最低 = 0,           最高 = 该玩家现金。
    """
    a = state.auction
    if a is None:
        return None
    p = current_player(state)
    money = state.players[p].money
    if state.phase in (Phase.BID_OPEN, Phase.BID_ONCE):
        return (a.high + 1, money)
    if state.phase == Phase.BID_SEALED:
        return (0, money)
    return None


# ══════════════════════════════════════════════════════════════════════════
# 合法动作
# ══════════════════════════════════════════════════════════════════════════

def legal_actions(state: GameState) -> list[Action]:
    """当前玩家的合法动作列表。

    含金额的动作(Bid/SealedBid/一口价的 price)以 amount/price=None 作为模板返回,
    调用方需在 [bid_bounds] 范围内填入具体金额后再交给 apply。
    """
    phase = state.phase
    if phase == Phase.CHOOSE:
        return _legal_choose(state)
    if phase == Phase.DOUBLE_OFFER:
        return _legal_double_offer(state)
    if phase in (Phase.BID_OPEN, Phase.BID_ONCE):
        return _legal_bid(state)
    if phase == Phase.BID_SEALED:
        return [SealedBid(amount=None)]
    if phase == Phase.BUY_FIXED:
        return _legal_buy_fixed(state)
    return []  # GAME_OVER


def _legal_choose(state: GameState) -> list[Action]:
    hand = state.players[state.active_player].hand
    acts: list[Action] = []
    for i, card in enumerate(hand):
        if card.auction == AuctionType.FIXED_PRICE:
            acts.append(ChooseCard(card_index=i, price=None))  # 需定价
        else:
            acts.append(ChooseCard(card_index=i))
    return acts


def _legal_double_offer(state: GameState) -> list[Action]:
    a = state.auction
    assert a is not None
    p = a.to_act
    hand = state.players[p].hand
    acts: list[Action] = [DeclineAdd()]
    for i, card in enumerate(hand):
        if card.artist != a.artist:
            continue
        if card.auction == AuctionType.DOUBLE:
            continue  # 不能用另一张双张作为配对牌
        if card.auction == AuctionType.FIXED_PRICE:
            acts.append(AddSecond(card_index=i, price=None))
        else:
            acts.append(AddSecond(card_index=i))
    return acts


def _legal_bid(state: GameState) -> list[Action]:
    a = state.auction
    assert a is not None
    p = a.to_act
    money = state.players[p].money
    acts: list[Action] = [PassBid()]
    if money >= a.high + 1:  # 有能力加价才允许出价
        acts.append(Bid(amount=None))
    return acts


def _legal_buy_fixed(state: GameState) -> list[Action]:
    a = state.auction
    assert a is not None
    p = a.to_act
    acts: list[Action] = [PassBuy()]
    if state.players[p].money >= a.high:  # a.high 存的是一口价定价
        acts.append(Buy())
    return acts


# ══════════════════════════════════════════════════════════════════════════
# apply:执行动作
# ══════════════════════════════════════════════════════════════════════════

def apply(state: GameState, action: Action) -> GameState:
    """执行动作,返回新状态(不修改入参)。非法动作抛 ValueError。"""
    s = state.clone()
    phase = s.phase
    if phase == Phase.GAME_OVER:
        raise ValueError("游戏已结束,无法行动")
    if phase == Phase.CHOOSE:
        _apply_choose(s, action)
    elif phase == Phase.DOUBLE_OFFER:
        _apply_double_offer(s, action)
    elif phase in (Phase.BID_OPEN, Phase.BID_ONCE):
        _apply_bid(s, action)
    elif phase == Phase.BID_SEALED:
        _apply_sealed(s, action)
    elif phase == Phase.BUY_FIXED:
        _apply_buy_fixed(s, action)
    else:
        raise ValueError(f"未知阶段 {phase}")
    return s


def _name(artist: ArtistId) -> str:
    # 日志中用颜色标识画家(与 UI/卡面一致,便于对照);附画家名。
    return f"{ARTIST_COLOR_ZH[artist]}({ARTIST_NAME_ZH[artist]})"


# ── CHOOSE ────────────────────────────────────────────────────────────────

def _apply_choose(s: GameState, action: Action) -> None:
    if not isinstance(action, ChooseCard):
        raise ValueError(f"CHOOSE 阶段只接受 ChooseCard,收到 {type(action).__name__}")
    p = s.active_player
    hand = s.players[p].hand
    i = action.card_index
    if not (0 <= i < len(hand)):
        raise ValueError(f"手牌索引越界:{i}")
    card = hand[i]

    if card.auction == AuctionType.FIXED_PRICE:
        _validate_price(s, p, action.price)

    hand.pop(i)
    triggered = _place_card(s, card.artist)
    if triggered:
        s.log.append(
            f"P{p} 打出第 {ROUND_END_TRIGGER_COUNT} 张【{_name(card.artist)}】"
            f"→ 第 {s.round} 回合结束,该画不拍卖"
        )
        _end_round(s, last_player=p)
        return

    if card.auction == AuctionType.DOUBLE:
        _start_double_offer(s, double_card=card, turn_holder=p)
    else:
        _start_auction(
            s,
            artist=card.artist,
            auction_type=card.auction,
            cards=[card],
            turn_holder=p,
            seller=p,
            price=action.price,
        )


def _validate_price(s: GameState, seller: int, price: int | None) -> None:
    if price is None:
        raise ValueError("一口价必须给出定价 price")
    if price < 0:
        raise ValueError("一口价定价不能为负")
    # 约束:定价不得超过卖家现金,保证无人购买时卖家能自购(金钱不为负)
    if price > s.players[seller].money:
        raise ValueError(
            f"一口价定价 {price} 超过卖家现金 {s.players[seller].money}(须保证可自购)"
        )


# ── 上桌计数 / 回合结束触发 ────────────────────────────────────────────────

def _place_card(s: GameState, artist: ArtistId) -> bool:
    """一张牌上桌:计入本回合名次统计。返回是否触发回合结束(达到第 5 张)。"""
    s.round_counts[artist] += 1
    return s.round_counts[artist] >= ROUND_END_TRIGGER_COUNT


# ── 开始一次拍卖 ──────────────────────────────────────────────────────────

def _rotation(n: int, start: int) -> list[int]:
    """从 start 起顺时针的完整玩家顺序。"""
    return [(start + k) % n for k in range(n)]


def _start_auction(
    s: GameState,
    artist: ArtistId,
    auction_type: AuctionType,
    cards: list[Card],
    turn_holder: int,
    seller: int,
    price: int | None = None,
) -> None:
    n = s.num_players
    a = Auction(
        artist=artist,
        auction_type=auction_type,
        cards=cards,
        turn_holder=turn_holder,
        seller=seller,
    )
    tag = f"P{seller} 拍卖【{_name(artist)}】×{len(cards)}({AUCTION_NAME_ZH[auction_type]})"

    if auction_type == AuctionType.OPEN:
        a.order = _rotation(n, (seller + 1) % n)  # 卖家左侧先叫,卖家自身也可参与
        a.active = list(range(n))
        a.high = 0
        a.high_bidder = None
        a.to_act = a.order[0]
        s.phase = Phase.BID_OPEN

    elif auction_type == AuctionType.ONCE_AROUND:
        a.order = _rotation(n, (seller + 1) % n)  # 卖家最后(rotation 末位即 seller)
        a.idx = 0
        a.high = 0
        a.high_bidder = None
        a.to_act = a.order[0]
        s.phase = Phase.BID_ONCE

    elif auction_type == AuctionType.SEALED:
        a.order = _rotation(n, seller)  # 卖家优先(平局时靠前者胜)
        a.idx = 0
        a.sealed_bids = {}
        a.to_act = a.order[0]
        s.phase = Phase.BID_SEALED

    elif auction_type == AuctionType.FIXED_PRICE:
        assert price is not None
        a.high = price  # 复用 high 存一口价定价
        a.order = [(seller + 1 + k) % n for k in range(n - 1)]  # 除卖家外顺时针
        a.idx = 0
        a.high_bidder = None
        a.to_act = a.order[0]
        tag += f" 定价 {price}"
        s.phase = Phase.BUY_FIXED

    else:
        raise ValueError(f"不能直接开始 {auction_type} 类型拍卖")

    s.auction = a
    s.log.append(tag)


# ── 双张补牌流程 ──────────────────────────────────────────────────────────

def _start_double_offer(s: GameState, double_card: Card, turn_holder: int) -> None:
    n = s.num_players
    a = Auction(
        artist=double_card.artist,
        auction_type=AuctionType.DOUBLE,  # 占位,补牌后改为第二张的类型
        cards=[],
        turn_holder=turn_holder,
        seller=turn_holder,
        double_card=double_card,
    )
    a.offer_order = _rotation(n, turn_holder)  # 出双张者优先补牌,然后顺时针
    a.idx = 0
    a.to_act = a.offer_order[0]
    s.phase = Phase.DOUBLE_OFFER
    s.auction = a
    s.log.append(
        f"P{turn_holder} 打出双张【{_name(double_card.artist)}】,依次询问补第二张…"
    )


def _apply_double_offer(s: GameState, action: Action) -> None:
    a = s.auction
    assert a is not None
    p = a.to_act
    assert p is not None

    if isinstance(action, DeclineAdd):
        a.idx += 1
        if a.idx < len(a.offer_order):
            a.to_act = a.offer_order[a.idx]
            return
        # 全体放弃补牌:原主持人免费获得该双张画(已计入名次)
        s.log.append(f"无人为双张【{_name(a.artist)}】补牌")
        _finish_no_sale(s)
        return

    if isinstance(action, AddSecond):
        hand = s.players[p].hand
        i = action.card_index
        if not (0 <= i < len(hand)):
            raise ValueError(f"手牌索引越界:{i}")
        second = hand[i]
        if second.artist != a.artist:
            raise ValueError("补牌必须为同一艺术家")
        if second.auction == AuctionType.DOUBLE:
            raise ValueError("不能用另一张双张作为配对牌")
        if second.auction == AuctionType.FIXED_PRICE:
            _validate_price(s, p, action.price)

        hand.pop(i)
        triggered = _place_card(s, a.artist)
        cards = [a.double_card, second]  # type: ignore[list-item]
        if triggered:
            s.log.append(
                f"P{p} 补牌使【{_name(a.artist)}】达第 {ROUND_END_TRIGGER_COUNT} 张 "
                f"→ 回合结束,双张不成交"
            )
            _end_round(s, last_player=p)
            return
        # 补牌者成为卖家;按第二张类型开拍,两张一起售出。
        # 所得在原主持人与补牌者之间「两人对分」(见 _award,本项目采用的版本规则)。
        s.log.append(f"P{p} 补上第二张【{_name(a.artist)}】,成为卖家(所得两人对分)")
        _start_auction(
            s,
            artist=a.artist,
            auction_type=second.auction,
            cards=cards,
            turn_holder=a.turn_holder,
            seller=p,
            price=action.price,
        )
        return

    raise ValueError(f"DOUBLE_OFFER 阶段只接受 AddSecond/DeclineAdd,收到 {type(action).__name__}")


# ── 公开拍卖 / 轮流出价 ────────────────────────────────────────────────────

def _apply_bid(s: GameState, action: Action) -> None:
    if s.phase == Phase.BID_OPEN:
        _apply_bid_open(s, action)
    else:
        _apply_bid_once(s, action)


def _check_bid_amount(s: GameState, p: int, amount: int | None, min_amount: int) -> int:
    if amount is None:
        raise ValueError("出价必须为具体金额")
    if amount < min_amount:
        raise ValueError(f"出价 {amount} 须 ≥ {min_amount}")
    if amount > s.players[p].money:
        raise ValueError(f"出价 {amount} 超过现金 {s.players[p].money}")
    return amount


def _apply_bid_open(s: GameState, action: Action) -> None:
    a = s.auction
    assert a is not None
    p = a.to_act
    assert p is not None

    if isinstance(action, Bid):
        amt = _check_bid_amount(s, p, action.amount, a.high + 1)
        a.high = amt
        a.high_bidder = p
        a.player_bids[p] = amt          # 记录该玩家当前出价(公开)
        a.pass_streak = 0               # 有人加价 → 重新计过牌
        s.log.append(f"P{p} 出价 {amt}")
    elif isinstance(action, PassBid):
        # 公开拍卖:过牌不淘汰,只是本轮不加价;之后仍可再喊
        a.pass_streak += 1
        s.log.append(f"P{p} 过(暂不加价)")
    else:
        raise ValueError(f"竞价阶段只接受 Bid/PassBid,收到 {type(action).__name__}")

    _advance_open(s)


def _advance_open(s: GameState) -> None:
    """公开拍卖推进:当所有"当前最高出价者以外"的玩家都对当前价连续过牌时成交。
    这样任何人都能"先过、之后再喊回来"——只要还有人加价,竞价就继续。"""
    a = s.auction
    assert a is not None
    n = s.num_players
    eligible_count = n - (1 if a.high_bidder is not None else 0)  # 需要连续过几次才收锤
    if a.pass_streak >= eligible_count:
        _resolve_high_bid(s)
        return
    # 下一位:顺时针下一个玩家,跳过当前最高出价者(不必与自己竞价)
    n_order = len(a.order)
    start_idx = a.order.index(a.to_act)
    for step in range(1, n_order + 1):
        q = a.order[(start_idx + step) % n_order]
        if q != a.high_bidder:
            a.to_act = q
            return
    _resolve_high_bid(s)  # 兜底


def _apply_bid_once(s: GameState, action: Action) -> None:
    a = s.auction
    assert a is not None
    p = a.to_act
    assert p is not None

    if isinstance(action, Bid):
        amt = _check_bid_amount(s, p, action.amount, a.high + 1)
        a.high = amt
        a.high_bidder = p
        a.player_bids[p] = amt          # 记录该玩家当前出价(公开)
        s.log.append(f"P{p} 出价 {amt}")
    elif isinstance(action, PassBid):
        s.log.append(f"P{p} 放弃")
    else:
        raise ValueError(f"竞价阶段只接受 Bid/PassBid,收到 {type(action).__name__}")

    a.idx += 1
    if a.idx < len(a.order):
        a.to_act = a.order[a.idx]
    else:
        _resolve_high_bid(s)  # 一圈结束(卖家已最后出价)


def _resolve_high_bid(s: GameState) -> None:
    """公开/轮流的结算:最高出价者获胜;无人出价则卖家以 0 价保留。"""
    a = s.auction
    assert a is not None
    if a.high_bidder is not None:
        _award(s, winner=a.high_bidder, price=a.high)
    else:
        _award(s, winner=a.seller, price=0)


# ── 暗标 ──────────────────────────────────────────────────────────────────

def _apply_sealed(s: GameState, action: Action) -> None:
    a = s.auction
    assert a is not None
    p = a.to_act
    assert p is not None
    if not isinstance(action, SealedBid):
        raise ValueError(f"暗标阶段只接受 SealedBid,收到 {type(action).__name__}")
    amt = action.amount
    if amt is None:
        raise ValueError("暗标必须给出出价金额")
    if amt < 0:
        raise ValueError("暗标出价不能为负")
    if amt > s.players[p].money:
        raise ValueError(f"暗标出价 {amt} 超过现金 {s.players[p].money}")

    a.sealed_bids[p] = amt
    a.idx += 1
    if a.idx < len(a.order):
        a.to_act = a.order[a.idx]
        return

    # 全员出价完毕:最高者胜;平局按 order(卖家优先)取靠前者
    best_bid = max(a.sealed_bids.values())
    win = next(q for q in a.order if a.sealed_bids[q] == best_bid)
    bids_str = ", ".join(f"P{q}:{a.sealed_bids[q]}" for q in a.order)
    s.log.append(f"暗标开标 [{bids_str}]")
    _award(s, winner=win, price=best_bid)


# ── 一口价 ────────────────────────────────────────────────────────────────

def _apply_buy_fixed(s: GameState, action: Action) -> None:
    a = s.auction
    assert a is not None
    p = a.to_act
    assert p is not None
    price = a.high

    if isinstance(action, Buy):
        if s.players[p].money < price:
            raise ValueError(f"现金 {s.players[p].money} 不足以按定价 {price} 购买")
        _award(s, winner=p, price=price)
        return

    if isinstance(action, PassBuy):
        a.idx += 1
        if a.idx < len(a.order):
            a.to_act = a.order[a.idx]
            return
        # 全体不买 → 卖家按定价自购(付给银行)
        s.log.append(f"无人按定价 {price} 购买 → 卖家 P{a.seller} 自购")
        _award(s, winner=a.seller, price=price)
        return

    raise ValueError(f"一口价阶段只接受 Buy/PassBuy,收到 {type(action).__name__}")


# ── 结算一次拍卖 ──────────────────────────────────────────────────────────

def _award(s: GameState, winner: int, price: int) -> None:
    """把画判给 winner,处理付款,画入其本回合购入。

    普通拍卖(卖家即当前主持人):买家付款给卖家;买家即卖家则付银行。
    双张由他人补牌(卖家=补牌者≠原主持人):采用「两人对分」版本规则 ——
      成交价在原主持人(turn_holder)与补牌者(seller)之间平分,
      奇数多出的 1 归补牌者(新主持人);某方自购时其应得的那份改付银行。
    """
    a = s.auction
    assert a is not None
    seller = a.seller
    th = a.turn_holder
    s.players[winner].money -= price
    s.players[winner].paid_total += price   # 公开成交价累计(激进/花钱信号)

    if seller == th:
        # 普通结算(含双张自补):买家付给卖家;买家即卖家则付银行
        if winner != seller:
            s.players[seller].money += price
        who = "自购(付银行)" if winner == seller else f"付给卖家 P{seller}"
    else:
        # 双张他人补牌 → 两人对分,奇数归补牌者
        h = price // 2          # 原主持人份额(小半)
        r = price - h           # 补牌者份额(大半,含奇数多出的 1)
        if winner == th:
            # 原主持人自购:自己那份(h)进银行,补牌者份(r)给补牌者
            s.players[seller].money += r
        elif winner == seller:
            # 补牌者自购:原主持人份(h)给原主持人,自己份(r,含奇数)进银行
            s.players[th].money += h
        else:
            # 第三方买下:两人对分
            s.players[th].money += h
            s.players[seller].money += r
        who = f"对分:原主持人 P{th} +{h}, 补牌者 P{seller} +{r}" + (
            "(其中自购部分付银行)" if winner in (th, seller) else ""
        )

    s.players[winner].purchases.extend(a.cards)
    s.log.append(
        f"→ P{winner} 以 {price} 得【{_name(a.artist)}】×{len(a.cards)} {who}"
    )
    s.auction = None
    # 下一位拍卖师从卖家左侧开始(双张若由他人补牌成交,则从补牌者=卖家左侧开始,
    # 中间放弃补牌者被跳过)
    _after_auction(s, from_player=seller)


def _finish_no_sale(s: GameState) -> None:
    """双张无人补牌:原主持人免费获得该画(结算时照价变现),从其左侧继续。"""
    a = s.auction
    assert a is not None
    th = a.turn_holder
    if a.double_card is not None:
        s.players[th].purchases.append(a.double_card)
        s.log.append(f"→ P{th} 免费获得【{_name(a.artist)}】(结算时照价变现)")
    s.auction = None
    _after_auction(s, from_player=th)


# ── 一次拍卖后:推进拍卖师 或 回合结束 ─────────────────────────────────────

def _after_auction(s: GameState, from_player: int) -> None:
    nxt = _first_with_cards_from(s, (from_player + 1) % s.num_players)
    if nxt is None:
        _end_round(s, last_player=from_player)  # 无人有手牌可打 → 回合结束
        return
    s.active_player = nxt
    s.phase = Phase.CHOOSE


def _first_with_cards_from(s: GameState, start: int) -> int | None:
    """从 start(含)起顺时针,返回第一个手牌非空的玩家;都空则 None。"""
    n = s.num_players
    for k in range(n):
        p = (start + k) % n
        if s.players[p].hand:
            return p
    return None


# ══════════════════════════════════════════════════════════════════════════
# 回合结算 / 换回合 / 终局
# ══════════════════════════════════════════════════════════════════════════

def _end_round(s: GameState, last_player: int) -> None:
    """结算本回合。last_player = 打出本回合最后一张牌(通常是第5张)的玩家;
    下一回合由其左侧玩家起始(官方规则)。"""
    s.auction = None
    counts = s.round_counts

    ranked = sorted(
        (a for a in ARTISTS if counts[a] > 0),
        key=lambda a: (-counts[a], ARTIST_PRIORITY[a]),
    )
    top3 = ranked[:3]
    for rank, artist in enumerate(top3):
        s.value_board[artist] += RANK_VALUES[rank]
        s.value_markers[artist][s.round - 1] = RANK_VALUES[rank]  # 记录本回合放的标码

    rank_str = ", ".join(
        f"{_name(a)}={counts[a]}张(+{RANK_VALUES[i]},累计{s.value_board[a]})"
        for i, a in enumerate(top3)
    )
    s.log.append(f"第 {s.round} 回合结算:前三 {rank_str or '(无成交)'}")

    top3_set = set(top3)
    for p in range(s.num_players):
        gained = 0
        for card in s.players[p].purchases:
            if card.artist in top3_set:
                gained += s.value_board[card.artist]  # 累计价值变现
        if gained:
            s.players[p].money += gained
            s.log.append(f"  P{p} 变现 +{gained}(现金 {s.players[p].money})")
        s.players[p].purchases.clear()

    # 重置本回合计数
    for a in ARTISTS:
        s.round_counts[a] = 0

    if s.round >= NUM_ROUNDS:
        s.phase = Phase.GAME_OVER
        scores = final_scores(s)
        s.log.append(f"游戏结束。最终现金:{scores}。赢家:{ [f'P{i}' for i in winner(s)] }")
        return

    # 进入下一回合:起始玩家 = 打出上一回合最后一张牌者的左侧
    s.round += 1
    s.start_player = (last_player + 1) % s.num_players
    _deal(s, s.round)
    s.log.append(f"—— 第 {s.round} 回合开始(起始玩家 P{s.start_player})——")
    _open_round(s)


def _open_round(s: GameState) -> None:
    """回合开始:把拍卖师定为从起始玩家起第一个有手牌者;都无手牌则本回合直接结算。"""
    first = _first_with_cards_from(s, s.start_player)
    if first is None:
        # 极端情况(通常仅第 4 回合可能出现):无人有手牌,本回合无内容
        _end_round(s, last_player=(s.start_player - 1) % s.num_players)
        return
    s.active_player = first
    s.phase = Phase.CHOOSE


# ══════════════════════════════════════════════════════════════════════════
# 观测(隐藏对手私有信息)
# ══════════════════════════════════════════════════════════════════════════

def observation(state: GameState, player: int) -> dict:
    """返回 player 视角的观测:自己完整,对手仅公开信息。用于 UI/AI。"""
    s = state
    a = s.auction
    sealed_public = None
    if s.phase == Phase.BID_SEALED and a is not None:
        # 只公开各玩家是否已出价,不泄露金额;自己的出价对自己可见
        sealed_public = {
            "submitted": sorted(a.sealed_bids.keys()),
            "my_bid": a.sealed_bids.get(player),
        }

    return {
        "you": player,
        "num_players": s.num_players,
        "round": s.round,
        "phase": s.phase.value,
        "to_act": current_player(s),
        "your_hand": [c.to_dict() for c in s.players[player].hand],
        "your_money": s.players[player].money,
        "your_purchases": [c.to_dict() for c in s.players[player].purchases],
        "value_board": {k.value: v for k, v in s.value_board.items()},
        "value_markers": {k.value: list(v) for k, v in s.value_markers.items()},
        "round_counts": {k.value: v for k, v in s.round_counts.items()},
        "players_public": [
            {
                "hand_size": len(s.players[q].hand),
                # 收藏区:本回合购入的画是明摆在桌上的公开信息(所有人可见)
                "purchases": [c.to_dict() for c in s.players[q].purchases],
                "purchases_count": len(s.players[q].purchases),
                # 累计成交付款(公开,反映花钱/激进程度);对手现金本身仍保密
                "paid_total": s.players[q].paid_total,
                "money": s.players[q].money if q == player else None,
            }
            for q in range(s.num_players)
        ],
        "auction": _auction_public(s, a) if a is not None else None,
        "sealed": sealed_public,
        "is_over": is_over(s),
        "scores": final_scores(s) if is_over(s) else None,
    }


def _auction_public(s: GameState, a: Auction) -> dict:
    d = {
        "artist": a.artist.value,
        "auction_type": a.auction_type.value,
        # 拍卖台上正在拍的实际牌面(公开)。双张联合时为 2 张(双张牌 + 补的第二张)
        "cards": [c.to_dict() for c in a.cards],
        "num_cards": len(a.cards) if a.cards else (2 if a.double_card else 1),
        "is_double_lot": len(a.cards) == 2,
        "seller": a.seller,
        "turn_holder": a.turn_holder,
        "to_act": a.to_act,
        "high": a.high,
        "high_bidder": a.high_bidder,
    }
    if s.phase == Phase.DOUBLE_OFFER:
        d["double_card"] = a.double_card.to_dict() if a.double_card else None
    return d
