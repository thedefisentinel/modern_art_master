"""所有游戏常量,集中于此,方便对照实体规则书逐处校正。

设计原则:引擎的**逻辑**不依赖这些数字的具体取值(它们只是数据),
所以即使某个数字要修正,也只改这一个文件,逻辑部分无需改动。

校验来源:已对照官方英文规则书(John Webley 译本,The Rules Bank / Hans im Glück,
经 Knizia 认可的版本)逐条核对。以下均已确认:
  * 张数 12/13/14/15/16,从左到右 = 平局优先级(绿/粉/蓝/红/黄);
  * 发牌数 3人10/6/6/0、4人9/4/4/0、5人8/3/3/0;初始资金 100(单位:千,即 100,000);
  * 计分 30/20/10 累计、仅前三变现、暗标平局向拍卖师方向取近者、
    一口价定价不得超过卖家现金、无人出价则拍卖师免费获得。

「每位艺术家各拍卖类型的具体张数」(CARD_DISTRIBUTION)来自新浪游戏公开的卡牌表
(games.sina.com.cn/zysj/news/20101009/25.html),各类型合计 16/14/14/14/12=70,自洽。
按张数(12/13/14/15/16)对应到本文件的 5 位艺术家。
"""

from __future__ import annotations

from enum import Enum


class AuctionType(str, Enum):
    """五种拍卖机制。值同时用作 JSON 序列化的字符串标识。"""

    OPEN = "open"            # 公开拍卖:自由喊价,价高者得
    ONCE_AROUND = "once"     # 轮流出价:从拍卖师左侧起每人一次,拍卖师最后
    SEALED = "sealed"        # 暗标:所有人同时秘密出价,价高者得
    FIXED_PRICE = "fixed"    # 一口价:拍卖师定价,依次决定是否买;无人买则拍卖师自购
    DOUBLE = "double"        # 双张:与同艺术家第二张牌合并,类型由第二张决定


class ArtistId(str, Enum):
    """五位虚构艺术家。牌张数量各不相同(12/13/14/15/16,共 70 张)。

    枚举的先后顺序即**固定优先级**:平局时靠前者获得更高名次
    (对应实体规则中"最左侧艺术家优先")。
    """

    LITE_METAL = "lite_metal"
    YOKO = "yoko"
    CHRISTIN_P = "christin_p"
    KARL_GITTER = "karl_gitter"
    KRYPTO = "krypto"


# 有序列表(顺序 = 平局优先级,靠前者优先)
ARTISTS: list[ArtistId] = list(ArtistId)
AUCTION_TYPES: list[AuctionType] = list(AuctionType)

# 艺术家在计分板/平局上的优先级(索引越小越优先)
ARTIST_PRIORITY: dict[ArtistId, int] = {a: i for i, a in enumerate(ARTISTS)}


# ── 中文显示名 ────────────────────────────────────────────────────────────
ARTIST_NAME_ZH: dict[ArtistId, str] = {
    ArtistId.LITE_METAL: "莱特·梅塔",
    ArtistId.YOKO: "八虎",
    ArtistId.CHRISTIN_P: "克里斯汀·P",
    ArtistId.KARL_GITTER: "卡尔·基特",
    ArtistId.KRYPTO: "克里普托",
}

# 每位艺术家配一个颜色(对应实体卡面颜色,已按实体版核对:
# 从左到右 = 平局优先级 = 张数 12/13/14/15/16 = 绿/粉/蓝/红/黄)
ARTIST_COLOR_ZH: dict[ArtistId, str] = {
    ArtistId.LITE_METAL: "绿",   # 12
    ArtistId.YOKO: "粉",         # 13
    ArtistId.CHRISTIN_P: "蓝",   # 14
    ArtistId.KARL_GITTER: "红",  # 15
    ArtistId.KRYPTO: "黄",       # 16
}

AUCTION_NAME_ZH: dict[AuctionType, str] = {
    AuctionType.OPEN: "公开拍卖",
    AuctionType.ONCE_AROUND: "轮流出价",
    AuctionType.SEALED: "暗标",
    AuctionType.FIXED_PRICE: "一口价",
    AuctionType.DOUBLE: "双张",
}


# ── 牌库分布(来自新浪游戏公开卡牌表,已核验自洽)──────────────────────────
# 来源:games.sina.com.cn/zysj/news/20101009/25.html
# 该版本艺术家名为 夏加尔/克里木特/马蒂斯/罗特列克/保罗·克利,按张数
# 12/13/14/15/16 对应到本文件的 LITE_METAL/YOKO/CHRISTIN_P/KARL_GITTER/KRYPTO。
#   artist -> { auction_type -> count }
def _dist(open_: int, once: int, sealed: int, fixed: int, double: int) -> dict[AuctionType, int]:
    return {
        AuctionType.OPEN: open_,
        AuctionType.ONCE_AROUND: once,
        AuctionType.SEALED: sealed,
        AuctionType.FIXED_PRICE: fixed,
        AuctionType.DOUBLE: double,
    }


CARD_DISTRIBUTION = {
    #                          OPEN ONCE SEALED FIXED DOUBLE   合计
    ArtistId.LITE_METAL:  _dist(3,   3,   2,     2,    2),   # 12  绿·夏加尔
    ArtistId.YOKO:        _dist(3,   2,   3,     3,    2),   # 13  粉·克里木特
    ArtistId.CHRISTIN_P:  _dist(3,   3,   3,     3,    2),   # 14  蓝·马蒂斯
    ArtistId.KARL_GITTER: _dist(3,   3,   3,     3,    3),   # 15  红·罗特列克
    ArtistId.KRYPTO:      _dist(4,   3,   3,     3,    3),   # 16  黄·保罗·克利
}
# 各类型合计:公开16 / 一轮14 / 暗标14 / 定价14 / 联合12 = 70(已核验自洽)

# 每位艺术家的总张数(由分布推导,并用于断言完整性)
CARDS_PER_ARTIST: dict[ArtistId, int] = {
    ArtistId.LITE_METAL: 12,
    ArtistId.YOKO: 13,
    ArtistId.CHRISTIN_P: 14,
    ArtistId.KARL_GITTER: 15,
    ArtistId.KRYPTO: 16,
}

TOTAL_CARDS = sum(CARDS_PER_ARTIST.values())  # 70


# ── 发牌数(已核实,官方英文规则书)─────────────────────────────────────
# 每回合发给每位玩家的**新增**手牌数(手牌跨回合保留,不清空)。
# 第 4 回合不发新牌。索引:CARDS_DEALT[人数][回合(1..4)]。
# 注:部分中文译本写"3人首轮11张",经与官方英文规则书(10张)核对,系误译,以此为准。
CARDS_DEALT: dict[int, dict[int, int]] = {
    3: {1: 10, 2: 6, 3: 6, 4: 0},
    4: {1: 9, 2: 4, 3: 4, 4: 0},
    5: {1: 8, 2: 3, 3: 3, 4: 0},
}

# 初始资金:每人 100(原版为 100,000,本项目按比例削去三个 0,数值更清爽,
# 出价以 1 为最小单位)。金钱全程保密。
STARTING_MONEY = 100

MIN_PLAYERS = 3
MAX_PLAYERS = 5
NUM_ROUNDS = 4

# 计分表:每回合结束时,当回合上架张数最多的前三位艺术家分别累加的价值。
# 只有本回合进入前三的艺术家,其画作本回合才能变现(累计价值)。
RANK_VALUES = [30, 20, 10]  # 第 1 / 2 / 3 名

# 触发回合结束:同一艺术家第 N 张被摆上桌面时,回合立即结束,该张不拍卖。
ROUND_END_TRIGGER_COUNT = 5


def validate_rules() -> None:
    """自检:牌库分布与声明的总张数一致。引擎初始化时调用。"""
    for artist in ARTISTS:
        dist_total = sum(CARD_DISTRIBUTION[artist].values())
        declared = CARDS_PER_ARTIST[artist]
        assert dist_total == declared, (
            f"艺术家 {artist} 分布合计 {dist_total} != 声明张数 {declared}"
        )
    assert TOTAL_CARDS == 70, f"牌库总数应为 70,实为 {TOTAL_CARDS}"
    for n in range(MIN_PLAYERS, MAX_PLAYERS + 1):
        assert n in CARDS_DEALT, f"缺少 {n} 人的发牌配置"
        assert set(CARDS_DEALT[n]) == {1, 2, 3, 4}, f"{n} 人发牌回合配置不全"


validate_rules()
