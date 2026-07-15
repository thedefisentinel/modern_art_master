# 现代艺术 · Modern Art(电子复刻)

[English](README.md) | **中文**

> Reiner Knizia 经典拍卖桌游《现代艺术》的开源、从零电子复刻:一套引擎驱动命令行、
> 零后端网页,以及一个整局在浏览器里跑(ONNX)的自对弈**强化学习 AI**。

线上版本：(https://modern-art-master.vercel.app/)

> ⚠️ **免责声明**:本项目是**非官方的粉丝复刻**,仅供学习与个人娱乐。《现代艺术 / Modern Art》
> 由 Reiner Knizia 设计,Hans im Glück / CMON 出版,本项目与其无任何关联,也不含任何官方美术或商标资产。
> 代码采用 **MIT 许可**(见 `LICENSE`)。
> **卡面美术不随仓库分发**——请自备图片或用 `tools/gen_art.py`(本地 ComfyUI)生成;缺图时卡面自动降级为纯色。

**当前进度:三期全部完成 —— 引擎 + 命令行 + 单元测试 + 网页 UI(逐帧动画 + 每牌独立画作)
+ 强化学习(自对弈 PPO,已全面超越所有手写 AI)。**

---

## 架构:单一引擎,多前端

游戏逻辑只写一遍(`engine/`),CLI、网页、强化学习环境都是它的薄封装。
"保证逻辑对" 只需保证这一处对。

```
modern_art/
├── engine/          纯逻辑,无 IO。唯一的规则真相来源
│   ├── rules.py     所有常量(牌库分布/发牌数/初始资金/计分表),集中可校正
│   ├── actions.py   玩家动作(可序列化)
│   ├── state.py     游戏状态(可克隆、可 JSON 序列化、含观测视图)
│   └── game.py      状态机:new_game / legal_actions / apply / …
├── agents/          AI 对手:random / heuristic / standard / aggressive(莽夫) /
│                    tight(铁公鸡) / rl(AI 大师,加载训练检查点)
├── cli/             命令行前端(人机混战 / AI 自对弈)
├── web/             网页前端(标准库 http.server + 单页 HTML)
│   └── static/      纯静态浏览器版:engine.js / agents.js / local.js / policy.onnx
└── rl/              强化学习:自对弈 PPO(encoding/env/model/train/eval/exploit)
tools/               ONNX 导出、美术生成/导入、Python↔JS 对拍工具
tests/               引擎单元测试 19 项 + RL 环境测试 4 项(零第三方依赖)
```

对外核心接口(见 `engine/game.py`):

```python
state = new_game(num_players, seed)     # 建局
acts  = legal_actions(state)            # 当前玩家合法动作
state = apply(state, action)            # 执行动作,返回新状态(不改入参)
current_player(state) / is_over(state) / final_scores(state)
observation(state, player)              # 隐藏对手手牌/现金/暗标的视角视图
```

---

## 运行

引擎、命令行、网页服务器、以及纯静态浏览器版**无需任何第三方依赖**(纯标准库,Python ≥ 3.10)。
强化学习训练/导出需要 `torch` / `numpy` / `onnx`(见 `requirements.txt`)。

```bash
# 命令行:交互式配置座位,人机对战
python -m modern_art.cli.play

# 命令行:指定座位(human / random / heuristic / standard / aggressive / tight / rl)
python -m modern_art.cli.play --seats human,heuristic,heuristic --seed 7

# 命令行:和最强的「AI 大师」对战(需先有 rl/checkpoints/policy_v5.pt)
python -m modern_art.cli.play --seats human,rl,rl

# 跑测试(引擎 19 项 + RL 环境 4 项)
python tests/test_engine.py
python tests/test_rl_env.py

# 网页版:浏览器开 http://127.0.0.1:8000,你坐 P0,下拉选难度
python -m modern_art.web.server --port 8000
```

> Windows 控制台若中文乱码,先执行 `set PYTHONIOENCODING=utf-8`
> (PowerShell:`$env:PYTHONIOENCODING="utf-8"`)。

---

## 纯静态版(可挂静态托管,朋友点开就玩,无需后端)

因为每人是各自单人对 AI,整局(引擎 + AI + 动画)都能在浏览器里跑:

- `engine.js`:引擎 + 观测编码的 JS 移植,与 Python **逐位对拍验证**
  (`tools/dump_parity.py` 导出 8600+ 步,`tools/check_parity.js` 复现,观测/掩码/后态零误差)。
- `agents.js`:基线 AI 的 JS 版 + `rl` 走 **ONNX**(`policy.onnx`,与 PyTorch 误差 1e-5)。
- `local.js`:浏览器内"本地后端"(payload / 逐帧事件 / 驱动 AI),对齐 `server.py`。
- `index.html`:用 `onnxruntime-web` 推理,不再 `fetch` 任何 `/api`。

**部署**:`modern_art/web/static/` 整个目录就是可部署的静态站——拖到 Netlify Drop /
Cloudflare Pages / GitHub Pages / Vercel 即得一个朋友能点开的网址。
更新模型:`python tools/export_onnx.py`(把 `policy_v5.pt` 导成 `web/static/policy.onnx`)。

---

## 规则实现说明

已实现并测试:

- 4 个回合;5 位艺术家,牌各 12/13/14/15/16 张,共 70 张。
- 五种拍卖:公开、轮流、暗标、一口价、双张。
- **一张牌上桌即计入名次统计**(拍卖前计数);某艺术家第 5 张上桌 → 回合立即结束,
  该张不拍卖但计入名次。
- 回合结算:当回合上架最多的前三名艺术家累加 30/20/10;价值**跨回合累计**;
  **只有本回合进入前三的艺术家其画作才能变现**(按累计价值),其余变现为 0。
- 平局:按艺术家固定优先级(靠前者优先)。
- 结算金流:买家付款给卖家;买家即卖家(自购)则付给银行;回合末银行按价值回购画作。

常量已对照**官方英文规则书**(John Webley 译本,The Rules Bank / Hans im Glück)与公开卡牌表核实;
牌库分布(公开 16 / 一轮 14 / 暗标 14 / 定价 14 / 双张 12 = 70)集中在 `engine/rules.py`。

**明确裁定**(可调):公开拍卖为自由喊价,**过牌不淘汰**——只要还有人加价,之后仍可再喊,
当所有人对当前最高价连续一圈都不加价时成交。**双张**成交所得由原主持人与补牌者**两人对分**
("对分"版本)。一口价定价不得超过卖家现金;每回合起始玩家 = 上一回合打出最后一张牌者的左侧。

---

## 强化学习(`modern_art/rl/`)

**为什么不是"求最优解"**:这是 3–5 人、不完全信息(对手手牌/金钱隐藏)、一般和的博弈,
不存在独立于对手的最优策略。目标改为:自对弈训练出**强且低可利用性**的通用策略,
用①对基线胜率 ②检查点对轰 ③可利用性(冻结策略再训最佳应对)来评估。

- `encoding.py`:定长观测(`OBS_DIM=161`:手牌 + 公开面板 + 各家收藏 + **对手出价/累计花钱** +
  拍卖上下文)、参数化动作(36 离散选牌 + 1 连续金额)、合法掩码。金额上限锚死
  **min(现金, 张数×(累计价值+30))** = 该画本回合最大可能价值,从动作空间剔除必亏出价。
- `env.py`:薄自对弈环境(相对分奖励 = 现金 − 全场平均,零和)。
- `model.py`:Actor-Critic —— 离散头 + Beta 连续头 + 价值头 + **两个辅助信念头**
  (预测对手手牌构成 / 本回合最终价值标码),逼网络形成"对对手与结局的信念"。
- `train.py`:参数共享 PPO 自对弈 + **多风格对手池**(standard/heuristic/莽夫/铁公鸡/随机
  + 历史自己快照)+ 辅助损失。
- `eval.py` / `exploit.py`:对基线全面评估 / 可利用性(训最佳应对)。

```bash
python -m modern_art.rl.train  --iters 200 --games 96 --device cpu
python -m modern_art.rl.eval    --ckpt modern_art/rl/checkpoints/policy_v5.pt --games 300
python -m modern_art.rl.exploit --ckpt modern_art/rl/checkpoints/policy_v5.pt
```

**当前最强:`policy_v5`(即「AI 大师」)战绩**(seat0 对 3 个同类,均势线 = 1/人数):

| 对手 | 3人 | 4人 | 5人 |
|---|---|---|---|
| 标准(手写最强) | 81% | 69% | 52% |
| 启发式 | 84% | 63% | 54% |
| 莽夫(乱抢高价) | 100% | 100% | 100% |
| 铁公鸡(能不买就不买) | 79% | 87% | 64% |
| 随机 | 100% | 98% | 95% |

- **一个高手 vs 一桌莽夫**:AI 大师夺冠率 ~100%,靠的是**把画高价卖给上头的对手**、
  而不是跟着一起超额出价——这正是纯自对弈学不到、非得靠"莽夫在对手池里"才能学会的通用能力。
- **可利用性**:专门训练来针对它的最佳应对,在预算内仍打不过它,未发现易被针对的漏洞。

关键设计来自两点博弈洞察:①**群体训练**(对手池含多种非理性性格,而非纯自对弈);
②观测带上**对手出价/累计花钱**,让暗标等非公开拍卖也能推断对手。
