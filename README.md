# 现代艺术 · Modern Art(电子复刻)

> An open-source, from-scratch electronic re-implementation of Reiner Knizia's
> classic auction board game **Modern Art** — with a single rules engine driving a
> CLI, a zero-backend web UI, and a self-play **reinforcement-learning AI** that runs
> entirely in the browser (ONNX).

Reiner Knizia 经典拍卖桌游《现代艺术》的电子版。命令行 + 网页 UI 双前端,
并用强化学习训练出一个 3–5 人通用的强 AI「AI 大师」与人对弈。

> ⚠️ **免责声明**:本项目是**非官方的粉丝复刻**,仅供学习与个人娱乐。《现代艺术 / Modern Art》
> 由 Reiner Knizia 设计,Hans im Glück / CMON 出版,本项目与其无任何关联,也不含任何官方美术或商标资产。
> 代码采用 **MIT 许可**(见 `LICENSE`)。
> **卡面美术不随仓库分发**——请自备图片或用 `tools/gen_art.py`(本地 ComfyUI)生成;缺图时卡面自动降级为纯色。

**当前进度:三期全部完成 —— 引擎 + 命令行 + 单元测试 + 网页 UI(逐帧动画 + 70 张 AI 生成画作)
+ 强化学习(自对弈 PPO,已全面超越所有手写 AI)。**

---

## 架构:单一引擎,多前端

游戏逻辑只写一遍(`engine/`),CLI、未来的网页、强化学习环境都是它的薄封装。
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
├── cli/             中文命令行前端(人机混战 / AI 自对弈)
├── web/             中文网页前端(http.server + 单页 HTML;逐帧动画 + 70 张画作)
│   └── static/art/  各牌的 AI 生成画作(<artist>_<id>.png)
└── rl/              强化学习:自对弈 PPO(encoding/env/model/train/eval/exploit)
tools/gen_art.py     用本地 ComfyUI 生成卡面画作(关键词可自改)
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

无需安装任何第三方依赖(纯标准库,Python ≥ 3.10)。

```bash
# 交互式配置座位,人机对战
python -m modern_art.cli.play

# 指定座位(human / random / heuristic),3~5 人
python -m modern_art.cli.play --seats human,heuristic,heuristic --seed 7

# 纯 AI 自对弈(不需输入)
python -m modern_art.cli.play --auto --seats heuristic,heuristic,random --seed 1

# 和最强的「AI 大师」对战(需先训练出 rl/checkpoints/policy_v4.pt)
python -m modern_art.cli.play --seats human,rl,rl

# 跑测试(引擎 19 项 + RL 环境 4 项)
python tests/test_engine.py
python tests/test_rl_env.py

# 启动网页版(浏览器开 http://127.0.0.1:8000,你坐 P0;下拉可选「AI 大师」)
python -m modern_art.web.server
python -m modern_art.web.server --port 8080     # 换端口
```

> Windows 控制台若中文显示为乱码,先执行 `set PYTHONIOENCODING=utf-8`
> (PowerShell:`$env:PYTHONIOENCODING="utf-8"`)。

---

## 规则实现说明

已实现并测试的规则:

- 4 个回合;5 位艺术家,牌各 12/13/14/15/16 张,共 70 张。
- 五种拍卖:公开、轮流、暗标、一口价、双张。
- **一张牌上桌即计入名次统计**(拍卖前计数);某艺术家第 5 张上桌 → 回合立即结束,
  该张不拍卖但计入名次。
- 回合结算:当回合上架最多的前三名艺术家累加 30/20/10;价值**跨回合累计**;
  **只有本回合进入前三的艺术家其画作才能变现**(按累计价值),其余变现为 0。
- 平局:按艺术家固定优先级(枚举顺序,靠前者优先)。
- 结算金流:买家付款给卖家;买家即卖家(自购)则付给银行;回合末银行按价值回购画作。

### 常量已对照权威来源核实(集中在 `engine/rules.py`)

规则机制已对照**官方英文规则书**(John Webley 译本,The Rules Bank / Hans im Glück,
经 Knizia 认可)逐条核对;牌库分布已对照**新浪游戏公开卡牌表**核实。

| 项 | 取值 | 来源 |
|---|---|---|
| 牌库分布 `CARD_DISTRIBUTION` | 见下表 | 新浪游戏卡牌表,各类型合计 16/14/14/14/12=70,自洽 |
| 每回合发牌数 `CARDS_DEALT` | 3人10/6/6,4人9/4/4,5人8/3/3(第4回合不发) | 官方规则书 |
| 初始资金 `STARTING_MONEY` | 100(原版 100,000,本项目削去三个 0) | 官方规则书 |

牌库分布(按艺术家张数 → 各拍卖类型):

| 艺术家(色·张数) | 公开 | 一轮 | 暗标 | 定价 | 双张 |
|---|---|---|---|---|---|
| 绿 12 | 3 | 3 | 2 | 2 | 2 |
| 粉 13 | 3 | 2 | 3 | 3 | 2 |
| 蓝 14 | 3 | 3 | 3 | 3 | 2 |
| 红 15 | 3 | 3 | 3 | 3 | 3 |
| 黄 16 | 4 | 3 | 3 | 3 | 3 |

### 引擎为可玩性/RL 化做的明确裁定(可按需调整)

- **公开拍卖**离散化为"顺时针轮流加价,放弃即退出,只剩最高出价者时成交"(拍卖师可参与,
  无人出价则免费得画)—— 这是为 CLI/RL 化对连续喊价过程做的等价离散建模。
- **双张**:出双张者优先补第二张,再顺时针询问;补牌者成为新主持人,两张一起售出。
  成交所得由**原主持人与补牌者两人对分**(奇数多出的 1 归补牌者;某方自购时其应得的
  那份改付银行)—— 这是"对分"版本规则(与官方英文"全归补牌者"不同,按本项目选择采用)。
  若无人补牌 → 原主持人**免费获得**该画(计入名次,结算时照价变现)。双张成交后,
  下一位拍卖师从补牌者(卖家)左侧起,中间放弃者被跳过。
- **一口价定价** 不得超过卖家现金(保证无人认购时卖家能自购,金钱不为负)。
- 每回合起始玩家 = 上一回合打出最后一张牌者的左侧(官方规则)。

---

## 第二期:网页 UI(`modern_art/web/`)

标准库 `http.server` 把同一个 `engine` 暴露为 JSON 后端,自包含中文单页前端,
已用 Chrome 浏览器自动化走完整局验证。特色:

- **逐帧演示动画**:AI 不再一瞬间算完;后端返回"从你出手到下次轮到你"的帧序列,
  前端每步停 1.5–2s 播放 —— 谁出价/放弃(玩家卡浮动气泡)、谁拍下(高亮 + 画进收藏)、
  谁上架新画(拍卖台滑入)、每回合结算(居中横幅),右下角可"跳过演示"。
- **桌游化界面**:价值板按"画家为列 · 回合为行"排布,每回合放 30/20/10 标码;
  各玩家有公开**收藏区**;手牌按颜色排序;拍卖台画作居中放大。
- **70 张 AI 生成画作**:每张牌一张独立画(`tools/gen_art.py` 调本地 ComfyUI/SDXL 生成)。
  引擎给每张牌加了 `art_id`(`compare=False`,不影响游戏逻辑,同类牌仍等价)。

### 纯静态版(可挂静态托管,朋友点开就玩,无需后端)

整局(引擎 + AI + 动画)都在浏览器里跑,因为每人是各自单人对 AI。

- `engine.js`:引擎 + 观测编码的 JS 移植,与 Python **逐位对拍验证**
  (`tools/dump_parity.py` 导出 8600+ 步,`tools/check_parity.js` 复现,观测/掩码/后态零误差)。
- `agents.js`:5 个基线 AI 的 JS 版 + `rl` 走 **ONNX**(`policy.onnx`,与 PyTorch 误差 1e-5)。
- `local.js`:浏览器内"本地后端"(payload / 逐帧事件 / 驱动 AI),对齐 `server.py`。
- `index.html`:用 `onnxruntime-web`(CDN)推理,不再 `fetch` 任何 `/api`。

**部署**:`modern_art/web/static/` 整个目录(~8MB)就是可部署的静态站。任选:
- Netlify Drop(app.netlify.com/drop)→ 拖目录进去,秒得 `*.netlify.app` 临时网址;
- Cloudflare Pages / GitHub Pages / Vercel → 上传或连仓库,得 `*.pages.dev` 等域名。

导出/更新模型:见 `tools/export_onnx`(把 `policy_v5.pt` 导成 `web/static/policy.onnx`)。

## 第三期:强化学习(`modern_art/rl/`)

**为什么不是"求最优解"**:这是 3–5 人、不完全信息(对手手牌/金钱隐藏)、一般和的博弈,
不存在独立于对手的最优策略。目标改为:自对弈训练出**强且低可利用性**的通用策略,
用①对基线胜率 ②检查点对轰 ③可利用性(冻结策略再训最佳应对)来评估。

- `encoding.py`:定长观测(手牌 + 公开面板 + 各家收藏 + **对手出价/累计花钱** + 拍卖上下文,
  **OBS_DIM=161**)、参数化动作(36 离散选牌 + 1 连续金额)、合法掩码。金额上限锚死
  **min(现金, 张数×(累计价值+30))** = 该画本回合最大可能价值,从动作空间剔除必亏出价。
  已单测:掩码内动作解码后**一定合法**。
- `env.py`:薄自对弈环境(相对分奖励 = 现金 − 全场平均,零和)。
- `model.py`:Actor-Critic —— 离散头 + Beta 连续头 + 价值头 + **两个辅助信念头**
  (预测对手手牌构成 / 本回合最终价值标码),逼网络形成"对对手与结局的信念"。
- `train.py`:参数共享 PPO 自对弈 + **多风格对手池**(standard/heuristic/莽夫/铁公鸡/随机
  + 历史自己快照)+ 辅助损失。
- `eval.py` / `exploit.py`:对基线全面评估 / 可利用性(训最佳应对)。
- `agents/rl_agent.py`:把检查点封成 Agent,已接入 CLI(`--seats human,rl,rl`)与网页(选"AI 大师")。

```bash
python -m modern_art.rl.train --iters 200 --games 96 --device cpu   # 训练(小网 CPU 反而更快)
python -m modern_art.rl.eval  --ckpt modern_art/rl/checkpoints/policy_v4.pt --games 300
python -m modern_art.rl.exploit --ckpt modern_art/rl/checkpoints/policy_v4.pt  # 可利用性
```

**当前最强:`policy_v4`(已设为「AI 大师」)战绩**(seat0 对 3 个同类,均势线 = 1/人数):

| 对手 | 3人 | 4人 | 5人 |
|---|---|---|---|
| 标准(手写最强) | 81% | 69% | 52% |
| 启发式 | 84% | 63% | 54% |
| 莽夫 | 100% | 100% | 100% |
| 铁公鸡 | 79% | 87% | 64% |
| 随机 | 100% | 98% | 95% |

- **v3 vs v4**:v4 用修正后的"真实莽夫"(顶格 ~85%、绝不超上限)对手池训练,泛化更好;
  2v2 正面对轰 **v4 61% : v3 39%**。
- **一个高手 vs 一桌莽夫**:v4 单挑 3–4 个莽夫,夺冠率 ~100%、相对分 +2~3 —— 验证了它学到的
  通用打法核心:**识别并剥削非理性对手(把画高价卖给莽夫),而不是跟着一起超额出价**。
  这正是纯自对弈学不到、非得靠"莽夫在对手池里"才能学会的能力。
- **可利用性**:专门训练的最佳应对在预算内打不过它(仍亏),未发现易被针对的漏洞。

关键设计来自两点博弈洞察:①**群体训练**(对手池含多种非理性性格,而非纯自对弈);
②观测带上**对手出价/累计花钱**,让暗标等非公开拍卖也能推断对手。
