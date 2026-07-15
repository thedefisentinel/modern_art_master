# Modern Art — Electronic Edition

**English** | [中文](README.zh.md)

A online version can be found at here: https://vercel.com/deisentinel/modern-art-master

> An open-source, from-scratch electronic re-implementation of Reiner Knizia's
> classic auction board game **Modern Art** — a single rules engine drives a CLI,
> a zero-backend web UI, and a self-play **reinforcement-learning AI** that runs
> entirely in the browser via ONNX.

> ⚠️ **Disclaimer.** This is an **unofficial, fan-made** re-implementation, for
> learning and personal play only. *Modern Art* is designed by Reiner Knizia and
> published by Hans im Glück / CMON; this project is not affiliated with them and
> contains no official artwork or trademarked assets. The **code is MIT-licensed**
> (see `LICENSE`). **Card artwork is not distributed with this repo** — bring your
> own images or generate them with `tools/gen_art.py` (local ComfyUI); cards fall
> back to plain colored tiles when art is missing.

**Status: all three phases complete** — engine + CLI + unit tests, web UI
(step-by-step animation, per-card artwork), and reinforcement learning (self-play
PPO that decisively beats every hand-written AI).

---

## Architecture: one engine, many front-ends

The game logic is written once (`engine/`); the CLI, the web UI, and the RL
environment are all thin wrappers over it. "Getting the rules right" only has to
be done in one place.

```
modern_art/
├── engine/          Pure logic, no I/O. The single source of truth for rules.
│   ├── rules.py     All constants (deck distribution, deal counts, scoring) — one place to fix.
│   ├── actions.py   Player actions (serializable).
│   ├── state.py     Game state (cloneable, JSON-serializable, with observation view).
│   └── game.py      State machine: new_game / legal_actions / apply / …
├── agents/          AI opponents: random / heuristic / standard / aggressive /
│                    tight / rl (the "AI Master", loads a trained checkpoint).
├── cli/             Command-line front-end (human vs AI / AI self-play).
├── web/             Web front-end (stdlib http.server + single-page HTML).
│   └── static/      Fully-static browser build: engine.js / agents.js / local.js / policy.onnx.
└── rl/              Reinforcement learning: self-play PPO (encoding/env/model/train/eval/exploit).
tools/               ONNX export, art generation/import, Python↔JS parity harness.
tests/               19 engine unit tests + 4 RL-env tests (zero third-party deps).
```

Core API (see `engine/game.py`):

```python
state = new_game(num_players, seed)     # start a game
acts  = legal_actions(state)            # legal actions for the current player
state = apply(state, action)            # returns a NEW state (input untouched)
current_player(state) / is_over(state) / final_scores(state)
observation(state, player)              # per-player view that hides opponents' hands/money/sealed bids
```

---

## Running it

The engine, CLI, web server, and the static browser build need **no third-party
packages** (pure standard library, Python ≥ 3.10). RL training/export needs
`torch` / `numpy` / `onnx` (see `requirements.txt`).

```bash
# CLI: interactive seat setup, play against AI
python -m modern_art.cli.play

# CLI: fixed seats (human / random / heuristic / standard / aggressive / tight / rl)
python -m modern_art.cli.play --seats human,heuristic,heuristic --seed 7

# CLI: play against the strongest "AI Master" (needs rl/checkpoints/policy_v5.pt)
python -m modern_art.cli.play --seats human,rl,rl

# Tests (19 engine + 4 RL-env)
python tests/test_engine.py
python tests/test_rl_env.py

# Web: open http://127.0.0.1:8000 — you are P0, pick a difficulty
python -m modern_art.web.server --port 8000
```

> On a Windows console showing mojibake, run `set PYTHONIOENCODING=utf-8`
> (PowerShell: `$env:PYTHONIOENCODING="utf-8"`).

---

## Fully-static browser build (host anywhere, no backend)

Because each player plays solo against the AI, an entire game (engine + AI +
animation) runs client-side:

- `engine.js` — a JS port of the engine + observation encoding, **verified
  bit-for-bit against Python** (`tools/dump_parity.py` dumps 8600+ steps,
  `tools/check_parity.js` replays them: observations, masks, and resulting states
  all match exactly).
- `agents.js` — JS ports of the baseline AIs, plus `rl` running the exported
  **ONNX** model (`policy.onnx`, within 1e-5 of PyTorch).
- `local.js` — an in-browser "backend" (payload / per-step events / AI driving),
  mirroring `server.py`.
- `index.html` — infers with `onnxruntime-web`; no `/api` calls.

**Deploy:** the `modern_art/web/static/` directory is the deployable static site.
Drop it on Netlify Drop, Cloudflare Pages, GitHub Pages, or Vercel to get a URL
friends can just open. Update the model with `python tools/export_onnx.py`
(exports `policy_v5.pt` → `web/static/policy.onnx`).

---

## Rules implementation

Implemented and unit-tested:

- 4 rounds; 5 artists with 12/13/14/15/16 cards each (70 total).
- Five auction types: **Open, Once-around, Sealed, Fixed-price, Double**.
- A card **counts toward the ranking the moment it hits the table** (before it
  sells); the **5th** card of any artist ends the round immediately (that card is
  not auctioned but still counts).
- End-of-round scoring: the three most-auctioned artists gain 30/20/10, which
  **accumulate across rounds**; a painting only cashes out if its artist is in the
  **top 3 this round** (at its accumulated value), otherwise it pays 0.
- Ties broken by a fixed artist precedence (earlier artist wins).
- Money flow: the buyer pays the seller; if the buyer *is* the seller (self-buy)
  they pay the bank; at round end the bank buys back paintings at their value.

Constants are cross-checked against the **official English rulebook** (John Webley
translation, The Rules Bank / Hans im Glück) and a public card-list; the deck
distribution (Open 16 / Once 14 / Sealed 14 / Fixed 14 / Double 12 = 70) is in
`engine/rules.py`.

**Explicit rulings** (adjustable): the open auction is modeled as free bidding
where *passing does not eliminate you* — you can re-enter as long as bidding
continues, and it resolves when everyone passes on the current high bid. The
**double** auction splits proceeds between the original player and whoever adds
the second card ("split" variant). Fixed prices can't exceed the seller's cash;
each round starts to the left of whoever played the last card.

---

## Reinforcement learning (`modern_art/rl/`)

**Why not "the optimal solution"?** This is a 3–5 player, imperfect-information
(hidden hands/money), general-sum game — there is no strategy that is optimal
independent of the opponents. The goal instead is a **strong, low-exploitability,
general** policy, evaluated by (1) win-rate vs baselines, (2) head-to-head between
checkpoints, and (3) exploitability (freeze the policy, train a best-response).

- `encoding.py` — fixed-length observation (`OBS_DIM=161`: your hand, public
  board, everyone's collections, **opponents' bids / cumulative spend**, auction
  context), a parameterized action (36 discrete choices + 1 continuous amount),
  and a legality mask. Bids are capped at `min(cash, n_cards × (accrued value +
  30))` — the painting's max possible value this round — which removes
  guaranteed-losing bids from the action space. Unit-tested: any masked action
  decodes to a legal move.
- `env.py` — a thin self-play environment (relative reward = cash − table mean,
  zero-sum).
- `model.py` — Actor-Critic: discrete head + Beta continuous head + value head +
  **two auxiliary belief heads** (predict opponents' hand composition / this
  round's final value markers), forcing the trunk to form beliefs about opponents
  and outcomes.
- `train.py` — parameter-shared self-play PPO + a **diverse opponent pool**
  (standard / heuristic / aggressive / tight / random + frozen past-self
  snapshots) + auxiliary losses.
- `eval.py` / `exploit.py` — full evaluation vs baselines / exploitability.

```bash
python -m modern_art.rl.train  --iters 200 --games 96 --device cpu   # a small net trains faster on CPU
python -m modern_art.rl.eval    --ckpt modern_art/rl/checkpoints/policy_v5.pt --games 300
python -m modern_art.rl.exploit --ckpt modern_art/rl/checkpoints/policy_v5.pt
```

**Current best: `policy_v5` (the "AI Master")** — win-rate as seat 0 vs three of
each opponent (fair share = 1 / players):

| Opponent | 3p | 4p | 5p |
|---|---|---|---|
| standard (strongest hand-written) | 81% | 69% | 52% |
| heuristic | 84% | 63% | 54% |
| aggressive ("reckless overbidder") | 100% | 100% | 100% |
| tight ("miser") | 79% | 87% | 64% |
| random | 100% | 98% | 95% |

- **One expert vs a table of reckless overbidders**: the AI Master wins ~100% by
  **selling paintings into their overpaying** rather than competing to overpay —
  the general skill that pure self-play never learns, but a diverse opponent pool
  does.
- **Exploitability**: a from-scratch best-response, trained specifically to beat it
  within budget, still loses — no easy exploit found.

Two design insights made it work: (1) **population training** (an opponent pool of
irrational personalities, not just self-play), and (2) putting **opponents' bids /
cumulative spend** in the observation so even sealed auctions can be reasoned
about.
