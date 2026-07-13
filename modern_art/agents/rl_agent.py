"""把训练好的 RL 策略封成 Agent,接入 CLI / 网页 / 评估。

    from modern_art.agents.rl_agent import RLAgent
    agent = RLAgent("modern_art/rl/checkpoints/policy.pt")   # 载入检查点
    action = agent.act(state)

推理默认用 CPU、贪心动作(argmax 离散 + Beta 均值金额),适合对战;
训练/评估里想要随机性可设 greedy=False。
"""

from __future__ import annotations

import torch

from ..engine import game
from ..engine.state import GameState
from ..engine.actions import Action
from ..rl import encoding
from ..rl.model import ActorCritic
from .base import Agent


class RLAgent(Agent):
    def __init__(self, checkpoint: str, device: str = "cpu", greedy: bool = True,
                 name: str = "rl", seed: int | None = None):
        super().__init__(name=name, seed=seed)
        self.device = torch.device(device)
        self.greedy = greedy
        self.net = ActorCritic().to(self.device)
        ckpt = torch.load(checkpoint, map_location=self.device)
        state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        if isinstance(ckpt, dict) and ckpt.get("obs_dim", encoding.OBS_DIM) != encoding.OBS_DIM:
            raise ValueError(
                f"检查点 OBS_DIM={ckpt['obs_dim']} 与当前 {encoding.OBS_DIM} 不一致(编码变了?需重训)")
        self.net.load_state_dict(state_dict)
        self.net.eval()

    @torch.no_grad()
    def act(self, state: GameState) -> Action:
        me = game.current_player(state)
        obs_np, mask_np = encoding.encode_obs(state, me), encoding.legal_mask(state, me)
        obs = torch.from_numpy(obs_np).to(self.device)
        mask = torch.from_numpy(mask_np).to(self.device)
        if self.greedy:
            d, x = self.net.act_greedy(obs, mask)
        else:
            out = self.net.act(obs, mask,
                               needs_amount_fn=lambda dd: encoding.needs_amount(state, me, dd))
            d, x = out["discrete"], out["amount01"]
        return encoding.decode_action(state, me, d, x)
