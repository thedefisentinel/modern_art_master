"""Actor-Critic 网络:参数共享,一张网吃 3–5 人。

躯干 MLP → 五个头:
  * 离散策略头(36,带掩码)—— 选牌 / 出价与否
  * 连续策略头 —— Beta(α,β)∈(0,1),映射到金额区间
  * 价值头 V —— PPO 优势估计
  * 辅助信念头①:预测各对手手牌构成(座位相对,回归)
  * 辅助信念头②:预测本回合各画家最终价值标码(回归)
辅助头用真实标签监督,逼躯干形成"对对手与结局的信念"。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical, Beta

from .encoding import OBS_DIM, NUM_DISCRETE, MAX_PLAYERS, NA, NT

_EPS = 1e-4
AUX_OPP_DIM = (MAX_PLAYERS - 1) * NA * NT   # 4 * 25 = 100
AUX_RANK_DIM = NA                            # 5


class ActorCritic(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.pi_discrete = nn.Linear(hidden, NUM_DISCRETE)
        self.pi_amount = nn.Linear(hidden, 2)        # -> softplus+1 => alpha,beta
        self.value = nn.Linear(hidden, 1)
        self.aux_opp = nn.Linear(hidden, AUX_OPP_DIM)
        self.aux_rank = nn.Linear(hidden, AUX_RANK_DIM)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        logits = self.pi_discrete(h)
        ab = torch.nn.functional.softplus(self.pi_amount(h)) + 1.0   # α,β ≥ 1(单峰)
        alpha, beta = ab[..., 0], ab[..., 1]
        value = self.value(h).squeeze(-1)
        return h, logits, alpha, beta, value

    @staticmethod
    def _masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
        neg = torch.finfo(logits.dtype).min
        logits = torch.where(mask, logits, torch.full_like(logits, neg))
        return Categorical(logits=logits)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, mask: torch.Tensor, needs_amount_fn=None):
        """单步采样。obs/mask 为 (D,)/(NUM_DISCRETE,)。返回选择及 logp/value。"""
        h, logits, alpha, beta, value = self.forward(obs.unsqueeze(0))
        dist = self._masked_categorical(logits, mask.unsqueeze(0))
        d = dist.sample()
        logp_d = dist.log_prob(d)
        bdist = Beta(alpha, beta)
        x = bdist.sample().clamp(_EPS, 1 - _EPS)
        logp_a = bdist.log_prob(x)
        d_int = int(d.item())
        needs = bool(needs_amount_fn(d_int)) if needs_amount_fn else False
        logp = logp_d + (logp_a if needs else 0.0)
        return {
            "discrete": d_int,
            "amount01": float(x.item()),
            "needs": needs,
            "logp": float(logp.item()),
            "value": float(value.item()),
        }

    @torch.no_grad()
    def act_greedy(self, obs: torch.Tensor, mask: torch.Tensor, needs_amount_fn=None):
        """贪心动作(评估/对战用):离散取 argmax,金额取 Beta 均值。"""
        h, logits, alpha, beta, value = self.forward(obs.unsqueeze(0))
        neg = torch.finfo(logits.dtype).min
        logits = torch.where(mask.unsqueeze(0), logits, torch.full_like(logits, neg))
        d_int = int(torch.argmax(logits, dim=-1).item())
        mean = (alpha / (alpha + beta)).clamp(_EPS, 1 - _EPS)
        return d_int, float(mean.item())

    def evaluate_actions(self, obs, mask, discrete, amount01, needs):
        """PPO 更新用:批量重算 logp/熵/价值 + 辅助头预测。"""
        h, logits, alpha, beta, value = self.forward(obs)
        dist = self._masked_categorical(logits, mask)
        logp_d = dist.log_prob(discrete)
        ent_d = dist.entropy()
        bdist = Beta(alpha, beta)
        x = amount01.clamp(_EPS, 1 - _EPS)
        logp_a = bdist.log_prob(x)
        ent_a = bdist.entropy()
        logp = logp_d + needs * logp_a
        entropy = ent_d + needs * ent_a
        aux_opp = self.aux_opp(h)
        aux_rank = self.aux_rank(h)
        return logp, entropy, value, aux_opp, aux_rank
