"""把训练好的检查点导出成浏览器可用的 ONNX(供纯静态版 onnxruntime-web 推理)。

    python tools/export_onnx.py                 # 默认 policy_v5.pt -> web/static/policy.onnx
    python tools/export_onnx.py policy_v4.pt

导出的是推理子图:obs -> (logits, alpha, beta)。贪心解码(掩码 argmax + Beta 均值)在
JS 端做(见 agents.js)。导出后会与 PyTorch 对拍验证。
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
from modern_art.rl.model import ActorCritic
from modern_art.rl.encoding import OBS_DIM

CKPT = sys.argv[1] if len(sys.argv) > 1 else "policy_v5.pt"
CK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "modern_art", "rl", "checkpoints")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "modern_art", "web", "static", "policy.onnx")

net = ActorCritic()
net.load_state_dict(torch.load(os.path.join(CK_DIR, CKPT), map_location="cpu")["model"])
net.eval()


class Infer(nn.Module):
    def __init__(self, n):
        super().__init__(); self.n = n

    def forward(self, obs):
        _h, logits, alpha, beta, _v = self.n(obs)
        return logits, alpha, beta


m = Infer(net).eval()
torch.onnx.export(
    m, torch.zeros(1, OBS_DIM), OUT,
    input_names=["obs"], output_names=["logits", "alpha", "beta"],
    dynamic_axes={"obs": {0: "B"}, "logits": {0: "B"}, "alpha": {0: "B"}, "beta": {0: "B"}},
    opset_version=17, dynamo=False)
print(f"导出 {CKPT} -> {OUT}  ({os.path.getsize(OUT)//1024} KB)")

# 对拍验证
import onnxruntime as ort
sess = ort.InferenceSession(OUT)
rng = np.random.default_rng(0); maxd = 0.0
for _ in range(20):
    x = rng.standard_normal((1, OBS_DIM)).astype(np.float32)
    t = [v.detach().numpy() for v in m(torch.from_numpy(x))]
    o = sess.run(None, {"obs": x})
    maxd = max(maxd, *[abs(a - b).max() for a, b in zip(t, o)])
print(f"ONNX vs PyTorch 最大误差 {maxd:.2e}")
