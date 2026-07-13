"""用本地 ComfyUI(SDXL)为 70 张牌各生成一张不同的画,存到网页静态目录。

前置:ComfyUI 已在 http://127.0.0.1:8188 运行(j:\\llm\\ComfyUI\\run.bat)。
运行:python tools/gen_art.py            # 生成全部 70 张(约 9 分钟)
     python tools/gen_art.py yoko      # 只重生成某位画家

每位画家有固定张数(12/13/14/15/16),各生成 <artist>_0.png .. <artist>_{n-1}.png,
用同一主题、不同姿势 + 不同随机种子来保证"每张不一样"。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
★ 想换成你要的画:直接改下面 BASE 里的关键词(主题)和 POSES(动作/表情)。
  关键词你随便填——这是你的本地模型,你说了算。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

COMFY = "http://127.0.0.1:8188"
CKPT = "sd_xl_base_1.0.safetensors"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "modern_art", "web", "static", "art")

SIZE, STEPS, CFG = 1024, 32, 6.5
QUALITY = ("masterpiece, best quality, highly detailed, clean bold outlines, vibrant colors, "
           "professional sticker illustration, centered subject, simple background")
NEG = ("ugly, deformed, blurry, low quality, jpeg artifacts, text, watermark, signature, "
       "extra limbs, bad anatomy, messy, muddy colors, scary, nsfw")

# 画家 -> (张数, 主题基础提示词)。改这里换主题。
ARTISTS = {
    "lite_metal":  (12, "a cute chubby cartoon bear mascot in a green forest"),
    "yoko":        (13, "an adorable round pink cartoon piglet"),
    "christin_p":  (14, "a goofy cartoon family together, sitcom cartoon style"),
    "karl_gitter": (15, "a funny expressive orange cartoon cat"),
    "krypto":      (16, "a cute chubby yellow baby dragon mascot"),
}
# 用不同姿势/表情让每张画不一样
POSES = [
    "sitting and smiling", "waving hello", "sleeping peacefully", "eating a snack",
    "running happily", "looking surprised", "dancing", "reading a book",
    "holding a balloon", "winking playfully", "jumping for joy", "giving a warm hug",
    "laughing out loud", "yawning sleepily", "peeking from behind a corner", "celebrating with confetti",
]


def _post(path, payload):
    req = urllib.request.Request(COMFY + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def _get(path) -> bytes:
    return urllib.request.urlopen(COMFY + path, timeout=60).read()


def wait_up(timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(COMFY + "/system_stats", timeout=5)
            return
        except Exception:
            time.sleep(3)
    raise RuntimeError("ComfyUI 未就绪")


def workflow(pos: str, seed: int) -> dict:
    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": f"{pos}, {QUALITY}", "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["4", 1]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": SIZE, "height": SIZE, "batch_size": 1}},
        "3": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": STEPS, "cfg": CFG, "sampler_name": "dpmpp_2m",
            "scheduler": "karras", "denoise": 1.0,
            "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ma_art", "images": ["8", 0]}},
    }


def generate(pos: str, seed: int) -> bytes:
    pid = _post("/prompt", {"prompt": workflow(pos, seed)})["prompt_id"]
    while True:
        hist = json.loads(_get(f"/history/{pid}"))
        if pid in hist and hist[pid].get("outputs"):
            img = hist[pid]["outputs"]["9"]["images"][0]
            return _get(f"/view?filename={img['filename']}&subfolder={img.get('subfolder','')}&type={img['type']}")
        time.sleep(1.2)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    os.makedirs(OUT_DIR, exist_ok=True)
    print("等待 ComfyUI 就绪…", flush=True)
    wait_up()
    total = sum(n for a, (n, _) in ARTISTS.items() if only is None or a == only)
    done = 0
    for ai, (artist, (count, base)) in enumerate(ARTISTS.items()):
        if only and artist != only:
            continue
        for i in range(count):
            pos = f"{base}, {POSES[i % len(POSES)]}"
            t0 = time.time()
            data = generate(pos, seed=1000 + ai * 100 + i)
            with open(os.path.join(OUT_DIR, f"{artist}_{i}.png"), "wb") as f:
                f.write(data)
            done += 1
            print(f"[{done}/{total}] {artist}_{i}.png  ({len(data)//1024}KB {time.time()-t0:.1f}s)", flush=True)
    print("全部完成。", flush=True)


if __name__ == "__main__":
    main()
