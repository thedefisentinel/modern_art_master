# -*- coding: utf-8 -*-
"""把 j:\\S1\\memes 下按张数命名的文件夹里的图片,导入为各牌的卡面。
每个文件夹 -> 一位画家;文件夹内图片按名字排序,依次对应 art_id 0..N-1。
统一居中裁方 + 缩到 256,存成 web/static/art/<artist>_<id>.png(并覆盖单图回退)。"""
import os, sys
from PIL import Image, ImageSequence

SRC = r"J:\S1\memes"
DST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "modern_art", "web", "static", "art")
SIZE = 256
EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".avif")

# 文件夹名 -> (画家 id, 张数)
FOLDERS = {
    "鸡汤12":   ("lite_metal", 12),
    "征服13":   ("yoko", 13),
    "杰哥不要14": ("christin_p", 14),
    "恶搞15":   ("karl_gitter", 15),
    "cats16":  ("krypto", 16),
}


def square_256(im):
    # 裁成正方形:横向居中,纵向保留顶部(裁掉底部)
    im = im.convert("RGB")
    w, h = im.size
    s = min(w, h)
    left = (w - s) // 2
    im = im.crop((left, 0, left + s, s))
    return im.resize((SIZE, SIZE), Image.LANCZOS)


def load_first_frame(path):
    im = Image.open(path)
    try:
        im.seek(0)  # gif/webp 取首帧
    except Exception:
        pass
    return im


def main():
    os.makedirs(DST, exist_ok=True)
    total = 0
    for folder, (artist, count) in FOLDERS.items():
        d = os.path.join(SRC, folder)
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(EXTS))
        if len(files) < count:
            print(f"⚠ {folder}: 只有 {len(files)} 张,少于 {count}")
        for i in range(count):
            f = files[i % len(files)]
            im = square_256(load_first_frame(os.path.join(d, f)))
            im.save(os.path.join(DST, f"{artist}_{i}.png"), "PNG", optimize=True)
            if i == 0:
                im.save(os.path.join(DST, f"{artist}.png"), "PNG", optimize=True)
            total += 1
        print(f"{folder} -> {artist}: {count} 张")
    print(f"完成,共导入 {total} 张。")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
