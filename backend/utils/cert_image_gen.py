"""证书图片生成工具。

将证书的名称/类型/日期渲染成真实的 PNG 图片，供视觉编码器(BLIP)提取真实图像特征。
合成数据集中证书名称是有限集合，因此按 (award_name, award_type) 渲染并缓存到磁盘，
每个候选人的证书通过 image_path 指向对应图片。
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

# 证书图片输出目录
CERT_IMAGE_DIR = Path(__file__).resolve().parents[2] / "data" / "cert_images"

# 不同证书类型用不同主色，让图像在视觉上可区分（视觉编码器据此产生区分性特征）
_TYPE_COLORS = {
    "竞赛获奖": (198, 40, 40),    # 红
    "资格证书": (21, 101, 192),   # 蓝
    "荣誉称号": (245, 124, 0),    # 橙
}
_DEFAULT_COLOR = (69, 90, 100)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """尽量加载中文字体，失败则用默认字体。"""
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _safe_name(*parts: str) -> str:
    raw = "__".join(p for p in parts if p)
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]
    return f"cert_{h}.png"


def render_certificate_image(
    award_name: str,
    award_type: str = "",
    award_date: str = "",
    out_dir: Optional[Path] = None,
) -> str:
    """渲染一张证书图片，返回图片绝对路径（已缓存则直接返回）。"""
    out_dir = Path(out_dir) if out_dir else CERT_IMAGE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = _safe_name(award_name, award_type)
    fpath = out_dir / fname
    if fpath.exists():
        return str(fpath)

    W, H = 512, 384
    color = _TYPE_COLORS.get(award_type, _DEFAULT_COLOR)
    img = Image.new("RGB", (W, H), (252, 250, 245))
    draw = ImageDraw.Draw(img)

    # 外边框
    draw.rectangle([8, 8, W - 8, H - 8], outline=color, width=6)
    draw.rectangle([20, 20, W - 20, H - 20], outline=color, width=2)
    # 顶部色带
    draw.rectangle([20, 20, W - 20, 70], fill=color)

    title_font = _load_font(30)
    name_font = _load_font(34)
    sub_font = _load_font(22)
    small_font = _load_font(18)

    def _center(text, y, font, fill):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((W - tw) / 2, y), text, font=font, fill=fill)

    _center("荣 誉 证 书", 30, title_font, (255, 255, 255))
    _center("CERTIFICATE", 110, sub_font, color)
    _center(award_name, 175, name_font, (40, 40, 40))
    if award_type:
        _center(f"类别：{award_type}", 240, small_font, (90, 90, 90))
    if award_date:
        _center(f"颁发日期：{award_date}", 280, small_font, (90, 90, 90))
    # 印章
    draw.ellipse([W - 130, H - 110, W - 40, H - 20], outline=color, width=4)
    _center_seal = "认证"
    sbbox = draw.textbbox((0, 0), _center_seal, font=small_font)
    sw = sbbox[2] - sbbox[0]
    draw.text((W - 85 - sw / 2, H - 78), _center_seal, font=small_font, fill=color)

    img.save(fpath, "PNG")
    return str(fpath)


# 合成数据集中出现的全部证书名称（与 run_experiments.generate_candidate 一致）
ALL_AWARD_NAMES = [
    "ACM-ICPC区域赛金奖", "数学建模一等奖", "软件设计师",
    "PMP认证", "AWS认证", "优秀毕业生",
]
ALL_AWARD_TYPES = ["竞赛获奖", "资格证书", "荣誉称号"]


def pregenerate_all(out_dir: Optional[Path] = None) -> dict:
    """预生成所有 (name, type) 组合的证书图片，返回 {(name,type): path}。"""
    mapping = {}
    for name in ALL_AWARD_NAMES:
        for atype in ALL_AWARD_TYPES:
            p = render_certificate_image(name, atype, out_dir=out_dir)
            mapping[(name, atype)] = p
    return mapping


if __name__ == "__main__":
    m = pregenerate_all()
    print(f"generated {len(m)} certificate images in {CERT_IMAGE_DIR}")
    for k, v in list(m.items())[:3]:
        print(k, "->", v)
