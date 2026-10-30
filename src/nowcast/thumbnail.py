"""1200x627 LinkedIn featured thumbnail in the portfolio house style
(dark-blue title band, light background, chart panel, 'Amir Khan' credit)."""
from PIL import Image, ImageDraw, ImageFont

from . import config as C

FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def make_thumbnail(title="KSA Consumer Spending Nowcast", chart="00_thumbnail_chart.png"):
    W, H = 1200, 627
    img = Image.new("RGB", (W, H), "#F4F6FA")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 90], fill="#0B3D5C")
    d.text((30, 45), title, font=ImageFont.truetype(FONT_B, 40), fill="white", anchor="lm")
    ch = Image.open(C.FIGURES / chart).convert("RGB")
    box_w, box_h = 1110, 480
    ch.thumbnail((box_w, box_h), Image.LANCZOS)
    x, y = (W - ch.width) // 2, 90 + (H - 90 - 40 - ch.height) // 2 + 5
    d.rectangle([x - 8, y - 8, x + ch.width + 8, y + ch.height + 8], fill="white")
    img.paste(ch, (x, y))
    d.text((W - 30, H - 22), "Amir Khan", font=ImageFont.truetype(FONT_R, 22), fill="#555555", anchor="rm")
    out = C.ROOT / "reports" / "linkedin_thumbnail.png"
    img.save(out)
    return out
