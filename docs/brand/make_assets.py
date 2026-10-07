"""Build Studio's brand files from the SVG set in docs/brand/source/ (the user's
"UnitWave Studio App Brand Assets", 2026-10-01; see its README). Not part of the app:
run it again after changing a master.

    .venv/bin/python docs/brand/make_assets.py    # macOS: the PNGs use QuickLook

Writes unitwave/studio/static/brand/:
- unitwave-app-icon-flat.svg: the tab icon and the top-bar mark (the flat version
  reads better small);
- unitwave-navbar-light.svg, unitwave-navbar-dark.svg: the header lockups, with the
  viewBox cropped to the artwork (the masters leave the right 40% of the canvas
  empty), otherwise unchanged;
- unitwave-splash-light.svg, unitwave-splash-dark.svg: the screen shown while a
  session opens, unchanged;
- icon-32.png, icon-180.png: PNG fallbacks for browsers without SVG tab icons and
  for Apple's home-screen icon. QuickLook renders the master on white, and the
  corners are cut back to the icon's own rounded square (rx 218 of 1024).
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source"
OUT = HERE.parents[1] / "unitwave" / "studio" / "static" / "brand"
# The navbar artwork's bounds in its 1200 x 240 master, measured in a browser
# (getBBox: x 43-745, y 27-169), plus room for the glow.
NAVBAR_VIEWBOX = (28, 12, 732, 172)
ICON_RX = 218 / 1024
RENDER_PX = 720
PNGS = {
    "icon-32.png": ("unitwave-app-icon-flat.svg", 32),
    "icon-180.png": ("unitwave-app-icon.svg", 180),
}


def cropped_navbar(text: str) -> str:
    """The master with its <svg> size and viewBox set to NAVBAR_VIEWBOX."""
    x, y, w, h = NAVBAR_VIEWBOX
    tag = 'width="1200" height="240" viewBox="0 0 1200 240"'
    if text.count(tag) != 1:
        raise ValueError("the navbar master's canvas changed: measure its bounds again")
    return text.replace(tag, f'width="{w}" height="{h}" viewBox="{x} {y} {w} {h}"')


def icon_png(master: str, size_px: int) -> Image.Image:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["qlmanage", "-t", "-s", str(RENDER_PX), "-o", tmp, str(SOURCE / master)],
            check=True,
            capture_output=True,
        )
        image = Image.open(Path(tmp) / f"{master}.png").convert("RGBA")
    if image.size != (RENDER_PX, RENDER_PX):
        raise ValueError(f"QuickLook rendered {master} at {image.size}")
    big = 4 * RENDER_PX  # a smooth mask, inset 1 px so no white rendering fringe is kept
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).rounded_rectangle((4, 4, big - 5, big - 5), radius=ICON_RX * big, fill=255)
    image.putalpha(mask.resize(image.size, Image.LANCZOS))
    return image.resize((size_px, size_px), Image.LANCZOS)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for name in (
        "unitwave-app-icon-flat.svg",
        "unitwave-splash-light.svg",
        "unitwave-splash-dark.svg",
    ):
        shutil.copyfile(SOURCE / name, OUT / name)
    for theme in ("light", "dark"):
        name = f"unitwave-navbar-{theme}.svg"
        (OUT / name).write_text(cropped_navbar((SOURCE / name).read_text()))
    for name, (master, size) in PNGS.items():
        icon_png(master, size).save(OUT / name, optimize=True)
    for path in sorted(OUT.iterdir()):
        print(f"{path.name}: {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
