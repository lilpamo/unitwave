"""Brand artwork (the user's SVG set, 2026-10-01): every file the pages use exists."""

import re
from pathlib import Path

from unitwave.studio.server import _static

STUDIO = Path(__file__).resolve().parents[1] / "unitwave" / "studio"


def test_every_brand_file_the_pages_and_styles_use_is_served():
    used = set()
    for path in [STUDIO / "home.html", STUDIO / "index.html", *(STUDIO / "static").glob("*.css")]:
        used |= set(re.findall(r"/static/(brand/[\w.-]+)", path.read_text()))
    # The tab icon and top-bar mark on both pages; the lockups and splashes on the homepage.
    assert {
        "brand/unitwave-app-icon-flat.svg",
        "brand/icon-32.png",
        "brand/icon-180.png",
        "brand/unitwave-navbar-light.svg",
        "brand/unitwave-navbar-dark.svg",
        "brand/unitwave-splash-light.svg",
        "brand/unitwave-splash-dark.svg",
    } <= used
    for ref in used:
        assert _static(f"/static/{ref}") is not None, ref


def test_both_pages_carry_the_tab_icon_and_the_mark():
    for page in ("home.html", "index.html"):
        html = (STUDIO / page).read_text()
        assert 'rel="icon"' in html and 'rel="apple-touch-icon"' in html, page
        assert (
            '<div class="brand"><img src="/static/brand/unitwave-app-icon-flat.svg" alt="">' in html
        ), page


def test_served_svgs_carry_no_script_or_outside_reference():
    for svg in (STUDIO / "static" / "brand").glob("*.svg"):
        text = svg.read_text().lower()
        assert "<script" not in text and "foreignobject" not in text, svg.name
        assert not re.search(r"href=\"(https?:)?//", text), svg.name
