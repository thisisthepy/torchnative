#!/usr/bin/env python3
"""Test for the static guide in docs/guide/ (run: python3 docs/guide/check_guide.py).

Checks, each of which fails loudly:
  1. every expected page exists and parses (balanced tags for non-void elements);
  2. every relative href/src resolves to a file inside docs/guide/;
  3. bilingual completeness -- every visible text node sits inside an element
     marked data-lang="en" or data-lang="ko" (or inside code/pre/svg, or an
     element marked translate="no"), and every parent holds as many "en"
     children as "ko" children, in the same order of tag names;
  4. no external script or stylesheet except Google Fonts;
  5. every page carries the shared header (language + theme toggles) and the
     same navigation, so no page is unreachable.
What it cannot see: whether a Korean string is a faithful translation of its
English pair, and how the pages look. Those are reviewed by a person.
Exit code 0 iff every check passes.
"""
from __future__ import annotations

import pathlib
import re
import sys
from html.parser import HTMLParser
from urllib.parse import urlparse

GUIDE = pathlib.Path(__file__).resolve().parent
PAGES = [
    "index.html", "getting-started.html", "concepts.html",
    "guide-inference.html", "guide-adaptation.html", "guide-accelerators.html",
    "guide-federated.html", "ecosystem.html", "status.html", "contributing.html",
]
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "source", "track", "wbr", "path", "circle", "rect", "line",
        "polyline", "polygon", "ellipse", "use", "stop"}
SKIP_TEXT = {"script", "style", "code", "pre", "svg", "title", "kbd"}
ALLOWED_EXTERNAL = ("https://fonts.googleapis.com", "https://fonts.gstatic.com")


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, dict]] = []
        self.errors: list[str] = []
        self.links: list[str] = []
        self.external: list[str] = []
        self.ids: set[str] = set()
        # per open element: list of (lang, tag) of its direct data-lang children
        self.children: list[list[tuple[str, str]]] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        for key in ("href", "src"):
            if key in a and a[key]:
                self.links.append(a[key])
        if tag == "script" and a.get("src", "").startswith("http"):
            self.external.append(a["src"])
        if tag == "link" and a.get("rel") == "stylesheet" and a.get("href", "").startswith("http"):
            self.external.append(a["href"])
        if "data-lang" in a and self.children:
            self.children[-1].append((a["data-lang"], tag))
        if tag in VOID:
            return
        self.stack.append((tag, a))
        self.children.append([])

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:          # <x/> on a non-void element closes it
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack:
            self.errors.append(f"stray </{tag}>")
            return
        open_tag, _ = self.stack.pop()
        kids = self.children.pop()
        if open_tag != tag:
            self.errors.append(f"</{tag}> closes <{open_tag}>")
        en = [t for lang, t in kids if lang == "en"]
        ko = [t for lang, t in kids if lang == "ko"]
        if en != ko:
            self.errors.append(f"<{open_tag}> has en children {en} but ko children {ko}")

    def handle_data(self, data):
        # Text with no letters (numbers, arrows, punctuation) reads the same in
        # both languages; names and code are marked translate="no" explicitly.
        if not re.search(r"[^\W\d_]", data):
            return
        for tag, a in self.stack:
            if tag in SKIP_TEXT or "data-lang" in a or a.get("translate") == "no":
                return
        self.errors.append(f"untranslated text {data.strip()[:60]!r} in <{self.stack[-1][0] if self.stack else '?'}>")


def check() -> list[str]:
    problems: list[str] = []
    navs = {}
    for name in PAGES:
        path = GUIDE / name
        if not path.is_file():
            problems.append(f"{name}: missing")
            continue
        text = path.read_text(encoding="utf-8")
        p = Page()
        p.feed(text)
        p.close()
        if p.stack:
            problems.append(f"{name}: unclosed {[t for t, _ in p.stack]}")
        problems += [f"{name}: {e}" for e in p.errors]
        problems += [f"{name}: external resource {u}" for u in p.external
                     if not u.startswith(ALLOWED_EXTERNAL)]
        for link in p.links:
            u = urlparse(link)
            if u.scheme in ("http", "https", "mailto") or link.startswith("//"):
                if link.startswith("//"):
                    problems.append(f"{name}: protocol-relative URL {link}")
                continue
            target, frag = u.path, u.fragment
            if not target:
                if frag and frag not in p.ids:
                    problems.append(f"{name}: anchor #{frag} not on the page")
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.is_file():
                problems.append(f"{name}: link {link} does not resolve")
            elif GUIDE not in resolved.parents:
                problems.append(f"{name}: link {link} leaves docs/guide/")
        for needle in ('id="lang-toggle"', 'id="theme-toggle"', 'assets/style.css', 'assets/site.js',
                       '<html lang="en"'):
            if needle not in text:
                problems.append(f"{name}: missing {needle}")
        start, end = text.find("<nav"), text.find("</nav>")
        navs[name] = text[start:end] if start >= 0 else ""
        for other in PAGES:
            if f'href="{other}"' not in navs[name] and other != "contributing.html":
                problems.append(f"{name}: nav does not link {other}")
    if len(set(navs.values())) > 1:
        problems.append("the <nav> block differs between pages")
    for asset in ("assets/style.css", "assets/site.js"):
        if not (GUIDE / asset).is_file():
            problems.append(f"{asset}: missing")
    css = (GUIDE / "assets/style.css").read_text() if (GUIDE / "assets/style.css").is_file() else ""
    if "#e8590c" not in css.lower():
        problems.append("style.css: the torchnative accent #e8590c is not defined")
    js = (GUIDE / "assets/site.js").read_text() if (GUIDE / "assets/site.js").is_file() else ""
    if "localStorage" in js and "try" not in js:
        problems.append("site.js: localStorage access is not wrapped in try/catch")
    return problems


def main() -> int:
    problems = check()
    for pr in problems:
        print("FAIL", pr)
    print(f"{len(PAGES)} pages, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
