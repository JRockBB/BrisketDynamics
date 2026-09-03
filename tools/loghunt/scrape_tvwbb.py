#!/usr/bin/env python3
"""Collect cook-log candidates from a public TVWBB (XenForo) thread.

Walks the thread's pages, records per post: permalink, author, date, a
plain-text snippet, and every image URL (attachments and proxied external
images). ``--fetch`` downloads the images into ``cache/`` (git-ignored).

Polite by construction: public thread pages only, one request per ~2 s,
an identified user agent. Content is (c) its authors — this tool only
gathers candidates; what gets USED is a small factual data extraction
with attribution (see README.md).

Usage (from tools/loghunt/):
    python scrape_tvwbb.py [THREAD_URL] [--pages N] [--fetch]

Writes candidates.json (and cache/<sha8>.<ext> with --fetch).
"""

import argparse
import hashlib
import html
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

HERE = Path(__file__).resolve().parent
DEFAULT_THREAD = "https://tvwbb.com/threads/post-your-heatermeter-graphs.42779/"
UA = (
    "BrisketDynamics-loghunt/1.0 (open-source cook-physics research; "
    "small factual data extraction with attribution)"
)
DELAY_S = 2.0

ART_RE = re.compile(r"<article[^>]*\bdata-content=\"(post-\d+)\"[^>]*>", re.DOTALL)
AUTHOR_RE = re.compile(r"data-author=\"([^\"]*)\"")
TIME_RE = re.compile(r"<time[^>]*datetime=\"([^\"]*)\"")
BODY_RE = re.compile(r"<div class=\"bbWrapper\">(.*?)</div>", re.DOTALL)
IMG_RE = re.compile(r"<img[^>]*\bsrc=\"([^\"]+)\"[^>]*>")
TAG_RE = re.compile(r"<[^>]+>")
LAST_PAGE_RE = re.compile(r"page-(\d+)")


def _get(session: requests.Session, url: str) -> str:
    r = session.get(url, timeout=30, headers={"User-Agent": UA})
    r.raise_for_status()
    time.sleep(DELAY_S)
    return r.text


def parse_posts(page_html: str, base: str) -> list[dict]:
    """Split a thread page into posts; regex-based, XenForo layout."""
    posts = []
    marks = list(ART_RE.finditer(page_html))
    for i, m in enumerate(marks):
        block = page_html[m.start() : marks[i + 1].start() if i + 1 < len(marks) else None]
        post_id = m.group(1)
        author = (AUTHOR_RE.search(block) or [None, "?"])[1]
        when = (TIME_RE.search(block) or [None, ""])[1]
        body_m = BODY_RE.search(block)
        body = body_m.group(1) if body_m else ""
        text = html.unescape(TAG_RE.sub(" ", body))
        text = re.sub(r"\s+", " ", text).strip()
        imgs = []
        for src in IMG_RE.findall(body):
            try:
                u = urljoin(base, html.unescape(src))
            except ValueError:  # data: URIs, template srcs, malformed hrefs
                continue
            if not u.startswith("http"):
                continue
            # keep chart-sized content: attachments and proxied externals;
            # skip smilies/avatars by path
            if any(s in u for s in ("smilies", "avatar", "badge", "emoji")):
                continue
            imgs.append(u)
        posts.append(
            {
                "post_url": urljoin(base, f"post-{post_id.split('-')[1]}"),
                "post_id": post_id,
                "author": author,
                "date": when,
                "text": text[:800],
                "images": sorted(set(imgs)),
            }
        )
    return posts


def fetch_images(session: requests.Session, posts: list[dict], cache: Path):
    """Download each post's images into cache/ (content-hash names)."""
    cache.mkdir(exist_ok=True)
    for p in posts:
        local = []
        for u in p["images"]:
            try:
                r = session.get(u, timeout=30, headers={"User-Agent": UA})
                r.raise_for_status()
            except requests.RequestException as e:  # dead externals are common
                print(f"  skip {u}: {e}", file=sys.stderr)
                continue
            time.sleep(DELAY_S)
            ctype = r.headers.get("content-type", "")
            ext = (
                ".png"
                if "png" in ctype
                else ".gif"
                if "gif" in ctype
                else ".jpg"
                if ("jpeg" in ctype or "jpg" in ctype)
                else None
            )
            if ext is None or len(r.content) < 20_000:  # not a chart
                continue
            name = hashlib.sha256(r.content).hexdigest()[:8] + ext
            (cache / name).write_bytes(r.content)
            local.append(name)
        p["cached"] = local
        if local:
            print(f"  {p['post_id']} ({p['author']}): {len(local)} image(s)")


def main():
    """Walk the thread pages and write candidates.json."""
    ap = argparse.ArgumentParser()
    ap.add_argument("thread", nargs="?", default=DEFAULT_THREAD)
    ap.add_argument("--pages", type=int, default=0, help="0 = all")
    ap.add_argument("--start-page", type=int, default=1)
    ap.add_argument("--fetch", action="store_true", help="download images to cache/")
    ap.add_argument("--out", default="candidates.json")
    args = ap.parse_args()

    session = requests.Session()
    first = _get(session, args.thread)
    last = max((int(n) for n in LAST_PAGE_RE.findall(first)), default=1)
    n_pages = last - args.start_page + 1 if args.pages == 0 else args.pages
    print(f"thread reports {last} page(s); walking {n_pages} from {args.start_page}")

    posts = []
    for k in range(n_pages):
        page = args.start_page + k
        url = args.thread if page == 1 else urljoin(args.thread, f"page-{page}")
        html_page = first if page == 1 else _get(session, url)
        got = parse_posts(html_page, args.thread)
        posts.extend(got)
        print(f"page {page}: {len(got)} posts, {sum(len(p['images']) for p in got)} images")

    if args.fetch:
        fetch_images(session, posts, HERE / "cache")

    (HERE / args.out).write_text(json.dumps(posts, indent=1), encoding="utf-8")
    with_img = sum(1 for p in posts if p["images"])
    print(f"wrote {args.out}: {len(posts)} posts, {with_img} with images")


if __name__ == "__main__":
    main()
