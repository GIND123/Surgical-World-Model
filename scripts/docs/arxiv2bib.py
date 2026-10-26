#!/usr/bin/env python
"""Generate BibTeX entries for arXiv papers from the arXiv API (so titles/authors/years are never typed by hand).

    python scripts/docs/arxiv2bib.py key1=2402.15391 key2=2503.02904 ... > paper/tmrb/arxiv.bib
A trailing "|Venue" sets the journal/booktitle note, e.g. surgwm=2503.02904|MICCAI Workshop DEMI 2025.
"""

from __future__ import annotations

import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def fetch(ids: list[str]) -> dict[str, ET.Element]:
    out = {}
    for i in range(0, len(ids), 40):
        chunk = ids[i:i + 40]
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"id_list": ",".join(chunk), "max_results": len(chunk)})
        root = ET.fromstring(urllib.request.urlopen(url, timeout=90).read())
        for e in root.findall("a:entry", NS):
            aid = e.find("a:id", NS).text.split("/abs/")[-1].split("v")[0]
            out[aid] = e
        time.sleep(3)
    return out


GREEK = {"α": r"$\alpha$", "β": r"$\beta$", "γ": r"$\gamma$", "δ": r"$\delta$", "π": r"$\pi$", "σ": r"$\sigma$",
         "τ": r"$\tau$", "μ": r"$\mu$", "λ": r"$\lambda$", "ε": r"$\epsilon$"}


def tex_escape(s: str) -> str:
    s = " ".join(s.split()).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
    for g, t in GREEK.items():
        s = s.replace(g, t)
    return s


def main() -> None:
    specs = []
    for arg in sys.argv[1:]:
        key, rest = arg.split("=", 1)
        aid, _, venue = rest.partition("|")
        specs.append((key, aid, venue))
    entries = fetch([a for _, a, _ in specs])
    for key, aid, venue in specs:
        e = entries.get(aid)
        if e is None:
            print(f"% MISSING {key} {aid}", file=sys.stderr)
            continue
        title = tex_escape(e.find("a:title", NS).text)
        authors = " and ".join(a.find("a:name", NS).text.strip() for a in e.findall("a:author", NS))
        year = e.find("a:published", NS).text[:4]
        note = f",\n  note = {{{tex_escape(venue)}}}" if venue else ""
        print(f"@article{{{key},\n  title = {{{{{title}}}}},\n  author = {{{tex_escape(authors)}}},\n"
              f"  journal = {{arXiv preprint arXiv:{aid}}},\n  year = {{{year}}}{note}\n}}\n")


if __name__ == "__main__":
    main()
