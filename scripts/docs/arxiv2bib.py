#!/usr/bin/env python
"""Generate BibTeX entries for arXiv papers from the arXiv API (so titles/authors/years are never typed by hand).

    python scripts/docs/arxiv2bib.py key1=2402.15391 key2=2503.02904 ... > paper/tmrb/arxiv.bib
A trailing "|Venue Year" marks a peer-reviewed paper, e.g. genie=2402.15391|ICML 2024: it is cited as published there
(IEEE style, "in Proc. Int. Conf. Mach. Learn. (ICML), 2024"), not as an arXiv preprint.
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
        s = s.replace(f"${g}$", t).replace(g, t)             # some arXiv titles already put the letter in math
    return s


def author(name: str) -> str:
    """One author; an organisation is braced so that BibTeX keeps it whole ("Team Wan", not "T. Wan")."""
    name = " ".join(name.split())
    return f"{{{name}}}" if set(name.split()) & {"Consortium", "Collaboration", "Team"} else name


PROCEEDINGS = [                                           # venue prefix -> IEEE-abbreviated proceedings name
    ("MICCAI Workshop on Data Engineering in Medical Imaging (DEMI)", "MICCAI Workshop Data Eng. Med. Imaging (DEMI)"),
    ("MICCAI Workshop EARTH", "MICCAI Workshop Embodied AI Robot. Healthcare (EARTH)"),
    ("MICCAI", "Int. Conf. Med. Image Comput. Comput. Assist. Interv. (MICCAI)"),
    ("IEEE International Symposium on Medical Robotics (ISMR)", "IEEE Int. Symp. Med. Robot. (ISMR)"),
    ("IEEE/CVF ICCV", "IEEE/CVF Int. Conf. Comput. Vis. (ICCV)"),
    ("ICCV", "IEEE/CVF Int. Conf. Comput. Vis. (ICCV)"),
    ("CVPR", "IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)"),
    ("ICML", "Int. Conf. Mach. Learn. (ICML)"),
    ("ICLR", "Int. Conf. Learn. Represent. (ICLR)"),
    ("NeurIPS", "Adv. Neural Inf. Process. Syst. (NeurIPS)"),
    ("AAAI", "AAAI Conf. Artif. Intell. (AAAI)"),
]


def proceedings(venue: str) -> tuple[str, str]:
    """("Proc. <abbreviated venue>", year) for a venue such as "ICML 2024" or "MICCAI Workshop EARTH 2024"."""
    year = venue.rstrip(" .")[-4:]
    for prefix, name in PROCEEDINGS:
        if venue.startswith(prefix):
            return f"Proc. {name}", year
    raise ValueError(f"unknown venue {venue!r}: add it to PROCEEDINGS")


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
        authors = " and ".join(author(a.find("a:name", NS).text) for a in e.findall("a:author", NS)
                               if any(c.isalpha() for c in a.find("a:name", NS).text))   # not a ":" separator
        head = f"  title = {{{{{title}}}}},\n  author = {{{tex_escape(authors)}}},\n"
        if venue:
            book, year = proceedings(venue)
            print(f"@inproceedings{{{key},\n{head}  booktitle = {{{book}}},\n  year = {{{year}}}\n}}\n")
        else:
            year = e.find("a:published", NS).text[:4]
            print(f"@article{{{key},\n{head}  journal = {{arXiv preprint arXiv:{aid}}},\n  year = {{{year}}}\n}}\n")


if __name__ == "__main__":
    main()
