# Submitting to IEEE T-MRB

Checked on 15 Nov 2026 against T-MRB's [Information for Authors](https://www.ieee-ras.org/publications/t-mrb/information-for-authors)
and the IEEE RAS [rules for the double-anonymous review process](https://www.ieee-ras.org/publications/rules-for-the-double-anonymous-review-process/).

## Requirements and where they are met

| Requirement | State |
|---|---|
| Double-anonymous review: no names, affiliations, acknowledgements or identifying links anywhere in the manuscript or supplementary material; desk rejection possible otherwise | `review.tex` sets `\reviewtrue` (the default): "Anonymous Authors", no acknowledgement, availability statement without the GitHub and Hugging Face links. Both PDFs scanned (text, metadata, all decompressed streams): no name, account or local path |
| Original article: 12 pages, double-column IEEE template; $175 per page beyond 12 | 12 pages including references |
| Abstract of at most 200 words | 193 |
| 2 to 5 keywords, chosen in the portal | Surgical robotics; World models; Video prediction; Latent actions; Robot-assisted surgery |
| Approval or exemption for research on humans or animals, stated in the manuscript and in the submission form | "Data Availability and Ethics" states that only previously published datasets of animal and human procedures are analysed and that no new data were collected. The form answer is the authors' (below) |
| Multimedia described in the supporting material | Supplementary Movie S1, described in the supplementary PDF (section "Rollouts") |
| References in IEEE style | Peer-reviewed papers are cited at their venue (`scripts/docs/arxiv2bib.py`) |
| Graphical abstract | Only after acceptance |

## Files to upload (at most 60 MB in total)

1. Manuscript: `paper/tmrb/main.pdf` (review build).
2. Supplementary material: `paper/tmrb/supplementary.pdf`.
3. Supplementary Movie S1: `outputs/videos/movie_s1.mp4` (made by job `0741a`; SurgVU frames, so not in Git).
4. Cover letter: `paper/tmrb/cover_letter.md`, with the bracketed fields filled. It goes to the editor only and may name the authors.

Rebuild both PDFs with `cd paper/tmrb && latexmk -pdf main.tex supplementary.tex`. Every number comes from
`scripts/eval/aggregate.py`; a missing one prints a red **??**, so search the PDFs for "??" before uploading.

## Entries only the authors can make

- **Authors.** Names, affiliations, ORCID iDs and the corresponding author go into the portal only, never into the PDFs.
- **Ethics question.** The study is a secondary analysis of public, de-identified datasets: SurgVU, porcine training
  procedures; SAR-RARP50 and AutoLaparo, human surgery video; JIGSAWS and Open-H-Embodiment, bench-top, phantom and
  ex-vivo tasks. "Not applicable" is the usual answer for such work. If your institution requires an exemption
  determination for secondary use of human surgical video, select the exemption instead. Then add the institution, the
  board and the date to the ethics sentence in `sections/discussion.tex`.
- **Conflicts of interest and funding.** These are declared in the Acknowledgment of the camera-ready version, and in
  the portal if it asks.
- **Suggested reviewers (optional).** Only people without conflicts of interest with the authors.
- **Anonymised code for the reviewers (optional).** Create a mirror of the GitHub repository at
  [anonymous.4open.science](https://anonymous.4open.science), which removes the terms you list, such as your name and
  account names. Put its URL into `\anonrepo` in `review.tex` and rebuild; the availability statement then prints it.

The public GitHub repository and Hugging Face page carry the paper and the author's name. The RAS rules allow preprints
and ask reviewers not to search for them, and the manuscript itself links to neither. Whether to keep them public during
review is your choice.

## After acceptance

1. Set `\reviewfalse` in `review.tex`. This restores the author block, the public links and the Acknowledgment section.
2. Fill the author block (`\thanks` in `main.tex`) and the Acknowledgment (funding, conflicts of interest).
3. Prepare the graphical abstract, complete the IEEE electronic copyright form, and upload the final files.
