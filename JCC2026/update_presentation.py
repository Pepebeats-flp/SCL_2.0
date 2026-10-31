#!/usr/bin/env python3
"""Simplify the language of PEARL_JCC2026.pptx for a non-musician audience and
add numbered citations plus a final References slide.

Works on the CURRENT deck (including any manual PowerPoint edits); a backup is
written to /tmp/opencode/PEARL_JCC2026_before_update.pptx before saving.

Usage:  python3 update_presentation.py
"""

from __future__ import annotations

import copy
import os
import shutil

import build_presentation as bp
from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

DECK = os.path.join(bp.HERE, "PEARL_JCC2026.pptx")
BACKUP = "/tmp/opencode/PEARL_JCC2026_before_update.pptx"
RED = bp.RED
BODY = bp.BODY
GREY = bp.GREY_TITLE
DARK_RED = bp.DARK_RED


def set_cell(cell, text):
    """Replace a table cell's text keeping its current formatting."""
    tf = cell.text_frame
    runs = [r for p in tf.paragraphs for r in p.runs]
    if runs:
        runs[0].text = text
        for r in runs[1:]:
            r._r.getparent().remove(r._r)
    else:
        tf.text = text


def retitle(slide, title, subtitle, n, ids=(119, 126, 123)):
    tsh = bp.by_id(slide, ids[0])
    bp.simple(tsh, title, bp.fit_size(title, 25.2, bp.Emu(tsh.width).inches),
              GREY)
    ssh = bp.by_id(slide, ids[1])
    ssh.width = Inches(8.80)
    bp.simple(ssh, subtitle, bp.fit_size(subtitle, 18.2, 8.80, min_size=13.0),
              GREY)
    bp.page_number(slide, n, ids[2])


def clone_slide(prs, source):
    """Append a copy of `source` (shapes + image relationships) to the deck."""
    new = prs.slides.add_slide(source.slide_layout)
    for sh in list(new.shapes):
        sh._element.getparent().remove(sh._element)
    id_map = {}
    for rel in source.part.rels.values():
        if rel.reltype.endswith("/slideLayout") or rel.reltype.endswith("/notesSlide"):
            continue
        if rel.is_external:
            new_rid = new.part.rels.get_or_add_ext_rel(rel.reltype, rel.target_ref)
        else:
            new_rid = new.part.relate_to(rel.target_part, rel.reltype)
        id_map[rel.rId] = new_rid
    for sh in source.shapes:
        el = copy.deepcopy(sh._element)
        new.shapes._spTree.append(el)
        for node in el.iter():
            for attr, val in list(node.attrib.items()):
                if attr.endswith("}embed") or attr.endswith("}link") or attr.endswith("}id"):
                    if val in id_map:
                        node.attrib[attr] = id_map[val]
    return new


def move_slide(prs, slide, index):
    """Move `slide` to `index` (0-based) in the presentation's slide order."""
    sld_ids = prs.slides._sldIdLst
    el = next(e for e in sld_ids if e.get("id") == str(slide.slide_id))
    sld_ids.remove(el)
    sld_ids.insert(index, el)


def renumber(prs):
    """Rewrite every page-number box according to the current slide order."""
    for i, slide in enumerate(prs.slides, 1):
        for sh in slide.shapes:
            if not sh.has_text_frame:
                continue
            if (sh.text_frame.text.strip().isdigit() and sh.top is not None
                    and sh.top > Inches(5.0) and sh.width < Inches(1.5)):
                bp.page_number(slide, i, sh.shape_id)


def main() -> None:
    shutil.copyfile(DECK, BACKUP)
    prs = Presentation(DECK)
    s = list(prs.slides)

    # ---- 1. Title: add a technical subtitle --------------------------------
    sl = s[0]
    bp.add_textbox(sl, 0.88, 2.62, 7.96, 0.35, [
        dict(text="A model that changes a chord progression to a chosen level "
                  "of complexity.", size=12, color=BODY, italic=True,
             align=PP_ALIGN.CENTER)])
    bp.notes(sl, "0:20 — This talk presents PEARL, an RVAE-LSTM that enriches "
                 "a chord progression to a user-selected level of perceptual "
                 "complexity. The level is measured by PCS, a score obtained "
                 "from a listening study. No musical background is assumed: "
                 "each term is defined when it first appears. Joint work with "
                 "Nicolás Rojas-Morales, UTFSM.")

    # ---- 2. Outline --------------------------------------------------------
    sl = s[1]
    items = [
        (80, 92, "Problem", "What is missing today"),
        (81, 93, "Complexity Score", "A score from listeners [3, 4]"),
        (82, 94, "Model", "How PEARL works"),
        (83, 95, "Setup and Metrics", "Data, baselines, measures"),
        (84, 96, "Results", "Comparison with earlier systems"),
        (85, 97, "Conclusions", "Limits and next steps"),
    ]
    for tid, did, t, d in items:
        tsh, dsh = bp.by_id(sl, tid), bp.by_id(sl, did)
        tsh.width = dsh.width = Inches(4.00)
        bp.simple(tsh, t, bp.fit_size(t, 17.0, 4.00, min_size=13.0, bold=True),
                  GREY, bold=True)
        bp.simple(dsh, d, bp.fit_size(d, 11.5, 4.00, min_size=9.5), GREY)
    for sid in (83, 84, 85, 95, 96, 97):  # keep a gap after the red square
        sh = bp.by_id(sl, sid)
        sh.left, sh.width = Inches(5.49), Inches(3.90)

    # ---- 3. Motivation -----------------------------------------------------
    sl = s[2]
    st = bp.by_id(sl, 112)
    st.top, st.height = Inches(3.00), Inches(1.00)
    bp.simple(st, "Most songs use a few simple chords. Reharmonization can "
                  "enrich them, but tools cannot control how much changes.",
              14.5, BODY)
    bp.by_id(sl, 110).top = Inches(3.95)
    bp.add_textbox(sl, 2.27, 4.02, 6.90, 0.40, [
        dict(text="•  Chord: three or more notes played together; "
                  "progression: a series of chords in order.", size=9.5,
             color=GREY),
        dict(text="•  Reharmonization: replacing the chords of a progression "
                  "with different ones.", size=9.5, color=GREY),
    ])
    pearl = bp.by_id(sl, 113)
    pearl.top = Inches(4.55)
    bp.notes(sl, "0:40 — Context: large-scale analyses of Western popular music, "
                 "such as the McGill Billboard corpus, show a narrow repertoire "
                 "of triads and repeating cadential formulas. Definitions used "
                 "in this talk follow standard tonal theory: a chord is three or "
                 "more notes sounded together; a progression is a series of "
                 "chords played in order; reharmonization replaces the chords "
                 "of a progression with different ones while keeping the song's "
                 "key, as in the enrichment line of work [2, 6]. Existing tools "
                 "can transform a progression, but they do not let the user "
                 "choose how much complexity to add or which property should "
                 "change.")

    # ---- 4. Gap & contributions -------------------------------------------
    sl = s[3]
    retitle(sl, "What Others Did and What Is Missing",
            "Earlier systems, the gap, and our proposal", 4)
    bp.write(bp.by_id(sl, 127).text_frame,
             [dict(text="Earlier systems", size=13, color=RED, bold=True,
                   space_after=6)] + bp.bullets([
                 ("Comanducci et al. [2]: ", "one complexity number from a "
                  "statistical model. 4 chord types; synthetic 5-chord loops in "
                  "C major/minor."),
                 ("Uemura et al. [6]: ", "moves chords inside a trained model; "
                  "no target level."),
                 ("MusicVAE [5]: ", "generates music; not for controlled "
                  "enrichment."),
             ], size=10.5, space_after=8))
    bp.write(bp.by_id(sl, 128).text_frame,
             [dict(text="The gap, and what we propose", size=13, color=RED,
                   bold=True, space_after=6)] + bp.bullets([
                 ("Missing: ", "a target, a choice of property, and a measure "
                  "based on listeners."),
                 ("PCS: ", "a complexity score from listening tests [3, 4]."),
                 ("PEARL: ", "reaches a chosen score on 877k real progressions "
                  "[1], 11 keys, 4–12 chords."),
             ], size=10.5, space_after=8))
    bp.notes(sl, "1:10 — Comanducci et al. [2] is the closest work: their "
                 "complexity is a single number produced by statistical models "
                 "that predict the next chord (prediction by partial matching, "
                 "HMM and RNN); their chord vocabulary has four types and their "
                 "data are 6,311 synthetic five-chord loops in C. Uemura et al. "
                 "[6] morph chords inside a VAE, without a target. MusicVAE [5] "
                 "organises a latent space for generation. What is missing is a "
                 "target expressed in perceptual terms, a choice of which "
                 "property changes, and validation on real, multi-key data. "
                 "That is what PCS and PEARL add.")

    # ---- 5. Encoding -------------------------------------------------------
    sl = s[4]
    retitle(sl, "How Chords Are Represented",
            "48 yes/no values per chord, readable as a chord name", 5)
    bp.write(bp.by_id(sl, 127).text_frame, bp.bullets([
        "48 yes/no values per chord.",
        ("Seven slots: ", "root, type, seventh, extensions, alterations, "
         "added notes, bass."),
        ("One row per chord; ", "the values decode back to a chord name."),
        ("877k real progressions [1]; ", "split 80/10/10 by song."),
    ], size=11, space_after=8))
    table = next(sh for sh in sl.shapes if sh.has_table).table
    meanings = ["Root pitch class (C–B)",
                "maj, min, dim, aug, sus, no3d",
                "none, dom7, maj7, dim7, aug7",
                "9, 11, 13 (multi-label)",
                "b9, #9, #11, b13",
                "add9, add11, add13",
                "Bass note (C–B) + inversion"]
    set_cell(table.cell(0, 2), "Meaning")
    for i, text in enumerate(meanings, start=1):
        set_cell(table.cell(i, 2), text)
    bp.notes(sl, "0:55 — The encoding is explicit, not learned: 48 binary "
                 "values in seven slots. The vector is reversible, so any "
                 "output can be read as a chord symbol. A progression of T "
                 "chords is a T by 48 matrix. Training data are 877k "
                 "progressions from Chordonomicon [1], and jSymbolic [3] "
                 "computes the features used for PCS. The split is by song, so "
                 "no song appears in two subsets.")

    # ---- 6. PCS -------------------------------------------------------------
    sl = s[5]
    retitle(sl, "A Complexity Score from Listeners (PCS)",
            "We asked listeners instead of writing a formula", 6)
    bp.write(bp.by_id(sl, 127).text_frame, bp.bullets([
        ("43 listeners; ", "1,032 pairwise comparisons on a 5-point scale."),
        ("11 properties measured [3]; ", "4 matched the listeners."),
        ("Score: ", "weighted average, 0 = simple, 1 = complex."),
        ("v1: ", "listener weights. v2: all four weighted equally."),
    ], size=11, space_after=8))
    table = next(sh for sh in sl.shapes if sh.has_table).table
    names = ["Seventh chords", "Notes per chord", "Interval spacing",
             "Dissonance"]
    meanings = ["How often they appear", "How much it varies",
                "Distance between them", "How often it appears"]
    set_cell(table.cell(0, 0), "Property")
    set_cell(table.cell(0, 1), "Meaning")
    for i, (name, text) in enumerate(zip(names, meanings), start=1):
        set_cell(table.cell(i, 0), name)
        set_cell(table.cell(i, 1), text)
    for j, width in enumerate([1.45, 1.65, 0.65]):
        table.columns[j].width = Inches(width)
    for row in table.rows:
        for para in row.cells[1].text_frame.paragraphs:
            for run in para.runs:
                run.font.size = Pt(9)
    bp.write(bp.by_id(sl, 129).text_frame, [
        dict(text="Seventh chords are the strongest signal of complexity: "
                  "twice the weight of dissonance.", size=10.5, color=BODY,
             italic=True)])
    bp.notes(sl, "1:40 — Harmonic complexity is defined as the degree to which "
                 "chord changes are varied, rich and unexpected. We do not use "
                 "a language-model proxy as [2] does; we run a paired-"
                 "comparison test: 53 participants, 2 control pairs, 43 valid, "
                 "1,032 pairwise judgements on a five-point scale, stratified "
                 "into three groups by the Goldsmiths index [4]. From 11 "
                 "jSymbolic features [3], four pass the two criteria (p < 0.05 "
                 "in at least one group; same sign across groups): 7C = share "
                 "of chords with a seventh; VNSPC = variability of simultaneous "
                 "pitch-class density; DTMCVI = distance between the two most "
                 "common vertical intervals; VDR = share of dissonant vertical "
                 "intervals. Weights are mean absolute correlations, normalised "
                 "by 1.360. 7C has twice the weight of VDR (0.443 vs 0.201) and "
                 "its correlation is r = 0.580, p = 0.003 at high training.")

    # ---- 7. Architecture ----------------------------------------------------
    sl = s[6]
    retitle(sl, "The PEARL Model",
            "A network that turns a progression into 32 numbers, then rebuilds "
            "it [2, 5]", 7)
    arch = next(sh for sh in sl.shapes if sh.name == "Picture 126")
    arch.crop_top = 4 / 464  # remove stray sliver from the source figure
    # keep the architecture only: drop the training-loss panel on the right
    arch.crop_right = 576 / 1871
    new_w = int(arch.width * (1 - 576 / 1871))
    arch.width = new_w
    arch.left = int((Inches(10) - new_w) / 2)
    bp.write(bp.by_id(sl, 128).text_frame, bp.bullets([
        ("Encoder: ", "a progression → 32 numbers (z)."),
        ("Decoder: ", "32 numbers → a progression."),
    ], size=10, space_after=4))
    bp.write(bp.by_id(sl, 129).text_frame, bp.bullets([
        ("Two predictors: ", "one reads complexity, one reads key."),
        ("We move z to change complexity; ", "the key stays readable."),
    ], size=10, space_after=4))
    bp.notes(sl, "1:20 — PEARL is a regression VAE with an LSTM encoder and "
                 "decoder [2, 5]. The encoder is a two-layer bidirectional LSTM "
                 "with 256 units and outputs the parameters of a 32-dimensional "
                 "Gaussian; z is a sample of it. The decoder is a two-layer "
                 "autoregressive LSTM that rebuilds the progression chord by "
                 "chord from z. Two auxiliary heads shape z: the complexity "
                 "predictor maps z to the four PCS values (MSE 0.0034), and the "
                 "key predictor maps z to 12 key classes (98.4% accuracy). "
                 "Training combines binary cross-entropy reconstruction, a KL "
                 "term and the two prediction losses.")

    # ---- 8. Inference -------------------------------------------------------
    sl = s[7]
    retitle(sl, "Reaching a Target Complexity",
            "The model is frozen; we only move the 32 numbers", 8)
    bp.write(bp.by_id(sl, 127).text_frame, bp.bullets([
        ("Goal: ", "move z until complexity = target."),
        ("About 20–30 small steps."),
        ("A penalty keeps z close to the original."),
        ("One property at a time: ", "sevenths, dissonance, or variety."),
    ], size=11, space_after=9))
    bp.write(bp.by_id(sl, 128).text_frame, [
        dict(text="Example: seventh chords, target 1.0", size=12,
             color=RED, bold=True, space_after=8),
        dict(text="Input", size=11, color=GREY, bold=True, space_after=2),
        dict(text="Cmin – Gmaj – Cmin – Fmin – Cmin", size=11, space_after=8),
        dict(text="Output", size=11, color=GREY, bold=True, space_after=2),
        dict(text="Cmaj7 – Dmin7 – Cmin7 – Gmin7 – Cmin7", size=11,
             space_after=8),
        dict(text="The same model serves any target in [0, 1]; no retraining.",
             size=10.5, italic=True),
    ])
    bp.notes(sl, "1:10 — Inference optimises z directly. We minimise the MSE "
                 "between the predicted PCS and the target, plus a "
                 "regularisation term that anchors z to the original code, with "
                 "λ_reg = 0.005. We use Adam with learning rate 0.5 for 90 "
                 "steps and gradient clipping at 1.0; in practice the target is "
                 "reached in 20 to 30 steps. The procedure is deterministic: no "
                 "sampling or reranking. Setting the loss to one dimension "
                 "targets it alone: 7C from 0.002 to 0.982 with five seventh "
                 "chords, VDR from 0.005 to 0.987 without seventh chords, "
                 "VNSPC from −0.002 to 0.983 with the other dimensions almost "
                 "unchanged.")

    # ---- 9. Setup ------------------------------------------------------------
    sl = s[8]
    retitle(sl, "Experimental Setup", "Data, training and baselines", 9)
    bp.write(bp.by_id(sl, 127).text_frame, bp.bullets([
        ("One model, ", "trained once on 877k real progressions [1]."),
        ("Split: ", "80/10/10 by song; results on unseen songs."),
        ("Same test for CVAE and RVAE [2]; ", "outputs from their authors."),
        ("Five targets: ", "0.00, 0.25, 0.50, 0.75, 1.00."),
    ], size=13, space_after=14))
    bp.notes(sl, "0:45 — Training uses 30 epochs on 877k progressions [1], "
                 "Adam with learning rate 5e-4 and batch 256; the full table is "
                 "in the paper. All 32 latent dimensions are active, with KL "
                 "around 6 to 7 nats, and the bypass delta of 2.77 BCE shows "
                 "that the decoder relies on z. The split is 80/10/10 by song, "
                 "and every reported number comes from the test set. Baselines "
                 "are CVAE and RVAE [2]; their weights are unavailable, so we "
                 "compare on the outputs their authors released. Everyone "
                 "performs the same task: enrich one loop at five PCS levels.")

    # ---- 9b. Evaluation metrics (new slide) ---------------------------------
    ms = clone_slide(prs, s[8])
    retitle(ms, "Evaluation Metrics",
            "The score comes from listeners, not statistics. These metrics "
            "check the target and the key", 10)
    body = bp.by_id(ms, 127)
    body._element.getparent().remove(body._element)
    rows = [
        ["Metric", "What it measures"],
        ["Target reached", "Complexity of the output vs the target."],
        ["Property changed", "Change in each of the four properties."],
        ["Stays in key", "Share of roots in the original key [7]."],
        ["Keeps roots", "Share of positions with the same root."],
        ["Keeps quality", "Share of chords with the same major/minor quality."],
        ["Distance moved", "How far z moved from the original."],
    ]
    gfx = bp.add_table(ms, 0.60, 2.02, 8.80, rows, [2.05, 6.75],
                       font_size=9.5, header_size=10)
    for row in gfx.table.rows:
        for para in row.cells[1].text_frame.paragraphs:
            para.alignment = PP_ALIGN.LEFT
    bp.notes(ms, "0:55 — Two groups of metrics. Target reach: PCS achieved vs "
                 "target; the change in each of the four dimensions; and the 7C "
                 "share, Δ7C divided by the sum of the other three changes. "
                 "Tonal preservation: scale adherence, the percentage of chord "
                 "roots in the input key's scale [7]; positional overlap, the "
                 "percentage of positions with the same root as the input; "
                 "quality preservation, the percentage of chords keeping the "
                 "input major/minor quality; key consistency from key_predictor; "
                 "and latent displacement, the Euclidean distance between the "
                 "optimised and original z. PCS itself is perceptual: it comes "
                 "from the listening study, not from a PPM/HMM/RNN language "
                 "model as in [2].")
    move_slide(prs, ms, 9)

    # ---- 10. Latent space -----------------------------------------------------
    sl = s[9]
    retitle(sl, "What the Model Learned",
            "3,000 progressions: complexity forms a gradient, keys form "
            "clusters", 11)
    note = bp.by_id(sl, 128)
    note.top, note.height = Inches(4.56), Inches(0.45)
    bp.write(note.text_frame, [
        dict(runs=[("Complexity and key are organised, ", True),
                   ("so we can move z in a controlled way.", False)],
             size=10, space_after=3),
        dict(text="32/32 dimensions used · complexity error 0.0034 · key "
                  "accuracy 98.4%", size=9.5, color=GREY),
    ])
    bp.notes(sl, "0:45 — This is a PCA of 3,000 encoded progressions. On the "
                 "left, colour is mean PCS: the gradient is smooth, so "
                 "complexity is organised in z. On the right, colour is the "
                 "detected key: progressions in the same key cluster together. "
                 "All 32 dimensions are active, c_predictor MSE is 0.0034, key "
                 "accuracy is 98.4%, and the first two components capture 42.6% "
                 "of the variance. This organisation is what makes gradient "
                 "ascent reliable.")

    # ---- 11. Complexity levels -------------------------------------------------
    sl = s[10]
    retitle(sl, "Results: Five Complexity Levels",
            "How many seventh chords each model uses at each level", 12)
    bp.write(bp.by_id(sl, 128).text_frame, bp.bullets([
        ("Both versions ", "reach the target in 20–30 steps."),
        ("v1: ", "about 1/5 of the change is seventh chords; v2: 1/10."),
        ("v2 ", "spreads the change over the four properties."),
    ], size=10, space_after=8))
    bp.write(bp.by_id(sl, 129).text_frame, [
        dict(runs=[("Baselines use a different scale; ", True),
                   ("seventh-chord counts still show how much each model "
                    "enriches.", False)], size=10)])
    bp.notes(sl, "1:05 — The chart counts seventh chords per model at the five "
                 "levels. Both PEARL variants reach the requested PCS in 20 to "
                 "30 steps. In v1 the perceptual weight of 7C is 0.443, and at "
                 "target 1.0, 20% of the total dimension change goes through 7C; "
                 "in v2 the share is 12%. v2 spreads the change over the four "
                 "dimensions and produces more suspended, diminished and no3d "
                 "chords. Latent displacement grows from about 0.1 at level 1 to "
                 "about 7 at level 5. The baselines use another complexity "
                 "scale, but both increase with level, so the seventh-chord "
                 "count is a coarse common indicator.")

    # ---- 12. Generalization -----------------------------------------------------
    sl = s[11]
    retitle(sl, "Results: Other Keys and Lengths",
            "11 keys (7 major, 4 minor) and 4–12 chords, target PCS = 0.50", 13)
    bp.write(bp.by_id(sl, 128).text_frame, bp.bullets([
        ("All 11 keys ", "reach the target (0.45–0.49)."),
        ("80% of roots ", "stay in key; 100% at every length."),
        ("Key kept in 6 of 11; ", "the others move to a related key."),
    ], size=9.5, space_after=1))
    bp.notes(sl, "0:50 — Generalisation: 11 keys, seven major and four minor, "
                 "and lengths from 4 to 12 chords, all enriched to PCS 0.50. "
                 "The target is reached in every key, between 0.453 and 0.485. "
                 "Scale adherence averages 80% and is 100% at every length. The "
                 "key_predictor keeps the input key in 6 of 11 cases, 55%; the "
                 "other five move to the relative or parallel key, so the roots "
                 "stay largely in scale but the tonal centre is not always "
                 "anchored.")

    # ---- 13. Tonal coherence -----------------------------------------------------
    sl = s[12]
    retitle(sl, "Results: Staying in Key",
            "Five levels, input in C minor", 14)
    bp.write(bp.by_id(sl, 127).text_frame, bp.bullets([
        ("88% of roots stay in key ", "(CVAE 84%, RVAE 76%) [2]."),
        ("32–36% keep the same root; ", "CVAE 84%."),
        ("32–44% keep major/minor quality; ", "CVAE 72%."),
        ("Low overlap is expected: ", "the penalty favours the target."),
    ], size=10, space_after=8))
    table = next(sh for sh in sl.shapes if sh.has_table).table
    for j, text in enumerate(["Model", "In key", "Same roots", "Same quality"]):
        cell = table.cell(0, j)
        set_cell(cell, text)
        for para in cell.text_frame.paragraphs:
            for run in para.runs:
                run.font.size = Pt(9)
    for j, width in enumerate([1.10, 0.85, 1.15, 1.35]):
        table.columns[j].width = Inches(width)
    bp.write(bp.by_id(sl, 129).text_frame, [
        dict(text="Means of five levels, input in C minor. 'Same roots' = same "
                  "root at the same position.", size=10, color=GREY,
             italic=True)])
    bp.notes(sl, "1:00 — Tonal coherence combines three measures. Scale "
                 "adherence is 88% for both variants, against 84% for CVAE and "
                 "76% for RVAE [2]. Positional overlap is 32–36%, against "
                 "64–84% for the baselines: most chord identities change, which "
                 "is expected when complexity rises. Quality preservation is 32% "
                 "for v1 and 44% for v2; CVAE keeps 72%, partly because it "
                 "copies the input at level 1. The key_predictor stays stable "
                 "across the five levels. The low overlap is a consequence of "
                 "λ_reg = 0.005, which favours reaching the target; tonal "
                 "structure is maintained by the key-organised latent space.")

    # ---- 14. Conclusions ---------------------------------------------------------
    sl = s[13]
    sub = bp.by_id(sl, 18)
    sub.width = Inches(8.80)
    bp.simple(sub, "Summary, limitations and future work", 18.2, GREY)
    bp.write(bp.by_id(sl, 19).text_frame, bp.bullets([
        ("Score: ", "from listeners, not statistics."),
        ("Control: ", "any target in under 30 steps, globally or per property."),
        ("Key: ", "88% of roots stay in key, across 11 keys and 4–12 chords."),
        ("Limits: ", "harmony only; 43 listeners; key kept in 6 of 11."),
        ("Next: ", "user-controlled penalty, rhythm, larger studies."),
    ], size=11.5, space_after=10))
    bp.notes(sl, "0:50 — Summary: PCS is a perceptual score with four "
                 "interpretable dimensions, obtained from 1,032 pairwise "
                 "judgements by 43 listeners. Gradient ascent on z reaches any "
                 "PCS target in under 30 steps, globally or per dimension, and "
                 "keeps 88% of roots in the input scale, against 84% for CVAE "
                 "and 76% for RVAE [2]. Limitations: harmony only, 43 "
                 "listeners, baselines compared on released outputs, and the "
                 "key is kept in 6 of 11 generalisation tests. Next: λ_reg as a "
                 "user parameter, stronger key anchoring, rhythm and structure "
                 "conditioning, and larger listening studies.")

    # ---- 15. Acknowledgments ------------------------------------------------------
    bp.notes(s[14], "0:20 — Thank the audience and invite questions, musical or "
                    "technical. The next slide lists the references, which can "
                    "be left on screen during Q&A.")

    # ---- 16. References -----------------------------------------------------------
    sl = clone_slide(prs, s[13])
    bp.simple(bp.by_id(sl, 16), "References", 25.2, GREY)
    sub = bp.by_id(sl, 18)
    sub.width = Inches(8.80)
    bp.simple(sub, "Works cited in this talk — full details in the paper", 18.2,
              GREY)
    refs = [
        "[1] S. Kantarelis et al., “Chordonomicon: A Dataset of 666,000 Songs "
        "and Their Chord Progressions,” arXiv:2410.22046, 2024.",
        "[2] L. Comanducci, D. Gioiosa, M. Zanoni, F. Antonacci and A. Sarti, "
        "“Variational Autoencoders for Chord Sequence Generation Conditioned on "
        "Western Harmonic Music Complexity,” EURASIP J. Audio Speech Music "
        "Process., 2023.",
        "[3] C. McKay, J. Cumming and I. Fujinaga, “jSymbolic 2.2: Extracting "
        "Features from Symbolic Music for Use in Musicological and MIR "
        "Research,” in Proc. ISMIR, 2018.",
        "[4] D. Müllensiefen, B. Gingras, J. Musil and L. Stewart, “The "
        "Musicality of Non-Musicians: An Index for Assessing Musical "
        "Sophistication in the General Population,” PLoS ONE, 2014.",
        "[5] A. Roberts, J. Engel and D. Eck, “Hierarchical Variational "
        "Autoencoders for Music,” in NeurIPS Workshop on Machine Learning for "
        "Creativity and Design, 2017.",
        "[6] A. Uemura and T. Kitahara, “Morphing-Based Reharmonization with "
        "VAE: Reducing Dissonance with Consonance-Based Loss Function,” in "
        "Proc. AIMC, 2022.",
        "[7] D. Temperley, The Cognition of Basic Musical Structures. "
        "MIT Press, 2004.",
    ]
    body = bp.by_id(sl, 19)
    body.top, body.height = Inches(1.95), Inches(3.00)
    repo = bp.by_id(sl, 20)  # repo line already appears on slides 14 and 15
    repo._element.getparent().remove(repo._element)
    specs = []
    for ref in refs:
        num, rest = ref.split("] ", 1)
        specs.append(dict(runs=[(num + "]  ", True), (rest, False)],
                          size=9.5, space_after=5))
    bp.write(body.text_frame, specs)
    bp.notes(sl, "0:00 — Reference slide. Not presented; useful during Q&A. "
                 "[1] dataset, [2] CVAE/RVAE baselines, [3] jSymbolic features, "
                 "[4] musical sophistication index, [5] VAEs for music, "
                 "[6] VAE reharmonization, [7] tonal theory.")

    renumber(prs)
    prs.save(DECK)
    print(f"updated {DECK}: {len(prs.slides.__iter__.__self__._sldIdLst)} slides, "
          f"{os.path.getsize(DECK)/1e6:.2f} MB")
    print(f"backup: {BACKUP}")


if __name__ == "__main__":
    main()
