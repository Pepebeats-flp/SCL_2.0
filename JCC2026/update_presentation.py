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
from pptx.dml.color import RGBColor
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


def delete_slide(prs, slide):
    """Remove a slide and its relationship from the presentation."""
    from pptx.oxml.ns import qn

    sld_ids = prs.slides._sldIdLst
    for el in list(sld_ids):
        if el.get("id") == str(slide.slide_id):
            rId = el.get(qn("r:id"))
            sld_ids.remove(el)
            prs.part.drop_rel(rId)
            return


def reorder(prs, slides):
    """Set the slide order to exactly `slides`."""
    sld_ids = prs.slides._sldIdLst
    order = [str(sl.slide_id) for sl in slides]
    by_id = {e.get("id"): e for e in list(sld_ids)}
    for el in list(sld_ids):
        sld_ids.remove(el)
    for sid in order:
        sld_ids.append(by_id[sid])


def link_repo(slide, url="https://github.com/Pepebeats-flp/pearl-sc"):
    """Make the repository URL clickable."""
    for sh in slide.shapes:
        if not sh.has_text_frame or "github.com" not in sh.text_frame.text:
            continue
        for para in sh.text_frame.paragraphs:
            for run in para.runs:
                if "github.com" in run.text:
                    run.hyperlink.address = url


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
    # clone the clean content layout before any slide is modified
    ms = clone_slide(prs, s[8])
    pp = clone_slide(prs, s[8])
    ts = clone_slide(prs, s[8])

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

    # ---- 2. Motivation (content layout with staff example) ------------------
    mot = clone_slide(prs, s[8])
    retitle(mot, "Motivation",
            "Richer chords simply have more notes: compare these two versions",
            2)
    mb = bp.by_id(mot, 127)
    mb.width = Inches(4.05)
    bp.write(mb.text_frame, bp.bullets([
        ("Chord: ", "3+ notes played together (C–E–G)."),
        ("Progression: ", "a series of chords in order."),
        ("Richer chord: ", "more notes (C–E–G–B)."),
        ("Reharmonization: ", "changing the chords."),
        ("The gap: ", "no tool lets you choose how much."),
    ], size=11, space_after=8))
    bp.add_picture_fit(mot, "slides_assets/enrichment_example.png",
                       4.35, 2.05, 5.05, 1.60)
    bp.add_textbox(mot, 4.35, 3.78, 5.05, 0.55, [
        dict(text="Hand-written example. Simple: 3 notes per chord. Enriched: "
                  "4–6 notes per chord.", size=9.5, color=GREY, italic=True)])
    bp.notes(mot, "0:40 — Context: large-scale analyses of Western popular music, "
                 "such as the McGill Billboard corpus, show a narrow repertoire "
                 "of triads and repeating cadential formulas. Definitions follow "
                 "standard tonal theory: a chord is three or more notes sounded "
                 "together; a progression is a series of chords in order; "
                 "reharmonization replaces the chords of a progression with "
                 "different ones. The staff shows the same progression twice: on "
                 "top, plain triads with three notes; below, richer chords with "
                 "four, five and six notes. Musicians do this by hand, but no "
                 "tool lets the user choose how much richer the result should "
                 "be.")

    # ---- 4. Gap & contributions -------------------------------------------
    sl = s[3]
    retitle(sl, "What Others Did and What Is Missing",
            "Can a model target what listeners perceive, one property at a "
            "time?", 4)
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
             [dict(text="The gap", size=13, color=RED, bold=True,
                   space_after=0)])
    rows = [
        ["", "Target", "Property", "Listener score"],
        ["Comanducci [2]", "5 levels", "No", "No"],
        ["Uemura [6]", "No", "No", "No"],
        ["MusicVAE [5]", "No", "No", "No"],
        ["PEARL", "Any", "Yes", "Yes"],
    ]
    gfx = bp.add_table(sl, 5.05, 2.42, 4.35, rows, [1.20, 0.78, 0.92, 1.45],
                       font_size=9, header_size=8.5, highlight=(4,))
    for r in range(1, 5):
        for c in range(1, 4):
            cell = gfx.table.cell(r, c)
            for para in cell.text_frame.paragraphs:
                for run in para.runs:
                    if run.text.strip() == "Yes":
                        run.font.color.rgb = RGBColor(0xE3, 0x05, 0x20)
                        run.font.bold = True
                    elif run.text.strip() == "No":
                        run.font.color.rgb = RGBColor(0x9A, 0x9A, 0x9A)
    bp.add_textbox(sl, 5.05, 4.12, 4.35, 0.55, [
        dict(text="No earlier system offers all three.", size=10.5,
             color=DARK_RED, bold=True)])
    bp.notes(sl, "1:10 — Comanducci et al. [2] is the closest work: their "
                 "complexity is a single number produced by statistical models "
                 "that predict the next chord (prediction by partial matching, "
                 "HMM and RNN); their chord vocabulary has four types and their "
                 "data are 6,311 synthetic five-chord loops in C. The table "
                 "compares the three capabilities that matter here: a target "
                 "level (Comanducci has five classes), the choice of which "
                 "property changes, and a score built from listening tests. "
                 "Uemura et al. [6] morph chords inside a VAE without a target; "
                 "MusicVAE [5] organises a latent space for generation. No "
                 "earlier system offers all three, which is what PCS and PEARL "
                 "add.")

    # ---- 6. PCS -------------------------------------------------------------
    sl = s[5]
    retitle(sl, "A Complexity Score from Listeners (PCS)",
            "A human filter and a statistical filter select four properties",
            6)
    body = bp.by_id(sl, 127)
    body._element.getparent().remove(body._element)
    bp.add_picture_fit(sl, "slides_assets/pcs_funnel.png",
                       0.55, 1.92, 4.55, 2.67)
    bp.add_textbox(sl, 0.55, 4.68, 4.55, 0.35, [
        dict(text="53 joined; 43 valid; 1,032 pairwise comparisons.",
             size=9, color=GREY, italic=True)])
    gfx_shape = next(sh for sh in sl.shapes if sh.has_table)
    gfx_shape.left, gfx_shape.top = Inches(5.15), Inches(1.95)
    note = bp.by_id(sl, 129)
    note.left, note.top, note.width, note.height = (
        Inches(5.15), Inches(4.00), Inches(4.25), Inches(0.55))
    table = gfx_shape.table
    names = ["Seventh chords", "Notes per chord", "Interval spacing",
             "Dissonance"]
    meanings = ["Share of chords with a seventh",
                "Variation in simultaneous note count",
                "Distance between the two common intervals",
                "Share of dissonant vertical intervals"]
    set_cell(table.cell(0, 0), "Property")
    set_cell(table.cell(0, 1), "Meaning")
    for i, (name, text) in enumerate(zip(names, meanings), start=1):
        set_cell(table.cell(i, 0), name)
        set_cell(table.cell(i, 1), text)
    for j, width in enumerate([1.55, 1.95, 0.70]):
        table.columns[j].width = Inches(width)
    for row in table.rows:
        for para in row.cells[1].text_frame.paragraphs:
            for run in para.runs:
                run.font.size = Pt(8.5)
    bp.write(bp.by_id(sl, 129).text_frame, [
        dict(text="v1 uses these weights; v2 weights all four equally. "
                  "Seventh chords weigh most.", size=10, color=BODY,
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

    # ---- 6b. Chord representation ------------------------------------------
    retitle(ts, "How a Chord Becomes 48 Numbers",
            "Seven slots, each with a musical meaning", 6)
    tb = bp.by_id(ts, 127)
    tb._element.getparent().remove(tb._element)
    bp.add_picture_fit(ts, "slides_assets/chord_tensor.png",
                       0.60, 1.95, 8.60, 3.00)
    bp.notes(ts, "0:45 — Each chord is a fixed vector of 48 yes/no values in "
                 "seven slots: root (12), quality (8), seventh (5), extensions "
                 "(3), alterations (4), added notes (3) and bass (13). The "
                 "example is Cmaj7: four values are active — root C, quality "
                 "major, seventh maj7 and bass root — and the other 44 are "
                 "zero. Because the slots are explicit, any vector decodes back "
                 "to a chord symbol, unlike piano-roll or multi-hot encodings. "
                 "Training uses 877k progressions from Chordonomicon [1].")

    # ---- 7. Architecture ----------------------------------------------------
    sl = s[6]
    retitle(sl, "The PEARL Model",
            "A network that turns a progression into 32 numbers, then rebuilds "
            "it [2, 5]", 7)
    arch = next(sh for sh in sl.shapes if sh.name == "Picture 126")
    arch._element.getparent().remove(arch._element)
    bp.add_picture_fit(sl, "slides_assets/pearl_pipeline.png",
                       0.60, 1.92, 8.80, 2.05)
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
            "The space is ordered by complexity; we move z toward the target",
            8)
    body = bp.by_id(sl, 127)
    body.width = Inches(4.05)
    bp.write(body.text_frame, bp.bullets([
        ("Goal: ", "move z until complexity = target."),
        ("About 20–30 small steps."),
        ("A penalty keeps z close to the original."),
        ("One property at a time.", ""),
    ], size=10.5, space_after=7))
    ex = bp.by_id(sl, 128)
    ex.left, ex.top, ex.width, ex.height = (
        Inches(0.60), Inches(3.30), Inches(4.05), Inches(1.55))
    bp.add_picture_fit(sl, "slides_assets/latent_control.png",
                       5.20, 1.90, 4.10, 3.00)
    bp.write(ex.text_frame, [
        dict(text="Example: seventh chords, target 1.0", size=12,
             color=RED, bold=True, space_after=8),
        dict(text="Input", size=11, color=GREY, bold=True, space_after=2),
        dict(text="Cmin – Gmaj – Cmin – Fmin – Cmin", size=11, space_after=8),
        dict(text="Output", size=11, color=GREY, bold=True, space_after=2),
        dict(text="Cmaj7 – Dmin7 – Cmin7 – Gmin7 – Cmin7", size=11,
             space_after=8),
        dict(text="Any target in [0, 1]; no retraining.", size=10.5,
             italic=True),
    ])
    bp.notes(sl, "1:10 — The plot is the real PCA of 3,000 encoded "
                 "progressions; colour is mean PCS, so complexity increases "
                 "from left to right. We start from the input point and move z "
                 "toward the target region; the decoder turns the selected "
                 "point into a new progression. Formally, inference optimises z "
                 "directly. We minimise the MSE "
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
    body = bp.by_id(sl, 127)
    body._element.getparent().remove(body._element)
    cards = [
        ("DATA", "877k", "progressions", "real songs [1]", 0.60),
        ("SPLIT", "80/10/10", "by song", "results on unseen songs", 2.85),
        ("TASK", "5", "targets", "from 0.00 to 1.00", 5.10),
        ("COMPARE", "2", "baselines", "CVAE · RVAE [2]", 7.35),
    ]
    for tag, num, caption, detail, x in cards:
        bp.add_textbox(sl, x, 2.00, 2.05, 0.30,
                       [dict(text=tag, size=9, color=RED, bold=True)])
        size = 30 if len(num) <= 4 else 24
        bp.add_textbox(sl, x, 2.22, 2.05, 0.62,
                       [dict(text=num, size=size, color=DARK_RED, bold=True)])
        bp.add_textbox(sl, x, 2.92, 2.05, 0.30,
                       [dict(text=caption, size=11, color=BODY, bold=True)])
        bp.add_textbox(sl, x, 3.20, 2.05, 0.55,
                       [dict(text=detail, size=9.5, color=GREY)])
    bp.add_textbox(sl, 0.60, 3.85, 8.80, 0.40, [
        dict(text="Every model gets the same input loop and the same five "
                  "targets.", size=11, color=BODY, italic=True)])
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
    # ---- 10. Per-property control ------------------------------------------
    retitle(pp, "Result: Choose What Changes",
            "Earlier systems return one number; PEARL can target one property",
            11)
    body = bp.by_id(pp, 127)
    bp.write(body.text_frame, bp.bullets([
        ("Seventh chords only: ", "score 0.002 → 0.982; all five chords get a "
         "seventh."),
        ("Dissonance only: ", "0.005 → 0.987; no seventh chords added."),
        ("Note variety only: ", "−0.002 → 0.983; other properties unchanged."),
    ], size=13, space_after=14))
    bp.add_picture_fit(pp, "slides_assets/property_outputs.png",
                       0.60, 3.35, 8.80, 1.62)
    bp.notes(pp, "0:50 — This is the capability earlier systems do not have "
                 "because they expose a single complexity number. Here we set "
                 "the loss to one property. Targeting seventh chords takes the "
                 "score from 0.002 to 0.982 and all five chords get a seventh. "
                 "Targeting dissonance takes it from 0.005 to 0.987 without "
                 "adding seventh chords. Targeting note variety takes it from "
                 "−0.002 to 0.983 with the other properties almost unchanged.")

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
        ("88% of roots stay in key ", "vs CVAE 84% and RVAE 76% [2]."),
        ("Overlap 32–36% ", "vs 84% for CVAE: PEARL changes more chords."),
        ("Low overlap is expected: ", "the anchor favours the target."),
    ], size=10.5, space_after=10))
    tbl_shape = next(sh for sh in sl.shapes if sh.has_table)
    tbl_shape._element.getparent().remove(tbl_shape._element)
    bp.add_picture_fit(sl, "slides_assets/overlap_vs_key.png",
                       4.95, 1.95, 4.35, 2.90)
    note = bp.by_id(sl, 129)
    note.left, note.top, note.width, note.height = (
        Inches(0.60), Inches(3.95), Inches(4.05), Inches(0.80))
    bp.write(note.text_frame, [
        dict(text="Mean of five levels, input in C minor. Overlap = same root "
                  "at the same position.", size=9.5, color=GREY, italic=True)])
    bp.notes(sl, "1:00 — The plot places each model by scale adherence "
                 "(horizontal) and overlap with the input (vertical). CVAE sits "
                 "top right: it copies the input, with 84% overlap and 84% "
                 "adherence. RVAE has 64% overlap and 76% adherence. PEARL sits "
                 "bottom right: it rewrites most chords (32–36% overlap) but "
                 "keeps 88% of roots in key, the highest adherence. Quality "
                 "preservation is 32% for v1 and 44% for v2; CVAE keeps 72%, "
                 "partly because it copies the input at level 1. The low overlap "
                 "is a consequence of the anchor weight, which favours reaching "
                 "the target; tonal structure is maintained by the key-"
                 "organised latent space.")

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
    link_repo(sl)
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
    link_repo(s[14])
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
    refs_slide = sl

    # ---- Final structure: drop outline, encoding and PCA; set the order -----
    for dead in (s[1], s[2], s[4], s[9]):
        delete_slide(prs, dead)
    reorder(prs, [s[0], mot, s[3], s[5], ts, s[6], s[7], s[8], ms, s[10],
                  pp, s[11], s[12], s[13], s[14], refs_slide])
    renumber(prs)
    prs.save(DECK)
    print(f"updated {DECK}: {len(prs.slides.__iter__.__self__._sldIdLst)} slides, "
          f"{os.path.getsize(DECK)/1e6:.2f} MB")
    print(f"backup: {BACKUP}")


if __name__ == "__main__":
    main()
