#!/usr/bin/env python3
"""Build PEARL_JCC2026.pptx from Plantilla2026.pptx (JCC 2026 template).

Usage:
    python3 build_presentation.py

Requires: python-pptx, Pillow, PyMuPDF (only if --render-figures is used).
The template is never modified; the deck is assembled by cloning the template
slides at the OOXML level (preserving every decorative shape) and then replacing
the placeholder text with the paper content.
"""

from __future__ import annotations

import copy
import os
import re
import shutil
import zipfile

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "Plantilla2026.pptx")
OUTPUT = os.path.join(HERE, "PEARL_JCC2026.pptx")
ASSETS = os.path.join(HERE, "slides_assets")
TMP_BASE = "/tmp/opencode/pearl_base15.pptx"

# Source template slide used for each of the 15 output slides.
SLIDE_SOURCES = [1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6]

# Template palette
RED = RGBColor(0xE3, 0x05, 0x20)
DARK_RED = RGBColor(0xB0, 0x04, 0x18)
GREY_TITLE = RGBColor(0x80, 0x80, 0x80)
BODY = RGBColor(0x59, 0x59, 0x59)
ORANGE = RGBColor(0xFF, 0x99, 0x00)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_ROW = RGBColor(0xF4, 0xF4, 0xF4)
PEARL_ROW = RGBColor(0xFD, 0xEC, 0xEF)
FONT = "Montserrat"


# --------------------------------------------------------------------------
# 1. Clone template slides at the XML level
# --------------------------------------------------------------------------
def build_base_deck() -> None:
    zin = zipfile.ZipFile(TEMPLATE)
    parts = {n: zin.read(n) for n in zin.namelist()}
    zin.close()

    out: dict[str, bytes] = {}
    for name, data in parts.items():
        # slides / notes are rebuilt below
        if re.match(r"ppt/slides/slide\d+\.xml$", name):
            continue
        if re.match(r"ppt/slides/_rels/slide\d+\.xml\.rels$", name):
            continue
        if name.startswith("ppt/notesSlides/"):
            continue
        if name in ("ppt/presentation.xml", "ppt/_rels/presentation.xml.rels",
                    "[Content_Types].xml"):
            continue
        out[name] = data

    for new_idx, src in enumerate(SLIDE_SOURCES, start=1):
        out[f"ppt/slides/slide{new_idx}.xml"] = parts[f"ppt/slides/slide{src}.xml"]
        rels_name = f"ppt/slides/_rels/slide{src}.xml.rels"
        rels = parts.get(rels_name, b"").decode("utf-8")
        # drop notesSlide relationships: fresh notes are created later
        rels = re.sub(r"<Relationship[^>]*notesSlide[^>]*/>", "", rels)
        out[f"ppt/slides/_rels/slide{new_idx}.xml.rels"] = rels.encode("utf-8")

    # presentation.xml.rels : keep everything but slide relationships
    rels = parts["ppt/_rels/presentation.xml.rels"].decode("utf-8")
    rels = re.sub(r"<Relationship[^>]*relationships/slide\"[^>]*/>", "", rels)
    slide_rels = "".join(
        f'<Relationship Id="rIdSlide{i}" '
        f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" '
        f'Target="slides/slide{i}.xml"/>'
        for i in range(1, len(SLIDE_SOURCES) + 1)
    )
    rels = rels.replace("</Relationships>", slide_rels + "</Relationships>")
    out["ppt/_rels/presentation.xml.rels"] = rels.encode("utf-8")

    # presentation.xml : rebuild the slide id list
    pres = parts["ppt/presentation.xml"].decode("utf-8")
    sld_ids = "".join(
        f'<p:sldId id="{255 + i}" r:id="rIdSlide{i}"/>'
        for i in range(1, len(SLIDE_SOURCES) + 1)
    )
    pres = re.sub(r"<p:sldIdLst>.*?</p:sldIdLst>",
                  f"<p:sldIdLst>{sld_ids}</p:sldIdLst>", pres, flags=re.S)
    out["ppt/presentation.xml"] = pres.encode("utf-8")

    # content types
    ct = parts["[Content_Types].xml"].decode("utf-8")
    ct = re.sub(r'<Override PartName="/ppt/slides/slide\d+\.xml"[^>]*/>', "", ct)
    ct = re.sub(r'<Override PartName="/ppt/notesSlides/[^"]*"[^>]*/>', "", ct)
    overrides = "".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" '
        f'ContentType="application/vnd.openxmlformats-officedocument.'
        f'presentationml.slide+xml"/>'
        for i in range(1, len(SLIDE_SOURCES) + 1)
    )
    ct = ct.replace("</Types>", overrides + "</Types>")
    out["[Content_Types].xml"] = ct.encode("utf-8")

    os.makedirs(os.path.dirname(TMP_BASE), exist_ok=True)
    with zipfile.ZipFile(TMP_BASE, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in out.items():
            zout.writestr(name, data)


# --------------------------------------------------------------------------
# 2. Text helpers
# --------------------------------------------------------------------------
def by_id(slide, shape_id):
    for sh in slide.shapes:
        if sh.shape_id == shape_id:
            return sh
    raise KeyError(f"shape {shape_id} not found")


def base_rpr(tf):
    for p in tf.paragraphs:
        for r in p.runs:
            rpr = r._r.find(qn("a:rPr"))
            if rpr is not None:
                return copy.deepcopy(rpr)
    return None


def clear_tf(tf):
    tx_body = tf._txBody
    paragraphs = tx_body.findall(qn("a:p"))
    for p in paragraphs[1:]:
        tx_body.remove(p)
    p0 = paragraphs[0]
    for child in list(p0):
        if child.tag != qn("a:pPr"):
            p0.remove(child)
    tf.word_wrap = True


def write(tf, specs, default_size=12.0, default_color=BODY, anchor_top=True):
    """specs: list of dicts with text / runs, size, bold, color, italic, align,
    space_after, line_spacing."""
    rpr = base_rpr(tf)
    clear_tf(tf)
    if anchor_top:
        tf.vertical_anchor = MSO_ANCHOR.TOP
    for i, spec in enumerate(specs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = spec.get("align", PP_ALIGN.LEFT)
        p.space_after = Pt(spec.get("space_after", 4))
        p.space_before = Pt(spec.get("space_before", 0))
        p.line_spacing = spec.get("line_spacing", 1.0)
        runs = spec.get("runs") or [(spec.get("text", ""), spec.get("bold", False))]
        for text, bold in runs:
            run = p.add_run()
            run.text = text
            if rpr is not None:
                run._r.insert(0, copy.deepcopy(rpr))
            f = run.font
            f.name = FONT
            f.size = Pt(spec.get("size", default_size))
            f.bold = bold
            f.italic = spec.get("italic", False)
            f.color.rgb = spec.get("color", default_color)
    return tf


def simple(shape, text, size, color=BODY, bold=False, align=PP_ALIGN.LEFT,
           italic=False):
    write(shape.text_frame,
          [dict(text=text, size=size, color=color, bold=bold, align=align,
                italic=italic)],
          anchor_top=False)


def bullets(items, size=12.0, space_after=6):
    """items: str or (lead, rest) tuples; strings starting with '>' become
    sub-bullets."""
    specs = []
    for it in items:
        if isinstance(it, tuple):
            lead, rest = it
            specs.append(dict(runs=[("•  " + lead, True), (rest, False)],
                              size=size, space_after=space_after))
        elif it.startswith(">"):
            specs.append(dict(text="      –  " + it[1:].strip(), size=size - 0.5,
                              space_after=space_after - 1))
        else:
            specs.append(dict(text="•  " + it, size=size, space_after=space_after))
    return specs


def add_textbox(slide, x, y, w, h, specs, default_size=12.0):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    write(box.text_frame, specs, default_size=default_size)
    return box


def add_picture_fit(slide, path, x, y, w, h):
    """Insert picture centred in the (x, y, w, h) box, preserving aspect ratio."""
    from PIL import Image
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    return slide.shapes.add_picture(path, Inches(x + (w - pw) / 2),
                                    Inches(y + (h - ph) / 2),
                                    width=Inches(pw), height=Inches(ph))


def add_table(slide, x, y, w, rows, col_widths, font_size=10.5,
              highlight=(), header_size=None):
    n_rows, n_cols = len(rows), len(rows[0])
    gfx = slide.shapes.add_table(n_rows, n_cols, Inches(x), Inches(y),
                                 Inches(w), Inches(0.3 * n_rows))
    table = gfx.table
    # kill default banding style so our fills rule
    tbl = table._tbl
    tbl.set("firstRow", "0")
    tbl.set("bandRow", "0")
    for j, cw in enumerate(col_widths):
        table.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        table.rows[i].height = Inches(0.30 if i else 0.34)
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = RED
            elif i in highlight:
                cell.fill.fore_color.rgb = PEARL_ROW
            else:
                cell.fill.fore_color.rgb = WHITE if i % 2 else LIGHT_ROW
            tf = cell.text_frame
            clear_tf(tf)
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            run = p.add_run()
            run.text = str(value)
            f = run.font
            f.name = FONT
            f.size = Pt((header_size or font_size) if i == 0 else font_size)
            f.bold = (i == 0) or (i in highlight and j == 0)
            f.color.rgb = WHITE if i == 0 else BODY
    return gfx


def _fit_font(size_pt, bold):
    """DejaVu is wider than Montserrat, so fitting with it is conservative."""
    import glob
    from PIL import ImageFont
    pattern = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = glob.glob(f"/usr/share/fonts/**/{pattern}", recursive=True)
    if not path:
        return None
    return ImageFont.truetype(path[0], int(round(size_pt * 4)))


def fit_size(text, max_size, width_in, min_size=13.0, bold=False):
    """Largest size (<= max_size) whose text fits one line in width_in."""
    size = max_size
    while size > min_size:
        f = _fit_font(size, bold)
        if f is None:
            return size
        if f.getlength(text) / 4.0 <= (width_in * 72 - 34):
            return size
        size -= 0.5
    return min_size


def page_number(slide, n, shape_id=123):
    try:
        sh = by_id(slide, shape_id)
    except KeyError:
        sh = None
    if sh is None:
        add_textbox(slide, 8.86, 5.23, 1.14, 0.41,
                    [dict(text=str(n), size=12, color=WHITE,
                          align=PP_ALIGN.CENTER)])
    else:
        simple(sh, str(n), 12, WHITE, align=PP_ALIGN.CENTER)
        # the template's red footer bar is drawn after the number: bring it forward
        el = sh._element
        tree = el.getparent()
        tree.remove(el)
        tree.append(el)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def heading(slide, title, subtitle, n, ids=(119, 126, 123)):
    title_id, subtitle_id, page_id = ids
    tsh = by_id(slide, title_id)
    simple(tsh, title, fit_size(title, 25.2, Emu(tsh.width).inches),
           GREY_TITLE)
    ssh = by_id(slide, subtitle_id)
    ssh.width = Inches(8.80)
    if subtitle:
        simple(ssh, subtitle, fit_size(subtitle, 18.2, 8.80, min_size=13.0),
               GREY_TITLE)
    else:
        ssh.text_frame.clear()
    page_number(slide, n, page_id)


# --------------------------------------------------------------------------
# 3. Slide content
# --------------------------------------------------------------------------
def fill_deck() -> None:
    prs = Presentation(TMP_BASE)
    s = list(prs.slides)
    prs.core_properties.title = (
        "Controllable Chord Progression Reharmonization via RVAE-LSTM "
        "using Perceptually-Grounded Complexity Score")
    prs.core_properties.author = "José Pinto-Muñoz; Nicolás Rojas-Morales"
    prs.core_properties.subject = "JCC 2026 — Talca, Chile"

    # ---- 1. Title ---------------------------------------------------------
    sl = s[0]
    title = by_id(sl, 54)
    title.top, title.height = Inches(1.05), Inches(1.95)
    simple(title,
           "Controllable Chord Progression Reharmonization via RVAE-LSTM "
           "using Perceptually-Grounded Complexity Score",
           20, BODY, bold=True, align=PP_ALIGN.CENTER)
    authors = [
        (60, "José Pinto-Muñoz", "jose.pintom@usm.cl", 1.45),
        (61, "Nicolás Rojas-Morales", "nicolas.rojasm@usm.cl", 5.25),
    ]
    for sid, name, mail, x in authors:
        sh = by_id(sl, sid)
        sh.left, sh.top, sh.width, sh.height = (Inches(x), Inches(3.05),
                                                Inches(3.3), Inches(1.45))
        write(sh.text_frame, [
            dict(text=name, size=14, color=RED, bold=True,
                 align=PP_ALIGN.CENTER, space_after=6),
            dict(text="Departamento de Informática", size=11.5,
                 align=PP_ALIGN.CENTER, space_after=2),
            dict(text="Universidad Técnica Federico Santa María", size=11.5,
                 align=PP_ALIGN.CENTER, space_after=2),
            dict(text=mail, size=11.5, align=PP_ALIGN.CENTER),
        ], anchor_top=False)
    # remove the unused third author box
    third = by_id(sl, 62)
    third._element.getparent().remove(third._element)
    notes(sl, "0:20 — Good morning. This work presents PEARL, a system for "
              "controllable chord progression reharmonization. Joint work with "
              "Nicolás Rojas-Morales, Departamento de Informática, UTFSM.")

    # ---- 2. Outline -------------------------------------------------------
    sl = s[1]
    simple(by_id(sl, 73), "OUTLINE", 27.2, GREY_TITLE, bold=True)
    items = [
        (80, 92, "Motivation & Gap", "Why controllable reharmonization?"),
        (81, 93, "Perceptual Complexity Score", "Listening study, N = 43"),
        (82, 94, "PEARL Architecture", "RVAE-LSTM + latent gradient ascent"),
        (83, 95, "Experimental Setup", "877k progressions, CVAE/RVAE baselines"),
        (84, 96, "Results", "Control, generalization, tonal coherence"),
        (85, 97, "Conclusions", "Limitations and future work"),
    ]
    for tid, did, t, d in items:
        tsh, dsh = by_id(sl, tid), by_id(sl, did)
        tsh.width = dsh.width = Inches(4.00)
        simple(tsh, t, fit_size(t, 17.0, 4.00, min_size=13.0, bold=True),
               GREY_TITLE, bold=True)
        simple(dsh, d, fit_size(d, 11.5, 4.00, min_size=9.5), GREY_TITLE)
    page_number(sl, 2, 77)
    notes(sl, "0:15 — Roadmap: motivation and the gap, the perceptual complexity "
              "score, the architecture, the experimental setup, results and "
              "conclusions.")

    # ---- 3. Motivation (statement slide) ----------------------------------
    sl = s[2]
    simple(by_id(sl, 109), "01", 27.2, GREY_TITLE, bold=True)
    simple(by_id(sl, 104), "Motivation", 27.2, GREY_TITLE)
    st = by_id(sl, 112)
    st.top, st.height = Inches(3.05), Inches(1.30)
    simple(st, "How can we enrich a simple chord progression while controlling "
               "how complex it becomes — without losing its tonal identity?",
           17, BODY)
    by_id(sl, 110).top = Inches(4.15)  # divider below the statement
    add_textbox(sl, 2.27, 4.35, 6.6, 0.45, [
        dict(text="PEARL: Perceptual Enrichment via Anchored Regularized "
                  "Latent-space", size=11.5, color=RED, bold=True)])
    notes(sl, "0:45 — Reharmonization is a core creative task: extensions, "
              "substitutions and denser voicings. DAWs such as FL Studio or "
              "Ableton Live offer transformations, but there is no explicit, "
              "perceptually meaningful knob for how far the transformation goes. "
              "That is exactly the control PEARL provides.")

    # ---- 4. Gap & contributions ------------------------------------------
    sl = s[3]
    heading(sl, "The Gap PEARL Addresses",
            "From generation to controllable enrichment", 4)
    body = by_id(sl, 127)
    body.left, body.top, body.width, body.height = (Inches(0.60), Inches(2.02),
                                                    Inches(4.20), Inches(2.95))
    write(body.text_frame,
          [dict(text="Limitations of prior work", size=13, color=RED, bold=True,
                space_after=8)] + bullets([
              ("Chord vocabulary: ", "only maj, min, 5 and 7 as 12-dim "
               "multi-hot vectors — no extensions, alterations or inversions."),
              ("Data: ", "6,311 synthetic five-chord progressions, anchored to "
               "C major/minor."),
              ("Complexity: ", "one scalar from a compound language model "
               "(PPM + HMM + RNN), with no interpretable decomposition."),
          ], size=11.5))
    add_textbox(sl, 5.05, 2.02, 4.35, 2.95,
                [dict(text="Our contributions", size=13, color=RED, bold=True,
                      space_after=8)] + bullets([
                    ("PCS: ", "four interpretable complexity dimensions derived "
                     "from a paired-comparison listening study (N = 43)."),
                    ("PEARL: ", "RVAE-LSTM with a 48-dim chord encoding and "
                     "gradient ascent on z for per-dimension control."),
                    ("Evaluation: ", "tonal coherence, latent organization and "
                     "five complexity levels vs. CVAE/RVAE, plus generalization "
                     "to 11 keys and 4–12 chords."),
                ], size=11.5))
    notes(sl, "1:00 — Explicit gap, following Comanducci et al. 2023: limited "
              "vocabulary, a small synthetic single-key dataset and a "
              "non-interpretable complexity scalar. Our three contributions map "
              "one-to-one onto those three limitations.")

    # ---- 5. Encoding & data ----------------------------------------------
    sl = s[4]
    heading(sl, "Chord Encoding and Data",
            "A 48-dimensional, fully interpretable representation", 5)
    body = by_id(sl, 127)
    body.left, body.top, body.width, body.height = (Inches(0.60), Inches(2.02),
                                                    Inches(4.55), Inches(2.95))
    write(body.text_frame, bullets([
        ("Factored encoding: ", "seven orthogonal slots per chord instead of "
         "independent chord tones."),
        ("Reversible: ", "each sub-vector decodes back to a readable chord "
         "symbol via argmax / thresholding."),
        ("Sequence: ", "a progression of T chords becomes X in {0,1}^(T x 48)."),
        ("Chordonomicon: ", "~666k songs -> 877k section-level progressions, "
         "labelled with jSymbolic 2.2."),
        ("Split: ", "80/10/10 stratified by song — no song appears in both "
         "train and test."),
    ], size=11))
    add_table(sl, 5.35, 2.02, 4.05, [
        ["Slot", "Dims", "Contents"],
        ["Root", "12", "Pitch class C–B"],
        ["Quality", "8", "maj, min, dim, aug, sus…"],
        ["Seventh", "5", "none, dom7, maj7, dim7, aug7"],
        ["Extensions", "3", "9, 11, 13"],
        ["Alterations", "4", "b9, #9, #11, b13"],
        ["Added notes", "3", "add9, add11, add13"],
        ["Bass", "13", "C–B + root / no slash"],
    ], [1.35, 0.60, 2.10], font_size=9.5)
    notes(sl, "1:00 — The encoding separates orthogonal musical attributes, so "
              "the model learns which extensions co-occur with which seventh "
              "types directly from data. Training uses 877k progressions from "
              "Chordonomicon, three orders of magnitude more than the 6,311 used "
              "by the baselines.")

    # ---- 6. PCS ------------------------------------------------------------
    sl = s[5]
    heading(sl, "Perceptual Complexity Score",
            "PCS: from 11 jSymbolic features to 4 perceptual dimensions", 6)
    body = by_id(sl, 127)
    body.left, body.top, body.width, body.height = (Inches(0.60), Inches(2.02),
                                                    Inches(4.85), Inches(2.95))
    write(body.text_frame, bullets([
        ("Listening study: ", "paired comparisons, 5-point scale; 53 "
         "participants, 43 valid after attention controls; 1,032 observations."),
        ("Stratification: ", "Goldsmiths Musical Sophistication Index — low, "
         "medium and high levels."),
        ("Feature retention: ", "p < 0.05 in at least one MSI level and a "
         "consistent correlation sign across levels (11 -> 4 features)."),
        ("Two variants: ", "PEARL-v1 uses perceptual weights; PEARL-v2 weights "
         "all four dimensions equally."),
    ], size=11) + [
        dict(text="PCS_w = (0.443·7C + 0.378·VNSPC + 0.338·DTMCVI + "
                  "0.201·VDR) / 1.360", size=11, color=DARK_RED, bold=True,
             space_before=6),
    ])
    add_table(sl, 5.65, 2.02, 3.75, [
        ["Dim.", "Musical meaning", "Weight"],
        ["7C", "Seventh chords", "0.443"],
        ["VNSPC", "Pitch-class density variability", "0.378"],
        ["DTMCVI", "Distance between common intervals", "0.338"],
        ["VDR", "Vertical dissonance ratio", "0.201"],
    ], [0.75, 2.30, 0.70], font_size=9.5)
    add_textbox(sl, 5.65, 3.85, 3.75, 1.0, [
        dict(text="Seventh chords weigh ~2.2x more than vertical dissonance, "
                  "matching how listeners judge complexity.",
             size=11, color=BODY, italic=True)])
    notes(sl, "1:40 — PCS is the core of the contribution. Complexity is defined "
              "perceptually, not mathematically: listeners compared pairs of "
              "excerpts. Only features that were significant and sign-consistent "
              "across sophistication levels survived, which is why vertical "
              "tritones or non-standard chords were discarded — statistically, "
              "not musically. Weights are the mean absolute correlations across "
              "MSI levels.")

    # ---- 7. Architecture ---------------------------------------------------
    sl = s[6]
    heading(sl, "PEARL Architecture",
            "RVAE-LSTM with auxiliary complexity and key predictors", 7)
    body = by_id(sl, 127)
    body._element.getparent().remove(body._element)
    add_picture_fit(sl, os.path.join(ASSETS, "fig2_arch.png"),
                    0.60, 1.92, 8.80, 2.10)
    add_textbox(sl, 0.60, 4.10, 4.30, 0.95, bullets([
        ("Encoder: ", "bidirectional LSTM 2x256; decoder: autoregressive LSTM "
         "2x256; z in R^32."),
        ("Conditioning: ", "c = 4 PCS + 12 key (16 dims), with a conditional "
         "ComplexityPrior."),
    ], size=10.5, space_after=4))
    add_textbox(sl, 5.05, 4.10, 4.35, 0.95, bullets([
        ("Auxiliary heads: ", "c_predictor MSE 0.0034; key_predictor accuracy "
         "98.4%."),
        ("Loss: ", "recon + beta·KL + 10·L_c + 0.5·L_key, with free bits and a "
         "cyclic KL schedule."),
    ], size=10.5, space_after=4))
    notes(sl, "1:40 — Architecture band of the pipeline (data preparation was "
              "covered in the previous slides, inference comes next). The "
              "auxiliary predictors are essential: c_predictor makes the latent "
              "space readable in PCS terms and provides the gradient used at "
              "inference; key_predictor organizes z by tonal centre. Word "
              "dropout of 70% and a bypass delta of 2.77 BCE confirm the decoder "
              "really relies on z.")

    # ---- 8. Inference ------------------------------------------------------
    sl = s[7]
    heading(sl, "Gradient Ascent on z",
            "Inference: enriching to a target PCS in 20–30 steps", 8)
    body = by_id(sl, 127)
    body.left, body.top, body.width, body.height = (Inches(0.60), Inches(2.02),
                                                    Inches(4.90), Inches(2.95))
    write(body.text_frame, bullets([
        ("Objective: ", "minimize MSE(PCS(c_pred(z)), target) + lambda_reg · "
         "MSE(z, z_orig), with lambda_reg = 0.005 anchoring the original "
         "progression."),
        ("Decode: ", "z* is decoded autoregressively into the enriched "
         "progression; latent displacement grows from ~0.1 at level 1 to ~7 at "
         "level 5."),
        ("Per-dimension control: ", "optimizing one dimension leaves the others "
         "almost unchanged."),
        ">7C only: 0.002 -> 0.982, five of five seventh chords.",
        ">VDR only: 0.005 -> 0.987, dissonance without forcing sevenths.",
        ">VNSPC only: -0.002 -> 0.983, density variability alone.",
    ], size=11, space_after=5))
    add_textbox(sl, 5.70, 2.02, 3.70, 2.60, [
        dict(text="Example — target 7C = 1.0", size=12.5, color=RED, bold=True,
             space_after=8),
        dict(text="Input", size=11, color=GREY_TITLE, bold=True, space_after=2),
        dict(text="Cmin – Gmaj – Cmin – Fmin – Cmin", size=11.5, space_after=8),
        dict(text="Output", size=11, color=GREY_TITLE, bold=True, space_after=2),
        dict(text="Cmaj7 – Dmin7 – Cmin7 – Gmin7 – Cmin7", size=11.5,
             space_after=8),
        dict(text="No retraining, no sampling: the same checkpoint serves any "
                  "target in [0, 1] and any subset of dimensions.",
             size=11, italic=True),
    ])
    notes(sl, "1:20 — At inference the weights are frozen: we optimize the latent "
              "vector, not the network. The regularization term is the anchor "
              "that keeps z* in a tonally compatible region. Emphasize "
              "per-dimension targeting, something neither baseline can do.")

    # ---- 9. Experimental setup --------------------------------------------
    sl = s[8]
    heading(sl, "Experimental Setup", "Training, baselines and metrics", 9)
    body = by_id(sl, 127)
    body.left, body.top, body.width, body.height = (Inches(0.60), Inches(2.02),
                                                    Inches(8.80), Inches(2.95))
    write(body.text_frame, bullets([
        ("Training: ", "30 epochs on 877k progressions; Adam lr 5e-4, batch 256, "
         "dropout 0.2, word dropout 0.7, free bits 1.0 / 0.25, 10-epoch KL "
         "cycle, scheduled sampling 1.0 -> 0.3."),
        ("Convergence: ", "val loss 4.24, KL ~6–7 nats, 32/32 latent dimensions "
         "active, bypass delta 2.77 BCE (no posterior collapse)."),
        ("Baselines: ", "CVAE and RVAE (Comanducci et al., 2023) on their "
         "publicly released outputs — original weights are unavailable."),
        ("Protocol: ", "input Cmin – Gmaj – Cmin – Fmin – Cmin; baseline levels "
         "1–5 mapped to PCS targets 0.00 / 0.25 / 0.50 / 0.75 / 1.00."),
        ("Metrics: ", "PCS achieved, per-dimension deltas, seventh-chord ratio, "
         "scale adherence, root overlap, quality preservation and latent "
         "displacement."),
    ], size=11.5, space_after=7))
    notes(sl, "0:50 — All results are computed on the held-out test set. Be "
              "transparent about the baseline comparison: we use their released "
              "outputs, which is why we add key and length generalization tests "
              "where their design cannot operate.")

    # ---- 10. Latent space --------------------------------------------------
    sl = s[9]
    heading(sl, "Latent Space Organization",
            "Complexity and key are structurally encoded in z", 10)
    body = by_id(sl, 127)
    body._element.getparent().remove(body._element)
    add_picture_fit(sl, os.path.join(ASSETS, "pca_colored.png"),
                    0.60, 1.92, 8.80, 2.62)
    add_textbox(sl, 0.60, 4.55, 8.80, 0.52, [
        dict(runs=[("32/32 active dimensions (KL 32.72 nats)  ·  ", True),
                   ("c_predictor MSE 0.0034  ·  key_predictor 98.4%  ·  "
                    "2 PCs explain 42.6% of variance", False)],
             size=10.5, color=BODY)])
    notes(sl, "0:55 — PCA of 3,000 encoded progressions. Left: a smooth "
              "complexity gradient. Right: clear tonal clustering. Specialized "
              "dimensions such as z26 (dissonance) or z15 (seventh chords) "
              "explain why gradient ascent converges so reliably.")

    # ---- 11. Complexity levels --------------------------------------------
    sl = s[10]
    heading(sl, "Five Complexity Levels",
            "Seventh-chord usage tracks the requested PCS target", 11)
    body = by_id(sl, 127)
    body._element.getparent().remove(body._element)
    add_picture_fit(sl, os.path.join(ASSETS, "seventh_comparison.png"),
                    0.45, 1.95, 6.05, 2.55)
    add_textbox(sl, 6.65, 2.02, 2.95, 2.70, bullets([
        ("Convergence: ", "both variants reach their targets in 20–30 steps."),
        ("v1 (weighted): ", "14% of the total dimension change flows through 7C "
         "at target 0.75 and 20% at 1.00 — v2 reaches only 6% and 12%."),
        ("Chord types: ", "v1 favours maj7 and altered dominants; v2 produces "
         "more suspended, diminished and no3d chords."),
    ], size=10.5))
    add_textbox(sl, 0.60, 4.55, 8.80, 0.52, [
        dict(runs=[("Baselines have no per-dimension control: ", True),
                   ("RVAE introduces chromatic alterations and drops to 76% "
                    "scale adherence.", False)], size=10.5)])
    notes(sl, "1:20 — Seventh-chord count is a coarse but directly comparable "
              "indicator of enrichment intensity across models. The weighted PCS "
              "steers enrichment towards seventh chords, exactly as the "
              "perceptual study suggests.")

    # ---- 12. Generalization -------------------------------------------------
    sl = s[11]
    heading(sl, "Generalization",
            "11 keys and progression lengths from 4 to 12 chords", 12)
    body = by_id(sl, 127)
    body._element.getparent().remove(body._element)
    add_picture_fit(sl, os.path.join(ASSETS, "key_generalization.png"),
                    0.60, 1.92, 8.80, 2.60)
    add_textbox(sl, 0.60, 4.53, 8.80, 0.54, [
        dict(runs=[("PCS reached in every key (0.453–0.485); 100% scale "
                    "adherence at all lengths. ", True),
                   ("Key preserved in 6/11 cases — drifts go to relative or "
                    "parallel keys.", False)], size=10.5)])
    notes(sl, "0:55 — This is where PEARL goes beyond the baselines by design. "
              "Honest limitation: the key predictor preserves the original key "
              "in 55% of the generalization tests, although drifts land on "
              "closely related keys and roots largely stay in scale. Stronger "
              "tonic anchoring is future work.")

    # ---- 13. Tonal coherence ------------------------------------------------
    sl = s[12]
    heading(sl, "Tonal Coherence",
            "PEARL vs. CVAE and RVAE: scale adherence and reharmonization", 13)
    body = by_id(sl, 127)
    body.left, body.top, body.width, body.height = (Inches(0.60), Inches(2.02),
                                                    Inches(4.05), Inches(2.95))
    write(body.text_frame, bullets([
        ("88% scale adherence ", "for both PEARL variants, vs. 84% (CVAE) and "
         "76% (RVAE)."),
        ("Low positional overlap ", "(32–36%) follows from the enrichment "
         "objective with lambda_reg = 0.005 — not from a tonal failure."),
        ("Key consistency: ", "key_predictor output stays stable across all "
         "five levels for both variants."),
        ("Trade-off: ", "exposing lambda_reg as a user control would let "
         "composers choose between subtle enrichment and radical "
         "reharmonization."),
    ], size=11, space_after=5))
    add_table(sl, 4.95, 2.10, 4.45, [
        ["Model", "Scale %", "Overlap %", "Quality %"],
        ["CVAE", "84%", "84%", "72%"],
        ["RVAE", "76%", "64%", "60%"],
        ["PEARL-v1", "88%", "32%", "32%"],
        ["PEARL-v2", "88%", "36%", "44%"],
    ], [1.45, 1.00, 1.00, 1.00], font_size=11, highlight=(3, 4))
    add_textbox(sl, 4.95, 3.90, 4.45, 0.90, [
        dict(text="Mean across the five complexity levels. Input key: C minor. "
                  "CVAE's high overlap partly comes from copying the input at "
                  "level 1.", size=10, color=GREY_TITLE, italic=True)])
    notes(sl, "1:10 — Anticipate the obvious question: why is overlap so low? "
              "Because enriching means changing chord identities — triads become "
              "seventh chords or gain extensions. Tonality stays anchored: 88% "
              "of roots remain in the input scale, the best result of the three "
              "models.")

    # ---- 14. Conclusions ----------------------------------------------------
    sl = s[13]
    simple(by_id(sl, 16), "Conclusions", 25.2, GREY_TITLE)
    conc_sub = by_id(sl, 18)
    conc_sub.width = Inches(8.80)
    simple(conc_sub, "What PEARL enables — and what comes next",
           fit_size("What PEARL enables — and what comes next", 18.2, 8.80),
           GREY_TITLE)
    body = by_id(sl, 19)
    body.left, body.top, body.width, body.height = (Inches(0.60), Inches(2.02),
                                                    Inches(8.80), Inches(2.60))
    write(body.text_frame, bullets([
        ("Perceptual control: ", "PCS distils 11 features into 4 interpretable "
         "dimensions validated by 43 listeners."),
        ("Controllable enrichment: ", "gradient ascent reaches any PCS target in "
         "under 30 steps and supports per-dimension targeting."),
        ("Tonal coherence and reach: ", "88% scale adherence, generalizing to 11 "
         "keys and 4–12 chords."),
        ("Limitations: ", "harmonic content only (no melody, rhythm or genre); "
         "PCS weights from 43 participants; baselines compared on released "
         "outputs; key preserved in 55% of generalization tests."),
        ("Future work: ", "user-facing lambda_reg, stronger tonic anchoring, "
         "rhythmic and structural conditioning, larger and more diverse "
         "listening studies, integration into composition tools."),
    ], size=11, space_after=6))
    add_textbox(sl, 0.60, 4.66, 8.80, 0.42, [
        dict(runs=[("Code, checkpoint, audio/MIDI and survey data:  ", True),
                   ("github.com/Pepebeats-flp/pearl-sc", False)],
             size=11.5, color=DARK_RED)])
    page_number(sl, 14)
    notes(sl, "1:05 — Close with the three takeaways and be upfront about the "
              "limitations. The repository contains everything needed to "
              "reproduce the results, including audio and MIDI examples.")

    # ---- 15. Acknowledgments -------------------------------------------------
    sl = s[14]
    by_id(sl, 4).height = Inches(1.95)  # grey panel behind the closing text
    simple(by_id(sl, 5), "Thank you — Questions?", 17.2, DARK_RED,
           align=PP_ALIGN.CENTER)
    ack_body = by_id(sl, 6)
    ack_body.height = Inches(1.95)
    write(ack_body.text_frame, [
        dict(text="PEARL — Controllable Chord Progression Reharmonization",
             size=12, bold=True, space_after=6),
        dict(text="José Pinto-Muñoz  ·  Nicolás Rojas-Morales", size=11.5,
             space_after=2),
        dict(text="Departamento de Informática, Universidad Técnica Federico "
                  "Santa María", size=11.5, space_after=6),
        dict(text="github.com/Pepebeats-flp/pearl-sc", size=11.5,
             color=DARK_RED),
    ], anchor_top=False)
    notes(sl, "0:20 — Thank the audience and invite questions. Likely questions: "
              "low positional overlap, 55% key preservation, baseline comparison "
              "without retraining, and why 7 of 11 jSymbolic features were "
              "discarded.")

    prs.save(OUTPUT)
    print(f"saved {OUTPUT} ({os.path.getsize(OUTPUT)/1024:.0f} KB, "
          f"{len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    build_base_deck()
    fill_deck()
