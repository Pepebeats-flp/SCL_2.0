#!/usr/bin/env python3
"""Ultra-simple PEARL pipeline diagram - just the essentials, big and clear."""
import subprocess, shutil, os
from pathlib import Path

OUT = Path("/home/pepebeats/SCL_2.0/JCC2026/img")
DRAWIO = str(OUT / "pearl_pipeline.drawio")

# Build manually with minimal, clean layout
xml = """<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net">
  <diagram name="PEARL Pipeline" id="p">
    <mxGraphModel dx="0" dy="0" grid="1" gridSize="10" guides="1" pageWidth="1000" pageHeight="750">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>

        <!-- TITLE -->
        <mxCell id="t" value="PEARL — Pipeline Overview" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=24;fontStyle=1;fontColor=#263238;" vertex="1" parent="1">
          <mxGeometry x="250" y="10" width="500" height="35" as="geometry"/>
        </mxCell>

        <!-- ====== ROW 1: DATA ====== -->
        <mxCell id="d_title" value="DATA PREPARATION" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#BBDEFB;strokeColor=#64B5F6;fontSize=16;fontStyle=1;fontColor=#1565C0;" vertex="1" parent="1">
          <mxGeometry x="30" y="55" width="200" height="35" as="geometry"/>
        </mxCell>
        <mxCell id="d1" value="Chordonomicon&lt;br&gt;~877k MIDI progressions" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#90CAF9;fontSize=13;fontStyle=0;fontColor=#212121;" vertex="1" parent="1">
          <mxGeometry x="30" y="100" width="190" height="55" as="geometry"/>
        </mxCell>
        <mxCell id="d2" value="Encoding &amp;amp; jSymbolic&lt;br&gt;48-dim + 11 features" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#90CAF9;fontSize=13;fontStyle=0;fontColor=#212121;" vertex="1" parent="1">
          <mxGeometry x="250" y="100" width="190" height="55" as="geometry"/>
        </mxCell>
        <mxCell id="d3" value="PCS via Perceptual Study&lt;br&gt;43 participants → 4 dims" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#90CAF9;fontSize=13;fontStyle=0;fontColor=#212121;" vertex="1" parent="1">
          <mxGeometry x="470" y="100" width="210" height="55" as="geometry"/>
        </mxCell>
        <mxCell id="d4" value="4 PCS dimensions&lt;br&gt;7C | VNSPC | DTMCVI | VDR" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#90CAF9;strokeColor=#64B5F6;fontSize=13;fontStyle=1;fontColor=#FFFFFF;" vertex="1" parent="1">
          <mxGeometry x="710" y="100" width="210" height="55" as="geometry"/>
        </mxCell>

        <!-- Data arrows -->
        <mxCell id="da1" style="edgeStyle=orthogonalEdgeStyle;html=1;strokeColor=#64B5F6;strokeWidth=2;endArrow=classic;rounded=1;" edge="1" parent="1" source="d1" target="d2">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="da2" style="edgeStyle=orthogonalEdgeStyle;html=1;strokeColor=#64B5F6;strokeWidth=2;endArrow=classic;rounded=1;" edge="1" parent="1" source="d2" target="d3">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="da3" style="edgeStyle=orthogonalEdgeStyle;html=1;strokeColor=#64B5F6;strokeWidth=2;endArrow=classic;rounded=1;" edge="1" parent="1" source="d3" target="d4">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>

        <!-- Arrow from data to architecture -->
        <mxCell id="da" style="edgeStyle=orthogonalEdgeStyle;html=1;strokeColor=#546E7A;strokeWidth=3;endArrow=classic;rounded=1;exitX=0.5;exitY=1;entryX=0.5;entryY=0;" edge="1" parent="1" source="d4" target="a_title">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>

        <!-- ====== ROW 2: ARCHITECTURE ====== -->
        <mxCell id="a_title" value="ARCHITECTURE — RVAE-LSTM" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFE0B2;strokeColor=#FFB74D;fontSize=16;fontStyle=1;fontColor=#E65100;" vertex="1" parent="1">
          <mxGeometry x="30" y="195" width="280" height="35" as="geometry"/>
        </mxCell>

        <!-- Encoder -->
        <mxCell id="enc" value="Bidir LSTM&lt;br&gt;Encoder&lt;br&gt;2×256" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF3E0;strokeColor=#FFB74D;fontSize=13;fontStyle=0;fontColor=#212121;" vertex="1" parent="1">
          <mxGeometry x="30" y="245" width="120" height="75" as="geometry"/>
        </mxCell>
        <!-- z -->
        <mxCell id="z" value="⏺ z&lt;br&gt;32-dim" style="ellipse;whiteSpace=wrap;html=1;fillColor=#FFB74D;strokeColor=#FB8C00;fontSize=14;fontStyle=1;fontColor=#FFFFFF;aspect=fixed;" vertex="1" parent="1">
          <mxGeometry x="185" y="252" width="60" height="60" as="geometry"/>
        </mxCell>
        <!-- Decoder -->
        <mxCell id="dec" value="Autoregressive&lt;br&gt;LSTM Decoder&lt;br&gt;2×256" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF3E0;strokeColor=#FFB74D;fontSize=13;fontStyle=0;fontColor=#212121;" vertex="1" parent="1">
          <mxGeometry x="280" y="245" width="150" height="75" as="geometry"/>
        </mxCell>

        <!-- Architecture arrows -->
        <mxCell id="ea1" style="edgeStyle=orthogonalEdgeStyle;html=1;strokeColor=#FFB74D;strokeWidth=2;endArrow=classic;rounded=1;" edge="1" parent="1" source="enc" target="z">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="ea2" style="edgeStyle=orthogonalEdgeStyle;html=1;strokeColor=#FFB74D;strokeWidth=2;endArrow=classic;rounded=1;" edge="1" parent="1" source="z" target="dec">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>

        <!-- c_predictor below z -->
        <mxCell id="cp" value="c_predictor: z → [7C, VNSPC, DTMCVI, VDR]" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF3E0;strokeColor=#FFB74D;fontSize=11;fontStyle=0;fontColor=#212121;" vertex="1" parent="1">
          <mxGeometry x="120" y="340" width="250" height="30" as="geometry"/>
        </mxCell>
        <mxCell id="kp" value="key_predictor: z → 12 key logits (98.4% acc)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF3E0;strokeColor=#FFB74D;fontSize=11;fontStyle=0;fontColor=#212121;" vertex="1" parent="1">
          <mxGeometry x="120" y="380" width="250" height="30" as="geometry"/>
        </mxCell>
        <mxCell id="eca" style="edgeStyle=orthogonalEdgeStyle;html=1;strokeColor=#FFB74D;strokeWidth=1.5;endArrow=classic;dashed=1;rounded=1;" edge="1" parent="1" source="z" target="cp">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="eka" style="edgeStyle=orthogonalEdgeStyle;html=1;strokeColor=#FFB74D;strokeWidth=1.5;endArrow=classic;dashed=1;rounded=1;" edge="1" parent="1" source="z" target="kp">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>

        <!-- Right side: ComplexityPrior + output -->
        <mxCell id="prior" value="ComplexityPrior&lt;br&gt;c → μ_prior, σ_prior" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF3E0;strokeColor=#FFB74D;fontSize=11;fontStyle=0;fontColor=#212121;" vertex="1" parent="1">
          <mxGeometry x="480" y="250" width="180" height="45" as="geometry"/>
        </mxCell>
        <mxCell id="out_a" value="48-dim chord&lt;br&gt;logits → decode" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFB74D;strokeColor=#FB8C00;fontSize=13;fontStyle=1;fontColor=#FFFFFF;" vertex="1" parent="1">
          <mxGeometry x="480" y="310" width="180" height="45" as="geometry"/>
        </mxCell>

        <!-- Arrow to inference -->
        <mxCell id="ai" style="edgeStyle=orthogonalEdgeStyle;html=1;strokeColor=#546E7A;strokeWidth=3;endArrow=classic;rounded=1;exitX=0.5;exitY=1;entryX=0.5;entryY=0;" edge="1" parent="1" source="dec" target="i_title">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>

        <!-- ====== ROW 3: INFERENCE ====== -->
        <mxCell id="i_title" value="INFERENCE — Gradient Ascent on z" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#C8E6C9;strokeColor=#81C784;fontSize=16;fontStyle=1;fontColor=#2E7D32;" vertex="1" parent="1">
          <mxGeometry x="30" y="450" width="320" height="35" as="geometry"/>
        </mxCell>

        <mxCell id="i1" value="① Encode input&lt;br&gt;x → z_orig" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F5E9;strokeColor=#A5D6A7;fontSize=13;fontStyle=0;fontColor=#212121;" vertex="1" parent="1">
          <mxGeometry x="30" y="500" width="150" height="55" as="geometry"/>
        </mxCell>
        <mxCell id="i2" value="② Gradient Ascent (90 steps)&lt;br&gt;min MSE(PCS(c_pred), target)&lt;br&gt;+ 0.005·MSE(z, z_orig)&lt;br&gt;Adam, lr=0.5" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F5E9;strokeColor=#A5D6A7;fontSize=13;fontStyle=0;fontColor=#212121;" vertex="1" parent="1">
          <mxGeometry x="210" y="495" width="250" height="70" as="geometry"/>
        </mxCell>
        <mxCell id="i3" value="③ Decode z★&lt;br&gt;→ enriched progression" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F5E9;strokeColor=#A5D6A7;fontSize=13;fontStyle=0;fontColor=#212121;" vertex="1" parent="1">
          <mxGeometry x="490" y="500" width="170" height="55" as="geometry"/>
        </mxCell>

        <mxCell id="ei1" style="edgeStyle=orthogonalEdgeStyle;html=1;strokeColor=#81C784;strokeWidth=2;endArrow=classic;rounded=1;" edge="1" parent="1" source="i1" target="i2">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
        <mxCell id="ei2" style="edgeStyle=orthogonalEdgeStyle;html=1;strokeColor=#81C784;strokeWidth=2;endArrow=classic;rounded=1;" edge="1" parent="1" source="i2" target="i3">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>

        <!-- Example -->
        <mxCell id="ex_in" value="IN:  Cm — Gm — Cm — Fm — Cm" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#A5D6A7;strokeColor=#81C784;fontSize=12;fontStyle=1;fontColor=#FFFFFF;" vertex="1" parent="1">
          <mxGeometry x="30" y="575" width="250" height="25" as="geometry"/>
        </mxCell>
        <mxCell id="ex_out" value="OUT: Cmaj7 — Gm7 — Cmaj7 — Fm7 — Cmaj7" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#A5D6A7;strokeColor=#81C784;fontSize=12;fontStyle=1;fontColor=#FFFFFF;" vertex="1" parent="1">
          <mxGeometry x="300" y="575" width="350" height="25" as="geometry"/>
        </mxCell>

        <!-- Variants -->
        <mxCell id="v1" value="PEARL-v1 (weighted):  PCS_w = Σ(w·c) ÷ 1.360" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#C8E6C9;strokeColor=#81C784;fontSize=12;fontStyle=0;fontColor=#212121;" vertex="1" parent="1">
          <mxGeometry x="30" y="615" width="320" height="28" as="geometry"/>
        </mxCell>
        <mxCell id="v2" value="PEARL-v2 (equal):  PCS_eq = mean(c)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#C8E6C9;strokeColor=#81C784;fontSize=12;fontStyle=0;fontColor=#212121;" vertex="1" parent="1">
          <mxGeometry x="370" y="615" width="280" height="28" as="geometry"/>
        </mxCell>

        <!-- ====== ROW 4: LOSSES ====== -->
        <mxCell id="l_title" value="TRAINING LOSSES" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCDD2;strokeColor=#E57373;fontSize=14;fontStyle=1;fontColor=#C62828;" vertex="1" parent="1">
          <mxGeometry x="700" y="450" width="200" height="35" as="geometry"/>
        </mxCell>
        <mxCell id="l1" value="L = L_recon + β·L_KL + λ_c·L_cpred + λ_k·L_key" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFEBEE;strokeColor=#EF9A9A;fontSize=12;fontStyle=0;fontColor=#212121;" vertex="1" parent="1">
          <mxGeometry x="690" y="500" width="290" height="28" as="geometry"/>
        </mxCell>
        <mxCell id="l2" value="BCE (length mask)  |  KL (free bits 1.0/0.25)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFEBEE;strokeColor=#EF9A9A;fontSize=10;fontStyle=0;fontColor=#212121;" vertex="1" parent="1">
          <mxGeometry x="690" y="540" width="290" height="22" as="geometry"/>
        </mxCell>
        <mxCell id="l3" value="MSE(c_pred, c_true)  |  CE(key_pred, key)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFEBEE;strokeColor=#EF9A9A;fontSize=10;fontStyle=0;fontColor=#212121;" vertex="1" parent="1">
          <mxGeometry x="690" y="570" width="290" height="22" as="geometry"/>
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""

with open(DRAWIO, "w") as f:
    f.write(xml)
print(f"Written: {DRAWIO}")

# Export to PDF
drawio = shutil.which("drawio")
if drawio:
    pdf = str(OUT / "pearl_pipeline.pdf")
    subprocess.run([drawio, "--export", "--format", "pdf", "--crop", "--output", pdf, DRAWIO], check=True)
    print(f"PDF: {pdf}")
else:
    print("No drawio CLI")
