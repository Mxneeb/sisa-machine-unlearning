"""Generate Proposal.pdf using reportlab."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)

# ── Output path ───────────────────────────────────────────────────────
OUTPUT = "Proposal.pdf"

# ── Colours ───────────────────────────────────────────────────────────
BRAND      = colors.HexColor("#1a56a0")
BRAND_LITE = colors.HexColor("#e8f0fe")
ACCENT     = colors.HexColor("#c0392b")
GOLD       = colors.HexColor("#856404")
GOLD_BG    = colors.HexColor("#fff3cd")
LIGHT_GREY = colors.HexColor("#f5f5f5")
MID_GREY   = colors.HexColor("#555555")
WHITE      = colors.white
BLACK      = colors.black

# ── Document ──────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=2.0*cm, rightMargin=2.0*cm,
    topMargin=1.8*cm,  bottomMargin=1.8*cm,
)
W = A4[0] - 4.0*cm   # usable width

# ── Base styles ───────────────────────────────────────────────────────
base = getSampleStyleSheet()

def S(name, parent="Normal", **kw):
    return ParagraphStyle(name, parent=base[parent], **kw)

title_style = S("Title2", fontSize=16, leading=20, textColor=BRAND,
                fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=4)
sub_style   = S("Sub",    fontSize=10, leading=13, textColor=MID_GREY,
                fontName="Helvetica", alignment=TA_CENTER, spaceAfter=2)
meta_style  = S("Meta",   fontSize=9,  leading=12, textColor=MID_GREY,
                fontName="Helvetica", alignment=TA_CENTER, spaceAfter=0)

h2_style = S("H2", fontSize=11, leading=14, textColor=WHITE,
             fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)

body_style = S("Body", fontSize=9.5, leading=13.5, textColor=BLACK,
               fontName="Helvetica", alignment=TA_JUSTIFY, spaceAfter=5)
body_bold  = S("BodyB", fontSize=9.5, leading=13.5, textColor=BLACK,
               fontName="Helvetica-Bold", alignment=TA_JUSTIFY, spaceAfter=3)

bullet_style = S("Bul", fontSize=9.5, leading=13, textColor=BLACK,
                 fontName="Helvetica", leftIndent=14, firstLineIndent=0,
                 alignment=TA_LEFT, spaceAfter=2)

footer_style = S("Footer", fontSize=8, leading=10, textColor=MID_GREY,
                 fontName="Helvetica", alignment=TA_CENTER)

def section_header(text):
    """Returns a dark-blue banner with white text as a 1-cell table."""
    cell = Paragraph(f"&nbsp;&nbsp;{text}", h2_style)
    t = Table([[cell]], colWidths=[W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), BRAND),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
    ]))
    return t

def coloured_table(data, col_widths, header=True):
    t = Table(data, colWidths=col_widths)
    style = [
        ("FONTNAME",  (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",  (0,0), (-1,-1), 9),
        ("LEADING",   (0,0), (-1,-1), 12),
        ("VALIGN",    (0,0), (-1,-1), "TOP"),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [LIGHT_GREY, WHITE]),
        ("GRID",      (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("RIGHTPADDING",  (0,0), (-1,-1), 6),
    ]
    if header:
        style += [
            ("BACKGROUND",  (0,0), (-1,0), BRAND),
            ("TEXTCOLOR",   (0,0), (-1,0), WHITE),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, LIGHT_GREY]),
        ]
    t.setStyle(TableStyle(style))
    return t

def bonus_badge():
    cell = Paragraph(
        "<b>★ Bonus Marks Eligible — Machine Unlearning Domain</b>",
        ParagraphStyle("badge", fontSize=9, leading=12, textColor=GOLD,
                       fontName="Helvetica-Bold", alignment=TA_CENTER)
    )
    t = Table([[cell]], colWidths=[W])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), GOLD_BG),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("BOX",           (0,0), (-1,-1), 1, colors.HexColor("#ffc107")),
    ]))
    return t

# ══════════════════════════════════════════════════════════════════════
# Build story
# ══════════════════════════════════════════════════════════════════════
story = []

# ── Title block ───────────────────────────────────────────────────────
story.append(Paragraph("Project Proposal", title_style))
story.append(Paragraph("Responsible Artificial Intelligence — Semester Project", sub_style))
story.append(Paragraph("Spring 2026", meta_style))
story.append(Spacer(1, 4))
story.append(Paragraph("Muneeb Ahmad &nbsp;&nbsp;22i-1889 &nbsp;&nbsp;|&nbsp;&nbsp; Saim Nadeem &nbsp;&nbsp;22i-1884", meta_style))
story.append(HRFlowable(width=W, thickness=2, color=BRAND, spaceAfter=6))
story.append(Spacer(1, 8))

# ── 1. Selected Research Paper ────────────────────────────────────────
story.append(section_header("1.  Selected Research Paper"))
story.append(Spacer(1, 4))

paper_data = [
    ["Field", "Details"],
    ["Title",   "Machine Unlearning"],
    ["Authors", "Lucas Bourtoule, Varun Chandrasekaran, Christopher A. Choquette-Choo,\n"
                "Hengrui Jia, Adelin Travers, Baiwu Zhang, David Lie, Nicolas Papernot"],
    ["Venue",   "IEEE Symposium on Security and Privacy (IEEE S&P 2021)\n"
                "Tier-1 Security & Privacy Conference"],
    ["Source",  "arXiv:1912.03817  |  DOI: 10.48550/arXiv.1912.03817"],
    ["Domain",  "Machine Unlearning"],
]
story.append(coloured_table(paper_data, [3.2*cm, W - 3.2*cm]))
story.append(Spacer(1, 8))

# ── 2. Problem Statement & Motivation ────────────────────────────────
story.append(section_header("2.  Problem Statement & Motivation"))
story.append(Spacer(1, 4))

story.append(Paragraph(
    "Modern machine learning models are trained on large collections of personal data. "
    "Regulations such as <b>GDPR Article 17</b> (Right to Erasure) and the <b>California "
    "Consumer Privacy Act (CCPA)</b> grant users the legal right to demand removal of their "
    "data from any system that holds it — including trained AI models.",
    body_style))

story.append(Paragraph(
    "The naive solution is to <b>retrain the model from scratch</b> after removing the "
    "requested data. For large-scale deployments this is prohibitively expensive: "
    "retraining a production model can take hours or days on a GPU cluster. "
    "This creates a fundamental tension:",
    body_style))

story.append(Paragraph(
    "<b>Privacy compliance demands data deletion, yet full retraining is "
    "computationally infeasible at scale.</b>",
    S("Bold", fontSize=9.5, leading=13, fontName="Helvetica-BoldOblique",
      textColor=ACCENT, alignment=TA_CENTER, spaceAfter=5)))

story.append(Paragraph(
    "Approximate alternatives (gradient-based influence subtraction, model editing) either "
    "fail to guarantee exact removal or destabilise model performance. "
    "There is therefore a critical need for an efficient, verifiable unlearning framework "
    "that is both mathematically exact and practically scalable.",
    body_style))
story.append(Spacer(1, 6))

# ── 3. Methodology Overview ───────────────────────────────────────────
story.append(section_header("3.  Overview of Methodology — SISA Training"))
story.append(Spacer(1, 4))

story.append(Paragraph(
    "Bourtoule et al. propose <b>SISA Training</b> (Sharded, Isolated, Sliced, and "
    "Aggregated), which restructures the training procedure so that future unlearning "
    "requests are fast and cheap by construction.",
    body_style))

sisa_data = [
    ["Component", "Mechanism", "Effect on Unlearning"],
    ["Sharding",     "Training data split into S disjoint shards;\none model trained per shard independently",
                     "A deletion request affects at most\n1 of S models"],
    ["Isolation",    "Shard models trained with no gradient\nsharing across shards",
                     "Unlearning in one shard is fully\nindependent of others"],
    ["Slicing",      "Each shard is trained slice-by-slice;\nmodel checkpoint saved after each slice",
                     "Only slices after the affected one\nneed to be replayed"],
    ["Aggregation",  "Inference averages softmax outputs\nacross all S shard models",
                     "Prediction quality recovered despite\nsmaller per-shard training sets"],
]
story.append(coloured_table(sisa_data, [2.8*cm, 6.5*cm, W - 9.3*cm]))
story.append(Spacer(1, 5))

story.append(Paragraph(
    "On a forget request for data point x, only the shard containing x is partially "
    "retrained — from the last checkpoint before the slice that introduced x. "
    "The expected unlearning cost is proportional to <b>N / (S × Q)</b> instead of N "
    "(full dataset size), giving an average speedup of <b>S × Q / 2</b>. "
    "For S = Q = 5 this yields a theoretical maximum of <b>12.5× speedup</b>.",
    body_style))
story.append(Spacer(1, 6))

# ── 4. Implementation Plan ────────────────────────────────────────────
story.append(section_header("4.  Implementation Plan"))
story.append(Spacer(1, 4))

plan_data = [
    ["Phase", "Tasks", "Timeline"],
    ["Phase 1\nSetup",
     "Implement SISA engine (ShardedDataset, shard trainer,\n"
     "checkpoint manager). Load CIFAR-10. Establish baseline.",
     "Week 1–2"],
    ["Phase 2\nCore Engine",
     "Build forget-request handler; partial shard retraining;\n"
     "aggregated inference across shard models.",
     "Week 3–4"],
    ["Phase 3\nVerification",
     "Implement Membership Inference Attack (MIA) as a\n"
     "post-unlearning privacy verification metric.",
     "Week 5"],
    ["Phase 4\nWeb Dashboard",
     "Build Flask + HTML/JS interactive dashboard;\n"
     "integrate all modules; test end-to-end.",
     "Week 6–7"],
    ["Phase 5\nReport & Demo",
     "Write final report; record demo; finalise repository.",
     "Week 8"],
]
story.append(coloured_table(plan_data, [2.4*cm, W - 5.6*cm, 3.2*cm]))
story.append(Spacer(1, 5))

tools_data = [
    ["Category", "Tools & Libraries"],
    ["Language & ML",    "Python 3.10+  |  PyTorch 2.x (training, checkpointing, inference)"],
    ["Dataset",          "CIFAR-10 — 10-class colour image dataset (torchvision)"],
    ["Backend API",      "Flask 3 — RESTful endpoints for train / forget / status"],
    ["Frontend",         "HTML5, Bootstrap 5, Chart.js — interactive real-time dashboard"],
    ["Verification",     "Membership Inference Attack (threshold-based, shadow model)"],
    ["Version Control",  "GitHub — source code repository"],
]
story.append(coloured_table(tools_data, [3.0*cm, W - 3.0*cm]))
story.append(Spacer(1, 6))

# ── 5. Expected Proof of Concept ─────────────────────────────────────
story.append(section_header("5.  Expected Proof of Concept (POC)"))
story.append(Spacer(1, 4))

story.append(Paragraph(
    "The POC is an <b>interactive web dashboard</b> titled <i>SISA Unlearning Studio</i> "
    "that demonstrates the complete machine unlearning lifecycle:",
    body_style))

poc_items = [
    ("<b>SISA Configuration Panel</b>",
     "User selects number of shards (S), slices (Q), and epochs. "
     "Dashboard visualises how data is partitioned into the S × Q checkpoint grid."),
    ("<b>Live Training Console</b>",
     "One-click SISA training with real-time log showing per-shard, per-slice "
     "accuracy and elapsed time."),
    ("<b>Forget Request Console</b>",
     "User submits a dataset index for deletion. System identifies the affected shard, "
     "retrains only the necessary slices, and reports exact unlearning time."),
    ("<b>Metrics Dashboard</b>",
     "Side-by-side comparison of (a) SISA unlearning time vs. full retraining time, "
     "(b) model confidence on the forget set before and after unlearning, "
     "(c) MIA attack accuracy before and after — confirming privacy restoration."),
    ("<b>GDPR Audit Log</b>",
     "Timestamped record of all deletion requests and outcomes, "
     "simulating a real-world compliance record."),
]

for label, desc in poc_items:
    story.append(Paragraph(
        f"• {label}: {desc}",
        bullet_style))

story.append(Spacer(1, 6))

# ── Footer rule ───────────────────────────────────────────────────────
story.append(HRFlowable(width=W, thickness=1, color=colors.HexColor("#cccccc"), spaceAfter=4))
story.append(Paragraph(
    "Responsible AI — Semester Project Proposal  |  Spring 2026  |  "
    "Base Paper: Bourtoule et al., IEEE S&amp;P 2021 (arXiv:1912.03817)",
    footer_style))

# ══════════════════════════════════════════════════════════════════════
# Build PDF
# ══════════════════════════════════════════════════════════════════════
doc.build(story)
print(f"PDF generated: {OUTPUT}")
