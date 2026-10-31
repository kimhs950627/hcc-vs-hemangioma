"""Build the figure submission files: figures_v5.docx and figures_v5.pptx.

KJFP Instructions for Authors require figures to be submitted as separate
files of 600 dpi or more in TIF, JPEG or PowerPoint format, and the e-Submission
steps say all images should be inserted into a PowerPoint file. Both vehicles
are therefore produced from the same images and captions:

  figures_v5.docx  one figure per A4 page, caption underneath (reading copy)
  figures_v5.pptx  one image per slide, in body citation order (upload vehicle)

Images are the 600 dpi delivery files in figures_kjfp/. Display width is capped
so the effective resolution never drops below 600 dpi:
width_cm = min(16.0, pixels / 600 * 2.54).

Captions come from captions_v5.py, the same source used by the manuscript, so
the two cannot drift apart. No revision number or working-path text appears in
either file.
"""
import pathlib

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.shared import Cm, Pt
from PIL import Image
from pptx import Presentation
from pptx.util import Cm as PCm, Pt as PPt

from captions_v5 import CAPTIONS

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / "figures_kjfp"

FILES = {
    1: "Figure1.jpg",
    2: "Figure2A-C.jpg",
    3: "Figure3.jpg",
    4: "Figure4.jpg",
    5: "Figure5.jpg",
}

DPI = 600
MAX_WIDTH_CM = 16.0
GREY = "#000000"


def width_cm(path):
    px = Image.open(path).size[0]
    return min(MAX_WIDTH_CM, px / DPI * 2.54)


# ---------------------------------------------------------------- docx
doc = docx.Document()
for s in doc.sections:
    s.page_width = Cm(21.0)
    s.page_height = Cm(29.7)
    s.top_margin = s.bottom_margin = Cm(2.0)
    s.left_margin = s.right_margin = Cm(2.5)

head = doc.add_paragraph()
head.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = head.add_run("Figures")
r.font.name = "Batang"
r.font.size = Pt(14)
r.font.bold = True

for idx, (title, desc, _rev) in enumerate(CAPTIONS, start=1):
    path = FIG / FILES[idx]
    if idx > 1:
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    ip = doc.add_paragraph()
    ip.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ip.paragraph_format.space_after = Pt(8)
    ip.add_run().add_picture(str(path), width=Cm(width_cm(path)))

    cp = doc.add_paragraph()
    cp.paragraph_format.space_before = Pt(4)
    cp.paragraph_format.space_after = Pt(4)
    cp.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r1 = cp.add_run(title + "\n")
    r1.font.name = "Batang"
    r1.font.size = Pt(10)
    r1.font.bold = True
    r2 = cp.add_run(desc)
    r2.font.name = "Batang"
    r2.font.size = Pt(9)

out_docx = HERE / "figures_v5.docx"
doc.save(str(out_docx))

# ---------------------------------------------------------------- pptx
prs = Presentation()
prs.slide_width = PCm(33.867)   # 13.33 in
prs.slide_height = PCm(19.05)   # 7.5 in
blank = prs.slide_layouts[6]
for idx, (title, _desc, _rev) in enumerate(CAPTIONS, start=1):
    path = FIG / FILES[idx]
    slide = prs.slides.add_slide(blank)

    lab = slide.shapes.add_textbox(PCm(1.0), PCm(0.6), PCm(20.0), PCm(1.2))
    p = lab.text_frame.paragraphs[0]
    run = p.add_run()
    run.text = title.split(". ")[0] + "."
    run.font.size = PPt(14)
    run.font.bold = True

    slide_w = 31.867
    slide_h = 17.0
    w_cm = width_cm(path)
    w_cm = min(w_cm, slide_w)
    px_w, px_h = Image.open(path).size
    h_cm = w_cm * px_h / px_w
    if h_cm > slide_h:
        h_cm = slide_h
        w_cm = h_cm * px_w / px_h
    left = (33.867 - w_cm) / 2
    top = 2.1 + (slide_h - h_cm) / 2
    slide.shapes.add_picture(str(path), PCm(left), PCm(top), PCm(w_cm), PCm(h_cm))

out_pptx = HERE / "figures_v5.pptx"
prs.save(str(out_pptx))

print(f"docx: {out_docx.name} ({out_docx.stat().st_size} bytes)")
print(f"pptx: {out_pptx.name} ({out_pptx.stat().st_size} bytes)")
for idx in FILES:
    path = FIG / FILES[idx]
    w = width_cm(path)
    px_w, px_h = Image.open(path).size
    print(f"  Figure {idx}: {px_w}x{px_h}px  display {w:.2f} cm "
          f"({px_w / (w / 2.54):.0f} dpi)")
