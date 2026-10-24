"""Generate the KJFP cover letter draft (docx + txt).

One page. States the title, that this is a resubmission after review, the three
changes the revision makes, and the standard declarations. Bracketed fields are
the ones only the authors can fill (corresponding author, ORCID, date, phone).

Run: python3 generate_cover_letter_v1.py
"""

from __future__ import annotations

import pathlib

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

OUT_DIR = pathlib.Path(__file__).resolve().parent

TITLE_KO = (
    "B-mode 복부 초음파에서 간세포암과 간혈관종 감별을 위한 딥러닝 모델의 "
    "Confidence Score와 임베딩 기반 유사도 점수: 단일기관 공개 데이터셋을 이용한 탐색적 연구"
)

PARAGRAPHS: list[tuple[str, str]] = [
    ("head", "대한가정의학회지 편집위원회 귀중"),
    ("body", "안녕하십니까. 아래 원고를 대한가정의학회지(Korean Journal of Family Practice)에 재제출합니다."),
    ("meta", f"제목: {TITLE_KO}"),
    ("meta", "투고 유형: 원저(Original Article) · 재제출(이전 심사 의견 반영)"),
    ("meta", "저자: 김현수¹, [공저자 이름·소속]¹ · 교신저자: [교신저자 이름] ([이메일])"),
    ("meta", "소속: ¹예수병원 가정의학과"),
    ("body", "이전 심사에서 지적하신 내용을 중심으로 다음 세 가지를 수정하였습니다."),
    ("list", "1. 검증 세트에서의 모델 비교를 실제 미학습 영상(n=417 HCC, n=381 혈관종)에 대한 대응표본 분석으로 바꾸고, 중심값 이동(Wilcoxon)과 분산 변화(Pitman-Morgan)를 함께 보고하였습니다."),
    ("list", "2. 두 닮음 지표의 정의와 산출식을 2.4절에 명시하고, 확률로 읽어서는 안 된다는 점을 본문과 표 각주에 적었습니다."),
    ("list", "3. 운영점은 관행값 0.50으로 미리 정하고, 닮음 지표의 임계값은 검증 세트에서만 유도하여 테스트 세트에는 변경 없이 적용한 탐색적 분석임을 밝혔습니다. 검증 세트 AUROC가 1.000이어서 임계값이 하나로 정해지지 않으므로, 같은 판정을 내는 등가 구간을 함께 제시하였습니다."),
    ("body", "세 출력의 테스트 AUROC가 모두 1.000이므로 지표 간 우열은 판정할 수 없으며, 본 원고는 우월성이나 비열등성을 주장하지 않습니다. 판정이 갈린 증례 수(6 대 1)의 차이도 통계적으로 유의하지 않았습니다(McNemar exact p = 0.125)."),
    ("body", "이 원고는 모든 저자가 내용을 확인하고 제출에 동의하였습니다. 이해상충은 없으며, 삼성서울병원이 공개한 익명 데이터셋(SMC-LUD)을 사용하여 별도의 기관생명윤리위원회 승인이 필요하지 않았습니다. 분석 코드와 재현 절차는 요청 시 제공할 수 있습니다."),
    ("body", "바쁘신 중에 검토해 주셔서 감사합니다."),
    ("sign", "김현수 드림"),
    ("sign", "예수병원 가정의학과 · [연락처] · [이메일]"),
    ("sign", "[날짜]"),
]


def build_txt() -> str:
    lines = []
    for kind, text in PARAGRAPHS:
        lines.append(text)
        if kind in {"head", "sign"}:
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_docx() -> docx.Document:
    doc = docx.Document()
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    for kind, text in PARAGRAPHS:
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = "Malgun Gothic"
        if kind == "head":
            run.font.size = Pt(11)
            run.font.bold = True
            p.paragraph_format.space_after = Pt(12)
        elif kind == "meta":
            run.font.size = Pt(10)
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.space_after = Pt(2)
        elif kind == "list":
            run.font.size = Pt(10)
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.line_spacing = 1.3
            p.paragraph_format.space_after = Pt(4)
        elif kind == "sign":
            run.font.size = Pt(10)
            p.paragraph_format.space_after = Pt(2)
        else:
            run.font.size = Pt(10)
            p.paragraph_format.line_spacing = 1.4
            p.paragraph_format.space_after = Pt(8)

    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    return doc


def main() -> None:
    txt_path = OUT_DIR / "Cover_Letter_v1.txt"
    txt_path.write_text(build_txt(), encoding="utf-8")
    docx_path = OUT_DIR / "Cover_Letter_v1.docx"
    try:
        build_docx().save(docx_path)
    except PermissionError:
        docx_path = OUT_DIR / "Cover_Letter_v1_updated.docx"
        build_docx().save(docx_path)
        print("WARNING: Cover_Letter_v1.docx is locked (open in Word); wrote the _updated copy")
    print(f"wrote {txt_path.name} and {docx_path.name} ({len(build_txt())} chars)")


if __name__ == "__main__":
    main()
