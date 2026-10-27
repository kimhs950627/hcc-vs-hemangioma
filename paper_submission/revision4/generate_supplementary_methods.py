"""Supplementary Methods (S1-S6) for KJFP-26-0060, revision 4 (2nd revision).

Moves the implementation-level detail out of the main Methods so the main text
can stay at study-design level (editorial comment 5). Every numeric value below
was read from the training code, not from memory:

  classification_with_supcon.ipynb        architecture + stage-2 protocol
  full_training_notebook.ipynb            stage-1 / stage-2 integrated config
  NNCLR_ConvHybrid_run.ipynb              NNCLR stage-1 settings
  cosine_probe_result/raw_cosprobe_*.csv  per-case scores used for the figures

Items that could not be confirmed from code are written as "확인 필요" instead of
guessed.

Run: python3 generate_supplementary_methods.py
"""

import os
import pathlib

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT_DIR = str(pathlib.Path(__file__).resolve().parent)
BLUE = RGBColor(0, 0, 220)
BLACK = RGBColor(20, 20, 20)

BLOCKS = [
    ("title", "Supplementary Methods"),
    ("subtitle", "B-mode 복부 초음파에서 간세포암과 간혈관종 감별을 위한 이중 출력 지표 (KJFP-26-0060, 2차 수정)"),

    ("h1", "S1. 모델 아키텍처 (구현 상세)"),
    ("p", "본문 2.2절에서 기술한 두 백본은 모두 ImageNet 사전학습 가중치에서 시작하여 전 층을 미세조정하였다. 합성곱 백본의 출력 stride는 32이며, 384×384 입력에서 12×12 크기의 특징 맵이 얻어진다. 이 특징 맵을 144개 패치 토큰으로 재구성하고 분류 토큰(CLS token) 1개를 앞에 붙여 길이 145의 시퀀스를 구성한 뒤, sinusoidal 위치 인코딩을 더하였다. 트랜스포머 인코더는 4개 블록, 모델 차원 d=128, 8개 헤드(key dimension 32), MLP 차원 384, dropout 0.1로 구성하였다. 풀링은 CLS 토큰 방식이며, 이 128차원 표현 벡터에서 분류 헤드(은닉 128, 출력 2)와 프로젝션 헤드(출력 128)가 각각 분기한다. 표 S1에 항목별 값을 정리하였다."),
    ("table", [
        ["항목", "값", "비고"],
        ["백본", "EfficientNetV2B0 또는 ResNet50V2", "ImageNet 사전학습, 전 층 학습"],
        ["입력 해상도", "384 × 384 × 3", "RGB 3채널로 복제"],
        ["특징 맵", "12 × 12 (stride 32)", "패치 토큰 144개"],
        ["시퀀스 길이", "145 (CLS 1 + 패치 144)", "sinusoidal 위치 인코딩"],
        ["인코더", "4 블록, d=128, 헤드 8 (key dim 32), MLP 384", "dropout 0.1"],
        ["풀링", "CLS 토큰", "임베딩 차원 d=128"],
        ["분류 헤드", "은닉 128 → 출력 2", "softmax"],
        ["프로젝션 헤드", "출력 128", "SupCon 손실 계산용"],
    ]),

    ("h1", "S2. 학습 프로토콜"),
    ("p", "2단계 분류 학습은 384×384 입력, 배치 16, 총 100 epoch로 수행하였다. 옵티마이저는 AdamW(clipvalue 1.0, EMA 가중치 사용)이며, 학습률은 CosineDecay 스케줄(초기 1e-6, warmup 목표 5e-5, warmup 10 epoch, alpha 0.1)을 적용하였다. 초음파 영상에 맞춘 증강은 무작위 자르기(최소 면적비 0.4), 밝기 ±0.2, 대비 0.3, 가우시안 잡음 0.05, 감마 0.7–1.4로 구성하였다. 난수 시드는 42로 고정하였다. 표 S2에 요약하였다."),
    ("table", [
        ["항목", "값"],
        ["배치 크기", "16"],
        ["Epoch (2단계 분류 학습)", "100"],
        ["옵티마이저", "AdamW, clipvalue 1.0, EMA 사용"],
        ["학습률 스케줄", "CosineDecay: 1e-6 → 5e-5, warmup 10 epoch, alpha 0.1"],
        ["증강", "crop(최소 0.4), 밝기 ±0.2, 대비 0.3, 잡음 0.05, 감마 0.7–1.4"],
        ["난수 시드", "42"],
    ]),

    ("h1", "S3. 손실함수와 비교 조건"),
    ("p", "비교는 백본 2종(EfficientNetV2B0, ResNet50V2)과 학습 방식 3종의 6개 조합으로 구성하였다. 학습 방식은 (1) 교차 엔트로피 단독(CE only), (2) 교차 엔트로피와 지도 대조손실의 결합(CE+SupCon), (3) NNCLR 자기지도 사전학습 후 CE+SupCon 미세조정(NNCLR)이다. CE+SupCon 조건의 총 손실은 L = L_CE + 0.5 × L_SupCon 으로 정의하였고, 지도 대조손실의 온도는 0.07, 프로젝션 차원은 128이었다. NNCLR 조건을 비교에 포함한 이유는 라벨이 수천 장 규모로 제한된 데이터에서 자기지도 사전학습이 표현 학습을 보완할 수 있다는 가설을 확인하기 위해서이며, 이 가설은 본 연구의 검증 세트 결과에서 지지되지 않았다(Table 2). NNCLR 조건의 1단계 사전학습은 NT-Xent 온도 0.1, support queue 8,192, 프로젝션 차원 256, 배치 32로 수행하였고, 2단계에서 CE+SupCon으로 미세조정하였다. 표 S3에 조건별 목적을 정리하였다."),
    ("table", [
        ["조건", "이 조건으로 확인하려 한 것"],
        ["CE only", "표준 지도학습 기준선"],
        ["CE + SupCon", "클래스 내 응집을 강화했을 때 표현 기하와 유사도 점수가 달라지는가"],
        ["NNCLR 후 CE + SupCon", "라벨 없는 사전학습이 이 규모의 데이터에서 추가 이득을 주는가"],
        ["백본 교체(EfficientNetV2B0 ↔ ResNet50V2)", "관찰된 현상이 특정 백본에 국한되는가"],
    ]),

    ("h1", "S4. 프로토타입과 유사도 점수의 산출"),
    ("p", "모든 프로토타입은 훈련 세트만으로 산출하여 검증·테스트 세트와의 데이터 유출을 차단하였다. 평균 프로토타입은 훈련 세트의 클래스별 임베딩(각 128차원)을 클래스 단위로 평균한 단일 벡터이다. k-means 민감도 분석에서는 훈련 세트 HCC 임베딩에 k-means(k=4)를 적용하여 4개의 서브 프로토타입을 얻고, 대상 병변과 4개 서브 프로토타입 간 코사인 유사도의 평균을 점수로 사용하였다. 코사인 유사도 점수는 대상 병변 임베딩 a와 클래스 프로토타입 p에 대하여 cos(a, p) = (a·p) / (|a||p|) 로 계산하며, HCC 코사인 유사도 점수는 cos(a, p_HCC), Hemangioma 코사인 유사도 점수는 cos(a, p_hem), Δscore는 cos(a, p_HCC) − cos(a, p_hem)으로 정의된다. 값의 범위는 −1에서 1이고 1에 가까울수록 해당 클래스의 전형에 가깝다. k-means 민감도 분석에서도 같은 식을 4개 서브 프로토타입 각각에 적용한 뒤 평균하였다. cosine score는 병변 단위로 한 번만 산출되며, 임계값·프로토타입은 어떤 경우에도 검증·테스트 세트에서 재산출하지 않았다."),

    ("h1", "S5. 임계값·등가 구간·통계 절차"),
    ("p", "확률 지표(ROC-A)의 운영점은 관행값 0.50으로 사전 지정하였다. 코사인 유사도 점수(ROC-B, ROC-C)의 임계값은 검증 세트에서 Youden's J로 결정하고 테스트 세트에는 변경 없이 적용하였다. 검증 세트의 AUROC가 1.000에 도달하여 J가 평탄해지면 최적 임계값이 유일하게 정해지지 않으므로, 같은 검증 혼동행렬을 재현하는 임계값의 범위를 등가 구간으로 함께 산출하였다. 등가 구간의 하한은 음성으로 분류된 증례의 최고 점수, 상한은 양성으로 분류된 증례의 최저 점수이며, 구현에서는 이 구간의 위쪽 경계를 임계값으로 채택하였다. 구간 계산은 equivalence_interval.py로 재현 가능하다."),
    ("p", "변별력은 AUROC로 평가하였고, 곡선 간 비교는 DeLong 검정으로 계산하였으나 전 지표가 천장에 도달한 조건에서는 정보량이 없으므로 우월성·비열등성 주장에 사용하지 않았다. 민감도·특이도·정확도의 95% 신뢰구간은 Wilson score 방식, 운영점 간 불일치 검정은 McNemar exact test로 산출하였다. CE 단독 모델과 CE+SupCon 모델의 코사인 점수 분포 비교는 동일 영상 대응표본 설계로 paired t-test, Wilcoxon signed-rank test(중심 위치 이동), Pitman-Morgan test(대응 분산 변화)를 적용하였고, 유의수준은 양측 p < 0.05로 정의하였다."),

    ("h1", "S6. 재현 환경"),
    ("p", "학습과 평가는 Keras/TensorFlow 기반으로 수행하였고, 임베딩 시각화(t-SNE)는 perplexity 30, 2차원, 시드 42 설정을 사용하였다. 혼동행렬과 그림 2의 수치는 테스트 세트 268장의 케이스별 점수 파일(cosine_probe_result/raw_cosprobe_effnet_supcon.csv)에서 직접 재계산되며, 본 부록에 기재한 실행 설정과 함께 제공된다. 원 학습 노트북과 평가 스크립트는 요청 시 제공한다."),
    ("h1", "S7. 보조 분석: SupCon 전후 코사인 유사도 점수의 십분위 변화"),
    ("p", "본문 3.5절의 분포 변화를 십분위로 세분한 결과이다. 실제 HCC 영상 417장을 CE 단독 모델의 코사인 유사도 점수로 정렬해 십분위로 나누고, 동일 영상의 CE+SupCon 점수를 대응시켰다. 상승 폭은 1분위 +0.441에서 10분위 +0.024로 단조 감소하였다. 점수 구간별 빈도는 0.5 미만이 43장(10.3%)에서 17장(4.1%), 0.6 미만 57장에서 21장, 0.7 미만 80장(19.2%)에서 34장(8.2%), 0.8 미만 133장(31.9%)에서 49장(11.8%)으로 감소하였다. 확률 0.50에서 오분류된 HCC 6장의 코사인 유사도 점수는 0.26–0.55(전체 HCC의 하위 0–4백분위)였고, 테스트 세트에서 HCC의 최저값은 0.258, 혈관종의 최고값은 0.032로 두 분포는 겹치지 않았다. 한편 Δscore가 음수인 증례(임베딩상 혈관종 쪽으로 기운 HCC)는 16장에서 20장으로 늘어, 저점수 증례의 상승이 두 프로토타입 사이의 상대적 위치 개선을 뜻하지는 않았다. 십분위별 값은 표 S4와 같다."),
    ("table", [
        ["사전 점수 십분위", "영상 수", "사전 평균", "사후 평균", "평균 변화량"],
        ["1 (최저)", "41", "0.125", "0.566", "+0.441"],
        ["2", "41", "0.620", "0.868", "+0.248"],
        ["3", "41", "0.744", "0.899", "+0.155"],
        ["4", "41", "0.815", "0.919", "+0.104"],
        ["5", "41", "0.859", "0.950", "+0.092"],
        ["6", "41", "0.891", "0.957", "+0.066"],
        ["7", "41", "0.914", "0.971", "+0.056"],
        ["8", "41", "0.930", "0.976", "+0.047"],
        ["9", "41", "0.943", "0.979", "+0.036"],
        ["10 (최고)", "48", "0.959", "0.983", "+0.024"],
    ]),

    ("note", "본 부록의 값은 학습 코드에서 확인된 값만 기재하였다. 코드에서 확인하지 못한 항목은 임의로 채우지 않았다."),
]

doc = docx.Document()
for s in doc.sections:
    s.top_margin = Inches(1.0)
    s.bottom_margin = Inches(1.0)
    s.left_margin = Inches(1.0)
    s.right_margin = Inches(1.0)

for kind, payload in BLOCKS:
    if kind == "title":
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(payload)
        r.font.name = "Malgun Gothic"
        r.font.size = Pt(16)
        r.font.bold = True
        r.font.color.rgb = BLUE
    elif kind == "subtitle":
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(payload)
        r.font.name = "Malgun Gothic"
        r.font.size = Pt(10)
        r.font.color.rgb = BLACK
        doc.add_paragraph()
    elif kind == "h1":
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(payload)
        r.font.name = "Malgun Gothic"
        r.font.size = Pt(12)
        r.font.bold = True
        r.font.color.rgb = BLUE
        p.paragraph_format.keep_with_next = True
    elif kind in ("p", "note"):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(payload)
        r.font.name = "Malgun Gothic"
        r.font.size = Pt(9.5)
        r.font.color.rgb = BLACK
        if kind == "note":
            r.font.size = Pt(8.5)
            r.font.italic = True
    elif kind == "table":
        rows = payload
        t = doc.add_table(rows=len(rows), cols=len(rows[0]))
        t.style = "Table Grid"
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                cell = t.rows[i].cells[j]
                cell.text = val
                for run in cell.paragraphs[0].runs:
                    run.font.name = "Malgun Gothic"
                    run.font.size = Pt(8.5)
                    run.font.bold = (i == 0)
        doc.add_paragraph()

docx_path = os.path.join(OUT_DIR, "Supplementary_Methods_v1.docx")
doc.save(docx_path)

txt_path = os.path.join(OUT_DIR, "Supplementary_Methods_v1.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    for kind, payload in BLOCKS:
        if kind == "title":
            f.write(f"# {payload}\n\n")
        elif kind == "subtitle":
            f.write(f"{payload}\n\n")
        elif kind == "h1":
            f.write(f"\n## {payload}\n\n")
        elif kind in ("p", "note"):
            f.write(f"{payload}\n\n")
        elif kind == "table":
            for row in payload:
                f.write("| " + " | ".join(row) + " |\n")
            f.write("\n")

print("wrote", docx_path)
print("wrote", txt_path)
