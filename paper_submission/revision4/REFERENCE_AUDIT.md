# 참고문헌 대조 결과 — KJFP-26-0060, 2차 수정 (Editorial comment 7)

검증 도구: Crossref REST API(`api.crossref.org/works/<DOI>`) + PubMed E-utilities. 확인일 2026-10-27.
판정: **확인** = 서지정보 원문 일치 / **수정** = 문헌은 존재하나 서지정보가 틀림 / **교체** = 문장을 직접 뒷받침하지 않음.

## 1. 지적된 3건

| # | 현재 기재 | 판정 | 근거 | 조치 |
|---|---|---|---|---|
| 6 | Chou CT, Chen YL, Wu HK, Chen RC. Atypical hemangioma mimicking mixed hepatocellular cholangiocarcinoma. Medicine (Baltimore). 2017;96(50):e9192. | **수정** | DOI 10.1097/MD.0000000000009192 — 제목·저널·권·호·e번호는 맞으나 **저자가 다름**. 실제 저자: Lin S, Zhang L, Li M, Cheng Q, Zhang L, Zheng S. | 저자 교체 → `Lin S, Zhang L, Li M, et al.` |
| 7 | Liu S, Wang Y, Yang X, et al. Deep Learning in Medical Ultrasound Analysis: A Review. Engineering. 2019;5(2):261-275. | **교체** | DOI 10.1016/j.eng.2018.11.020 — 서지정보 자체는 정확하나, 이 문헌은 **의료초음파 딥러닝 전반의 리뷰**로 "비전형 혈관종의 B-mode 소견"을 뒷받침하지 못함(심사자가 지적한 바로 그 문제). | 혈관종 B-mode 소견을 직접 다룬 문헌으로 교체 |
| 8 | American College of Radiology. CT/MRI LI-RADS v2018. ACR; 2018. | **교체** | LI-RADS는 **CT/MRI 조영증강 기반** 체계로, B-mode 초음파 소견 문장의 근거가 될 수 없음. | 위와 동일하게 교체 |

## 2. 교체 문헌 (B-mode 초음파 비전형 혈관종)

- **Moody AR, Wilson SR.** Atypical hepatic hemangioma: a suggestive sonographic morphology. *Radiology.* 1993;188(2):413-417. DOI 10.1148/radiology.188.2.8327687 — PMID 8327687.
  → 비전형 혈관종의 **B-mode 형태학**을 직접 다룬 고전 문헌.
- **Choi JC, Baek YH, Jeong JS, Lee SW, Han SY, Cho JH.** Discrete hypoechoic ring in hepatic cavernous hemangioma resembling a malignant tumor: correlation with histologic features. *Gut Liver.* 2009;3(3):226-230. DOI 10.5009/gnl.2009.3.3.226 — PMID 20431752.
  → 원고 문장의 "저에코 테두리를 갖는 비전형 혈관종"과 정확히 대응.

두 문헌 모두 PubMed/Crossref 원문 서지 확인 완료. 교체 후에도 인용 번호는 7·8로 유지되어 본문 인용 번호 변경은 없음(9번 이후 번호 불변).

## 3. 전수 서지 재확인 (17건)

| # | 확인 | 비고 |
|---|---|---|
| 1 | 확인 | GLOBOCAN 2020, CA Cancer J Clin 2021;71(3):209-249 |
| 2 | 확인 | EASL 2018 HCC guideline, J Hepatol 69(1):182-236 |
| 3 | 확인 | KLCA-NCC 2022, Clin Mol Hepatol 28(4):583-705 |
| 4 | 확인 | Minami 2023, Liver Cancer 12(2):103-115 (온라인 2022-12-06, 권호 2023) |
| 5 | 확인(부분) | Med Ultrason 2015;17(4):444-450 — DOI 10.11152/mu.2013.2066.174.csu; Crossref에 페이지 정보 없음, PubMed 페이지 444-50과 일치 |
| 6 | **수정** | 저자 교체(위 §1) |
| 7 | **교체** | Moody & Wilson 1993 |
| 8 | **교체** | Choi JC 2009 |
| 9 | 확인 | Yang Q 2020, EBioMedicine 56:102777, DOI 10.1016/j.ebiom.2020.102777 |
| 10 | 확인 | Du Z 2025, eClinicalMedicine 81:103098 |
| 11 | 확인 | Guo C 2017, ICML PMLR 70:1321-1330 (DOI 없음, 절차록) |
| 12 | 확인 | Khosla P 2020, NeurIPS 33 (절차록) |
| 13 | 확인 | Tak J 2026, Sci Data 13:649, DOI 10.1038/s41597-026-07023-7 — 단, Crossref 기준 권 13, 호 1, **문서번호 미표기**(원고는 13:649로 기재 → 게재본 확인 후 확정 필요) |
| 14 | 확인 | Tan M, Le QV 2021, ICML (절차록) |
| 15 | 확인 | DeLong 1988, Biometrics 44(3):837-845 |
| 16 | 확인 | Sambyal AS 2023, Comput Methods Programs Biomed 242:107816 |
| 17 | **수정** | 원고 기재 `PLoS One. 2022;17(12):e0278455`는 **오류**. DOI 10.1371/journal.pone.0278455는 각막원추증 논문. 실제: **PLoS One. 2022;17(1):e0262838**, DOI 10.1371/journal.pone.0262838 |

## 4. 남은 확인 1건

- **ref 13(SMC-LUD)**: Crossref에는 권 13, 호 1, 페이지/문서번호 미표기로 나온다. 원고 기재 `13:649`는 PubMed 레코드와 함께 확인이 필요하다. 게재본(PDF) 확보 후 확정한다.
- **ref 11·12·14(학회 절차록)**: DOI가 없는 Proceedings 인용이라 서지 확인은 학회 proceedings 페이지 기준으로만 가능. 현재 표기는 통용 표기와 일치.

## 5. 함께 손본 것

- 참고문헌 6·7·8·17은 `manuscript_v4_marked.docx`에서 **파란색**으로 표시됨(이번 라운드 규칙).
- Introduction의 `[7,8]` 인용 위치는 그대로 두고 문헌만 교체했으므로, 본문 인용 번호 체계는 1-17로 불변.

## 6. 재배열 반영 (2026-10-27)

인용 적합성 점검 결과에 따라 최종 18개로 재배열했다. 본 문서 §1~§3의 번호는 **재배열 전** 번호이며, 최종 대응은 다음과 같다.

| 최종 # | 문헌 | 재배열 전 # |
|---|---|---|
| 5 | Rapaccini GL, et al. Liver Int. 2004;24(2):124-130 (신규) | — |
| 6 | Moody AR, Wilson SR. Radiology. 1993;188(2):413-417 | 7 |
| 7 | Choi JC, et al. Gut Liver. 2009;3(3):226-230 | 8 |
| 8 | Lin S, et al. Medicine. 2017;96(50):e9192 | 6 |
| 9 | Sirli R, et al. Med Ultrason. 2015;17(4):444-450 | 5 |
| 10–18 | Yang / Du / Guo / Khosla / Tak / Tan / DeLong / Sambyal / Rajaraman | 9–17 |

본문 인용도 함께 바뀌었다: 서론 3문장 `[4]`, 고에코 HCC `[5]`, 비전형 혈관종 `[6,7,8]`, CEUS `[9]`, 딥러닝 `[10,11]`.
