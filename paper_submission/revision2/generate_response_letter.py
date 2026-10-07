"""
Generate detailed Response to Reviewers for Korean Journal of Family Practice (KJFP).
File: Response_to_Reviewers.docx & Response_to_Reviewers.txt
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT_DIR = r"d:\fm_paper_works\hcc-vs-hemangioma\paper_submission\revision2"
os.makedirs(OUT_DIR, exist_ok=True)

responses = [
    ("Title", "저자 답변서 (Response to Reviewers)"),
    ("Meta", "논문 제목: B-mode 복부 초음파에서 간세포암과 간혈관종 감별을 위한 딥러닝 모델의 Confidence Score와 임베딩 기반 유사도 점수: 단일기관 공개 데이터셋을 이용한 탐색적 연구\n저널: Korean Journal of Family Practice (대한가정의학회지)\n"),
    
    ("Intro", 
     "본 논문에 대한 심사위원님들의 심도 있고 건설적인 비평과 지도에 깊이 감사드립니다. "
     "심사위원님들의 지적에 전적으로 동의하며, 이를 바탕으로 원고 전반에 걸쳐 연구의 임상적 범위를 재설정하고, "
     "데이터셋 불일치와 기술적 오류를 바로잡았으며, 통계 검증과 한계점을 대폭 보강하였습니다. "
     "수정된 모든 부분은 원고 본문에 붉은색(Red)으로 표시하였습니다. "
     "아래에 각 심사의견에 대한 상세한 답변과 수정 사항을 보고드립니다.\n"),
     
    ("Reviewer_Header", "● Reviewer A 심사의견 및 답변"),
    
    ("Q", 
     "주요 코멘트 1:\n"
     "본 연구는 HCC 감시를 임상적 배경으로 제시하지만, 실제 과제는 이미 발견된 병변에서 HCC와 혈관종을 구분하는 이진 분류입니다. "
     "HCC 병변의 중앙값도 2.90 cm이며, HCC는 병리학적으로, 혈관종은 영상기준으로 진단되어 spectrum bias 가능성이 있습니다. "
     "연구의 범위를 탐색적 HCC–혈관종 감별 연구로 제한하고, 혈관종의 진단기준과 원 데이터셋의 제약을 명확히 기술해 주십시오."),
    ("A",
     "[답변]: 심사위원님의 지적에 전적으로 동의합니다.\n"
     "1) 연구 범위 재조정: 기존 논문 제목, 초록 및 서론에서 강조되었던 '간암 감시(surveillance)' 및 '조기 발견 선별'에 관한 주장을 전면 배제하고, '이미 초음파상에서 발견된 국소 병변에 대한 간세포암과 혈관종 간의 감별 진단 보조 탐색 연구'로 연구의 범위를 명확히 축소하였습니다.\n"
     "2) 참조표준 비대칭 기술: Methods 2.1절에 간세포암(HCC)은 수술적 절제 또는 생검을 통한 병리학적 확진 기준인 반면, 간혈관종(hemangioma)은 복부 영상의학과 전문의의 전형적 영상 소견 및 공식 판독 보고서를 기준으로 진단된 비침습적 영상 기준임을 명시하였습니다. 또한 원저(Tak et al. Sci Data 2026)의 데이터 선별 과정에서 확진 시점에 근접한 전형적 영상 1장이 대표로 선별되었으며, HCC 병변 직경의 중앙값도 2.90 cm로 비교적 컸음을 기술하였습니다.\n"
     "3) 편향(Spectrum/Selection bias) 고찰: Discussion 첫 번째 고찰 문단에 이러한 참조표준의 차이와 전형적 병변 선별 과정으로 인한 스펙트럼 편향 및 선택 편향을 본 연구의 가장 핵심적인 제한점으로 서술하였으며, 본 연구에서 관찰된 극단적으로 높은 정확도(99.6% 이상)가 이러한 정제된 데이터 조건에서 기인한 결과일 가능성을 명확히 고찰하였습니다.\n"
     "[수정 위치: 제목, 초록 배경, 서론 1절, 방법 2.1절, 고찰 3번째 문단]"),
     
    ("Q",
     "주요 코멘트 2:\n"
     "실제 감시 대상인 만성 B형간염 및 간경변 환자와 지방간 등 배경 간 에코가 달라지는 환경에서의 성능이 제시되지 않았습니다. "
     "관련 층화 분석과 Grad-CAM 또는 attention map을 제시하거나, 시행이 불가능하다면 모델이 배경 간의 특성을 학습했을 가능성을 핵심 제한점으로 기술해 주십시오."),
    ("A",
     "[답변]: 배경 간 실질 에코 환경의 영향에 대한 매우 중요한 지적에 감사드립니다.\n"
     "1) 층화 분석의 기술적 한계 명시: 본 연구에서 활용한 공개 데이터셋(SMC-LUD)의 제공 메타데이터에는 환자의 병리학적 병기(T stage)와 종양 크기만이 포함되어 있으며, B형/C형 간염 이환 여부, 간경변 유무, 지방간 등 배경 간 실질 질환에 관한 임상 정보가 수록되어 있지 않아 실제 층화 분석을 수행할 수 없는 원천적 제약이 있었습니다.\n"
     "2) 핵심 제한점 서술: 이에 따라 Discussion 네 번째 문단에 만성 간염, 간경변증, 지방간 등 다양한 배경 간 에코 환경에 따른 세부 평가를 시행하지 못한 점을 핵심 제한점으로 명시하였습니다. 특히 딥러닝 모델이 종양 결절 자체의 에코 특성뿐만 아니라 간세포암이 호발하는 거친 간경변 배경 실질이나 혈관종이 호발하는 정상 간 실질의 배경 음영을 일종의 단축 경로(shortcut feature)로 학습했을 가능성을 배제할 수 없음을 가감 없이 기술하였습니다.\n"
     "[수정 위치: 방법 2.1절, 고찰 4번째 문단]"),

    ("Q",
     "주요 코멘트 3:\n"
     "독립적인 외부 검증이 시행되지 않아 다른 기관, 장비 및 검사자에 대한 일반화 가능성을 평가할 수 없습니다. "
     "외부 검증을 시행하거나, 불가능하다면 임상적 적용과 일반화 성능에 관한 주장을 대폭 축소해 주십시오."),
    ("A",
     "[답변]: 외부 검증 부재에 따른 일반화 한계 지적에 깊이 공감합니다.\n"
     "본 연구는 단일 3차 의료기관의 후향적 단일 코호트에 국한된 내부 검증 연구로서, 현재 단계에서는 타 기관의 독립 외부 데이터셋이 포함되지 못하였습니다. 따라서 원고 전반에서 '우수한 일반화 성능 달성', '임상 현장 즉시 배포 가능' 등 일반화와 관련된 과장된 서술을 전면 삭제하였으며, 초록, 고찰 및 결론에 걸쳐 '타 기관, 타 초음파 기기 및 검사자 환경에서의 일반화 가능성을 입증하기 위해 향후 독립적 외부 검증이 반드시 선행되어야 한다'는 점을 명확히 한계점으로 재정립하였습니다.\n"
     "[수정 위치: 초록 결론, 서론 말미, 고찰 5번째 문단, 결론]"),

    ("Q",
     "주요 코멘트 4:\n"
     "744명의 환자가 2,656장의 영상을 제공했으므로 영상 단위 분석은 환자 내 상관성을 반영하지 못할 수 있습니다. "
     "환자 또는 병변 단위로 예측값을 집계하거나 환자 단위 cluster bootstrap을 적용해 주십시오. "
     "아울러 동일 검사에서 추출된 근접 중복 프레임이나 반복 영상이 데이터 세트 간에 분산되지 않았는지 확인해 주십시오."),
    ("A",
     "[답변]: 환자 내 상관성 및 데이터 분할의 독립성에 대한 정밀한 지적에 감사드립니다.\n"
     "1) 환자 분할(Split) 준용: 원 데이터셋 발표 논문(Tak et al. 2026)에서는 70% train, 20% validation, 10% test 분할 시 환자 단위(patient-level) 분할을 적용하여 동일 환자의 영상이 서로 다른 세트로 분산되는 데이터 누출을 방지하였다고 보고하였습니다. 본 연구는 원저에서 지정하여 제공한 Clean subset 분할(Train 1,858장, Val 530장, Test 268장)을 변경 없이 그대로 사용하였습니다.\n"
     "2) 평가 단위(Evaluation unit) 한계 기술: 다만 공개 데이터셋의 이미지 파일명에는 환자 식별 번호가 매핑되어 있지 않아, 본 연구팀이 개별 영상의 환자 교차 여부를 독립적으로 재검증하거나 환자 단위 cluster bootstrap을 수행하지는 못하였습니다. 이에 따라 원고에 제시된 모든 신뢰구간은 '영상 단위(image-level) Clopper-Pearson exact 95% CI'임을 표와 본문에 명확히 명시하였고, 동일 환자 영상 간 상관성으로 인해 실제보다 신뢰구간이 좁게 추정되었을 수 있음을 고찰에 상세히 기술하였습니다.\n"
     "[수정 위치: 방법 2.1절, 방법 2.4절, Table 3 주석, 고찰 6번째 문단]"),

    ("Q",
     "주요 코멘트 5:\n"
     "검증 세트의 민감도 99.64%, NPV 99.61% 및 F1 score 0.9982는 TP 276, FP 0, FN 1, TN 253 및 정확도 99.81%와 일치하지만, "
     "Table 3에는 TP 277, FN 0 및 정확도 100%로 제시되어 있습니다. 테스트 세트의 Table 3과 최종 Figure 2 범례는 FN 0, FP 1로 서로 일치합니다. "
     "다만 Results 3.3절의 Figure 2 설명에는 “위양성 0건, 위음성 1건”으로 반대로 기술되어 있으므로, "
     "해당 본문 문구를 수정하고 원고 전체의 성능지표를 다시 확인해 주십시오."),
    ("A",
     "[답변]: 수치 불일치를 예리하게 지적해 주셔서 대단히 감사드립니다.\n"
     "원인 확인 결과, 초기 실험에서 수행되었던 ResNet50V2 백본 모델의 결과(Val: TP 276, FN 1; cut=0.0002)와 본 연구의 주력 모델인 EfficientNetV2B0 백본 모델의 결과(Val: TP 277, FN 0; cut=0.00109)가 원고 기술 과정에서 혼용되어 발생한 중대한 전사 오류였습니다.\n"
     "원시 예측값 로그(raw_cosprobe_effnet_supcon.csv)를 바탕으로 원고 전체의 지표를 전면 재계산하여 다음과 같이 일관되게 정정하였습니다:\n"
     "- 검증 세트 (n=530): TP 277, FP 0, FN 0, TN 253 (정확도 100.0%, 민감도 100.0%, 특이도 100.0%, NPV 100.0%, F1 1.0000)\n"
     "- 테스트 세트 (n=268): TP 140, FP 1, FN 0, TN 127 (정확도 99.63%, 민감도 100.0%, 특이도 99.22%, PPV 99.29%, NPV 100.0%, F1 0.9964)\n"
     "- 3.3절 본문 문구 수정: '위양성 0건, 위음성 1건'으로 잘못 표기되었던 본문 문구를 '혈관종 영상 1장이 위양성(FP=1)으로 판정되었으며 위음성(FN=0)은 발생하지 않았다'로 바로잡았습니다. Table 2, Table 3, Figure 2 캡션의 모든 수치를 완전히 통일하였습니다.\n"
     "[수정 위치: 초록 결과, 3.2절, 3.3절, Table 2, Table 3, Figure 2 캡션]"),

    ("Q",
     "주요 코멘트 6:\n"
     "3.5절의 true HCC cases n=381은 검증 및 테스트 세트의 HCC 합계 417장이 아니라 혈관종 합계 381장과 일치합니다. "
     "이는 Figure 4, 5의 score 분포가 Table 4의 임계값 및 위음성 0건과 양립하기 어려운 이유일 수 있습니다. "
     "3.5절과 Figure 4, 5의 실제 분석대상, 클래스 라벨, score 산출 방법 및 임계값을 확인해 주십시오."),
    ("A",
     "[답변]: 매우 정밀하고 정확한 지적에 진심으로 감사드립니다.\n"
     "확인 결과, 이전 분석 스크립트에서 라벨 필터링이 누락된 채 검증 및 테스트 세트 전체 798장(HCC 417장 + 혈관종 381장)을 대상으로 분석을 수행하면서, 혈관종 수(n=381)가 HCC 표본 수로 잘못 전사되었고 두 클래스가 섞인 이봉 분포가 그래프에 표출되었음을 확인하였습니다.\n"
     "이에 따라 실제 검증 및 테스트 세트의 '진짜 간세포암 영상 전체(true HCC cases, n=417; Val 277장 + Test 140장)'만을 추출하여 원시 예측 데이터를 바탕으로 통계 및 그래프를 전면 재생성하였습니다:\n"
     "1) 분석 대상: True HCC cases (n=417)\n"
     "2) 재분석 결과 (paired t-test):\n"
     "   - HCC cosine score: EfficientNetV2B0에서 CE 단독 시 중앙값 0.882 → SupCon 적용 시 0.970으로 유의하게 상승 (p < 0.001)\n"
     "   - Δscore: EfficientNetV2B0에서 CE 단독 시 중앙값 1.508 → SupCon 적용 시 1.076으로 유의하게 감소 (p < 0.001)\n"
     "3) 그래프 재생성: Figure 4 및 Figure 5를 실제 HCC n=417 데이터에 맞추어 개별 데이터 포인트가 표시된 박스플롯으로 완전히 새로 생성하여 교체하였습니다. 3.5절 본문 설명도 해당 수치와 해석으로 전면 개정하였습니다.\n"
     "[수정 위치: 3.5절 본문, Figure 4, Figure 5]"),

    ("Q",
     "주요 코멘트 7:\n"
     "DeLong 검정의 p>0.05는 비열등성이나 통계적 동등성을 입증하지 않습니다. 특히 두 AUROC가 모두 1.000인 상황에서는 해당 검정이 실질적인 비교 정보를 제공하기 어렵습니다. "
     "비열등성 주장을 삭제하고, 주요 성능지표에는 환자 단위 bootstrap 95% 신뢰구간을 제시해 주십시오."),
    ("A",
     "[답변]: 통계적 개념에 관한 정확한 지적에 감사드리며 적극 반영하였습니다.\n"
     "1) 비열등성 주장 삭제: DeLong 검정 결과(p=1.000)를 근거로 주장했던 '비열등성(non-inferiority)' 및 '통계적 동등성' 표현을 제목, 초록, 본문 및 결론에서 전면 삭제하였습니다. 두 AUROC 곡선이 모두 1.000에 도달하는 천장 효과(ceiling effect) 환경에서는 DeLong 검정이 실질적인 판별 정보를 제공하지 못함을 Methods 2.4절과 Discussion 7번째 문단에 명시하였습니다.\n"
     "2) 95% 신뢰구간 제시: Table 3에 Clopper-Pearson exact binomial 방식을 적용하여 정확도, 민감도, 특이도에 대한 95% 신뢰구간을 산출하여 표기하였습니다 (Val: 민감도 100.0% [98.7–100.0], 특이도 100.0% [98.6–100.0]; Test: 민감도 100.0% [97.4–100.0], 특이도 99.2% [95.7–100.0]).\n"
     "[수정 위치: 초록, 방법 2.4절, 3.4절, Table 3, 고찰 7번째 문단]"),

    ("Q",
     "주요 코멘트 8:\n"
     "SupCon의 필수성은 현재 결과로 입증되지 않습니다. CE-only 모델에서도 HCC cosine score와 Δscore의 AUROC가 1.000이었고, Δscore는 SupCon 적용 전후 차이가 없었습니다. "
     "또한 동일 영상을 두 모델로 평가했다면 독립표본 t-test가 아닌 대응표본 분석을 적용해야 하며, Δscore가 다른 출력에 추가되는 증분적 이득도 제시해 주십시오."),
    ("A",
     "[답변]: 지적에 전적으로 동의하며 관련 서술과 분석을 수정하였습니다.\n"
     "1) 'SupCon 필수성' 주장 삭제: CE 단독 모델에서도 이미 cosine 지표들의 AUROC가 1.000에 달하므로, SupCon이 분류 성능 향상에 '필수적'이라는 단정적 주장을 전면 철회하였습니다. SupCon은 분류 성능 자체를 개선하기보다는 임베딩 기하 구조에서 HCC 점수의 절대적 분포를 이동시키는 특성으로 완화하여 기술하였습니다.\n"
     "2) 대응표본 분석(Paired analysis) 적용: 동일 영상에 대한 비교이므로 기존 독립표본 t-검정 대신 대응표본 t-검정(paired t-test)을 적용하여 재분석하였습니다. HCC cosine score는 paired t-test 상 p < 0.001로 유의하게 증가하였으나, Δscore는 오히려 p < 0.001로 유의하게 감소함을 확인하고 본문에 가감 없이 보고하였습니다.\n"
     "3) 증분적 이득 평가의 한계 보고: 테스트 세트(n=268)를 확인한 결과, 오분류된 단 1건의 영상(index 576, 실제 혈관종)이 confidence score, HCC cosine score, Δscore 모두에서 완전히 동일하게 위양성으로 판정되었습니다. 즉 오분류 사례 수가 극히 적어 cosine 기반 점수가 confidence score에 추가적인 증분 이득(incremental benefit)을 제공함을 현재 데이터 구조로는 증명할 수 없음을 솔직하게 고찰에 기술하였습니다.\n"
     "[수정 위치: 초록 결과/결론, 방법 2.4절, 3.4절, 3.5절, 고찰 2, 7번째 문단]"),

    ("Q",
     "마이너 코멘트 1:\n"
     "Confidence score 0.4–0.6인 사례는 운영 임계값 0.0011에 따르면 이미 모두 HCC로 분류되므로 cosine score의 “구제 효과”가 성립하지 않습니다. "
     "CE-only 모델의 7건과 SupCon 모델의 5건도 동일한 사례군이 아니므로 직접 비교할 수 없습니다. 분석 규칙을 재정의하거나 해당 분석을 삭제해 주십시오."),
    ("A",
     "[답변]: 지적해 주신 바와 같이 운영 임계값(0.00109) 기준으로는 0.4~0.6 구간의 사례들이 이미 모두 HCC로 분류되므로, 임계값 0.5를 사후적으로 가정한 기존 3.6절의 '구제 효과(salvage effect)' 분석은 논리적 정당성이 결여되어 있었습니다. 이에 심사위원님의 권고에 따라 기존 3.6절 분석 전체와 관련 고찰 문구 및 핵심 기여 항목을 원고에서 완전히 삭제하였습니다.\n"
     "[수정 위치: 기존 3.6절 전면 삭제, 고찰 내 구제 효과 문구 삭제]"),

    ("Q",
     "마이너 코멘트 2:\n"
     "Table 1은 전체 데이터셋 1,021명의 특성을 제시하므로 실제 Clean subset 744명을 대표하지 않습니다. "
     "Clean subset의 특성, 영상 제외기준 및 혈관종 병변 크기를 제시하고, 환자 수의 전사 오류 여부를 확인해 주십시오. "
     "Prototype 정의와 Figure 1의 데이터 세트 표기도 원고 전체에서 일관되게 수정해 주십시오."),
    ("A",
     "[답변]: 데이터셋 표기 및 프로토타입 정의를 다음과 같이 명확히 바로잡았습니다.\n"
     "1) Table 1 및 Table 1B 명확화: Table 1의 제목을 'Table 1. Patient and Lesion Demographics of Full Cohort (SMC-LUD, n=1,021)'로 명확히 수정하여 전체 코호트의 인구통계학적 특성임을 명시하였습니다. 원저에서 혈관종의 병변 크기가 제공되지 않았음을 각주에 기재하였습니다. 아울러 분석에 실제 사용된 Clean subset(744명, 2,656장)의 세트별 분할 현황은 Table 1B로 분리하여 명확히 대조 제시하였습니다.\n"
     "2) 영상 제외기준 정정: 원저에 기술되지 않았던 'poor echo window' 등의 주관적 제외 표현을 삭제하고, 원저 기준인 'caliper 등 주석 인공음영이 포함되지 않은 clean 영상'으로 정확히 정정하였습니다.\n"
     "3) 프로토타입 정의 통일: Methods 2.3절에 훈련 세트 평균 벡터 기반의 '단일 mean prototype'을 주 분석으로 정의하고, k-means(k=4) 서브 프로토타입은 보완적 민감도 분석임을 명시하였습니다.\n"
     "4) Figure 1 표기 정정: Figure 1 캡션을 실제 그래프와 일치하도록 훈련 및 검증 세트(Train+Val) 임베딩 분포 및 HCC(red) / Hemangioma(teal) 표기로 일관되게 수정하였습니다.\n"
     "[수정 위치: 방법 2.1절, 2.3절, Table 1, Table 1B, Figure 1 캡션]"),

    ("Q",
     "마이너 코멘트 3:\n"
     "원고의 구성과 표, 그림 제시 형식을 정리해 주십시오. "
     "Introduction과 Discussion의 불필요한 소제목을 삭제하고, Results에서는 Methods의 반복을 줄여 핵심 결과만 기술하는 것이 적절합니다. "
     "본문에 독립적으로 삽입된 표, 그림 제목과 별표 설명은 해당 표, 그림의 제목 또는 주석으로 이동해 주십시오."),
    ("A",
     "[답변]: 논문 양식 및 가독성 개선 권고에 감사드립니다.\n"
     "1) 소제목 삭제: Introduction(1.1, 1.2 등) 및 Discussion(4.1~4.7) 내 불필요한 번호형 소제목을 모두 삭제하고 자연스러운 문단 흐름으로 재구성하였습니다.\n"
     "2) Results 간소화: Results 절에서 Methods 내용의 중복 기술을 대폭 삭제하고 핵심적인 정량 결과 위주로 간결하게 서술하였습니다.\n"
     "3) 표 및 그림 캡션 정리: 본문 중간에 독립적으로 삽입되어 있던 표/그림 제목과 설명 문구들을 모두 해당 표의 상단 제목 및 하단 각주(footnote)와 그림 캡션(caption)으로 일괄 이동 정리하였습니다.\n"
     "[수정 위치: 원고 전반 (서론, 결과, 고찰, 표 및 그림 문서)]"),
     
    ("Reviewer_Header", "● Reviewer B 심사의견 및 답변"),
    
    ("Q",
     "심사의견:\n"
     "논문 양식에 맞추어 작성 부탁드리며, 전반적인 내용도 전달하고자 하는 바를 명확하게 논리적으로 서술하여 정리해야 겠습니다."),
    ("A",
     "[답변]: 심사위원님의 고견에 감사드립니다.\n"
     "1) 저널 투고 양식 준수: Korean Journal of Family Practice (대한가정의학회지)의 공식 논문 투고 규정에 맞추어 원고의 전반적인 구성을 표준 원저(Original Article) 양식으로 정비하였습니다. 구조화된 국문 초록(연구 배경, 방법, 결과, 결론), 본문 구성(서론, 대상 및 방법, 결과, 고찰, 결론), 밴쿠버 양식의 최신 참고문헌 번호 체계, 그리고 본문 말미의 표 및 그림 정렬 형식을 엄격히 적용하였습니다.\n"
     "2) 논리적 서술 및 전달력 강화: 논문의 핵심 메시지가 '간암 감시 선별'이라는 지나치게 광범위한 주장으로 분산되지 않도록, 'B-mode 초음파에서 발견된 국소 병변의 HCC-혈관종 감별을 위한 이중 출력 영상표지자의 기술적 타당성 평가'라는 하나의 명확한 주제로 서술 논리를 일관되게 재편하였습니다. 불필요한 수식어와 과장된 해석을 지양하고, 실제 실험 데이터에 근거한 객관적 서술과 엄밀한 제한점 고찰로 연구의 전달력을 높였습니다.\n"
     "[수정 위치: 원고 전체 구조 및 문맥 재정비]")
]

# Write plain text
txt_path = os.path.join(OUT_DIR, "Response_to_Reviewers.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    for kind, text in responses:
        if kind == "Title":
            f.write(f"=== {text} ===\n\n")
        elif kind == "Reviewer_Header":
            f.write(f"\n\n=======================================================\n{text}\n=======================================================\n\n")
        elif kind == "Q":
            f.write(f"-------------------------------------------------------\n{text}\n-------------------------------------------------------\n")
        elif kind == "A":
            f.write(f"{text}\n\n")
        else:
            f.write(f"{text}\n\n")

print(f"Saved plain text response to: {txt_path}")

# Write docx
doc = docx.Document()
for s in doc.sections:
    s.top_margin = Inches(1.0)
    s.bottom_margin = Inches(1.0)
    s.left_margin = Inches(1.0)
    s.right_margin = Inches(1.0)

BLUE = RGBColor(0, 51, 153)
DARK_GRAY = RGBColor(80, 80, 80)

for kind, text in responses:
    p = doc.add_paragraph()
    
    if kind == "Title":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.name = "Malgun Gothic"
        run.font.size = Pt(16)
        run.font.bold = True
        
    elif kind == "Meta":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(14)
        run = p.add_run(text)
        run.font.name = "Malgun Gothic"
        run.font.size = Pt(10)
        run.font.italic = True
        
    elif kind == "Intro":
        p.paragraph_format.line_spacing = 1.3
        p.paragraph_format.space_after = Pt(12)
        run = p.add_run(text)
        run.font.name = "Malgun Gothic"
        run.font.size = Pt(10)
        
    elif kind == "Reviewer_Header":
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(8)
        run = p.add_run(text)
        run.font.name = "Malgun Gothic"
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = BLUE
        
    elif kind == "Q":
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.left_indent = Inches(0.2)
        run = p.add_run(text)
        run.font.name = "Malgun Gothic"
        run.font.size = Pt(9.5)
        run.font.bold = True
        run.font.color.rgb = DARK_GRAY
        
    elif kind == "A":
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(10)
        p.paragraph_format.line_spacing = 1.3
        run = p.add_run(text)
        run.font.name = "Malgun Gothic"
        run.font.size = Pt(9.5)

docx_path = os.path.join(OUT_DIR, "Response_to_Reviewers.docx")
doc.save(docx_path)
print(f"Saved docx response to: {docx_path}")
