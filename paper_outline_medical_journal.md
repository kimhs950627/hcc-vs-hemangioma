# Paper Outline — Medical Journal Submission

> **Status**: Narrative rewrite for journal-style manuscript outline (rev. 2026-06-29)
> **Target journal tier**: PubMed-indexed, SCIE Q1–Q2  
> *(e.g., Ultrasonics, Diagnostics, Frontiers in Oncology, JMIR Medical Informatics)*

## Proposed Title

**Hybrid Vision Transformer 기반 B-mode 복부 초음파 이중 출력 HCC 진단 모델의 개발 및 검증: Confidence Score와 Cosine Similarity 기반 HCC Score의 상호보완적 활용**

*영문 부제:*  
**Dual-Output HCC Scoring from B-mode Ultrasound via a Hybrid Vision Transformer: Confidence Score and Embedding-Based Cosine Similarity as Complementary Radiologic Markers**

## Structured Abstract

### Background
B-mode 초음파는 간세포암(hepatocellular carcinoma, HCC) 감시에서 가장 널리 사용되는 1차 영상 도구이지만, 조기 병변에 대한 민감도는 제한적이며 검사자 의존성 또한 크다. 최근 딥러닝 기반 초음파 분류 모델들이 보고되고 있으나, 대부분 softmax 기반 단일 출력을 그대로 “확률”로 해석하고 있으며, 임상가가 판독 과정에서 사용하는 유사성 기반 추론과의 연결은 충분히 제시되지 못하였다. 특히 초음파 영상에서는 캘리퍼, 측정 눈금, 텍스트 annotation과 같은 비병변 artifact가 모델의 shortcut learning을 유발할 수 있어, high-confidence prediction 자체만으로는 임상적으로 충분한 신뢰를 제공하기 어렵다.

### Methods
본 연구는 단일 기관 후향적 연구로서, 삼성서울병원에서 구축된 공개 B-mode 간 초음파 데이터셋 SMC-LUD를 사용하였다. 본 분석에서는 환자 단위 분리를 유지한 상태에서 train, validation, test 세트로 영상을 구분하여 학습과 평가를 수행하였다. 모델은 CNN backbone과 transformer encoder를 결합한 hybrid vision transformer로 설계하였으며, classification-only 학습과 classification plus supervised contrastive learning(SupCon) 학습을 비교하였다. 본 연구에서는 단일 모델로부터 두 가지 상호보완적 출력을 정의하였다. 첫째, softmax 출력값은 고전적 의미의 확률이 아니라 모델의 상대적 확신 정도를 반영하는 **confidence score**로 정의하였다. 둘째, 임베딩 공간에서 HCC prototype과의 cosine similarity를 이용하여 **HCC cosine score**를 산출하였고, 여기에 hemangioma prototype과의 차이를 반영한 **Δscore**를 추가로 계산하였다. 최적 임계값은 validation 세트에서만 Youden’s J 통계를 이용하여 결정하였고, test 세트는 맹검 평가에만 사용하였다.

### Results
Validation 세트에서 confidence score 기반 분류는 AUROC 1.000, accuracy 97.0%, sensitivity 94.3%, specificity 100.0%, positive predictive value 100.0%를 보였다. t-SNE 시각화에서는 HCC와 hemangioma의 임베딩 군집이 뚜렷하게 분리되었으며, 이는 cosine similarity 기반 점수가 의미 있는 표현 공간 위에서 계산되고 있음을 시사하였다. 본 연구에서는 confidence score 기반 ROC뿐 아니라 HCC cosine score 및 Δscore 기반 ROC를 함께 비교하여, softmax confidence와 embedding similarity가 서로 다른 진단 정보를 제공할 수 있음을 분석하고자 하였다.

### Conclusions
Hybrid vision transformer 기반 이중 출력 접근은 B-mode 초음파만으로도 높은 변별력을 보였으며, softmax confidence에 더해 cosine similarity 기반 HCC score를 병용함으로써 기존 CAM 기반 설명이 가지는 한계를 보완할 수 있는 가능성을 제시하였다. 이러한 dual-output framework는 모델의 예측을 단일 스칼라 확률로 환원하지 않고, 임상가가 실제로 사용하는 “전형적 병변과의 유사성”이라는 해석 축을 함께 제공한다는 점에서 방법론적, 임상적 의의를 가진다.

**Keywords**: hepatocellular carcinoma; ultrasound; deep learning; supervised contrastive learning; vision transformer; cosine similarity; calibration; explainable artificial intelligence

## 1. Introduction

Hepatocellular carcinoma is the predominant histologic subtype of primary liver cancer and remains a major cause of cancer-related mortality worldwide. In clinical practice, prognosis is highly dependent on stage at diagnosis, because early lesions are amenable to curative treatment, including surgical resection, ablation, or liver transplantation, whereas advanced disease is associated with a markedly poorer survival rate.[1][2] For this reason, international and domestic practice guidelines recommend surveillance in at-risk populations, most commonly using abdominal ultrasonography at regular intervals.[2][3]

Among available surveillance tools, B-mode ultrasonography remains the most accessible and most widely implemented modality, particularly in routine outpatient care. Its advantages are obvious: it is noninvasive, inexpensive, repeatable, and immediately available at the point of care. However, these practical strengths coexist with substantial limitations. The sensitivity of ultrasound alone for early HCC remains unsatisfactory, and although serum biomarkers such as alpha-fetoprotein may improve detection when used in combination, their performance is still suboptimal.[4][5] In addition, the diagnostic quality of ultrasound is heavily influenced by operator experience, scanning technique, and interpretation skill. This operator dependency is particularly problematic in real-world primary care and community practice, where subspecialty radiologists are not always available.

These limitations create a clinically relevant unmet need. In many frontline settings, physicians must make decisions on whether a hepatic lesion appears sufficiently suspicious to justify referral for contrast-enhanced imaging or specialist evaluation. A deep learning model that could extract diagnostic information directly from B-mode ultrasound might therefore serve as a useful adjunct. However, an AI system would only be clinically persuasive if its output aligns not merely with statistical classification performance but also with the way clinicians actually reason about lesions.

Recent studies have reported promising performance for deep learning–based classification of focal liver lesions on ultrasound.[6][7] Even so, much of the literature remains limited in several respects. First, most models provide only a single softmax output, which is commonly described as a probability despite the fact that it is not necessarily a calibrated probability.[18] Second, threshold selection is not always cleanly separated from final evaluation, raising the risk of information leakage and overly optimistic performance estimates. Third, many studies rely on conventional CAM-based visual explanation tools without sufficiently discussing whether the highlighted regions correspond to true lesion characteristics or merely to dataset-specific artifacts. Finally, architecture selection has often been restricted either to conventional convolutional neural networks or to plain vision transformers, each of which has known disadvantages in relatively small medical imaging datasets.[9][10][11]

The interpretability problem deserves special emphasis in ultrasound imaging. In a typical liver ultrasound frame, information unrelated to the lesion itself—including calipers, measurement markers, text overlays, and device-specific acquisition artifacts—may be present. If the model learns to associate these spurious cues with class labels, then apparently excellent discrimination may in fact reflect shortcut learning rather than robust lesion understanding. In such a setting, a high softmax output does not necessarily mean that the model has captured clinically meaningful imaging features. Rather, it may simply indicate that the model is highly confident in an internal pattern that is not radiologically valid.[12][19][20]

This distinction is important because radiologists do not diagnose liver lesions by a single scalar sense of certainty alone. Instead, they compare the visual features of the target lesion against previously learned prototypes or patterns: how much does this lesion resemble a typical HCC, and how much does it resemble a benign hemangioma? From this perspective, similarity-based reasoning is not incidental but central to radiologic diagnosis. Supervised contrastive learning offers a way to encode this kind of geometry into the embedding space by encouraging same-class lesions to cluster and different-class lesions to separate.[15] If such a representation is achieved, then cosine similarity to class prototypes may function as a meaningful diagnostic marker rather than a mere post hoc numerical convenience.

Accordingly, the present study was designed with two aims. The first was to develop and validate a hybrid vision transformer for differentiating HCC from hemangioma on B-mode ultrasound. The second was to extend the usual single-output classification framework into a dual-output system consisting of a confidence score and an embedding-based HCC cosine score, together with a discriminative Δscore. By doing so, this study sought to evaluate whether similarity-based outputs could provide clinically interpretable information complementary to softmax confidence, and whether such outputs might mitigate the overreliance on shortcut-prone confidence signals in near-perfect classification settings.

## 2. Materials and Methods

### 2.1 Study design

This investigation was designed as a retrospective single-center observational study using archived B-mode liver ultrasound images collected at Samsung Medical Center, Seoul, Republic of Korea. The manuscript is intended to be reported in accordance with the TRIPOD statement for prediction model studies.[13] Institutional review board approval number and waiver details will be inserted after final administrative confirmation.

### 2.2 Dataset and study population

The study used the Samsung Medical Center–Liver Ultrasound Dataset (SMC-LUD), a publicly released B-mode ultrasound dataset comprising hepatic focal lesions collected between 2015 and 2024.[14] The full dataset contains 5,385 grayscale images from 1,021 patients, including 2,716 images of pathologically confirmed HCC and 2,669 images of hemangioma diagnosed by imaging criteria. All images were standardized to 384 × 384 pixels and stored as grayscale floating-point arrays.

For the present analysis, a patient-level split was used to avoid cross-contamination of images from the same patient across development and evaluation phases. The final split consisted of 1,858 training images, 530 validation images, and 268 test images. The class distribution remained balanced across splits, with HCC accounting for approximately 52% of images and hemangioma for approximately 48%. Clinical metadata such as age, sex, cirrhosis status, and lesion size will be incorporated into the final manuscript if complete metadata linkage is confirmed.

### 2.3 Model architecture

The proposed model was a hybrid vision transformer that combined a convolutional neural network backbone with a transformer encoder. This design was chosen to exploit the complementary strengths of both paradigms. Convolutional layers preserve local inductive bias and are well suited for extracting texture and boundary information from ultrasound images, whereas transformer layers can model global contextual relationships across the feature map.[9][10][11]

Either ResNet50V2 or EfficientNetV2B0, initialized with ImageNet-pretrained weights, was used as the CNN backbone. The resulting feature map was reshaped into patch-like tokens and supplied to a transformer encoder with varying depths. A classification token was appended and its final representation was used for downstream prediction. This same embedding was also used for similarity-based post hoc scoring, enabling the model to produce both a conventional classification output and an embedding-derived radiologic similarity output within a unified framework.

### 2.4 Training strategy

Two training conditions were compared. In the first condition, the model was optimized using cross-entropy loss alone. In the second condition, cross-entropy loss was combined with supervised contrastive loss. Cross-entropy provides the standard discriminative signal for binary classification, whereas supervised contrastive learning explicitly structures the embedding space by pulling same-class samples together and pushing different-class samples apart.[15]

The cross-entropy loss was defined in the conventional manner using the predicted class distribution and the one-hot encoded target label. The supervised contrastive loss was implemented following the formulation of Khosla and colleagues, where normalized embeddings belonging to the same class are treated as positives and all other embeddings in the batch are treated as negatives.[15] The total loss in the contrastive setting was the weighted sum of cross-entropy and supervised contrastive loss.

The conceptual importance of this choice lies in the geometry of the learned representation. When supervised contrastive learning is applied, cosine similarity between embeddings is no longer an arbitrary post hoc measure. Rather, it becomes a quantity that the training objective has explicitly encouraged to be meaningful, because the loss itself favors high cosine similarity among lesions of the same class and low cosine similarity across classes. For this reason, cosine similarity to class prototypes can be interpreted as a theoretically coherent extension of the training objective.

All models were trained using 384 × 384 grayscale input images. The optimizer was Adam with an initial learning rate of 1 × 10^-4, and learning rate scheduling followed a cosine annealing schedule with warmup. Data augmentation included horizontal and vertical flipping, random crop and resize, brightness and contrast perturbation, and Gaussian blur. Training was conducted under practical computational constraints within a cloud GPU environment.

### 2.5 Dual-output metrics

A central feature of this study was the explicit separation of two different types of model output. The first output, termed the **confidence score**, was defined as the softmax value corresponding to the HCC class. Although this value lies between 0 and 1, it should not automatically be interpreted as a true probability. Modern deep neural networks are known to produce systematically overconfident softmax outputs, and post hoc calibration such as temperature scaling is often required before probabilistic interpretation becomes justified.[18] For that reason, the present manuscript intentionally refers to the softmax output as a confidence score rather than a probability.

The second output, termed the **HCC cosine score**, was defined from the embedding space. After model training, the mean embedding vector of the HCC training samples was computed to form an HCC prototype, and likewise the mean embedding vector of hemangioma training samples was computed to form a hemangioma prototype. For a given lesion image, its embedding vector was compared with the HCC prototype by cosine similarity. This score was intended to reflect how closely the lesion resembled the geometric center of learned HCC representations in the embedding space.

To refine discrimination further, a **Δscore** was defined as the difference between the HCC cosine score and the hemangioma cosine score. This difference was designed to capture not only resemblance to HCC but also dissimilarity from hemangioma. In practical terms, the Δscore can be understood as a relative similarity margin between the two competing disease prototypes.

This dual-output structure was motivated by clinical reasoning. A radiologist does not simply ask whether a lesion is malignant with a certain internal certainty; rather, the radiologist considers whether the lesion resembles the visual characteristics of a typical HCC more than those of a benign mimic such as hemangioma. The cosine-based outputs therefore offer a representation that is conceptually closer to human pattern comparison, whereas the confidence score reflects the model’s internal decisional certainty. Studying the agreement and disagreement between these two outputs was considered a major objective of this work.

### 2.6 Threshold determination and prevention of leakage

To minimize optimistic bias, all operating thresholds were determined exclusively from the validation set. The main threshold selection method was Youden’s J statistic, defined as the point that maximizes the sum of sensitivity and specificity minus one on the validation ROC curve. A sensitivity-prioritized strategy was additionally considered as a secondary analysis. Once determined in the validation set, the threshold was fixed and applied unchanged to the test set. This procedure was used for the confidence score, the HCC cosine score, and the Δscore separately.

### 2.7 Visualization and explainability analysis

To investigate the model’s decision patterns, Grad-CAM and transformer attention visualization were used as auxiliary explanation tools.[12] Grad-CAM was applied to the final convolutional stage of the CNN backbone to produce heat maps indicating regions that contributed strongly to the predicted class. Attention visualization was generated from the transformer encoder to inspect interactions between the classification token and image tokens.

These visual tools were included because they remain familiar to readers in medical imaging. However, they were not treated as definitive evidence of causal reasoning. Particular attention was paid to cases in which highlighted regions appeared to overlap with non-lesion structures such as calipers, scale markers, or text annotations. Such cases were interpreted as potential evidence of shortcut learning. The dual-output framework was therefore positioned not as a replacement for visual explanation, but as an additional safeguard against misinterpreting high-confidence outputs as necessarily trustworthy.

### 2.8 Statistical analysis

Model discrimination was assessed primarily using the area under the receiver operating characteristic curve. Confidence intervals for AUROC were to be estimated using the DeLong method.[16] Three types of ROC analyses were planned: ROC-A based on the confidence score, ROC-B based on the HCC cosine score, and ROC-C based on the Δscore. Pairwise comparisons among these correlated ROC curves were to be performed using the DeLong test.

Threshold-dependent performance metrics included sensitivity, specificity, positive predictive value, negative predictive value, F1 score, and confusion matrix components. In addition, a dual-output scatter plot was planned to visualize the joint distribution of confidence score and Δscore, with special attention to discordant patterns. Decision curve analysis was also planned to evaluate potential clinical net benefit across threshold probabilities.[17] Bootstrap resampling with 1,000 iterations was planned for confidence interval estimation of test-set metrics.

## 3. Results

### 3.1 Dataset composition

The analytic dataset comprised 1,858 training images, 530 validation images, and 268 test images. Across all splits, the proportion of HCC and hemangioma remained stable, with a slight predominance of HCC. This class balance reduced the risk that performance metrics would be driven by extreme class imbalance. A detailed demographic and lesion-level summary table will be added after confirmation of metadata availability.

### 3.2 Diagnostic performance of the confidence score

Using the validation-derived threshold based on Youden’s J statistic, the confidence score achieved an AUROC of 1.000 on the validation set, with an optimal cutoff of 0.004. At this operating point, the model yielded an accuracy of 97.0%, sensitivity of 94.3%, specificity of 100.0%, positive predictive value of 100.0%, negative predictive value of 94.1%, and F1 score of 0.971. The absence of false-positive cases at this cutoff suggests that the model, when positive by this criterion, identified HCC with extremely high precision in the internal validation set.

### 3.3 Embedding structure and implications for cosine scoring

Visualization of the embedding space using t-SNE demonstrated clear separation between HCC and hemangioma clusters. This finding is important not merely as an aesthetic confirmation of class separation but because it directly underpins the validity of cosine-based scoring. If the learned embedding space did not preserve disease-specific geometry, then cosine similarity to class prototypes would have little interpretive value. The observed cluster separation therefore supports the claim that prototype similarity is a meaningful descriptor in this setting.

### 3.4 Planned comparison of ROC-A, ROC-B, and ROC-C

The central comparative analysis of this manuscript is the contrast between three ROC paradigms: the confidence score, the HCC cosine score, and the Δscore. At present, the validation performance of the confidence score is available and has shown near-perfect discrimination. The cosine-based ROC analyses are being incorporated into the final result tables and figures. The purpose of this comparison is not only to determine whether cosine-based scores match or exceed the confidence score in AUROC, but also to establish whether they offer complementary information in cases where the model’s internal certainty may be unreliable.

### 3.5 Error pattern analysis under the dual-output framework

Beyond conventional ROC analysis, the dual-output framework enables a clinically interpretable error taxonomy. Cases in which both confidence and Δscore are high may be interpreted as concordant high-likelihood HCC cases. Cases in which both are low may represent concordant benign cases or indeterminate lesions requiring additional evaluation. More importantly, discordant cases—particularly those with high confidence but low Δscore—may indicate overconfident predictions driven by non-robust cues. Conversely, low-confidence but high-Δscore cases may represent lesions that resemble known HCC prototypes despite a more cautious classifier output. These discordant patterns are expected to be central to the clinical interpretation of the model.

### 3.6 Explainability findings and shortcut learning

Preliminary Grad-CAM inspection identified cases in which the model appeared to attend to non-lesion artifacts, including calipers and scale markings, rather than to lesion parenchyma itself. These observations raise concern that at least part of the model’s apparent confidence may be linked to shortcut features. This issue motivated the use of cosine-based outputs as an auxiliary interpretive axis. Representative cases and their corresponding dual-output patterns will be incorporated into the final figure set.

## 4. Discussion

This study proposes a journal-style conceptual shift in how AI-based liver ultrasound classification outputs should be interpreted. Rather than treating the softmax value as a self-sufficient probability, the study distinguishes between a **confidence score**, which reflects the model’s decisional certainty, and a **cosine-based disease similarity score**, which reflects how closely the current lesion resembles the class-specific geometry learned from training data. This distinction is not semantic but methodological. It addresses a central weakness in much of the medical AI literature, namely the tacit assumption that a bounded softmax value is equivalent to a clinically interpretable probability.

The first major implication concerns calibration. Modern deep neural networks are well known to be overconfident, and a high softmax value may not correspond to the actual empirical frequency of correctness.[18] In a medical context, this matters because clinicians may naturally interpret outputs such as 0.90 or 0.95 as if they represented trustworthy disease probabilities. By renaming this quantity as a confidence score, the present study avoids overstating what the model has actually learned. This conceptual clarification is especially important in scenarios where performance appears near perfect, because the temptation to overinterpret the score is strongest precisely when the model seems most impressive.

The second major implication concerns interpretability. Visual explanation methods such as Grad-CAM remain useful but are not sufficient on their own, especially in ultrasound images where shortcut learning can arise from non-lesion artifacts. If a model highlights calipers or text overlays, a clinician is left with the difficult question of whether the prediction is driven by genuine lesion morphology or by spurious acquisition-related cues. Cosine-based prototype similarity does not solve this issue completely, but it provides a different and clinically intuitive perspective: whether the lesion resides geometrically close to the cluster of known HCC cases. In that sense, it serves as a form of radiologist-aligned interpretive support.

The third implication concerns the role of supervised contrastive learning. In conventional cross-entropy training, the embedding space is optimized only indirectly through the classification objective, and geometric distances between samples are not guaranteed to be meaningful. By contrast, supervised contrastive learning explicitly encourages same-class lesions to occupy neighboring regions of the embedding space and different-class lesions to diverge. This makes cosine similarity a theoretically coherent measure of resemblance, not merely an exploratory visualization tool. For the present study, that property provides the rationale for defining the HCC cosine score and Δscore as structured outputs rather than optional post hoc analyses.

From a clinical perspective, the proposed dual-output framework may be especially useful when confidence and cosine similarity disagree. Such discordance may flag cases in which the model’s internal certainty is not well supported by disease-prototype geometry. In practical terms, this could help physicians resist overtrust in apparently strong AI predictions and encourage closer review or additional imaging when warranted. The framework may therefore offer a modest but meaningful safeguard against inappropriate reliance on black-box outputs.

Several limitations should be acknowledged. First, the study is retrospective and based on a single-center dataset, which limits external validity. Second, the current binary comparison between HCC and hemangioma does not reflect the full spectrum of focal liver lesion differentials encountered in practice. Third, although shortcut activation was qualitatively observed, a fully quantitative artifact-localization analysis has not yet been completed. Fourth, clinical metadata and head-to-head comparison against radiologists are not yet integrated into the present version of the outline. Finally, prototype-based similarity itself may be sensitive to institutional bias if the training distribution does not adequately represent external populations.

Despite these limitations, the study offers a potentially useful methodological contribution to medical AI. It reframes the output of an ultrasound classifier from a single allegedly probabilistic number into a dual-axis decision aid that better matches both the mathematical properties of deep networks and the clinical reasoning process of radiologists. If externally validated, this framework may support more interpretable and more cautious use of AI in ultrasound-based liver lesion assessment.

## 5. Conclusion

In summary, this study develops a hybrid vision transformer for differentiating HCC from hemangioma on B-mode ultrasound and proposes a dual-output interpretation framework consisting of a confidence score and an embedding-based HCC cosine score. The model demonstrated excellent internal validation performance, and the learned embedding structure supports the plausibility of cosine-based similarity as a meaningful radiologic marker.

More importantly, the study argues that these two outputs should not be viewed as redundant. The confidence score captures the model’s internal decisional certainty, whereas the cosine-based score reflects lesion similarity to learned class prototypes. Their agreement may strengthen trust in the prediction, and their disagreement may reveal cases requiring greater caution. This dual-output framework therefore provides not only a performance-oriented classifier but also a more clinically aligned interpretive structure for future ultrasound AI systems.

## References

1. Sung H, Ferlay J, Siegel RL, et al. Global Cancer Statistics 2020: GLOBOCAN Estimates of Incidence and Mortality Worldwide for 36 Cancers in 185 Countries. *CA Cancer J Clin*. 2021;71(3):209–249.
2. Korean Liver Cancer Association (KLCA); National Cancer Center (NCC) Korea. 2022 KLCA-NCC Korea Practice Guidelines for the Management of Hepatocellular Carcinoma. *Clin Mol Hepatol*. 2022;28(4):583–705.
3. European Association for the Study of the Liver (EASL). EASL Clinical Practice Guidelines: Management of Hepatocellular Carcinoma. *J Hepatol*. 2018;69(1):182–236.
4. Tzartzeva K, Obi J, Rich NE, et al. Surveillance Imaging and Alpha Fetoprotein for Early Detection of Hepatocellular Carcinoma in Patients with Cirrhosis: A Meta-analysis. *Gastroenterology*. 2018;154(6):1706–1718.
5. Tsuchiya N, Sawada Y, Endo I, et al. Biomarkers for the Early Diagnosis of Hepatocellular Carcinoma. *World J Gastroenterol*. 2015;21(37):10573–10583.
6. Yang Q, Wei J, Hao X, et al. Improving B-mode Ultrasound Diagnostic Performance for Focal Liver Lesions Using Deep Learning: A Multicentre Study. *EBioMedicine*. 2020;56:102777.
7. Zhang J, Zhu Q, Zhong T, et al. Deep Learning–based Automatic Segmentation and Classification of Focal Liver Lesions on Ultrasound Images. *Abdom Radiol*. 2022;47(2):763–773.
8. *(삼성서울병원 CEUS 딥러닝 연구 — 해당 논문 확인 후 삽입)*
9. Dosovitskiy A, Beyer L, Kolesnikov A, et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR*. 2021.
10. Chen CF, Fan Q, Panda R. CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification. *ICCV*. 2021:357–366.
11. *(EfficientNetV2 + ViT hybrid ultrasound classification — Diagnostics 2026, 삽입 예정)*
12. Selvaraju RR, Cogswell M, Das A, et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*. 2017:618–626.
13. Collins GS, Reitsma JB, Altman DG, Moons KGM. Transparent Reporting of a Multivariable Prediction Model for Individual Prognosis or Diagnosis (TRIPOD): The TRIPOD Statement. *BMJ*. 2015;350:g7594.
14. *(SMC-LUD: Samsung Medical Center Liver Ultrasound Dataset. Sci Data, Nature Portfolio, 2026 — DOI 삽입 예정)*
15. Khosla P, Tian Y, Wang X, et al. Supervised Contrastive Learning. *NeurIPS*. 2020;33:18661–18673.
16. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach. *Biometrics*. 1988;44(3):837–845.
17. Vickers AJ, Elkin EB. Decision Curve Analysis: A Novel Method for Evaluating Prediction Models. *Med Decis Making*. 2006;26(6):565–574.
18. Guo C, Pleiss G, Sun Y, Weinberger KQ. On Calibration of Modern Neural Networks. *ICML*. 2017;70:1321–1330.
19. Nguyen A, Yosinski J, Clune J. Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images. *CVPR*. 2015:427–436.
20. Ribeiro MT, Singh S, Guestrin C. "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD*. 2016:1135–1144.
