package com.kimhs.liverdx.ui

/** Which image layer the full-screen viewer shows. */
enum class HeatmapLayer(val title: String, val chip: String, val note: String) {
    ORIGINAL(
        "RAW FRAME",
        "RAW",
        "EXIF 회전 보정 후 디코드된 원본. 모델은 이 프레임을 384×384 로 리샘플해 사용",
    ),
    CLEANED(
        "CLEANED FRAME",
        "CLEANED",
        "반사광 제거 · 조명 평탄화 · 그레이스케일 정렬 · 노이즈 제거 · 대비 정규화 적용본",
    ),
    ATTENTION(
        "CA ATTENTION ROLLOUT",
        "ATTN",
        "4개 cross-attention 블록의 head-mean rollout, min-max 정규화 (inferno)",
    ),
    PROTOTYPE(
        "HCC PROTOTYPE SIMILARITY",
        "MAP",
        "patch 별 HCC k-means(k=4) 프로토타입 평균 코사인, min-max 정규화 (turbo)",
    ),
}
