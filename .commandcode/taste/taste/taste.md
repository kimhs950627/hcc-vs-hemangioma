# Taste
- Writes in Korean and expects the assistant to communicate and produce deliverables in Korean. Confidence: 0.75
- Prefers the assistant to lay out a plan before executing (explicitly asks for a plan to be drawn up first rather than jumping straight to output). Confidence: 0.6
- Favors HTML as the delivery format for presentation/slide material. Confidence: 0.55
- Domain context: medical/clinical research; audiences are fellow physicians at academic conferences, so content should be pitched at specialist-level peers. Confidence: 0.5
- Expects revisions to be applied to both the HTML and PDF versions of the same deliverable — keep the two formats in sync ("html과 pdf 모두 공히 적용"). Confidence: 0.7
- Prefers presentation HTML to be fully self-contained with images embedded as base64 data URIs, so no separate figures/ directory is needed to open or share the deck. Confidence: 0.75
- Prefers real rendered tables (HTML table markup) over screenshots/images of tables in slides. Confidence: 0.65
- Expects numeric claims and citations to be verified against the primary source documents (e.g. the original dataset paper PDF) and sources referenced explicitly in the deliverable. Confidence: 0.6
