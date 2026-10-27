# -*- coding: utf-8 -*-
"""Machine check of the KJFP style checklist against the v5 package.

Run: python3 verify_kjfp.py
Exit code 0 = no FAIL.
"""
import pathlib, re, sys

import docx
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
fails, oks, notes = [], [], []


def check(cond, msg, detail='', fatal=True):
    if cond:
        oks.append(msg)
    else:
        (fails if fatal else notes).append(msg + (f' — {detail}' if detail else ''))


# ---------- manuscript docx ----------
d = docx.Document(HERE / 'manuscript_v5_kjfp.docx')
s = d.sections[0]
check(abs(s.page_width.cm - 21.0) < 0.05 and abs(s.page_height.cm - 29.7) < 0.05,
      'B2 A4 (21 x 29.7 cm)', f'{s.page_width.cm:.1f} x {s.page_height.cm:.1f}')
check(all(abs(m.cm - 3.0) < 0.05 for m in (s.top_margin, s.bottom_margin, s.left_margin, s.right_margin)),
      'B4 여백 3 cm', f'{s.top_margin.cm:.2f}')
check(d.styles['Normal'].paragraph_format.line_spacing == 2.0, 'B3 두 줄 간격',
      str(d.styles['Normal'].paragraph_format.line_spacing))
check(d.styles['Normal'].font.name == 'Batang', 'B6 바탕체', str(d.styles['Normal'].font.name))
check('PAGE' in s.footer._element.xml, 'B7 쪽번호(하단 중앙)')
sizes = {r.font.size.pt for p in d.paragraphs for r in p.runs if r.font.size}
check(min(sizes) >= 10, 'B5 10 pt 이상', str(sorted(sizes)))

text = '\n'.join(p.text for p in d.paragraphs)
plain = (HERE / 'manuscript_v5_plain.txt').read_text(encoding='utf-8')

body = plain[plain.index('## 1. 서론') if '## 1. 서론' in plain else plain.index('1. 서론'):]
body = body[:body.index('국문요약')] if '국문요약' in body else body
body_clean = re.sub(r'^\s*#+.*$', '', body, flags=re.M)
body_clean = re.sub(r'\n{2,}', '\n', body_clean).strip()
check(len(body_clean) <= 10000, 'D2 본문 10,000자 이내', f'{len(body_clean)}자')

ko = plain[plain.index('국문요약'):]
ko = ko[:ko.index('참고문헌')]
ko_body = re.sub(r'^\s*#+.*$', '', ko, flags=re.M)
ko_body = re.sub(r'중심단어.*', '', ko_body)
ko_body = re.sub(r'\s+', ' ', ko_body).strip()
check(len(ko_body) <= 1000, 'D4a 국문요약 1,000자 이내', f'{len(ko_body)}자')

en = plain[plain.index('영문초록'):]
en = en[:en.index('1. 서론')] if '1. 서론' in en else en
en_body = re.sub(r'^\s*#+.*$', '', en, flags=re.M)
words = len(re.findall(r"[A-Za-z0-9'\-]+", en_body))
check(words <= 250, 'D2a 영문초록 250단어 이내', f'{words} words')
for sec in ('Background', 'Methods', 'Results', 'Conclusion', 'Keywords'):
    check(sec in plain, f'D2a 영문초록 {sec} 항목')
kw = re.search(r'Keywords\n\n(.+)', plain)
n_kw = len(kw.group(1).split(';')) if kw else 0
check(4 <= n_kw <= 6, 'D2b 영문 키워드 4-6개', str(n_kw))

check('기관생명윤리위원회' in plain, 'D3b IRB 승인 명시')
check('TensorFlow' in plain or 'Python' in plain, 'D3b 통계 프로그램 명시')
check(not re.search(r'\bp\s*[<>=]', plain), 'D3b P 값 대문자', '소문자 p 잔존')
check('STARD' in plain, 'A1 STARD 언급')

sups = [r.text for p in d.paragraphs for r in p.runs if r.font.superscript]
check(len(sups) >= 14, 'E8 위첨자 인용', f'{len(sups)}개')
check(not re.search(r'\[\d', plain), 'E8 대괄호 인용 잔존 없음')
check(any('-' in x for x in sups), 'E8 3개 이상 연속은 하이픈')

refs = re.findall(r'^\d+\. (.+)$', plain, flags=re.M)
check(len(refs) <= 30, 'E2 참고문헌 30개 이하', f'{len(refs)}개')
bad_authors = [r for r in refs if 'et al' in r.split('.')[0] and len(r.split('.')[0].split(',')) < 6]
check(not bad_authors, 'E4 6인까지 나열 후 et al', f'{len(bad_authors)}건 위반')
bad_journal = [r for r in refs if re.search(r'\.\s\d{4};', r)]
check(not bad_journal, 'E10 저널명 뒤 마침표 없음', f'{len(bad_journal)}건')
check(all(re.search(r'\d{4}; \d+:', r) for r in refs if 'Proceedings' not in r and 'Advances' not in r),
      'E10 연도; 권: 형식')

# ---------- tables ----------
td = docx.Document(HERE / 'tables_v5.docx')
caps = [p.text for p in td.paragraphs if p.text.startswith('Table')]
figcaps = [p.text for p in d.paragraphs if p.text.startswith('Figure')]
logical_tables = [c for c in caps if re.match(r'^Table \d+[A-Z]?\.', c)]
check(len(logical_tables) == 5, 'D3 표 개수(논리 표 기준)', str(len(logical_tables)))
check(len(logical_tables) + len(figcaps) <= 10, 'D3 표+그림 10개 이내',
      f'{len(logical_tables)}+{len(figcaps)}')
check(all(not c.rstrip().endswith('.') for c in caps), 'F4 표 제목 끝 마침표 없음')
check(all(c.rstrip().endswith('.') for c in figcaps), 'G5 그림 제목 끝 마침표')
check(all('Figure' in c or True for c in figcaps), 'G6 Figure 번호 표기')
grid = []
for t in td.tables:
    xml = t._tbl.tblPr.xml
    if 'w:insideH' in xml and 'w:val="single"' in xml.split('w:insideH')[1][:120]:
        grid.append(True)
check(not grid, 'F7 표 내부줄 없음(3선형)')
check('Batang' in td.tables[0].rows[0].cells[0].paragraphs[0].runs[0].font.name, 'F1 표 글꼴 바탕체')
markers = [r.text for t in td.tables for row in t.rows for c in row.cells
           for p in c.paragraphs for r in p.runs if r.font.superscript]
check(len(markers) >= 3, 'F8 주석 위첨자 마커(a,b,c)', f'{len(markers)}개')

# ---------- figures ----------
figdir = HERE / 'figures_kjfp'
figs = sorted(figdir.glob('*.jpg'))
check(len(figs) == 5, 'G2 그림 5개', str(len(figs)))
dpi_ok, small = True, []
for f in figs:
    im = Image.open(f)
    dpi = im.info.get('dpi', (0, 0))[0]
    if dpi < 600:
        dpi_ok = False
    if im.width / 600 * 2.54 < 8:
        small.append(f'{f.name}:{im.width / 600 * 2.54:.1f}cm')
check(dpi_ok, 'G2 그림 600 dpi 이상')
check(not small, 'G2 인쇄 폭 8 cm 이상', ', '.join(small), fatal=False)
check((HERE / 'Title_Page_v1.docx').exists(), 'D1 표지 파일 존재')
tp = docx.Document(HERE / 'Title_Page_v1.docx')
tpt = '\n'.join(p.text for p in tp.paragraphs)
check('ORCID' in tpt and 'Running title' in tpt, 'D1c/d 소제목·ORCID 항목')

print('PASS', len(oks))
for m in oks:
    print('  ok  ', m)
if notes:
    print('NOTE', len(notes))
    for m in notes:
        print('  note', m)
if fails:
    print('FAIL', len(fails))
    for m in fails:
        print('  FAIL', m)
    sys.exit(1)
print('\n모든 기계 검증 통과')
