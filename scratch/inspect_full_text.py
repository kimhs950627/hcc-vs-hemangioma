"""
scratch/inspect_full_text.py
Print all sentences in generate_revised_manuscript.py with index.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
root = Path(r"d:\fm_paper_works\hcc-vs-hemangioma")
sys.path.append(str(root / "paper_submission" / "revision2"))
import generate_revised_manuscript as grm

for i, (ptype, chunks) in enumerate(grm.doc_structure):
    print(f"\n==================== [{i}] {ptype} ====================")
    for c_idx, (text, is_rev) in enumerate(chunks):
        print(f"Chunk {c_idx} (initially {'RED' if is_rev else 'BLACK'}):")
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines:
            print(f"  {line}")
