"""Diagnostic: where does the Erfc op come from in the serving graph?"""
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensorflow as tf  # noqa: E402

from export_web_demo import save_saved_model  # noqa: E402

from inference.explainable_predictor import ExplainableHCCPredictor  # noqa: E402
from export_tflite import PROBE_DIR, WEIGHTS_PATH  # noqa: E402

predictor = ExplainableHCCPredictor(weights_path=WEIGHTS_PATH, probe_dir=PROBE_DIR)
with tempfile.TemporaryDirectory() as temporary:
    target = Path(temporary) / "saved_model"
    save_saved_model(predictor, target)

    from tensorflow.core.protobuf import saved_model_pb2

    saved = saved_model_pb2.SavedModel()
    saved.ParseFromString((target / "saved_model.pb").read_bytes())
    meta_graph = next(m for m in saved.meta_graphs if "serve" in m.meta_info_def.tags)
    graph_def = meta_graph.graph_def

    nodes = list(graph_def.node)
    for function in graph_def.library.function:
        nodes.extend(function.node_def)
    node_by_name = {}
    for node in nodes:
        node_by_name[node.name] = node
        if "/" in node.name:
            node_by_name[node.name.split("/")[-1]] = node
    print(f"[diag] {len(nodes)} node(s) including function bodies")

    def describe(node, depth=0):
        prefix = "  " * depth
        print(f"{prefix}{node.op} :: {node.name}")
        if depth >= 8:
            return
        for input_name in node.input:
            key = input_name.lstrip("^").split(":")[0]
            upstream = node_by_name.get(key) or node_by_name.get(key.split("/")[-1])
            if upstream is not None:
                describe(upstream, depth + 1)

    roots = [node for node in nodes if "Erfc" in node.op or "Erf" in node.op]
    print(f"[diag] {len(roots)} Erf/Erfc node(s)")
    for node in roots[:3]:
        describe(node)
        print("---")
    print("[diag] scopes:", sorted({n.name.split("/")[0] for n in roots}))
