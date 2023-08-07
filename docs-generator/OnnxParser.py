from typing import Dict, List
from onnx import defs
from onnx.defs import OpSchema
from collections import defaultdict


def parse_onnx_defs() -> Dict[str, List[OpSchema]]:
    onnx_ops_by_domain: Dict[str, List[OpSchema]] = defaultdict(list)
    onnx_ops: List[OpSchema] = defs.get_all_schemas()

    for schema in onnx_ops:
        onnx_ops_by_domain[schema.domain].append(schema)

    onnx_ops_by_domain["ai.onnx"] = onnx_ops_by_domain[""]
    onnx_ops_by_domain.pop("")

    for domain in onnx_ops_by_domain:
        ops_list = onnx_ops_by_domain[domain]
        ops_list_sorted = sorted(ops_list, key=lambda op: op.name)
        onnx_ops_by_domain[domain] = ops_list_sorted

    return onnx_ops_by_domain
