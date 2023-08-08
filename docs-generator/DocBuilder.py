from typing import List, Dict
from onnx.defs import OpSchema


def op_string(onnx_operator: OpSchema, ki_ops: List[str]) -> str:
    op_name = onnx_operator.name
    is_implemented = op_name in ki_ops
    tick = "x" if is_implemented else " "

    return f"- [{tick}] [{op_name}](https://github.com/onnx/onnx/blob/main/docs/Operators.md#{op_name})\n"


def doc_string(ki_ops: List[str], onnx_ops: Dict[str, List[OpSchema]]) -> str:
    out_string = ""
    for domain in sorted(onnx_ops.keys()):
        out_string += f"### {domain}\n"

        domain_ops = onnx_ops[domain]
        implemented_ops = 0
        for onnx_op in domain_ops:
            if onnx_op.name in ki_ops:
                implemented_ops += 1

        out_string += f"#### Supported ops: {implemented_ops}/{len(domain_ops)}\n"

        function_ops = list(filter(lambda op: op.has_function, domain_ops))
        basic_ops = list(filter(lambda op: not op.has_function, domain_ops))

        if len(function_ops) > 0:
            out_string += "_Common operators_:\n"

        for operator in basic_ops:
            out_string += op_string(operator, ki_ops)

        if len(function_ops) > 0:
            out_string += "\n_Functions_:\n"
            for operator in function_ops:
                out_string += op_string(operator, ki_ops)

        out_string += "\n"

    return out_string
