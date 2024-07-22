from KiParser import parse_kinference_file
from OnnxParser import parse_onnx_defs
from DocBuilder import doc_string
from pathlib import Path
import argparse
import os


def generate_file(path: str, text: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as out_file:
        out_file.write(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--core_factory_file",
        help="Path to core operator factory",
        default="inference/inference-core/src/jvmMain/kotlin/io/kinference.core/operators/KIOperatorFactory.kt",
        required=False
    )

    parser.add_argument(
        "--tfjs_factory_file",
        help="Path to TFJS operator factory",
        default="inference/inference-tfjs/src/jsMain/kotlin/io.kinference.tfjs/operators/TFJSOperatorFactory.kt",
        required=False
    )

    parser.add_argument(
        "--core_doc_file",
        help="Path to core operators documentation",
        default="docs/CoreOperators.md",
        required=False
    )

    parser.add_argument(
        "--tfjs_doc_file",
        help="Path to TFJS operators documentation",
        default="docs/TfjsOperators.md",
        required=False
    )

    args = parser.parse_args()

    assert os.path.exists(args.core_factory_file)
    assert os.path.exists(args.tfjs_factory_file)

    core_ops = parse_kinference_file(args.core_factory_file)
    tfjs_ops = parse_kinference_file(args.tfjs_factory_file)
    onnx_defs = parse_onnx_defs()

    core_doc_string = doc_string(core_ops, onnx_defs)
    tfjs_doc_string = doc_string(tfjs_ops, onnx_defs)

    generate_file(args.core_doc_file, core_doc_string)
    generate_file(args.tfjs_doc_file, tfjs_doc_string)


if __name__ == "__main__":
    main()

