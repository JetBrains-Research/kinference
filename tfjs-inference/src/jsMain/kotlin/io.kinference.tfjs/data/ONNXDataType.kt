package io.kinference.tfjs.data

//TODO: sparse tensors, maps, unknown
enum class ONNXDataType {
    ONNX_TENSOR,
    ONNX_SEQUENCE,
    ONNX_MAP;

    companion object {
        fun fromString(str: String) = when (str) {
            "ONNX_TENSOR" -> ONNX_TENSOR
            "ONNX_SEQUENCE" -> ONNX_SEQUENCE
            "ONNX_MAP" -> ONNX_MAP
            else -> error("Unknown data type: $str")
        }
    }
}
