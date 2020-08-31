package io.kinference.data

import io.kinference.types.ValueInfo

//TODO: sparse tensors, maps, unknown
enum class ONNXDataType {
    ONNX_TENSOR,
    ONNX_SEQUENCE
}

abstract class ONNXData(val type: ONNXDataType, val info: ValueInfo) {
    abstract fun rename(newName: String): ONNXData
}
