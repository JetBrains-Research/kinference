package org.jetbrains.research.kotlin.inference.data

import org.jetbrains.research.kotlin.inference.types.ValueInfo

//TODO: sparse tensors, maps, unknown
enum class ONNXDataType {
    ONNX_TENSOR,
    ONNX_SEQUENCE
}

abstract class ONNXData(val type: ONNXDataType, val info: ValueInfo) {
    abstract fun clone(newName: String): ONNXData
    abstract fun rename(newName: String): ONNXData
}
