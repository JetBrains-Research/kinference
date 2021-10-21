package io.kinference.data

interface ONNXData<T> {
    val name: String?
    val type: ONNXDataType
    val data: T

    fun rename(name: String): ONNXData<T>
}

abstract class ONNXTensor<T>(override val name: String?, override val data: T) : ONNXData<T> {
    override val type: ONNXDataType = ONNXDataType.ONNX_TENSOR
}

abstract class ONNXSequence<T>(override val name: String?, override val data: T) : ONNXData<T> {
    override val type: ONNXDataType = ONNXDataType.ONNX_SEQUENCE
}

abstract class ONNXMap<T>(override val name: String?, override val data: T) : ONNXData<T> {
    override val type: ONNXDataType = ONNXDataType.ONNX_MAP
}
