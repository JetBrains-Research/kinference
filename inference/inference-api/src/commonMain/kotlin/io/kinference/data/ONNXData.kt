package io.kinference.data

import io.kinference.BackendInfo
import io.ktor.utils.io.core.*

interface BaseONNXData<T> {
    val name: String?
    val type: ONNXDataType
    val data: T

    fun rename(name: String): BaseONNXData<T>
}

interface ONNXData<T, B : BackendInfo> : BaseONNXData<T>, Closeable {
    val backend: B
    override fun rename(name: String): ONNXData<T, B>
}

abstract class ONNXTensor<T, B : BackendInfo>(override val name: String?, override val data: T) : ONNXData<T, B> {
    override val type: ONNXDataType = ONNXDataType.ONNX_TENSOR
}

abstract class ONNXSequence<T, B : BackendInfo>(override val name: String?, override val data: T) : ONNXData<T, B> {
    override val type: ONNXDataType = ONNXDataType.ONNX_SEQUENCE
}

abstract class ONNXMap<T, B : BackendInfo>(override val name: String?, override val data: T) : ONNXData<T, B> {
    override val type: ONNXDataType = ONNXDataType.ONNX_MAP
}
