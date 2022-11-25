package io.kinference.data

import io.kinference.BackendInfo
import io.kinference.utils.Closeable

/**
 * Base interface for ONNX data wrapper classes.
 *
 * @param T wrapped data type.
 */
interface BaseONNXData<T> {
    val name: String?
    val type: ONNXDataType
    val data: T

    fun rename(name: String): BaseONNXData<T>
}

/**
 * Interface for ONNX data wrapper classes.
 * [data] type should be supported by one of the available KInference backends.
 *
 * @param T wrapped data type.
 * @param B corresponding backend information.
 */
interface ONNXData<T, B : BackendInfo> : BaseONNXData<T>, Closeable {
    val backend: B
    override fun rename(name: String): ONNXData<T, B>
}

/**
 * Wrapper class for data of type [ONNXDataType.ONNX_TENSOR], where [data] contains tensor data.
 */
abstract class ONNXTensor<T, B : BackendInfo>(override val name: String?, override val data: T) : ONNXData<T, B> {
    override val type: ONNXDataType = ONNXDataType.ONNX_TENSOR
}

/**
 * Wrapper class for [ONNXDataType.ONNX_SEQUENCE], where [data] contains sequence data.
 */
abstract class ONNXSequence<T, B : BackendInfo>(override val name: String?, override val data: T) : ONNXData<T, B> {
    override val type: ONNXDataType = ONNXDataType.ONNX_SEQUENCE
}

/**
 * Wrapper class for [ONNXDataType.ONNX_MAP], where [data] contains map data.
 */
abstract class ONNXMap<T, B : BackendInfo>(override val name: String?, override val data: T) : ONNXData<T, B> {
    override val type: ONNXDataType = ONNXDataType.ONNX_MAP
}
