package org.jetbrains.research.kotlin.inference.types

import org.jetbrains.research.kotlin.inference.onnx.TensorProto.DataType
import kotlin.reflect.KClass

fun DataType.resolveKClass(): KClass<*> = when (this) {
    DataType.FLOAT -> Float::class
    DataType.DOUBLE -> Double::class
    DataType.INT64 -> Long::class
    DataType.INT32 -> Int::class
    else -> error("Unsupported data type")
}

fun <T : Any> KClass<T>.resolveDataType(): DataType = when (this) {
    Float::class -> DataType.FLOAT
    Double::class -> DataType.DOUBLE
    Long::class -> DataType.INT64
    Int::class -> DataType.INT32
    else -> error("Unsupported data type")
}
