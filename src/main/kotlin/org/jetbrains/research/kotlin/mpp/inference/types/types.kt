package org.jetbrains.research.kotlin.mpp.inference.types

import TensorProto.DataType
import kotlin.reflect.KClass

fun DataType.resolveKClass() : KClass<*> = when (this) {
    DataType.FLOAT -> Float::class
    DataType.DOUBLE -> Double::class
    DataType.INT64 -> Long::class
    DataType.INT32, DataType.INT8, DataType.UINT8, DataType.UINT16,
    DataType.INT16, DataType.BOOL, DataType.FLOAT16 -> Int::class
    else -> throw IllegalStateException("Unsupported data type")
}
