package org.jetbrains.research.kotlin.mpp.inference.types

import TensorProto.DataType
import kotlin.reflect.KClass

fun DataType.resolveKClass(): KClass<*> = when (this) {
    DataType.FLOAT -> Float::class
    DataType.DOUBLE -> Double::class
    DataType.INT64 -> Long::class
    DataType.INT32 -> Int::class
    else -> error("Unsupported data type")
}
