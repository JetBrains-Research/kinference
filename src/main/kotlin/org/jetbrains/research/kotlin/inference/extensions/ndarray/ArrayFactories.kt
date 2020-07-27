package org.jetbrains.research.kotlin.inference.extensions.ndarray

import TensorProto.DataType
import org.jetbrains.research.kotlin.inference.data.ndarray.*
import org.jetbrains.research.kotlin.inference.data.tensors.Strides

inline fun createArray(type: DataType, size: Int, noinline init: (Int) -> Any): Any {
    return when (type) {
        DataType.DOUBLE -> DoubleArray(size) { init(it) as Double }
        DataType.FLOAT -> FloatArray(size) { init(it) as Float }
        DataType.INT64 -> LongArray(size) { init(it) as Long }
        DataType.INT32 -> IntArray(size) { init(it) as Int }
        DataType.INT16 -> ShortArray(size) { init(it) as Short }
        else -> Array(size, init)
    }
}

inline fun createNDArray(type: DataType, strides: Strides = Strides.empty(), noinline init: (Int) -> Any): NDArray {
    return when (type) {
        DataType.DOUBLE -> DoubleNDArray(createArray(type, strides.linearSize, init) as DoubleArray, strides)
        DataType.FLOAT -> FloatNDArray(createArray(type, strides.linearSize, init) as FloatArray, strides)
        DataType.INT64 -> LongNDArray(createArray(type, strides.linearSize, init) as LongArray, strides)
        DataType.INT32 -> IntNDArray(createArray(type, strides.linearSize, init) as IntArray, strides)
        DataType.INT16 -> ShortNDArray(createArray(type, strides.linearSize, init) as ShortArray, strides)
        //else -> Array(size, init)
        else -> error("Unsupported data type $type")
    }
}

inline fun createScalarNDArray(type: DataType, value: Any): NDArray {
    return when (type) {
        DataType.DOUBLE -> DoubleNDArray(doubleArrayOf(value as Double))
        DataType.FLOAT -> FloatNDArray(floatArrayOf(value as Float))
        DataType.INT64 -> LongNDArray(longArrayOf(value as Long))
        DataType.INT32 -> IntNDArray(intArrayOf(value as Int))
        DataType.INT16 -> ShortNDArray(shortArrayOf(value as Short))
        //else -> Array(size, init)
        else -> error("Unsupported data type $type")
    }
}

inline fun zerosNDArray(type: DataType, strides: Strides): NDArray {
    return when (type) {
        DataType.DOUBLE -> DoubleNDArray(DoubleArray(strides.linearSize) { 0.0 }, strides)
        DataType.FLOAT -> FloatNDArray(FloatArray(strides.linearSize) { 0.0f }, strides)
        DataType.INT64 -> LongNDArray(LongArray(strides.linearSize) { 0L }, strides)
        DataType.INT32 -> IntNDArray(IntArray(strides.linearSize) { 0 }, strides)
        DataType.INT16 -> ShortNDArray(ShortArray(strides.linearSize) { (0).toShort() }, strides)
        //else -> Array(size, init)
        else -> error("Unsupported data type $type")
    }
}

inline fun allocateNDArray(type: DataType, strides: Strides): NDArray {
    return when (type) {
        DataType.DOUBLE -> DoubleNDArray(DoubleArray(strides.linearSize), strides)
        DataType.FLOAT -> FloatNDArray(FloatArray(strides.linearSize), strides)
        DataType.INT64 -> LongNDArray(LongArray(strides.linearSize), strides)
        DataType.INT32 -> IntNDArray(IntArray(strides.linearSize), strides)
        DataType.INT16 -> ShortNDArray(ShortArray(strides.linearSize), strides)
        else -> error("")
    }
}

val SUPPORTED_TYPES = setOf(DataType.DOUBLE, DataType.FLOAT, DataType.INT64, DataType.INT32, DataType.INT16)
fun inferType(type1: DataType, type2: DataType): DataType {
    return when {
        type1 !in SUPPORTED_TYPES || type2 !in SUPPORTED_TYPES -> error("Unsupported type")
        type1 == DataType.DOUBLE || type2 == DataType.DOUBLE -> DataType.DOUBLE
        type1 == DataType.FLOAT || type2 == DataType.FLOAT -> DataType.FLOAT
        type1 == DataType.INT64 || type2 == DataType.INT64 -> DataType.INT64
        type1 == DataType.INT32 || type2 == DataType.INT32 -> DataType.INT32
        type1 == DataType.INT16 || type2 == DataType.INT16 -> DataType.INT16
        else -> error("Unsupported type")
    }
}
