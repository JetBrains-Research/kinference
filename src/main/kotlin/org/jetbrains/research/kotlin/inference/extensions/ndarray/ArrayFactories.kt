package org.jetbrains.research.kotlin.inference.extensions.ndarray

import org.jetbrains.research.kotlin.inference.data.ndarray.*
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.onnx.TensorProto.DataType

inline fun <reified T> createArray(type: DataType, size: Int, noinline init: (Int) -> T): Any {
    return when (type) {
        DataType.DOUBLE -> DoubleArray(size) { init(it) as Double }
        DataType.FLOAT -> FloatArray(size) { init(it) as Float }
        DataType.INT64 -> LongArray(size) { init(it) as Long }
        DataType.INT32 -> IntArray(size) { init(it) as Int }
        DataType.INT16 -> ShortArray(size) { init(it) as Short }
        DataType.BOOL -> BooleanArray(size) { init(it) as Boolean }
        else -> Array(size, init)
    }
}

@Suppress("UNCHECKED_CAST")
fun <T> createScalarNDArray(type: DataType, value: Any): NDArray<T> {
    return when (type) {
        DataType.DOUBLE -> DoubleNDArray(doubleArrayOf(value as Double))
        DataType.FLOAT -> FloatNDArray(floatArrayOf(value as Float))
        DataType.INT64 -> LongNDArray(longArrayOf(value as Long))
        DataType.INT32 -> IntNDArray(intArrayOf(value as Int))
        DataType.INT16 -> ShortNDArray(shortArrayOf(value as Short))
        DataType.BOOL -> BooleanNDArray(booleanArrayOf(value as Boolean))
        //else -> Array(size, init)
        else -> error("Unsupported data type $type")
    } as NDArray<T>
}

@Suppress("UNCHECKED_CAST")
fun allocateNDArray(type: DataType, strides: Strides): NDArray<Any> {
    return when (type) {
        DataType.DOUBLE -> DoubleNDArray(DoubleArray(strides.linearSize), strides)
        DataType.FLOAT -> FloatNDArray(FloatArray(strides.linearSize), strides)
        DataType.INT64 -> LongNDArray(LongArray(strides.linearSize), strides)
        DataType.INT32 -> IntNDArray(IntArray(strides.linearSize), strides)
        DataType.INT16 -> ShortNDArray(ShortArray(strides.linearSize), strides)
        DataType.BOOL -> BooleanNDArray(BooleanArray(strides.linearSize), strides)
        else -> error("Unsupported type")
    } as NDArray<Any>
}

@Suppress("UNCHECKED_CAST")
fun createLateInitArray(type: DataType, strides: Strides): LateInitArray {
    return createLateInitArray(type, strides.linearSize)
}

fun createLateInitArray(type: DataType, size: Int): LateInitArray {
    return when (type) {
        DataType.DOUBLE -> LateInitDoubleArray(size)
        DataType.FLOAT -> LateInitFloatArray(size)
        DataType.INT64 -> LateInitLongArray(size)
        DataType.INT32 -> LateInitIntArray(size)
        DataType.INT16 -> LateInitShortArray(size)
        DataType.BOOL -> LateInitBooleanArray(size)
        else -> error("Unsupported type")
    }
}

@Suppress("UNCHECKED_CAST")
// TODO move into LateInitArray
fun createNDArrayFromLateInitArray(type: DataType, array: LateInitArray, strides: Strides): NDArray<Any> {
    return when (type) {
        DataType.DOUBLE -> DoubleNDArray((array as LateInitDoubleArray).getArray(), strides)
        DataType.FLOAT -> FloatNDArray((array as LateInitFloatArray).getArray(), strides)
        DataType.INT64 -> LongNDArray((array as LateInitLongArray).getArray(), strides)
        DataType.INT32 -> IntNDArray((array as LateInitIntArray).getArray(), strides)
        DataType.INT16 -> ShortNDArray((array as LateInitShortArray).getArray(), strides)
        DataType.BOOL -> BooleanNDArray((array as LateInitBooleanArray).getArray(), strides)
        else -> error("")
    } as NDArray<Any>
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
