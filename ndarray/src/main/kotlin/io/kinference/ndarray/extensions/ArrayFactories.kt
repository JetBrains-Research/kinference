package io.kinference.ndarray.extensions

import io.kinference.ndarray.LateInitArray
import io.kinference.ndarray.MutableNDArray
import io.kinference.ndarray.NDArray
import io.kinference.ndarray.Strides
import io.kinference.ndarray.arrays.BooleanNDArray
import io.kinference.ndarray.arrays.LateInitBooleanArray
import io.kinference.ndarray.arrays.MutableBooleanNDArray
import io.kinference.primitives.types.DataType
import io.kinference.ndarray.*
import io.kinference.ndarray.arrays.*

inline fun <reified T> createArray(type: DataType, size: Int, noinline init: (Int) -> T): Any {
    return when (type) {
        DataType.DOUBLE -> DoubleArray(size) { init(it) as Double }
        DataType.FLOAT -> FloatArray(size) { init(it) as Float }
        DataType.LONG -> LongArray(size) { init(it) as Long }
        DataType.INT -> IntArray(size) { init(it) as Int }
        DataType.SHORT -> ShortArray(size) { init(it) as Short }
        DataType.BOOLEAN -> BooleanArray(size) { init(it) as Boolean }
        else -> Array(size, init)
    }
}

@ExperimentalUnsignedTypes
fun createMutableNDArray(type: DataType, value: Any, strides: Strides, offset: Int = 0): MutableNDArray {
    return when (type) {
        DataType.DOUBLE -> MutableDoubleNDArray(value as DoubleArray, strides, offset)
        DataType.FLOAT -> MutableFloatNDArray(value as FloatArray, strides, offset)
        DataType.LONG -> MutableLongNDArray(value as LongArray, strides, offset)
        DataType.INT -> MutableIntNDArray(value as IntArray, strides, offset)
        DataType.SHORT -> MutableShortNDArray(value as ShortArray, strides, offset)
        DataType.BOOLEAN -> MutableBooleanNDArray(value as BooleanArray, strides, offset)
        //else -> Array(size, init)
        else -> error("Unsupported data type $type")
    }
}

@ExperimentalUnsignedTypes
fun createMutableNDArray(type: DataType, value: Any, shape: IntArray, offset: Int = 0): MutableNDArray {
    return createMutableNDArray(type, value, Strides(shape), offset)
}

@ExperimentalUnsignedTypes
fun createNDArray(type: DataType, value: Any, strides: Strides, offset: Int = 0): NDArray {
    return when (type) {
        DataType.DOUBLE -> DoubleNDArray(value as DoubleArray, strides, offset)
        DataType.FLOAT -> FloatNDArray(value as FloatArray, strides, offset)
        DataType.LONG -> LongNDArray(value as LongArray, strides, offset)
        DataType.INT -> IntNDArray(value as IntArray, strides, offset)
        DataType.SHORT -> ShortNDArray(value as ShortArray, strides, offset)
        DataType.BOOLEAN -> BooleanNDArray(value as BooleanArray, strides, offset)
        //else -> Array(size, init)
        else -> error("Unsupported data type $type")
    }
}

@ExperimentalUnsignedTypes
fun createNDArray(type: DataType, value: Any, shape: IntArray, offset: Int = 0): NDArray {
    return createNDArray(type, value, Strides(shape), offset)
}

fun createZerosArray(type: DataType, size: Int): Any {
    return when (type) {
        DataType.DOUBLE -> DoubleArray(size)
        DataType.FLOAT -> FloatArray(size)
        DataType.LONG -> LongArray(size)
        DataType.INT -> IntArray(size)
        DataType.SHORT -> ShortArray(size)
        DataType.BOOLEAN -> BooleanArray(size)
        else -> error("Unsupported data type $type")
    }
}

@ExperimentalUnsignedTypes
fun createScalarNDArray(type: DataType, value: Any): NDArray {
    return when (type) {
        DataType.DOUBLE -> DoubleNDArray(doubleArrayOf(value as Double))
        DataType.FLOAT -> FloatNDArray(floatArrayOf(value as Float))
        DataType.LONG -> LongNDArray(longArrayOf(value as Long))
        DataType.INT -> IntNDArray(intArrayOf(value as Int))
        DataType.SHORT -> ShortNDArray(shortArrayOf(value as Short))
        DataType.BOOLEAN -> BooleanNDArray(booleanArrayOf(value as Boolean))
        DataType.BYTE -> ByteNDArray(byteArrayOf(value as Byte))
        //else -> Array(size, init)
        else -> error("Unsupported data type $type")
    }
}

@ExperimentalUnsignedTypes
fun allocateNDArray(type: DataType, strides: Strides): MutableNDArray {
    return when (type) {
        DataType.DOUBLE -> MutableDoubleNDArray(DoubleArray(strides.linearSize), strides)
        DataType.FLOAT -> MutableFloatNDArray(FloatArray(strides.linearSize), strides)
        DataType.LONG -> MutableLongNDArray(LongArray(strides.linearSize), strides)
        DataType.INT -> MutableIntNDArray(IntArray(strides.linearSize), strides)
        DataType.SHORT -> MutableShortNDArray(ShortArray(strides.linearSize), strides)
        DataType.BOOLEAN -> MutableBooleanNDArray(BooleanArray(strides.linearSize), strides)
        else -> error("Unsupported data type $type")
    }
}

@ExperimentalUnsignedTypes
fun createLateInitArray(type: DataType, strides: Strides): LateInitArray {
    return createLateInitArray(type, strides.linearSize)
}

@ExperimentalUnsignedTypes
fun createLateInitArray(type: DataType, size: Int): LateInitArray {
    return when (type) {
        DataType.DOUBLE -> LateInitDoubleArray(size)
        DataType.FLOAT -> LateInitFloatArray(size)
        DataType.LONG -> LateInitLongArray(size)
        DataType.INT -> LateInitIntArray(size)
        DataType.SHORT -> LateInitShortArray(size)
        DataType.BOOLEAN -> LateInitBooleanArray(size)
        else -> error("Unsupported data type $type")
    }
}

@ExperimentalUnsignedTypes
// TODO move into LateInitArray
fun createNDArrayFromLateInitArray(type: DataType, array: LateInitArray, strides: Strides): NDArray {
    return when (type) {
        DataType.DOUBLE -> DoubleNDArray((array as LateInitDoubleArray).getArray(), strides)
        DataType.FLOAT -> FloatNDArray((array as LateInitFloatArray).getArray(), strides)
        DataType.LONG -> LongNDArray((array as LateInitLongArray).getArray(), strides)
        DataType.INT -> IntNDArray((array as LateInitIntArray).getArray(), strides)
        DataType.SHORT -> ShortNDArray((array as LateInitShortArray).getArray(), strides)
        DataType.BOOLEAN -> BooleanNDArray((array as LateInitBooleanArray).getArray(), strides)
        else -> error("")
    }
}

val SUPPORTED_TYPES = setOf(DataType.DOUBLE, DataType.FLOAT, DataType.LONG, DataType.INT, DataType.SHORT)
fun inferType(type1: DataType, type2: DataType): DataType {
    return when {
        type1 !in SUPPORTED_TYPES || type2 !in SUPPORTED_TYPES -> error("Unsupported type")
        type1 == DataType.DOUBLE || type2 == DataType.DOUBLE -> DataType.DOUBLE
        type1 == DataType.FLOAT || type2 == DataType.FLOAT -> DataType.FLOAT
        type1 == DataType.LONG || type2 == DataType.LONG -> DataType.LONG
        type1 == DataType.INT || type2 == DataType.INT -> DataType.INT
        type1 == DataType.SHORT || type2 == DataType.SHORT -> DataType.SHORT
        else -> error("Unsupported type")
    }
}
