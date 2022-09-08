package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

fun NDArray.map(func: PrimitiveToPrimitiveFunction): MutableNDArray {
    return when (type) {
        DataType.DOUBLE -> (this as DoubleNDArray).map(func)
        DataType.FLOAT -> (this as FloatNDArray).map(func)
        DataType.INT -> (this as IntNDArray).map(func)
        DataType.LONG -> (this as LongNDArray).map(func)
        DataType.BYTE -> (this as ByteNDArray).map(func)
        DataType.UBYTE -> (this as UByteNDArray).map(func)
        DataType.SHORT -> (this as ShortNDArray).map(func)
        DataType.USHORT -> (this as UShortNDArray).map(func)
        DataType.UINT -> (this as UIntNDArray).map(func)
        DataType.ULONG -> (this as ULongNDArray).map(func)
        DataType.BOOLEAN -> (this as BooleanNDArray).map(func)
        else -> error("Unsupported data type: $type.")
    }
}

fun NDArray.mapMutable(func: PrimitiveToPrimitiveFunction): MutableNDArray {
    return when (type) {
        DataType.DOUBLE -> (this as MutableDoubleNDArray).mapMutable(func)
        DataType.FLOAT -> (this as MutableFloatNDArray).mapMutable(func)
        DataType.INT -> (this as MutableIntNDArray).mapMutable(func)
        DataType.LONG -> (this as MutableLongNDArray).mapMutable(func)
        DataType.BYTE -> (this as MutableByteNDArray).mapMutable(func)
        DataType.UBYTE -> (this as MutableUByteNDArray).mapMutable(func)
        DataType.SHORT -> (this as MutableShortNDArray).mapMutable(func)
        DataType.USHORT -> (this as MutableUShortNDArray).mapMutable(func)
        DataType.UINT -> (this as MutableUIntNDArray).mapMutable(func)
        DataType.ULONG -> (this as MutableULongNDArray).mapMutable(func)
        DataType.BOOLEAN -> (this as MutableBooleanNDArray).mapMutable(func)
        else -> error("Unsupported data type: $type.")
    }
}

fun NDArray.map(func: PrimitiveToPrimitiveFunction, dest: MutableNDArray): MutableNDArray {
    return when (type) {
        DataType.DOUBLE -> (this as DoubleNDArray).map(func, dest)
        DataType.FLOAT -> (this as FloatNDArray).map(func, dest)
        DataType.INT -> (this as IntNDArray).map(func, dest)
        DataType.LONG -> (this as LongNDArray).map(func, dest)
        DataType.BYTE -> (this as ByteNDArray).map(func, dest)
        DataType.UBYTE -> (this as UByteNDArray).map(func, dest)
        DataType.SHORT -> (this as ShortNDArray).map(func, dest)
        DataType.USHORT -> (this as UShortNDArray).map(func, dest)
        DataType.UINT -> (this as UIntNDArray).map(func, dest)
        DataType.ULONG -> (this as ULongNDArray).map(func, dest)
        DataType.BOOLEAN -> (this as BooleanNDArray).map(func, dest)
        else -> error("Unsupported data type: $type.")
    }
}
