package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

fun NDArray.view(vararg axes: Int): NDArray {
    return when (type) {
        DataType.DOUBLE -> (this as DoubleNDArray).view(*axes)
        DataType.FLOAT -> (this as FloatNDArray).view(*axes)
        DataType.INT -> (this as IntNDArray).view(*axes)
        DataType.LONG -> (this as LongNDArray).view(*axes)
        DataType.BYTE -> (this as ByteNDArray).view(*axes)
        DataType.UBYTE -> (this as UByteNDArray).view(*axes)
        DataType.SHORT -> (this as ShortNDArray).view(*axes)
        DataType.USHORT -> (this as UShortNDArray).view(*axes)
        DataType.UINT -> (this as UIntNDArray).view(*axes)
        DataType.ULONG -> (this as ULongNDArray).view(*axes)
        DataType.BOOLEAN -> (this as BooleanNDArray).view(*axes)
        else -> error("Unsupported data type: $type.")
    }
}

fun NDArray.viewMutable(vararg axes: Int): MutableNDArray {
    return when (type) {
        DataType.DOUBLE -> (this as MutableDoubleNDArray).viewMutable(*axes)
        DataType.FLOAT -> (this as MutableFloatNDArray).viewMutable(*axes)
        DataType.INT -> (this as MutableIntNDArray).viewMutable(*axes)
        DataType.LONG -> (this as MutableLongNDArray).viewMutable(*axes)
        DataType.BYTE -> (this as MutableByteNDArray).viewMutable(*axes)
        DataType.UBYTE -> (this as MutableUByteNDArray).viewMutable(*axes)
        DataType.SHORT -> (this as MutableShortNDArray).viewMutable(*axes)
        DataType.USHORT -> (this as MutableUShortNDArray).viewMutable(*axes)
        DataType.UINT -> (this as MutableUIntNDArray).viewMutable(*axes)
        DataType.ULONG -> (this as MutableULongNDArray).viewMutable(*axes)
        DataType.BOOLEAN -> (this as MutableBooleanNDArray).viewMutable(*axes)
        else -> error("Unsupported data type: $type.")
    }
}

fun NumberNDArray.view(vararg axes: Int): NumberNDArray = (this as NDArray).view(*axes) as NumberNDArray
fun NumberNDArray.viewMutable(vararg axes: Int): MutableNumberNDArray = (this as NDArray).viewMutable(*axes) as MutableNumberNDArray
