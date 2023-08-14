package io.kinference.ndarray.extensions.mod

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

suspend fun NumberNDArrayCore.fmod(other: NumberNDArrayCore): NumberNDArrayCore {
    return when (type) {
        DataType.FLOAT -> (this as FloatNDArray).fmod(other as FloatNDArray)
        DataType.DOUBLE -> (this as DoubleNDArray).fmod(other as DoubleNDArray)
        DataType.BYTE -> (this as ByteNDArray).fmod(other as ByteNDArray)
        DataType.SHORT -> (this as ShortNDArray).fmod(other as ShortNDArray)
        DataType.INT -> (this as IntNDArray).fmod(other as IntNDArray)
        DataType.LONG -> (this as LongNDArray).fmod(other as LongNDArray)
        DataType.UBYTE -> (this as UByteNDArray).fmod(other as UByteNDArray)
        DataType.USHORT -> (this as UShortNDArray).fmod(other as UShortNDArray)
        DataType.UINT -> (this as UIntNDArray).fmod(other as UIntNDArray)
        DataType.ULONG -> (this as ULongNDArray).fmod(other as ULongNDArray)
        else -> error("")
    }
}
