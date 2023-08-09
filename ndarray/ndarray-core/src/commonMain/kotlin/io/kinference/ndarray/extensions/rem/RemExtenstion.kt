package io.kinference.ndarray.extensions.rem

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

suspend operator fun NumberNDArrayCore.rem(other: NumberNDArrayCore): NumberNDArrayCore {
    return when (type) {
        DataType.BYTE -> (this as ByteNDArray).rem(other as ByteNDArray)
        DataType.SHORT -> (this as ShortNDArray).rem(other as ShortNDArray)
        DataType.INT -> (this as IntNDArray).rem(other as IntNDArray)
        DataType.LONG -> (this as LongNDArray).rem(other as LongNDArray)
        DataType.UBYTE -> (this as UByteNDArray).rem(other as UByteNDArray)
        DataType.USHORT -> (this as UShortNDArray).rem(other as UShortNDArray)
        DataType.UINT -> (this as UIntNDArray).rem(other as UIntNDArray)
        DataType.ULONG -> (this as ULongNDArray).rem(other as ULongNDArray)
        DataType.FLOAT -> (this as FloatNDArray).rem(other as FloatNDArray)
        DataType.DOUBLE -> (this as DoubleNDArray).rem(other as DoubleNDArray)
        else -> error("Rem operation is only applicable to numeric tensors, current type: $type")
    }
}
