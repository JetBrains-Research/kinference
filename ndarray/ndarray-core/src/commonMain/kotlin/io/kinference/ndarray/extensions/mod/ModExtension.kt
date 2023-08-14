package io.kinference.ndarray.extensions.mod

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.mod.unsigned.mod
import io.kinference.primitives.types.DataType

suspend fun NumberNDArrayCore.mod(other: NumberNDArrayCore): NumberNDArrayCore {
    return when (type) {
        DataType.FLOAT -> (this as FloatNDArray).mod(other as FloatNDArray)
        DataType.DOUBLE -> (this as DoubleNDArray).mod(other as DoubleNDArray)
        DataType.BYTE -> (this as ByteNDArray).mod(other as ByteNDArray)
        DataType.SHORT -> (this as ShortNDArray).mod(other as ShortNDArray)
        DataType.INT -> (this as IntNDArray).mod(other as IntNDArray)
        DataType.LONG -> (this as LongNDArray).mod(other as LongNDArray)
        DataType.UBYTE -> (this as UByteNDArray).mod(other as UByteNDArray)
        DataType.USHORT -> (this as UShortNDArray).mod(other as UShortNDArray)
        DataType.UINT -> (this as UIntNDArray).mod(other as UIntNDArray)
        DataType.ULONG -> (this as ULongNDArray).mod(other as ULongNDArray)
        else -> error("mod operation is only applicable to numeric tensors, current type: $type")
    }
}
