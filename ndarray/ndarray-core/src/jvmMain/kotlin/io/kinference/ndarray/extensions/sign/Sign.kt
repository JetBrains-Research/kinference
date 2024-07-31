package io.kinference.ndarray.extensions.sign

import io.kinference.ndarray.arrays.*

suspend fun NumberNDArrayCore.sign(): NumberNDArrayCore {
    return when (this) {
        is UIntNDArray -> signIntegerUInt(this)
        is UShortNDArray -> signIntegerUShort(this)
        is UByteNDArray -> signIntegerUByte(this)
        is ULongNDArray -> signIntegerULong(this)
        is FloatNDArray -> signFPFloat(this)
        is DoubleNDArray -> signFPDouble(this)
        is IntNDArray -> signIntegerInt(this)
        is LongNDArray -> signIntegerLong(this)
        is ShortNDArray -> signIntegerShort(this)
        is ByteNDArray -> signIntegerByte(this)
        else -> error("Unsupported data type for sign operation: ${this.type}")
    }
}
