package io.kinference.ndarray.extensions.neg

import io.kinference.ndarray.arrays.*

suspend fun NumberNDArrayCore.neg(): NumberNDArrayCore {
    return when (this) {
        is FloatNDArray -> negFloat(this)
        is DoubleNDArray -> negDouble(this)
        is IntNDArray -> negInt(this)
        is LongNDArray -> negLong(this)
        is ShortNDArray -> negIntegerShort(this)
        is ByteNDArray -> negIntegerByte(this)
        else -> error("Unsupported data type: $type")
    }
}

suspend operator fun NumberNDArrayCore.unaryMinus() = this.neg()
