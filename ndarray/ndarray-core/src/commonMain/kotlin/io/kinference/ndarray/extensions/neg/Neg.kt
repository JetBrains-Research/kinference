package io.kinference.ndarray.extensions.neg

import io.kinference.ndarray.arrays.*

fun NumberNDArrayCore.neg(): NumberNDArrayCore {
    return when (this) {
        is FloatNDArray -> negFloat(this)
        is DoubleNDArray -> negDouble(this)
        is IntNDArray -> negInt(this)
        is LongNDArray -> negLong(this)
        is ShortNDArray -> negShort(this)
        is ByteNDArray -> negByte(this)
        else -> error("Unsupported data type: $type")
    }
}

operator fun NumberNDArrayCore.unaryMinus() = this.neg()
