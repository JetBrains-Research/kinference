package io.kinference.ndarray.extensions.pow

import io.kinference.ndarray.arrays.*

suspend fun NumberNDArrayCore.pow(powArray: NumberNDArrayCore): NumberNDArrayCore {
    return when (this) {
        is FloatNDArray -> this.powArray(powArray)
        is DoubleNDArray -> this.powArray(powArray)
        is IntNDArray -> this.powArray(powArray)
        is LongNDArray -> this.powArray(powArray)
        else -> error("Unsupported input array data type: ${powArray.type}")
    }
}
