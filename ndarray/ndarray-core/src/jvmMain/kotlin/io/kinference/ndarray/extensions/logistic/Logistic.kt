package io.kinference.ndarray.extensions.logistic

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

suspend fun logistic(
    input: NumberNDArrayCore,
): MutableNumberNDArrayCore {
    require(input.type == DataType.FLOAT || input.type == DataType.DOUBLE)
    { "Logistic operation supported only for FLOAT and DOUBLE tensors, actual type is ${input.type}" }

    return when(input.type) {
        DataType.FLOAT -> logisticFloat(input as FloatNDArray)
        DataType.DOUBLE ->  logisticDouble(input as DoubleNDArray)
        else -> error("Logistic operation supported only for FLOAT and DOUBLE tensors, actual type is ${input.type}")
    }
}

suspend fun logistic(
    input: NumberNDArrayCore,
    dest: MutableNumberNDArrayCore,
): MutableNumberNDArrayCore {
    require(input.type == dest.type)
    require(input.type == DataType.FLOAT || input.type == DataType.DOUBLE)
    { "Logistic operation supported only for FLOAT and DOUBLE tensors, actual type is ${input.type}" }

    return when(input.type) {
        DataType.FLOAT -> logisticFloat(input as FloatNDArray, dest as MutableFloatNDArray)
        DataType.DOUBLE ->  logisticDouble(input as DoubleNDArray, dest as MutableDoubleNDArray)
        else -> error("Logistic operation supported only for FLOAT and DOUBLE tensors, actual type is ${input.type}")
    }
}
