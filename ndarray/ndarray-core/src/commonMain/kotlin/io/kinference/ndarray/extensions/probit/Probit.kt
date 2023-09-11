package io.kinference.ndarray.extensions.probit

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

suspend fun probit(
    input: NumberNDArrayCore,
): MutableNumberNDArrayCore {
    require(input.type == DataType.FLOAT || input.type == DataType.DOUBLE)
    { "Probit operation supported only for FLOAT and DOUBLE tensors, actual type is ${input.type}" }

    return when(input.type) {
        DataType.FLOAT -> probitFloat(input as FloatNDArray)
        DataType.DOUBLE ->  probitDouble(input as DoubleNDArray)
        else -> error("Probit operation supported only for FLOAT and DOUBLE tensors, actual type is ${input.type}")
    }
}

suspend fun probit(
    input: NumberNDArrayCore,
    dest: MutableNumberNDArrayCore,
): MutableNumberNDArrayCore {
    require(input.type == dest.type)
    require(input.type == DataType.FLOAT || input.type == DataType.DOUBLE)
    { "Probit operation supported only for FLOAT and DOUBLE tensors, actual type is ${input.type}" }

    return when(input.type) {
        DataType.FLOAT -> probitFloat(input as FloatNDArray, dest as MutableFloatNDArray)
        DataType.DOUBLE ->  probitDouble(input as DoubleNDArray, dest as MutableDoubleNDArray)
        else -> error("Probit operation supported only for FLOAT and DOUBLE tensors, actual type is ${input.type}")
    }
}
