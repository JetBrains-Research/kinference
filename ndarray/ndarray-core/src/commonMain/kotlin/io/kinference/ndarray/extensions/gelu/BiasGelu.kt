package io.kinference.ndarray.extensions.gelu

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

suspend fun biasGelu(input: NumberNDArrayCore, bias: NumberNDArrayCore): MutableNumberNDArrayCore {
    require(
        (input.type == DataType.FLOAT && bias.type == DataType.FLOAT) ||
            (input.type == DataType.DOUBLE && bias.type == DataType.DOUBLE)
    ) { "BiasGelu operation supported only for DoubleNDArray and FloatNDArray" }

    return when(input.type) {
        DataType.FLOAT -> computeGeluFloat(input as FloatNDArray, bias as FloatNDArray)
        DataType.DOUBLE -> computeGeluDouble(input as DoubleNDArray, bias as DoubleNDArray)
        else -> error("BiasGelu operation supported only for DoubleNDArray and FloatNDArray")
    }
}
