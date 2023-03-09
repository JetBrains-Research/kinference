package io.kinference.ndarray.extensions.gelu

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

suspend fun fastGelu(input: NumberNDArrayCore, bias: NumberNDArrayCore?): MutableNumberNDArrayCore {
    require(bias == null || input.type == bias.type)
        { "Input and bias must have the same data type. Input data type: ${input.type}, bias data type: ${bias?.type}" }

    require(
        (input.type == DataType.FLOAT && (bias == null || bias.type == DataType.FLOAT)) ||
            (input.type == DataType.DOUBLE && (bias == null || bias.type == DataType.DOUBLE))
    ) { "FastGelu operation supports only Double and Float type arrays. Input data type: ${input.type}, bias data type: ${bias?.type}" }

    return when(input.type) {
        DataType.FLOAT -> fastGeluFloat(input as FloatNDArray, bias as FloatNDArray?)
        DataType.DOUBLE -> fastGeluDouble(input as DoubleNDArray, bias as DoubleNDArray?)
        else -> error { "FastGelu operation supports only Double and Float type arrays. Input data type: ${input.type}, bias data type: ${bias?.type}" }
    }
}
