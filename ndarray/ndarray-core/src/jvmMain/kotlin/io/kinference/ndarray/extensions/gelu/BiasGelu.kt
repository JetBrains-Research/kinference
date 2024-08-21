package io.kinference.ndarray.extensions.gelu

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

suspend fun biasGelu(input: NumberNDArrayCore, bias: NumberNDArrayCore): MutableNumberNDArrayCore {
    require(input.type == bias.type)
        { "Input and Bias types should be equal, actual input type is ${input.type}, actual bias type is ${bias.type}" }

    require(input.type == DataType.FLOAT || input.type == DataType.DOUBLE)
        { "BiasGelu operation supported only for FLOAT and DOUBLE tensors, actual types is ${input.type}" }

    return when(input.type) {
        DataType.FLOAT -> computeGeluFloat(input as FloatNDArray, bias as FloatNDArray)
        DataType.DOUBLE -> computeGeluDouble(input as DoubleNDArray, bias as DoubleNDArray)
        else -> error("BiasGelu operation supported only for FLOAT and DOUBLE tensors, actual types is ${input.type}")
    }
}

suspend fun biasGelu(input: NumberNDArrayCore, bias: NumberNDArrayCore, dest: MutableNumberNDArrayCore): MutableNumberNDArrayCore {
    require(input.type == bias.type)
        { "Input and Bias types should be equal, actual input type is ${input.type}, actual bias type is ${bias.type}" }

    require(input.type == DataType.FLOAT || input.type == DataType.DOUBLE)
        { "BiasGelu operation supported only for FLOAT and DOUBLE tensors, actual types is ${input.type}" }

    return when(input.type) {
        DataType.FLOAT -> computeGeluFloat(input as FloatNDArray, bias as FloatNDArray, dest as MutableFloatNDArray)
        DataType.DOUBLE -> computeGeluDouble(input as DoubleNDArray, bias as DoubleNDArray, dest as MutableDoubleNDArray)
        else -> error("BiasGelu operation supported only for FLOAT and DOUBLE tensors, actual types is ${input.type}")
    }
}
