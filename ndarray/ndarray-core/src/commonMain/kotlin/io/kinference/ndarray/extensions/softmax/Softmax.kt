package io.kinference.ndarray.extensions.softmax

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

suspend fun softmax(
    input: NumberNDArrayCore,
    axis: Int = 0,
): MutableNumberNDArrayCore {
    require(input.type == DataType.FLOAT || input.type == DataType.DOUBLE)
        { "Softmax operation supported only for FLOAT and DOUBLE tensors, actual type is ${input.type}" }

    val actualAxis = input.indexAxis(axis)
    val rows = input.computeBlockSize(toDim = actualAxis)
    val columns = input.computeBlockSize(fromDim = actualAxis)

    return when(input.type) {
        DataType.FLOAT -> softmaxFloat(input as FloatNDArray, rows, columns)
        DataType.DOUBLE ->  softmaxDouble(input as DoubleNDArray, rows, columns)
        else -> error("Softmax operation supported only for FLOAT and DOUBLE tensors, actual type is ${input.type}")
    }
}

suspend fun softmax(
    input: NumberNDArrayCore,
    dest: MutableNumberNDArrayCore,
    axis: Int = 0,
): MutableNumberNDArrayCore {
    require(input.type == dest.type)
    require(input.type == DataType.FLOAT || input.type == DataType.DOUBLE)
    { "Softmax operation supported only for FLOAT and DOUBLE tensors, actual type is ${input.type}" }

    val actualAxis = input.indexAxis(axis)
    val rows = input.computeBlockSize(toDim = actualAxis)
    val columns = input.computeBlockSize(fromDim = actualAxis)

    return when(input.type) {
        DataType.FLOAT -> softmaxFloat(input as FloatNDArray, dest as MutableFloatNDArray, rows, columns)
        DataType.DOUBLE ->  softmaxDouble(input as DoubleNDArray, dest as MutableDoubleNDArray, rows, columns)
        else -> error("Softmax operation supported only for FLOAT and DOUBLE tensors, actual type is ${input.type}")
    }
}
