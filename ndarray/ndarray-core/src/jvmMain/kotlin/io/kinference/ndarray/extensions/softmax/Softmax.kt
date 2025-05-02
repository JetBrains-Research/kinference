package io.kinference.ndarray.extensions.softmax

import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType

suspend fun softmax(
    input: NumberNDArrayCore,
    axis: Int = -1,
): MutableNumberNDArrayCore {
    require(input.type == DataType.FLOAT || input.type == DataType.DOUBLE)
        { "Softmax operation supported only for FLOAT and DOUBLE tensors, actual type is ${input.type}" }

    val actualAxis = input.indexAxis(axis)
    val rows = input.computeBlockSize(toDim = actualAxis)
    val columns = input.computeBlockSize(fromDim = actualAxis)
    val stride = if(actualAxis == input.rank - 1) 1 else input.computeBlockSize(fromDim = actualAxis + 1)

    return when(input.type) {
        DataType.FLOAT -> when(stride) {
            1 -> softmaxVer1Float(input as FloatNDArray, rows, columns)
            else -> softmaxVer13Float(input as FloatNDArray, rows, columns, stride)
        }
        DataType.DOUBLE -> when(stride) {
            1 -> softmaxVer1Double(input as DoubleNDArray, rows, columns)
            else -> softmaxVer13Double(input as DoubleNDArray, rows, columns, stride)
        }
        else -> error("Softmax operation supported only for FLOAT and DOUBLE tensors, actual type is ${input.type}")
    }
}

suspend fun softmax(
    input: NumberNDArrayCore,
    dest: MutableNumberNDArrayCore,
    axis: Int = -1,
): MutableNumberNDArrayCore {
    require(input.type == dest.type)
    require(input.type == DataType.FLOAT || input.type == DataType.DOUBLE)
    { "Softmax operation supported only for FLOAT and DOUBLE tensors, actual type is ${input.type}" }

    val actualAxis = input.indexAxis(axis)
    val rows = input.computeBlockSize(toDim = actualAxis)
    val columns = input.computeBlockSize(fromDim = actualAxis)
    val stride = if(actualAxis == input.rank - 1) 1 else input.computeBlockSize(fromDim = actualAxis + 1)

    return when(input.type) {
        DataType.FLOAT -> when(stride) {
            1 -> softmaxVer1Float(input as FloatNDArray, dest as MutableFloatNDArray, rows, columns)
            else -> softmaxVer13Float(input as FloatNDArray, dest as MutableFloatNDArray, rows, columns, stride)
        }
        DataType.DOUBLE -> when(stride) {
            1 -> softmaxVer1Double(input as DoubleNDArray, dest as MutableDoubleNDArray, rows, columns)
            else -> softmaxVer13Double(input as DoubleNDArray, dest as MutableDoubleNDArray, rows, columns, stride)
        }
        else -> error("Softmax operation supported only for FLOAT and DOUBLE tensors, actual type is ${input.type}")
    }
}

suspend fun softmaxVer1(
    input: NumberNDArrayCore,
    axis: Int = 0,
): MutableNumberNDArrayCore {
    require(input.type == DataType.FLOAT || input.type == DataType.DOUBLE)
        { "Softmax operation supported only for FLOAT and DOUBLE tensors, actual type is ${input.type}" }

    val actualAxis = input.indexAxis(axis)
    val rows = input.computeBlockSize(toDim = actualAxis)
    val columns = input.computeBlockSize(fromDim = actualAxis)

    return when(input.type) {
        DataType.FLOAT -> softmaxVer1Float(input as FloatNDArray, rows, columns)
        DataType.DOUBLE ->  softmaxVer1Double(input as DoubleNDArray, rows, columns)
        else -> error("Softmax operation supported only for FLOAT and DOUBLE tensors, actual type is ${input.type}")
    }
}

suspend fun softmaxVer1(
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
        DataType.FLOAT -> softmaxVer1Float(input as FloatNDArray, dest as MutableFloatNDArray, rows, columns)
        DataType.DOUBLE ->  softmaxVer1Double(input as DoubleNDArray, dest as MutableDoubleNDArray, rows, columns)
        else -> error("Softmax operation supported only for FLOAT and DOUBLE tensors, actual type is ${input.type}")
    }
}
