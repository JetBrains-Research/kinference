package org.jetbrains.research.kotlin.inference.extensions.ndarray

import org.jetbrains.research.kotlin.inference.data.ndarray.*
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.onnx.TensorProto
import kotlin.math.ceil

fun computeSplitShape(shape: IntArray, axis: Int, split: Int, keepDims: Boolean): IntArray {
    val newShape: IntArray
    if (keepDims) {
        newShape = shape.copyOf()
        newShape[axis] = split
    } else {
        newShape = IntArray(shape.size - 1)
        shape.copyInto(newShape, 0, 0, axis)
        shape.copyInto(newShape, axis, axis + 1)
    }
    return newShape
}


fun <T> NDArray<T>.splitWithAxis(parts: Int, axis: Int = 0, keepDims: Boolean = true): List<NDArray<T>> {
    require(axis in shape.indices) { "Index $axis out of shape bound: (0, ${rank - 1}" }
    val actualAxis = indexAxis(axis)
    val elementsByIndex = shape[actualAxis]
    val mainSplit = ceil(elementsByIndex.toDouble() / parts).toInt()
    val split = IntArray(parts) { mainSplit }

    val tail = elementsByIndex % parts
    if (tail != 0) split[parts - 1] = tail

    return splitWithAxis(split, actualAxis, keepDims)
}

fun <T> NDArray<T>.splitWithAxis(split: IntArray, axis: Int, keepDims: Boolean = true): List<NDArray<T>> {
    val beforeAxisDims = computeBlockSize(toDim = axis)
    val fromAxisDims = computeBlockSize(fromDim = axis)
    val afterAxisDims = if (axis + 1 == rank) 1 else computeBlockSize(fromDim = axis + 1)

    var inputOffset = 0

    return List(split.size) { i ->
        val splitSize = split[i]
        val outputDims = computeSplitShape(strides.shape, axis, split[i], keepDims)
        val outStrides = Strides(outputDims)
        val fragmentSize = splitSize * afterAxisDims

        val dst = this.splitFragment(beforeAxisDims, fromAxisDims, fragmentSize, outStrides, inputOffset)
        inputOffset += fragmentSize
        dst
    }
}

fun <T> NDArray<T>.splitFragment(beforeAxisDims: Int, fromAxisDims: Int, fragmentSize: Int, splitStrides: Strides, offset: Int): NDArray<T> {
    return when (type) {
        TensorProto.DataType.DOUBLE -> DoubleNDArray((array as DoubleArray).copySplitFragment(offset, beforeAxisDims, fragmentSize, fromAxisDims), splitStrides)
        TensorProto.DataType.FLOAT -> FloatNDArray((array as FloatArray).copySplitFragment(offset, beforeAxisDims, fragmentSize, fromAxisDims), splitStrides)
        TensorProto.DataType.INT64 -> LongNDArray((array as LongArray).copySplitFragment(offset, beforeAxisDims, fragmentSize, fromAxisDims), splitStrides)
        TensorProto.DataType.INT32 -> IntNDArray((array as IntArray).copySplitFragment(offset, beforeAxisDims, fragmentSize, fromAxisDims), splitStrides)
        TensorProto.DataType.INT16 -> ShortNDArray((array as ShortArray).copySplitFragment(offset, beforeAxisDims, fragmentSize, fromAxisDims), splitStrides)
        else -> error("")
    } as NDArray<T>
}

fun <T> NDArray<T>.splitArray(parts: Int, strides: Strides): List<NDArray<T>> {
    return when (array) {
        is FloatArray -> splitParts(array, parts, strides)
        else -> throw UnsupportedOperationException()
    } as List<NDArray<T>>
}
