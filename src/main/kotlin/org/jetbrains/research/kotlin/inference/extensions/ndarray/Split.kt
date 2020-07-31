package org.jetbrains.research.kotlin.inference.extensions.ndarray

import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import kotlin.math.ceil

inline fun <reified T> NDArray<T>.computeSplitShape(axis: Int, split: Int, keepDims: Boolean): IntArray {
    val newShape: IntArray
    if (keepDims) {
        newShape = strides.shape.copyOf()
        newShape[axis] = split
    } else {
        newShape = IntArray(strides.shape.size - 1)
        strides.shape.copyInto(newShape, 0, 0, axis)
        strides.shape.copyInto(newShape, axis, axis + 1)
    }
    return newShape
}

inline fun <reified T> NDArray<T>.copySplitFragment(dst: NDArray<T>, srcOffset: Int, numFragments: Int, sliceLen: Int, fragOffset: Int, fragSize: Int): NDArray<T> {
    if (fragOffset == sliceLen && fragSize == sliceLen) {
        val slice = this.slice(numFragments * sliceLen, 0)
        return dst.apply { placeAll(0, slice) }
    }

    repeat(numFragments) {
        dst.placeAll(it * sliceLen, slice(sliceLen, srcOffset + fragOffset * it))
    }
    return dst
}

inline fun <reified T> NDArray<T>.splitWithAxis(parts: Int, axis: Int = 0, keepDims: Boolean = true): List<NDArray<T>> {
    require(axis in shape.indices) { "Index $axis out of shape bound: (0, ${rank - 1}" }
    val actualAxis = indexAxis(axis)
    val elementsByIndex = shape[actualAxis]
    val mainSplit = ceil(elementsByIndex.toDouble() / parts).toInt()
    val split = IntArray(parts) { mainSplit }

    val tail = elementsByIndex % parts
    if (tail != 0) split[parts - 1] = tail

    return splitWithAxis(split, actualAxis, keepDims).toList()
}

inline fun <reified T> NDArray<T>.splitWithAxis(split: IntArray, axis: Int, keepDims: Boolean = true): Array<NDArray<T>> {
    val beforeAxisDims = computeBlockSize(toDim = axis)
    val fromDimsAxis = computeBlockSize(fromDim = axis)
    val afterAxisDims = if (axis + 1 == rank) 1 else computeBlockSize(fromDim = axis + 1)

    var inputOffset = 0

    return Array(split.size) { i ->
        val splitSize = split[i]
        val outputDims = computeSplitShape(axis, split[i], keepDims)
        val dst = allocateNDArray(type, Strides(outputDims)) as NDArray<T>
        val fragmentSize = splitSize * afterAxisDims
        copySplitFragment(dst, inputOffset, beforeAxisDims, fragmentSize, fromDimsAxis, fragmentSize)
        inputOffset += fragmentSize
        dst
    }
}
