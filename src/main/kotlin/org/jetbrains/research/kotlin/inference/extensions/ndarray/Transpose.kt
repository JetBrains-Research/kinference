package org.jetbrains.research.kotlin.inference.extensions.ndarray

import org.jetbrains.research.kotlin.inference.data.ndarray.*
import org.jetbrains.research.kotlin.inference.data.tensors.Strides

fun transpose(array: FloatArray, strides: Strides, permutation: IntArray): FloatNDArray {
    val newArray = FloatArray(array.size)

    val newShape = IntArray(strides.shape.size)
    for ((i, axis) in permutation.withIndex()) {
        newShape[i] = strides.shape[axis]
    }
    val newStrides = Strides(newShape)


    for (i in newArray.indices) {
        val indices = newStrides.index(i)
        val newIndices = IntArray(indices.size)
        for ((id, axis) in permutation.withIndex()) {
            newIndices[axis] = indices[id]
        }
        newArray[i] = array[strides.offset(newIndices)]
    }


    return FloatNDArray(newArray, newStrides)
}

fun transpose(array: DoubleArray, strides: Strides, permutation: IntArray): DoubleNDArray {
    val newArray = DoubleArray(array.size)

    val newShape = IntArray(strides.shape.size)
    for ((i, axis) in permutation.withIndex()) {
        newShape[i] = strides.shape[axis]
    }
    val newStrides = Strides(newShape)


    for (i in newArray.indices) {
        val indices = newStrides.index(i)
        val newIndices = IntArray(indices.size)
        for ((id, axis) in permutation.withIndex()) {
            newIndices[axis] = indices[id]
        }
        newArray[i] = array[strides.offset(newIndices)]
    }


    return DoubleNDArray(array, newStrides)
}

fun transpose(array: IntArray, strides: Strides, permutation: IntArray): IntNDArray {
    val newArray = IntArray(array.size)

    val newShape = IntArray(strides.shape.size)
    for ((i, axis) in permutation.withIndex()) {
        newShape[i] = strides.shape[axis]
    }
    val newStrides = Strides(newShape)


    for (i in newArray.indices) {
        val indices = newStrides.index(i)
        val newIndices = IntArray(indices.size)
        for ((id, axis) in permutation.withIndex()) {
            newIndices[axis] = indices[id]
        }
        newArray[i] = array[strides.offset(newIndices)]
    }


    return IntNDArray(newArray, newStrides)
}

fun transpose(array: LongArray, strides: Strides, permutation: IntArray): LongNDArray {
    val newArray = LongArray(array.size)

    val newShape = IntArray(strides.shape.size)
    for ((i, axis) in permutation.withIndex()) {
        newShape[i] = strides.shape[axis]
    }
    val newStrides = Strides(newShape)


    for (i in newArray.indices) {
        val indices = newStrides.index(i)
        val newIndices = IntArray(indices.size)
        for ((id, axis) in permutation.withIndex()) {
            newIndices[axis] = indices[id]
        }
        newArray[i] = array[strides.offset(newIndices)]
    }


    return LongNDArray(newArray, newStrides)
}

fun transpose(array: ShortArray, strides: Strides, permutation: IntArray): ShortNDArray {
    val newArray = ShortArray(array.size)

    val newShape = IntArray(strides.shape.size)
    for ((i, axis) in permutation.withIndex()) {
        newShape[i] = strides.shape[axis]
    }
    val newStrides = Strides(newShape)


    for (i in newArray.indices) {
        val indices = newStrides.index(i)
        val newIndices = IntArray(indices.size)
        for ((id, axis) in permutation.withIndex()) {
            newIndices[axis] = indices[id]
        }
        newArray[i] = array[strides.offset(newIndices)]
    }


    return ShortNDArray(newArray, newStrides)
}

fun NDArray.transpose(permutation: IntArray): NDArray {
    return when (array) {
        is IntArray -> transpose(array, strides, permutation)
        is FloatArray -> transpose(array, strides, permutation)
        is ShortArray -> transpose(array, strides, permutation)
        is DoubleArray -> transpose(array, strides, permutation)
        is LongArray -> transpose(array, strides, permutation)
        else -> throw UnsupportedOperationException()
    }
}
